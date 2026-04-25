"""Main self-play training engine."""

from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from .baseline import BaselineJudge, EnoEBaselineJudge, NoOpBaselineJudge
from .checkpointing import save_checkpoint, write_frozen_registry
from .config import AppConfig
from .curriculum import (
    bootstrap_easy_tasks_active,
    curriculum_gate_passed,
    curriculum_live_summary,
    get_designer_training_mode,
    pick_consumer_task_id,
    resolve_gate_metric_values,
    sample_curriculum_designer_opponent,
    should_use_weak_designer_pool,
    validate_curriculum_ranges,
)
from .dataset_utils import EpisodeArchive
from .elo import EloRatings, update_elo
from .env_client import RemoteEnvClient
from .evaluation import run_holdout_eval
from .hf_utils import ensure_hf_login
from .model_utils import PolicyModel, load_policy
from .rollout import evaluate_designer_episode, generate_designer_episode, run_consumer_episode
from .selfplay import LeaguePools, OpponentEntry, phase_for_round, sample_opponent
from .state_store import load_state, save_state
from .ui_state import StateHub
from .wandb_utils import init_wandb


def _write_metrics_csv(path: Path, metrics: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not metrics:
        return
    keys = sorted({k for row in metrics for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=keys)
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)


class TrainerEngine:
    def __init__(self, cfg: AppConfig, state_hub: StateHub):
        self.cfg = cfg
        self.state_hub = state_hub
        self.cfg.paths.ensure()
        self.rng = random.Random(cfg.training.seed)
        self.env = RemoteEnvClient(
            base_url=cfg.connection.env_base_url,
            timeout_s=cfg.connection.timeout_s,
            hf_token=cfg.connection.hf_token,
            max_retries=cfg.connection.max_retries,
            min_request_interval_s=cfg.connection.min_request_interval_s,
            retry_backoff_s=cfg.connection.retry_backoff_s,
        )
        self.consumer = load_policy("consumer", cfg.models.consumer_base_model, cfg.models.consumer_adapter_repo, cfg.models.consumer_checkpoint_override)
        self.designer = load_policy("designer", cfg.models.designer_base_model, cfg.models.designer_adapter_repo, cfg.models.designer_checkpoint_override)
        self.consumer.router_enabled = cfg.training.use_local_action_router
        self.designer.router_enabled = cfg.training.use_local_action_router
        self.archive = EpisodeArchive(max_size=cfg.training.replay_buffer_size)
        self.league = LeaguePools()
        self.elo = EloRatings()
        self.best_holdout = float("-inf")
        self.rollback_events = 0
        self.baseline: BaselineJudge = EnoEBaselineJudge() if cfg.training.use_baseline else NoOpBaselineJudge()
        self.metrics: list[dict[str, Any]] = []
        self._latest_train_safe_rate: float | None = None
        self._latest_train_mean_reward: float | None = None
        self._latest_eval_safe_rate: float | None = None
        self._latest_eval_reward: float | None = None
        self._latest_eval_baseline_reward: float = 0.0
        self._eval_history: list[tuple[float, float]] = []
        self._alternate_designer_phase_counter: int = 0
        self.wandb, self.wandb_status = init_wandb(
            cfg.connection.wandb_token,
            cfg.training.use_wandb,
            project="darkguard-selfplay-trainer",
            config=cfg.as_dict(),
        )
        for err in validate_curriculum_ranges(cfg.training.curriculum):
            self.log(f"[curriculum config] {err}")

    def log(self, msg: str) -> None:
        ts = datetime.now(UTC).strftime("%H:%M:%S")
        self.state_hub.append_log(f"[{ts}] {msg}")

    def _stop_requested(self) -> bool:
        return self.state_hub.snapshot().stop_requested

    def _record_metric(self, payload: dict[str, Any]) -> None:
        self.metrics.append(payload)
        self.state_hub.append_metric(payload)
        self.wandb.log(payload, step=int(payload.get("round", 0)))

    def _save_state(self, round_idx: int, phase: str) -> None:
        save_state(
            self.cfg.paths.state_file,
            {
                "round": round_idx,
                "phase": phase,
                "elo": asdict(self.elo),
                "rollback_events": self.rollback_events,
                "metrics_tail": self.metrics[-20:],
            },
        )

    def _stability_guard(self, eval_reward: float) -> bool:
        if self.best_holdout == float("-inf"):
            self.best_holdout = eval_reward
            return True
        degrade = (self.best_holdout - eval_reward) / max(1e-6, abs(self.best_holdout) + 1e-6)
        if degrade > self.cfg.training.rollback_threshold:
            self.rollback_events += 1
            return False
        self.best_holdout = max(self.best_holdout, eval_reward)
        return True

    def _format_gate_log(self, row: dict[str, Any]) -> str:
        cur = self.cfg.training.curriculum
        parts: list[str] = []
        if cur.require_safe_rate_gate:
            t = cur.safe_rate_threshold
            v = float(row.get("gate_safe_rate_value", 0.0))
            parts.append(f"safe_rate({v:.3f})>{t}")
        if cur.require_eval_reward_gate:
            cmp_ = ">" if cur.eval_reward_must_exceed_baseline else ">="
            v = float(row.get("gate_eval_reward_value", 0.0))
            b = self._latest_eval_baseline_reward
            parts.append(f"eval_reward({v:.3f}){cmp_}baseline({b:.3f})")
        if not parts:
            return ""
        joiner = f" {cur.gate_combiner} "
        return joiner.join(parts)

    def _resolve_curriculum_row(
        self,
        round_idx: int,
        phase: str,
    ) -> dict[str, Any]:
        """Snapshot of curriculum state for logging/metrics (before phase runs)."""
        cur = self.cfg.training.curriculum
        sr_v, er_v = resolve_gate_metric_values(
            cur,
            self._latest_train_safe_rate,
            self._latest_train_mean_reward,
            self._latest_eval_safe_rate,
            self._latest_eval_reward,
            self._eval_history,
        )
        gates = True
        if cur.enable_curriculum:
            gates = curriculum_gate_passed(
                sr_v,
                er_v,
                self._latest_eval_baseline_reward,
                cur,
            )
        else:
            gates = True
        mode = get_designer_training_mode(round_idx, cur, gates)
        weak_consumer = should_use_weak_designer_pool(mode, cur)
        easy = bootstrap_easy_tasks_active(round_idx, mode, gates, cur)
        designer_train = False
        skip_reason = ""
        if phase == "train_designer" and not self.cfg.training.freeze_designer:
            if not cur.enable_curriculum:
                designer_train = True
            elif mode == "frozen":
                designer_train = False
                skip_reason = "curriculum_frozen"
            elif mode == "alternate":
                self._alternate_designer_phase_counter += 1
                n = max(1, cur.designer_train_every_n_designer_phases)
                designer_train = (self._alternate_designer_phase_counter % n) == 0
                if not designer_train:
                    skip_reason = "schedule"
            else:
                designer_train = True
        elif phase == "train_designer" and self.cfg.training.freeze_designer:
            designer_train = False
            skip_reason = "global_freeze_designer"
        return {
            "curriculum_mode": mode,
            "designer_training_enabled": designer_train,
            "curriculum_gate_passed": gates,
            "gate_safe_rate_value": sr_v,
            "gate_eval_reward_value": er_v,
            "bootstrap_easy_tasks_policy": easy,
            "weak_pool_for_consumer": weak_consumer,
            "curriculum_skip_reason": skip_reason,
        }

    def _train_consumer_phase(self, round_idx: int, *, weak_pool: bool, easy_bootstrap: bool) -> dict[str, float]:
        _ = round_idx
        cur = self.cfg.training.curriculum
        rewards: list[float] = []
        safe: list[float] = []
        invalid: list[float] = []
        fp: list[float] = []
        sampled_from_weak_policy = False
        used_easy = False
        for _ in range(self.cfg.training.consumer_steps_per_round):
            if self._stop_requested():
                self.log("Stop requested during consumer phase.")
                break
            if weak_pool:
                opponent, sampled_from_weak_policy = sample_curriculum_designer_opponent(
                    self.elo.consumer, self.league.designer_pool, self.rng, cur, weak_mode=True
                )
            else:
                opponent = sample_opponent(self.elo.consumer, self.league.designer_pool, self.rng)
            if opponent:
                self.log(f"Consumer sampled designer opponent: {opponent.name} ({opponent.elo:.1f})")
            task_id = pick_consumer_task_id(self.rng, cur, easy_bootstrap)
            if easy_bootstrap:
                used_easy = True
            reset_payload = {"task_id": task_id, "seed": self.rng.randint(1, 999999)}
            result = run_consumer_episode(self.env, self.consumer, reset_payload, max_steps=24)
            if "error" in result.trace:
                self.log(f"Consumer rollout warning: {result.trace['error']}")
            rewards.append(result.total_reward)
            safe.append(1.0 if result.safe_completion else 0.0)
            invalid.append(result.invalid_action_rate)
            fp.append(result.false_positive_rate)
            self.archive.add(result.trace)
            self.consumer.improve(0.01)
        self._latest_train_safe_rate = mean(safe) if safe else 0.0
        self._latest_train_mean_reward = mean(rewards) if rewards else 0.0
        return {
            "mean_reward": self._latest_train_mean_reward,
            "safe_rate": self._latest_train_safe_rate,
            "invalid_rate": mean(invalid) if invalid else 0.0,
            "fp_rate": mean(fp) if fp else 0.0,
            "weak_pool_sampling_used": bool(weak_pool and sampled_from_weak_policy),
            "bootstrap_easy_tasks_only": used_easy,
        }

    def _train_designer_phase(self, round_idx: int, *, do_improve: bool) -> dict[str, float]:
        _ = round_idx
        rewards: list[float] = []
        validity: list[float] = []
        challenge: list[float] = []
        cur = self.cfg.training.curriculum
        if not do_improve:
            self.log("Designer frozen by curriculum: skipping optimizer steps this phase.")
        for _ in range(self.cfg.training.designer_steps_per_round):
            if self._stop_requested():
                self.log("Stop requested during designer phase.")
                break
            opponent = sample_opponent(self.elo.designer, self.league.consumer_pool, self.rng)
            if opponent:
                self.log(f"Designer sampled consumer opponent: {opponent.name} ({opponent.elo:.1f})")
            episode_cfg, _prompt = generate_designer_episode(self.designer, self.rng)
            novelty = self.rng.uniform(0.0, 0.3)
            score, trace = evaluate_designer_episode(self.env, self.consumer, episode_cfg, novelty=novelty)
            if "error" in trace.get("rollout", {}):
                self.log(f"Designer eval warning: {trace['rollout']['error']}")
            rewards.append(score)
            validity.append(1.0 if episode_cfg else 0.0)
            challenge.append(float(trace.get("challenge_delta", 0.0)))
            if do_improve:
                self.designer.improve(0.008 if score > 0 else -0.005)
        return {
            "designer_reward": mean(rewards) if rewards else 0.0,
            "designer_validity": mean(validity) if validity else 0.0,
            "designer_challenge": mean(challenge) if challenge else 0.0,
        }

    def _snapshot_phase(self, round_idx: int, metric: dict[str, float]) -> None:
        cons_score = float(metric.get("mean_reward", 0.0))
        des_score = float(metric.get("designer_reward", 0.0))
        cons_meta = save_checkpoint(self.cfg.paths.checkpoints_dir, "consumer", round_idx, {"skill_bias": self.consumer.skill_bias}, cons_score)
        des_meta = save_checkpoint(self.cfg.paths.checkpoints_dir, "designer", round_idx, {"skill_bias": self.designer.skill_bias}, des_score)
        self.league.add(OpponentEntry(name=cons_meta.name, elo=self.elo.consumer, role="consumer", checkpoint=cons_meta.path))
        self.league.add(OpponentEntry(name=des_meta.name, elo=self.elo.designer, role="designer", checkpoint=des_meta.path))
        write_frozen_registry(
            self.cfg.paths.frozen_registry,
            {
                "consumer_pool": [asdict(x) for x in self.league.consumer_pool],
                "designer_pool": [asdict(x) for x in self.league.designer_pool],
            },
        )
        self.log(f"Snapshot saved: {cons_meta.name}, {des_meta.name}")

    def run(self) -> None:
        self.state_hub.update(running=True, stop_requested=False, active_phase="starting")
        try:
            self.log(ensure_hf_login(self.cfg.connection.hf_token))
            self.log(self.wandb_status)
            self.log(f"Curriculum summary:\n{curriculum_live_summary(self.cfg.training.curriculum)}")
            try:
                health = self.env.health()
                self.log(f"Environment connected: {health}")
            except Exception as exc:
                self.log(f"Environment connection failed: {exc}")
                self.state_hub.update(active_phase="error")
                return

            resume = load_state(self.cfg.paths.state_file)
            if resume:
                self.log(f"Loaded resume state: round={resume.get('round')} phase={resume.get('phase')}")

            for round_idx in range(1, self.cfg.training.total_rounds + 1):
                if self._stop_requested():
                    self.log("Stop requested. Exiting training loop safely.")
                    break
                phase = phase_for_round(round_idx)
                self.state_hub.update(current_round=round_idx, active_phase=phase)
                self.log(f"Round {round_idx}/{self.cfg.training.total_rounds} phase={phase}")
                row: dict[str, Any] = {"round": round_idx, "phase": phase, "ts": datetime.now(UTC).isoformat()}
                cur_snap = self._resolve_curriculum_row(round_idx, phase)
                row.update(
                    {
                        "curriculum_mode": cur_snap["curriculum_mode"],
                        "designer_training_enabled": cur_snap["designer_training_enabled"],
                        "curriculum_gate_passed": cur_snap["curriculum_gate_passed"],
                        "gate_safe_rate_value": cur_snap["gate_safe_rate_value"],
                        "gate_eval_reward_value": cur_snap["gate_eval_reward_value"],
                        "bootstrap_easy_tasks_only": False,
                        "weak_pool_sampling_used": False,
                        "curriculum_skip_reason": cur_snap["curriculum_skip_reason"],
                    }
                )

                if phase == "train_consumer" and not self.cfg.training.freeze_consumer:
                    cons_extra = self._train_consumer_phase(
                        round_idx,
                        weak_pool=cur_snap["weak_pool_for_consumer"],
                        easy_bootstrap=cur_snap["bootstrap_easy_tasks_policy"],
                    )
                    row.update({k: v for k, v in cons_extra.items() if k in ("mean_reward", "safe_rate", "invalid_rate", "fp_rate")})
                    row["weak_pool_sampling_used"] = cons_extra.get("weak_pool_sampling_used", False)
                    row["bootstrap_easy_tasks_only"] = cons_extra.get("bootstrap_easy_tasks_only", False)
                elif phase == "train_designer":
                    if self.cfg.training.freeze_designer:
                        self.log("Designer phase skipped: global freeze_designer is enabled.")
                        row["designer_training_enabled"] = False
                        row["curriculum_skip_reason"] = "global_freeze_designer"
                    else:
                        do_improve = bool(cur_snap["designer_training_enabled"])
                        if not do_improve and cur_snap["curriculum_skip_reason"] == "schedule":
                            self.log("Designer skipped by curriculum schedule (alternate mode).")
                        row.update(self._train_designer_phase(round_idx, do_improve=do_improve))
                        row["designer_training_enabled"] = do_improve
                elif phase == "evaluation":
                    eval_summary = run_holdout_eval(
                        self.env,
                        self.consumer,
                        self.baseline,
                        seeds=[101, 202, 303, 404, 505],
                        stop_checker=self._stop_requested,
                    )
                    if self._stop_requested():
                        self.log("Stop requested during evaluation phase.")
                        break
                    self._latest_eval_safe_rate = eval_summary.safe_rate
                    self._latest_eval_reward = eval_summary.mean_reward
                    self._latest_eval_baseline_reward = eval_summary.baseline_score
                    self._eval_history.append((eval_summary.safe_rate, eval_summary.mean_reward))
                    row.update(
                        {
                            "eval_reward": eval_summary.mean_reward,
                            "eval_safe_rate": eval_summary.safe_rate,
                            "eval_invalid_rate": eval_summary.invalid_rate,
                            "baseline_score": eval_summary.baseline_score,
                        }
                    )
                    ok = self._stability_guard(eval_summary.mean_reward)
                    if not ok:
                        self.consumer.improve(-0.05)
                        self.designer.improve(-0.02)
                        self.log("Rollback guard triggered: performance degraded on holdout.")
                else:
                    self._snapshot_phase(round_idx, row)

                if self._stop_requested():
                    self.log("Stop requested after phase completion.")
                    break

                # ELO updates against opposite role using latest phase score.
                phase_score = float(row.get("mean_reward", row.get("eval_reward", row.get("designer_reward", 0.0))))
                normalized = max(0.0, min(1.0, 0.5 + phase_score / 8.0))
                self.elo.consumer, self.elo.designer = update_elo(
                    self.elo.consumer, self.elo.designer, normalized, self.cfg.training.elo_k_factor
                )
                row.update(
                    {
                        "consumer_elo": self.elo.consumer,
                        "designer_elo": self.elo.designer,
                        "baseline_elo": self.elo.baseline,
                        "rollback_events": self.rollback_events,
                    }
                )
                gate_note = self._format_gate_log(row)
                self.log(
                    f"Round {round_idx}: curriculum_mode={row['curriculum_mode']}, "
                    f"designer_train={row.get('designer_training_enabled', False)}, "
                    f"weak_pool={row.get('weak_pool_sampling_used', False)}, "
                    f"easy_tasks_only={row.get('bootstrap_easy_tasks_only', False)}"
                    + (f", skip_reason={row.get('curriculum_skip_reason')}" if row.get("curriculum_skip_reason") else "")
                    + (f", gate={gate_note}" if gate_note else "")
                )
                self._record_metric(row)
                self._save_state(round_idx, phase)
                time.sleep(0.05)
        except Exception as exc:  # pragma: no cover - runtime safety net
            self.log(f"[FATAL] Trainer crashed: {exc}")
            self.state_hub.update(active_phase="error")
        finally:
            _write_metrics_csv(self.cfg.paths.metrics_csv, self.metrics)
            self.state_hub.update(
                running=False,
                stop_requested=False,
                active_phase="idle",
                artifacts={
                    "metrics_csv": str(self.cfg.paths.metrics_csv),
                    "checkpoints_dir": str(self.cfg.paths.checkpoints_dir),
                    "frozen_registry": str(self.cfg.paths.frozen_registry),
                    "wandb_run_url": self.wandb.run_url,
                },
            )
            self.wandb.finish()


def test_connection(cfg: AppConfig) -> str:
    env = RemoteEnvClient(cfg.connection.env_base_url, timeout_s=cfg.connection.timeout_s, hf_token=cfg.connection.hf_token)
    try:
        health = env.health()
        env.reset({"task_id": "easy_safe_signup", "seed": 1})
        return f"Connected. Health={json.dumps(health)}"
    except Exception as exc:
        return f"Connection failed: {exc}"
