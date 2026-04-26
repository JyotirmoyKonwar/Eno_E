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
        self.archive = EpisodeArchive(max_size=cfg.training.replay_buffer_size)
        self.league = LeaguePools()
        self.elo = EloRatings()
        self.best_holdout = float("-inf")
        self.rollback_events = 0
        self.baseline: BaselineJudge = EnoEBaselineJudge() if cfg.training.use_baseline else NoOpBaselineJudge()
        self.metrics: list[dict[str, Any]] = []
        self.wandb, self.wandb_status = init_wandb(
            cfg.connection.wandb_token,
            cfg.training.use_wandb,
            project="darkguard-selfplay-trainer",
            config=cfg.as_dict(),
        )

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

    def _train_consumer_phase(self, round_idx: int) -> dict[str, float]:
        _ = round_idx
        rewards: list[float] = []
        safe: list[float] = []
        invalid: list[float] = []
        fp: list[float] = []
        for _ in range(self.cfg.training.consumer_steps_per_round):
            if self._stop_requested():
                self.log("Stop requested during consumer phase.")
                break
            opponent = sample_opponent(self.elo.consumer, self.league.designer_pool, self.rng)
            if opponent:
                self.log(f"Consumer sampled designer opponent: {opponent.name} ({opponent.elo:.1f})")
            reset_payload = {"task_id": "custom_episode", "seed": self.rng.randint(1, 999999)}
            result = run_consumer_episode(self.env, self.consumer, reset_payload, max_steps=24)
            if "error" in result.trace:
                self.log(f"Consumer rollout warning: {result.trace['error']}")
            rewards.append(result.total_reward)
            safe.append(1.0 if result.safe_completion else 0.0)
            invalid.append(result.invalid_action_rate)
            fp.append(result.false_positive_rate)
            self.archive.add(result.trace)
            self.consumer.improve(0.01)
        return {
            "mean_reward": mean(rewards) if rewards else 0.0,
            "safe_rate": mean(safe) if safe else 0.0,
            "invalid_rate": mean(invalid) if invalid else 0.0,
            "fp_rate": mean(fp) if fp else 0.0,
        }

    def _train_designer_phase(self, round_idx: int) -> dict[str, float]:
        _ = round_idx
        rewards: list[float] = []
        validity: list[float] = []
        challenge: list[float] = []
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

                if phase == "train_consumer" and not self.cfg.training.freeze_consumer:
                    row.update(self._train_consumer_phase(round_idx))
                elif phase == "train_designer" and not self.cfg.training.freeze_designer:
                    row.update(self._train_designer_phase(round_idx))
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
                normalized = 1.0 if phase_score > 0 else 0.0
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
