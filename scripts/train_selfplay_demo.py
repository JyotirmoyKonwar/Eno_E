"""
DarkGuard self-play training demo runner.

This script is intentionally dataset-agnostic. It performs online rollouts in
the environment and logs training/eval metrics (optionally to Weights & Biases).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repo root is importable when run as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.darkguard.env import DarkGuardEnv
from src.darkguard.models import DarkGuardAction

TASK_IDS = ["easy_safe_signup", "medium_fair_checkout", "hard_cancel_maze"]


def _safe_import_wandb():
    try:
        import wandb  # type: ignore
    except Exception:
        return None
    return wandb


@dataclass
class EpisodeSummary:
    episode_idx: int
    task_id: str
    episode_score: float
    cumulative_reward: float
    steps: int
    consumer_elo: float
    designer_elo: float
    subtlety: int


def _extract_visible_ids(obs: Any) -> List[str]:
    return [e.get("id") for e in obs.elements if isinstance(e, dict) and e.get("id")]


def _choose_action(obs: Any, rng: random.Random) -> DarkGuardAction:
    """
    Simple baseline policy:
    - inspect first on likely trap elements
    - flag suspicious elements
    - toggle known traps off
    - click/submit fallback
    """
    ids = set(_extract_visible_ids(obs))
    inspected = " ".join(obs.event_log[-8:])

    # friction gate handling
    if "friction_continue_btn" in ids:
        return DarkGuardAction(action_type="click", element_id="friction_continue_btn")

    suspicious = [
        "auto_renew_checkbox",
        "seat_insurance_toggle",
        "platform_fee_line",
        "dont_care_btn",
        "countdown_timer",
        "more_options_link",
        "price_label",
        "total_label",
    ]
    for sid in suspicious:
        if sid in ids and sid not in inspected:
            return DarkGuardAction(action_type="inspect", element_id=sid)

    for sid in ("auto_renew_checkbox", "seat_insurance_toggle", "platform_fee_line", "countdown_timer"):
        if sid in ids and f"[FLAG] {sid}" not in inspected:
            return DarkGuardAction(action_type="flag", element_id=sid, note="Potential dark pattern")

    for tid in ("auto_renew_checkbox", "seat_insurance_toggle"):
        if tid in ids:
            return DarkGuardAction(action_type="toggle", element_id=tid)

    for cid in (
        "pay_btn",
        "final_confirm_btn",
        "submit_btn",
        "continue_btn",
        "proceed_btn",
        "add_to_cart_btn",
        "plan_free_trial",
        "account_settings_link",
        "billing_link",
        "manage_plan_link",
        "more_options_link",
        "cancel_link",
        "dont_care_btn",
        "submit_survey_btn",
        "confirm_cancel_btn",
    ):
        if cid in ids:
            return DarkGuardAction(action_type="click", element_id=cid)

    if "submit_btn" in ids:
        return DarkGuardAction(action_type="submit", element_id="submit_btn")
    if "pay_btn" in ids:
        return DarkGuardAction(action_type="submit", element_id="pay_btn")

    # Last-resort random click or go_back
    clickables = [x for x in ids if x.endswith("_btn") or x.endswith("_link")]
    if clickables:
        return DarkGuardAction(action_type="click", element_id=rng.choice(clickables))
    return DarkGuardAction(action_type="go_back", element_id=None)


def run_episode(
    env: DarkGuardEnv,
    episode_idx: int,
    task_id: str,
    subtlety: int,
    rng: random.Random,
) -> EpisodeSummary:
    obs = env.reset(task_id=task_id, self_play=True, subtlety=subtlety)
    done = obs.done
    steps = 0
    while not done:
        action = _choose_action(obs, rng)
        obs = env.step(action)
        done = obs.done
        steps += 1

    sp = obs.metadata.get("self_play", {})
    return EpisodeSummary(
        episode_idx=episode_idx,
        task_id=task_id,
        episode_score=float(obs.metadata.get("episode_score") or 0.0),
        cumulative_reward=float(obs.cumulative_reward),
        steps=steps,
        consumer_elo=float(sp.get("consumer_elo", 1200.0)),
        designer_elo=float(sp.get("designer_elo", 1200.0)),
        subtlety=int(sp.get("designer_subtlety", subtlety)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DarkGuard self-play training demo with optional wandb logging.")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="darkguard-arena")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--role-swap-every", type=int, default=10)
    parser.add_argument(
        "--subtlety-schedule",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated subtlety values to cycle through.",
    )
    args = parser.parse_args()

    subtlety_schedule = [max(1, min(5, int(x.strip()))) for x in args.subtlety_schedule.split(",") if x.strip()]
    if not subtlety_schedule:
        subtlety_schedule = [1]

    rng = random.Random(args.seed)
    env = DarkGuardEnv()
    env.reset(task_id="easy_safe_signup", self_play=True, role_swap_every=args.role_swap_every)

    wandb_run = None
    if args.use_wandb:
        wandb = _safe_import_wandb()
        if wandb is None:
            raise RuntimeError("wandb is not installed. Install extras: `uv sync --extra train`.")
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            config={
                "episodes": args.episodes,
                "seed": args.seed,
                "role_swap_every": args.role_swap_every,
                "subtlety_schedule": subtlety_schedule,
            },
        )

    summaries: List[EpisodeSummary] = []
    wins = 0
    for i in range(1, args.episodes + 1):
        task_id = TASK_IDS[(i - 1) % len(TASK_IDS)]
        subtlety = subtlety_schedule[(i - 1) % len(subtlety_schedule)]
        summary = run_episode(env, i, task_id, subtlety, rng)
        summaries.append(summary)
        wins += 1 if summary.episode_score >= 0.6 else 0

        rolling_win_rate = wins / i
        if rolling_win_rate > 0.60 and subtlety_schedule[-1] < 5:
            subtlety_schedule.append(min(5, subtlety_schedule[-1] + 1))

        payload = {
            "episode": summary.episode_idx,
            "task_id": summary.task_id,
            "episode_score": summary.episode_score,
            "cumulative_reward": summary.cumulative_reward,
            "steps": summary.steps,
            "consumer_elo": summary.consumer_elo,
            "designer_elo": summary.designer_elo,
            "designer_subtlety": summary.subtlety,
            "rolling_win_rate": rolling_win_rate,
        }
        print(json.dumps(payload), flush=True)
        if wandb_run is not None:
            wandb_run.log(payload)

    out_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "selfplay_metrics.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for s in summaries:
            f.write(json.dumps(s.__dict__) + "\n")

    if wandb_run is not None:
        wandb_run.log({"artifact/selfplay_metrics_path": str(out_path)})
        wandb_run.finish()

    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
