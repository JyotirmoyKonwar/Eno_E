"""Staged self-play curriculum: designer freeze / alternate / full + gates."""

from __future__ import annotations

from statistics import mean
from typing import Literal

from .config import SelfPlayCurriculumConfig
from .selfplay import OpponentEntry, sample_opponent

CurriculumMode = Literal["frozen", "alternate", "full"]


def validate_curriculum_ranges(c: SelfPlayCurriculumConfig) -> list[str]:
    """Return human-readable validation errors (empty if OK)."""
    errs: list[str] = []
    if c.freeze_designer_from_round > c.freeze_designer_to_round:
        errs.append("Bootstrap freeze: from_round must be <= to_round.")
    if c.alternate_designer_from_round > c.alternate_designer_to_round:
        errs.append("Alternate-train: from_round must be <= to_round.")
    if c.enable_curriculum:
        if c.freeze_designer_to_round >= c.alternate_designer_from_round:
            errs.append("Freeze range must end strictly before alternate range starts (freeze_to < alternate_from).")
        if c.alternate_designer_to_round >= c.full_designer_training_from_round:
            errs.append("Alternate range must end strictly before full training starts (alternate_to < full_from).")
    if c.designer_train_every_n_designer_phases < 1:
        errs.append("designer_train_every_n_designer_phases must be >= 1.")
    if c.gate_rolling_eval_k < 1:
        errs.append("gate_rolling_eval_k must be >= 1.")
    if c.gate_combiner not in ("AND", "OR"):
        errs.append("gate_combiner must be 'AND' or 'OR'.")
    if c.gate_metric_source not in ("latest_training", "latest_eval", "rolling_eval_k"):
        errs.append("gate_metric_source must be latest_training, latest_eval, or rolling_eval_k.")
    return errs


def get_designer_training_mode(
    round_idx: int,
    curriculum: SelfPlayCurriculumConfig,
    gates_passed: bool,
) -> CurriculumMode:
    """High-level curriculum regime for this loop iteration (not per-phase)."""
    if not curriculum.enable_curriculum:
        return "full"
    if curriculum.freeze_designer_from_round <= round_idx <= curriculum.freeze_designer_to_round:
        return "frozen"
    if curriculum.alternate_designer_from_round <= round_idx <= curriculum.alternate_designer_to_round:
        return "alternate"
    if round_idx < curriculum.full_designer_training_from_round:
        return "frozen"
    if gates_passed:
        return "full"
    return "frozen"


def resolve_gate_metric_values(
    curriculum: SelfPlayCurriculumConfig,
    latest_train_safe_rate: float | None,
    latest_train_mean_reward: float | None,
    latest_eval_safe_rate: float | None,
    latest_eval_reward: float | None,
    eval_history: list[tuple[float, float]],
) -> tuple[float, float]:
    """Return (safe_rate_value, eval_reward_value) used for gating."""
    src = curriculum.gate_metric_source
    if src == "latest_training":
        sr = latest_train_safe_rate if latest_train_safe_rate is not None else 0.0
        er = latest_train_mean_reward if latest_train_mean_reward is not None else 0.0
        return sr, er
    if src == "latest_eval":
        sr = latest_eval_safe_rate if latest_eval_safe_rate is not None else 0.0
        er = latest_eval_reward if latest_eval_reward is not None else 0.0
        return sr, er
    k = max(1, curriculum.gate_rolling_eval_k)
    if eval_history:
        window = eval_history[-k:]
        return mean(s for s, _ in window), mean(r for _, r in window)
    sr = latest_eval_safe_rate if latest_eval_safe_rate is not None else 0.0
    er = latest_eval_reward if latest_eval_reward is not None else 0.0
    return sr, er


def curriculum_gate_passed(
    safe_rate_value: float,
    eval_reward_value: float,
    baseline_eval_reward: float,
    curriculum: SelfPlayCurriculumConfig,
) -> bool:
    """True if enabled gates pass according to gate_combiner (default OR)."""
    checks: list[bool] = []
    if curriculum.require_safe_rate_gate:
        checks.append(safe_rate_value > curriculum.safe_rate_threshold)
    if curriculum.require_eval_reward_gate:
        if curriculum.eval_reward_must_exceed_baseline:
            checks.append(eval_reward_value > baseline_eval_reward)
        else:
            checks.append(eval_reward_value >= baseline_eval_reward)
    if not checks:
        return True
    if curriculum.gate_combiner == "AND":
        return all(checks)
    return any(checks)


def should_use_weak_designer_pool(mode: CurriculumMode, curriculum: SelfPlayCurriculumConfig) -> bool:
    if not curriculum.enable_curriculum:
        return False
    return mode != "full"


def bootstrap_easy_tasks_active(
    round_idx: int,
    mode: CurriculumMode,
    gates_passed: bool,
    curriculum: SelfPlayCurriculumConfig,
) -> bool:
    if not curriculum.enable_curriculum or not curriculum.bootstrap_easy_tasks_only:
        return False
    if curriculum.freeze_designer_from_round <= round_idx <= curriculum.freeze_designer_to_round:
        return True
    if curriculum.alternate_designer_from_round <= round_idx <= curriculum.alternate_designer_to_round:
        return True
    if round_idx >= curriculum.full_designer_training_from_round and not gates_passed:
        return True
    return False


def pick_consumer_task_id(
    rng: random.Random,
    curriculum: SelfPlayCurriculumConfig,
    easy_bootstrap_active: bool,
) -> str:
    if easy_bootstrap_active and curriculum.bootstrap_task_allowlist:
        return rng.choice(curriculum.bootstrap_task_allowlist)
    if easy_bootstrap_active:
        return "easy_safe_signup"
    return "custom_episode"


def filter_weak_designer_pool(
    full_pool: list[OpponentEntry],
    curriculum: SelfPlayCurriculumConfig,
) -> list[OpponentEntry]:
    """Subset of designer opponents matching fixed checkpoint name/id substrings."""
    if not curriculum.use_fixed_designer_pool or not curriculum.fixed_designer_checkpoints:
        return list(full_pool)
    needles = [n.strip().lower() for n in curriculum.fixed_designer_checkpoints if n.strip()]
    if not needles:
        return list(full_pool)
    out: list[OpponentEntry] = []
    for e in full_pool:
        hay = f"{e.name} {e.checkpoint}".lower()
        if any(n in hay for n in needles):
            out.append(e)
    return out


def sample_curriculum_designer_opponent(
    active_elo: float,
    league_pool: list[OpponentEntry],
    rng: random.Random,
    curriculum: SelfPlayCurriculumConfig,
    weak_mode: bool,
) -> tuple[OpponentEntry | None, bool]:
    """
    Returns (opponent, used_weak_policy).
    When weak_mode and pool filter empty, may return baseline synthetic entry (logged only).
    """
    if not weak_mode:
        return sample_opponent(active_elo, league_pool, rng), False
    filtered = filter_weak_designer_pool(league_pool, curriculum)
    if filtered:
        return sample_opponent(active_elo, filtered, rng), True
    if curriculum.fallback_to_baseline_designer:
        return OpponentEntry(name="baseline_designer", elo=800.0, role="designer", checkpoint=""), True
    return None, True


def curriculum_live_summary(curriculum: SelfPlayCurriculumConfig) -> str:
    errs = validate_curriculum_ranges(curriculum)
    if errs:
        return "Invalid curriculum: " + "; ".join(errs)
    if not curriculum.enable_curriculum:
        return "Curriculum disabled: full designer training every designer phase."
    c = curriculum
    gate_bits: list[str] = []
    if c.require_safe_rate_gate:
        gate_bits.append(f"safe_rate > {c.safe_rate_threshold:.2f}")
    if c.require_eval_reward_gate:
        cmp_ = ">" if c.eval_reward_must_exceed_baseline else ">="
        gate_bits.append(f"eval_reward {cmp_} baseline")
    gate_txt = f" {c.gate_combiner} ".join(gate_bits) if gate_bits else "no gates"
    lines = [
        f"Rounds {c.freeze_designer_from_round}–{c.freeze_designer_to_round}: frozen",
        (
            f"Rounds {c.alternate_designer_from_round}–{c.alternate_designer_to_round}: "
            f"train Designer every {c.designer_train_every_n_designer_phases} designer phase(s)"
        ),
        f"Rounds {c.full_designer_training_from_round}+: full training when {gate_txt}",
    ]
    return "\n".join(lines)


def explain_designer_skip(
    phase: str,
    mode: CurriculumMode,
    designer_training_enabled: bool,
    curriculum: SelfPlayCurriculumConfig,
) -> str:
    if phase != "train_designer":
        return ""
    if not curriculum.enable_curriculum:
        return ""
    if designer_training_enabled:
        return ""
    if mode == "frozen":
        return "curriculum_frozen"
    if mode == "alternate":
        return "schedule"
    return "curriculum_frozen"
