"""Self-play curriculum mode and gate logic."""

from dataclasses import replace

from darkguard_trainer.config import SelfPlayCurriculumConfig
from darkguard_trainer.curriculum import (
    curriculum_gate_passed,
    curriculum_live_summary,
    get_designer_training_mode,
    resolve_gate_metric_values,
    validate_curriculum_ranges,
)


def _default_curriculum(**kwargs: object) -> SelfPlayCurriculumConfig:
    return replace(SelfPlayCurriculumConfig(), **kwargs)


def test_validate_ordering_ok() -> None:
    c = _default_curriculum()
    assert validate_curriculum_ranges(c) == []


def test_validate_overlap_errors() -> None:
    c = _default_curriculum(freeze_designer_to_round=10, alternate_designer_from_round=9)
    errs = validate_curriculum_ranges(c)
    assert any("freeze" in e.lower() for e in errs)


def test_mode_default_progression() -> None:
    c = _default_curriculum()
    assert get_designer_training_mode(5, c, gates_passed=False) == "frozen"
    assert get_designer_training_mode(10, c, gates_passed=False) == "alternate"
    assert get_designer_training_mode(13, c, gates_passed=False) == "frozen"
    assert get_designer_training_mode(13, c, gates_passed=True) == "full"


def test_mode_disabled_always_full() -> None:
    c = _default_curriculum(enable_curriculum=False)
    assert get_designer_training_mode(1, c, gates_passed=False) == "full"


def test_gate_or_default() -> None:
    c = _default_curriculum(
        require_safe_rate_gate=True,
        require_eval_reward_gate=True,
        safe_rate_threshold=0.10,
        eval_reward_must_exceed_baseline=True,
        gate_combiner="OR",
    )
    assert curriculum_gate_passed(0.15, 0.0, 0.5, c) is True
    assert curriculum_gate_passed(0.0, 0.6, 0.5, c) is True
    assert curriculum_gate_passed(0.0, 0.5, 0.5, c) is False


def test_gate_and() -> None:
    c = _default_curriculum(
        require_safe_rate_gate=True,
        require_eval_reward_gate=True,
        gate_combiner="AND",
        safe_rate_threshold=0.10,
        eval_reward_must_exceed_baseline=True,
    )
    assert curriculum_gate_passed(0.15, 0.6, 0.5, c) is True
    assert curriculum_gate_passed(0.15, 0.5, 0.5, c) is False


def test_resolve_rolling_eval() -> None:
    c = _default_curriculum(gate_metric_source="rolling_eval_k", gate_rolling_eval_k=2)
    hist = [(0.0, 0.0), (0.2, 1.0), (0.4, 2.0)]
    sr, er = resolve_gate_metric_values(c, None, None, None, None, hist)
    assert abs(sr - 0.3) < 1e-6
    assert abs(er - 1.5) < 1e-6


def test_live_summary_contains_ranges() -> None:
    s = curriculum_live_summary(_default_curriculum())
    assert "frozen" in s
    assert "every 2" in s or "2 designer" in s


def test_self_play_curriculum_config_dataclass_slots_dict() -> None:
    c = SelfPlayCurriculumConfig()
    assert c.freeze_designer_from_round == 1
    assert "easy_safe_signup" in c.bootstrap_task_allowlist
