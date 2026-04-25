import random

from darkguard_trainer.selfplay import OpponentEntry, phase_for_round, sample_opponent


def test_phase_cycle() -> None:
    assert phase_for_round(1) == "train_consumer"
    assert phase_for_round(2) == "train_designer"
    assert phase_for_round(3) == "evaluation"
    assert phase_for_round(4) == "snapshot"


def test_sample_opponent() -> None:
    rng = random.Random(1)
    pool = [
        OpponentEntry(name="a", elo=1200, role="designer"),
        OpponentEntry(name="b", elo=1300, role="designer"),
        OpponentEntry(name="c", elo=1100, role="designer"),
    ]
    choice = sample_opponent(1210, pool, rng)
    assert choice is not None
