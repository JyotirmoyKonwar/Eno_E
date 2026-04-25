from darkguard_trainer.elo import expected_score, update_elo


def test_expected_score_symmetric() -> None:
    a = expected_score(1200, 1200)
    b = expected_score(1200, 1200)
    assert round(a, 4) == round(b, 4) == 0.5


def test_update_elo_reward_win() -> None:
    ra, rb = update_elo(1200, 1200, score_a=1.0, k_factor=20.0)
    assert ra > 1200
    assert rb < 1200
