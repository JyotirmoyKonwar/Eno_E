from darkguard_trainer.rewards import compute_consumer_reward, compute_designer_reward


def test_consumer_reward_components() -> None:
    obs = {
        "reward": 0.2,
        "last_action_result": "safe completion",
        "reward_breakdown": {"false_positive": -0.1, "invalid_action": 0.0, "loop_penalty": 0.0, "efficiency": -0.01},
    }
    action = {"action_type": "flag", "target_id": "x"}
    reward = compute_consumer_reward(obs, action)
    assert reward.total > 0


def test_designer_invalid_penalty() -> None:
    r = compute_designer_reward(valid=False, challenge_delta=0.2, novelty=0.1, leakage_detected=False, impossible=False)
    assert r < 0
