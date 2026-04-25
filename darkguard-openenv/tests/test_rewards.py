from darkguard_openenv.environment import DarkGuardEnvironment


def test_reward_signs_safe_vs_harmful() -> None:
    env = DarkGuardEnvironment()
    env.reset(task_id="easy_safe_signup", seed=11)
    safe_obs = env.step({"action_type": "click", "target_id": "continue_clean"})

    env2 = DarkGuardEnvironment()
    env2.reset(task_id="easy_safe_signup", seed=11)
    harmful_obs = env2.step({"action_type": "click", "target_id": "accept_all"})

    assert safe_obs["done"] is True
    assert harmful_obs["done"] is True
    assert safe_obs["reward"] > harmful_obs["reward"]


def test_spam_flag_penalized() -> None:
    env = DarkGuardEnvironment()
    env.reset(task_id="hard_cancel_maze", seed=2)
    obs = env.step({"action_type": "flag", "target_id": "non_existing", "flag_category": "hidden-costs"})
    assert obs["reward"] < 0
