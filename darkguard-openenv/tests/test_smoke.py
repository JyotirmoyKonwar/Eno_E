from DarkVader_openenv.environment import DarkGuardEnvironment


def test_reset_returns_valid_observation() -> None:
    env = DarkGuardEnvironment()
    obs = env.reset(task_id="easy_safe_signup", seed=7)
    assert obs["episode_id"]
    assert obs["task_id"] == "easy_safe_signup"
    assert obs["screen_id"] == "signup_start"
    assert isinstance(obs["elements"], list)
    assert obs["done"] is False


def test_step_progresses_and_invalid_action_safe() -> None:
    env = DarkGuardEnvironment()
    env.reset(task_id="easy_safe_signup", seed=1)
    obs = env.step({"action_type": "inspect", "target_id": "marketing_checkbox"})
    assert obs["step_count"] == 1
    obs2 = env.step({"action_type": "click", "target_id": "does_not_exist"})
    assert obs2["step_count"] == 2
    assert obs2["done"] in {True, False}
    assert "invalid" in obs2["last_action_result"].lower() or "missing" in obs2["last_action_result"].lower()


def test_terminates_on_max_steps() -> None:
    env = DarkGuardEnvironment()
    env.reset(task_id="medium_fair_checkout", max_steps=2, seed=3)
    env.step({"action_type": "inspect", "target_id": "discount_toggle"})
    obs = env.step({"action_type": "inspect", "target_id": "discount_toggle"})
    assert obs["done"] is True
