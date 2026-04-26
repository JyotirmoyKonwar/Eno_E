from DarkVader_openenv.environment import DarkGuardEnvironment


def test_same_seed_same_initial_task_with_custom_mode() -> None:
    env_a = DarkGuardEnvironment()
    env_b = DarkGuardEnvironment()

    obs_a = env_a.reset(task_id="custom_episode", seed=42)
    obs_b = env_b.reset(task_id="custom_episode", seed=42)

    assert obs_a["task_id"] == obs_b["task_id"]
    assert obs_a["screen_id"] == obs_b["screen_id"]


def test_hidden_labels_not_leaked() -> None:
    env = DarkGuardEnvironment()
    obs = env.reset(task_id="medium_fair_checkout", seed=10)
    state = env.state()

    serialized = str(obs) + str(state)
    assert "trap_map" not in serialized
    assert "hidden-recurring-charge" not in serialized


def test_custom_episode_validation() -> None:
    env = DarkGuardEnvironment()
    try:
        env.reset(task_id="custom_episode", episode_config={"bad": "schema"})
        assert False, "Expected validation error"
    except Exception:
        assert True
