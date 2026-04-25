"""Local smoke script for direct environment usage."""

from darkguard_openenv.environment import DarkGuardEnvironment


def main() -> None:
    env = DarkGuardEnvironment()
    obs = env.reset(task_id="hard_cancel_maze", seed=123, max_steps=10)
    print("RESET:", obs["task_id"], obs["screen_id"], obs["visible_summary"])

    actions = [
        {"action_type": "inspect", "target_id": "keep_plan"},
        {"action_type": "click", "target_id": "manage_plan"},
        {"action_type": "flag", "target_id": "pause_plan", "flag_category": "friction-cancellation"},
        {"action_type": "click", "target_id": "cancel_small_link"},
        {"action_type": "click", "target_id": "confirm_cancel"},
    ]

    for action in actions:
        obs = env.step(action)
        print(
            "STEP:",
            obs["step_count"],
            "screen=",
            obs["screen_id"],
            "reward=",
            obs["reward"],
            "done=",
            obs["done"],
        )
        if obs["done"]:
            break

    print("STATE:", env.state())


if __name__ == "__main__":
    main()
