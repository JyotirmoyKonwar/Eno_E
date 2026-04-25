"""Remote smoke script for a deployed DarkGuard OpenEnv Space."""

from darkguard_openenv.client import DarkGuardClient


def main() -> None:
    # Replace with your Space URL, e.g. https://<user>-darkguard-openenv.hf.space
    base_url = "http://localhost:7860"
    client = DarkGuardClient(base_url=base_url)

    obs = client.reset(task_id="medium_fair_checkout", seed=99, max_steps=12)
    print("RESET:", obs["task_id"], obs["screen_id"])

    obs = client.step("ACTION: inspect | TARGET: discount_toggle")
    print("STEP1:", obs["last_action_result"], obs["reward"])
    obs = client.step({"action_type": "click", "target_id": "review_billing"})
    print("STEP2:", obs["screen_id"], obs["reward"])
    print("STATE:", client.state())


if __name__ == "__main__":
    main()
