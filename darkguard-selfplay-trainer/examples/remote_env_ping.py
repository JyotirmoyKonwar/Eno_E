from darkguard_trainer.env_client import RemoteEnvClient


def main() -> None:
    client = RemoteEnvClient("https://jyo-k-darkguard-openenv.hf.space")
    print("health:", client.health())
    obs = client.reset({"task_id": "easy_safe_signup", "seed": 7})
    print("reset:", obs["task_id"], obs["screen_id"])


if __name__ == "__main__":
    main()
