from darkguard_trainer.config import AppConfig
from darkguard_trainer.training import test_connection


def main() -> None:
    cfg = AppConfig()
    print(test_connection(cfg))


if __name__ == "__main__":
    main()
