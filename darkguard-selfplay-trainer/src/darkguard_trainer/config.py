"""Configuration models for DarkGuard self-play trainer."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


DEFAULT_ENV_URL = "https://jyo-k-darkguard-openenv.hf.space"


@dataclass(slots=True)
class ConnectionConfig:
    env_base_url: str = DEFAULT_ENV_URL
    hf_token: str | None = field(
        default_factory=lambda: (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or "").strip() or None
    )
    wandb_token: str | None = field(
        default_factory=lambda: (os.getenv("WANDB_API_KEY") or "").strip() or None
    )
    timeout_s: float = 45.0
    max_retries: int = 5
    min_request_interval_s: float = 0.2
    retry_backoff_s: float = 0.8


@dataclass(slots=True)
class ModelConfig:
    consumer_adapter_repo: str = "honestlyanubhav/darkguard-consumer"
    designer_adapter_repo: str = "honestlyanubhav/darkguard-designer"
    consumer_base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    designer_base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    consumer_checkpoint_override: str = ""
    designer_checkpoint_override: str = ""


@dataclass(slots=True)
class TrainingConfig:
    total_rounds: int = 20
    consumer_steps_per_round: int = 4
    designer_steps_per_round: int = 2
    num_generations: int = 2
    batch_size: int = 2
    max_prompt_length: int = 1024
    max_completion_length: int = 256
    consumer_lr: float = 2e-5
    designer_lr: float = 1e-5
    elo_k_factor: float = 12.0
    eval_interval: int = 2
    replay_buffer_size: int = 500
    rollback_threshold: float = 0.2
    freeze_consumer: bool = False
    freeze_designer: bool = False
    use_baseline: bool = True
    use_wandb: bool = False
    seed: int = 42


@dataclass(slots=True)
class RuntimePaths:
    root: Path = field(default_factory=lambda: Path.cwd())
    outputs_dir: Path = field(default_factory=lambda: Path("outputs"))
    checkpoints_dir: Path = field(default_factory=lambda: Path("outputs/checkpoints"))
    metrics_csv: Path = field(default_factory=lambda: Path("outputs/metrics.csv"))
    frozen_registry: Path = field(default_factory=lambda: Path("outputs/frozen_pool_registry.json"))
    state_file: Path = field(default_factory=lambda: Path("outputs/state_store.json"))

    def ensure(self) -> None:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class AppConfig:
    connection: ConnectionConfig = field(default_factory=ConnectionConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: RuntimePaths = field(default_factory=RuntimePaths)

    def as_dict(self) -> dict[str, object]:
        return {
            "connection": asdict(self.connection),
            "models": asdict(self.models),
            "training": asdict(self.training),
            "paths": {k: str(v) for k, v in asdict(self.paths).items()},
        }
