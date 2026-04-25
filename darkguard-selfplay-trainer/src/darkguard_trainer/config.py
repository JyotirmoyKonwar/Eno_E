"""Configuration models for DarkGuard self-play trainer."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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
    consumer_base_model: str = "unsloth/Qwen3-4B-Thinking-2507-FP8"
    designer_base_model: str = "unsloth/Qwen3-4B-Thinking-2507-FP8"
    consumer_checkpoint_override: str = ""
    designer_checkpoint_override: str = ""


@dataclass(slots=True)
class SelfPlayCurriculumConfig:
    enable_curriculum: bool = True
    freeze_designer_from_round: int = 1
    freeze_designer_to_round: int = 8
    alternate_designer_from_round: int = 9
    alternate_designer_to_round: int = 12
    designer_train_every_n_designer_phases: int = 2
    full_designer_training_from_round: int = 13
    require_safe_rate_gate: bool = True
    safe_rate_threshold: float = 0.10
    require_eval_reward_gate: bool = True
    eval_reward_must_exceed_baseline: bool = True
    gate_combiner: str = "OR"
    gate_metric_source: str = "latest_eval"
    gate_rolling_eval_k: int = 3
    use_fixed_designer_pool: bool = False
    fixed_designer_checkpoints: list[str] = field(default_factory=list)
    fallback_to_baseline_designer: bool = True
    bootstrap_easy_tasks_only: bool = True
    bootstrap_task_allowlist: list[str] = field(default_factory=lambda: ["easy_safe_signup"])


@dataclass(slots=True)
class TrainingConfig:
    total_rounds: int = 20
    consumer_steps_per_round: int = 20
    designer_steps_per_round: int = 10
    num_generations: int = 4
    batch_size: int = 4
    max_prompt_length: int = 1024
    max_completion_length: int = 256
    consumer_lr: float = 2e-5
    designer_lr: float = 1e-5
    elo_k_factor: float = 20.0
    eval_interval: int = 2
    replay_buffer_size: int = 300
    rollback_threshold: float = 0.2
    use_local_action_router: bool = False
    freeze_consumer: bool = False
    freeze_designer: bool = False
    use_baseline: bool = True
    use_wandb: bool = False
    seed: int = 42
    curriculum: SelfPlayCurriculumConfig = field(default_factory=SelfPlayCurriculumConfig)


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


def _serialize_training(cfg: TrainingConfig) -> dict[str, Any]:
    d = asdict(cfg)
    d["curriculum"] = asdict(cfg.curriculum)
    return d


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
            "training": _serialize_training(self.training),
            "paths": {k: str(v) for k, v in asdict(self.paths).items()},
        }
