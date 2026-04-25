"""
DarkGuard TRL training demo (dataset hook only).

This script intentionally does not hardcode any dataset source. Pass one via
--dataset-path when available; otherwise it exits with setup instructions.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _safe_import_train_stack():
    try:
        from datasets import load_dataset  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        from trl import SFTConfig, SFTTrainer  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Training stack missing. Install with `uv sync --extra train`."
        ) from exc
    return load_dataset, AutoModelForCausalLM, AutoTokenizer, SFTConfig, SFTTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset-driven TRL demo trainer for DarkGuard.")
    parser.add_argument("--dataset-path", type=str, default="", help="Local JSON/JSONL dataset path. Leave empty until dataset is ready.")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir", type=str, default="artifacts/trl-demo")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="darkguard-arena")
    args = parser.parse_args()

    if not args.dataset_path:
        print("No dataset provided yet. Pass --dataset-path when your dataset is available.")
        print("Expected schema: a text column (default name: 'text') containing training prompts/completions.")
        return

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    load_dataset, AutoModelForCausalLM, AutoTokenizer, SFTConfig, SFTTrainer = _safe_import_train_stack()

    if args.use_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        os.environ.setdefault("WANDB_LOG_MODEL", "false")

    ds = load_dataset("json", data_files={args.dataset_split: str(dataset_path)})[args.dataset_split]
    if args.text_column not in ds.column_names:
        raise ValueError(f"Dataset missing text column '{args.text_column}'. Columns: {ds.column_names}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    train_cfg = SFTConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=5,
        save_steps=50,
        report_to="wandb" if args.use_wandb else "none",
        bf16=False,
        fp16=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=train_cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        formatting_func=lambda ex: ex[args.text_column],
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved TRL demo model to {args.output_dir}")


if __name__ == "__main__":
    main()
