import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported


DEFAULT_MODEL = "unsloth/Qwen3-4B-Thinking-2507-FP8"
FALLBACK_MODELS = [
    "unsloth/Qwen2.5-4B-Instruct-bnb-4bit",
    "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DarkGuard SFT with Unsloth + TRL.")
    parser.add_argument("--role", choices=["consumer", "designer"], required=True)
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--train_file", default=None)
    parser.add_argument("--val_file", default=None)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--max_seq_length", type=int, default=3072)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--load_in_4bit", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", default=None)
    return parser.parse_args()


def resolve_data_files(role: str, train_file: str, val_file: str) -> Tuple[Path, Path]:
    default_train = Path("darkguard_preprocessed") / f"{role}_train.jsonl"
    default_val = Path("darkguard_preprocessed") / f"{role}_val.jsonl"
    train_path = Path(train_file) if train_file else default_train
    val_path = Path(val_file) if val_file else default_val
    return train_path, val_path


def ensure_files_exist(paths: List[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Required file(s) not found: {missing}")


def safe_chat_fallback(messages: List[Dict[str, str]]) -> str:
    chunks = []
    for msg in messages:
        role = str(msg.get("role", "unknown")).strip().lower()
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        if role == "system":
            chunks.append(f"<|system|>\n{content}")
        elif role == "user":
            chunks.append(f"<|user|>\n{content}")
        elif role == "assistant":
            chunks.append(f"<|assistant|>\n{content}")
        else:
            chunks.append(f"<|{role}|>\n{content}")
    return "\n".join(chunks).strip()


def apply_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as exc:
            print(f"[WARN] apply_chat_template failed, using fallback formatter: {exc}")
    return safe_chat_fallback(messages)


def build_text_mapper(tokenizer):
    def _map_fn(example: Dict) -> Dict:
        messages = example.get("messages", [])
        if not isinstance(messages, list):
            raise ValueError(f"Expected `messages` list, got: {type(messages)}")
        return {"text": apply_template(tokenizer, messages)}

    return _map_fn


def pick_model_with_fallback(
    model_name: str,
    max_seq_length: int,
    load_in_4bit: bool,
):
    candidates = [model_name] + [m for m in FALLBACK_MODELS if m != model_name]
    errors = {}
    for candidate in candidates:
        try:
            print(f"[INFO] Trying model: {candidate}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=candidate,
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=load_in_4bit,
            )
            if candidate != model_name:
                print(
                    "[WARN] Requested model was not available. "
                    f"Using fallback model: {candidate}"
                )
            return candidate, model, tokenizer
        except Exception as exc:
            errors[candidate] = str(exc)
            print(f"[WARN] Failed to load {candidate}: {exc}")

    error_blob = json.dumps(errors, indent=2)
    raise RuntimeError(
        "Could not load requested model or fallbacks with Unsloth.\n"
        "Tried candidates:\n"
        f"{error_blob}\n"
        "Please pass --model_name with a supported Unsloth checkpoint."
    )


def main() -> None:
    args = parse_args()
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("[INFO] Logged into Hugging Face Hub using HF_TOKEN from .env")
    else:
        print("[WARN] HF_TOKEN not found. Proceeding without hub login.")

    train_path, val_path = resolve_data_files(args.role, args.train_file, args.val_file)
    ensure_files_exist([train_path, val_path])

    used_model_name, model, tokenizer = pick_model_with_fallback(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    data_files = {"train": str(train_path), "validation": str(val_path)}
    raw_ds = load_dataset("json", data_files=data_files)

    mapper = build_text_mapper(tokenizer)
    train_ds = raw_ds["train"].map(mapper, remove_columns=raw_ds["train"].column_names)
    val_ds = raw_ds["validation"].map(mapper, remove_columns=raw_ds["validation"].column_names)

    role_output_dir = Path(args.output_dir) / args.role
    role_output_dir.mkdir(parents=True, exist_ok=True)

    report_to = "wandb" if args.use_wandb else "none"
    bf16 = is_bfloat16_supported()
    fp16 = not bf16

    print("\n[INFO] ===== Training Configuration =====")
    print(f"[INFO] role={args.role}")
    print(f"[INFO] requested_model={args.model_name}")
    print(f"[INFO] used_model={used_model_name}")
    print(f"[INFO] train_file={train_path}")
    print(f"[INFO] val_file={val_path}")
    print(f"[INFO] output_dir={role_output_dir}")
    print(f"[INFO] max_seq_length={args.max_seq_length}")
    print(f"[INFO] batch_size={args.batch_size}")
    print(f"[INFO] grad_accum={args.grad_accum}")
    print(f"[INFO] learning_rate={args.learning_rate}")
    print(f"[INFO] epochs={args.epochs}")
    print(f"[INFO] lora_r={args.lora_r}")
    print(f"[INFO] load_in_4bit={args.load_in_4bit}")
    print(f"[INFO] precision={'bf16' if bf16 else 'fp16'}")
    print(f"[INFO] report_to={report_to}")
    print("[INFO] ==================================\n")

    training_args = SFTConfig(
        output_dir=str(role_output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=bf16,
        fp16=fp16,
        optim="adamw_8bit",
        weight_decay=0.01,
        seed=args.seed,
        gradient_checkpointing=True,
        report_to=report_to,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    print(f"[INFO] Training complete. Metrics: {train_result.metrics}")

    eval_metrics = trainer.evaluate()
    print(f"[INFO] Validation metrics: {eval_metrics}")

    adapter_dir = role_output_dir / "adapter"
    merged_dir = role_output_dir / "merged_16bit"
    tokenizer_dir = role_output_dir / "tokenizer"

    print(f"[INFO] Saving adapter to: {adapter_dir}")
    trainer.model.save_pretrained(str(adapter_dir))

    print(f"[INFO] Saving tokenizer to: {tokenizer_dir}")
    tokenizer.save_pretrained(str(tokenizer_dir))

    try:
        print(f"[INFO] Saving merged 16-bit model to: {merged_dir}")
        trainer.model.save_pretrained_merged(
            str(merged_dir),
            tokenizer=tokenizer,
            save_method="merged_16bit",
        )
    except Exception as exc:
        print(f"[WARN] Could not export merged 16-bit model: {exc}")
        print("[WARN] Adapter and tokenizer were saved successfully.")

    print("[INFO] Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
