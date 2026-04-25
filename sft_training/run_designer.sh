#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python3 "${SCRIPT_DIR}/train_sft.py" \
  --role designer \
  --model_name unsloth/Qwen3-4B-Thinking-2507-FP8 \
  --train_file "${PROJECT_ROOT}/darkguard_preprocessed/designer_train.jsonl" \
  --val_file "${PROJECT_ROOT}/darkguard_preprocessed/designer_val.jsonl" \
  --output_dir "${PROJECT_ROOT}/outputs" \
  --max_seq_length 3072 \
  --batch_size 4 \
  --grad_accum 8 \
  --learning_rate 2e-4 \
  --epochs 2 \
  --lora_r 32 \
  --load_in_4bit true
