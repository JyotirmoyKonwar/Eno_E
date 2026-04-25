# DarkGuard SFT Training (Unsloth + TRL)

This folder contains a complete Stage 1 supervised fine-tuning setup for DarkGuard using Unsloth + TRL on Hugging Face Spaces A100-large hardware.

## 1) Select A100-large hardware in Hugging Face Spaces

1. Open your Space.
2. Go to **Settings** -> **Hardware**.
3. Select **Nvidia A100 - large** (80 GB VRAM, 142 GB RAM, 1000 GB disk).
4. Restart/rebuild the Space.

## 2) Create `.env`

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Set your token in `.env`:

```env
HF_TOKEN=hf_xxx
WANDB_PROJECT=darkguard-sft
```

`train_sft.py` uses `load_dotenv()` and `os.getenv("HF_TOKEN")`, and logs in to Hub only when `HF_TOKEN` is present.

## 3) Install dependencies

From your project root:

```bash
cd sft_training
pip install -r requirements.txt
```

## 4) Run training

### Consumer

```bash
cd /path/to/your/project
bash sft_training/run_consumer.sh
```

### Designer

```bash
cd /path/to/your/project
bash sft_training/run_designer.sh
```

## 5) Direct CLI usage

You can run the training script directly:

```bash
python3 sft_training/train_sft.py \
  --role consumer \
  --model_name unsloth/Qwen3-4B-Thinking-2507-FP8 \
  --train_file darkguard_preprocessed/consumer_train.jsonl \
  --val_file darkguard_preprocessed/consumer_val.jsonl \
  --output_dir outputs \
  --max_seq_length 3072 \
  --batch_size 4 \
  --grad_accum 8 \
  --learning_rate 2e-4 \
  --epochs 2 \
  --lora_r 32 \
  --load_in_4bit true
```

Supported CLI args:

- `--role {consumer,designer}`
- `--model_name`
- `--train_file`
- `--val_file`
- `--output_dir`
- `--max_seq_length`
- `--batch_size`
- `--grad_accum`
- `--learning_rate`
- `--epochs`
- `--lora_r`
- `--load_in_4bit`
- `--use_wandb`
- `--resume_from_checkpoint`

## 6) What gets saved

Outputs are role-specific:

- `outputs/consumer/` or `outputs/designer/`
- Checkpoints from Trainer save steps
- `adapter/` (LoRA adapter)
- `tokenizer/`
- `merged_16bit/` (best-effort merged export; if unsupported, adapter/tokenizer are still saved)

## 7) Resume from checkpoint

Pass a checkpoint path:

```bash
python3 sft_training/train_sft.py \
  --role consumer \
  --resume_from_checkpoint outputs/consumer/checkpoint-100
```

## 8) Optional: Push artifacts to Hub

After training, you can push adapter or merged model (if present):

```bash
python3 -c "from huggingface_hub import HfApi; api=HfApi(); api.upload_folder(folder_path='outputs/consumer/adapter', repo_id='YOUR_USERNAME/darkguard-consumer-adapter', repo_type='model')"
```

And for merged model:

```bash
python3 -c "from huggingface_hub import HfApi; api=HfApi(); api.upload_folder(folder_path='outputs/consumer/merged_16bit', repo_id='YOUR_USERNAME/darkguard-consumer-merged', repo_type='model')"
```

## 9) Model fallback behavior

The script first tries your requested `--model_name` (default: `unsloth/Qwen3-4B-Thinking-2507-FP8`).
If it fails to load in Unsloth, it automatically tries these fallbacks:

1. `unsloth/Qwen2.5-4B-Instruct-bnb-4bit`
2. `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`

The chosen model is printed at startup, and fallback usage is clearly logged.
