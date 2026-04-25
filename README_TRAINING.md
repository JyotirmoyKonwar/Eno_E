# DarkGuard — End-to-End Training Pipeline
## `unsloth/Qwen3-4B-Thinking-2507-FP8` → SFT → GRPO + ELO Self-Play

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: SFT (Supervised Fine-Tuning)                          │
│                                                                 │
│  4 Layered passes on Qwen3-4B-Thinking via Unsloth              │
│                                                                 │
│  Layer 1: Recognition   ← itsbaivab + darkbench                 │
│  Layer 2: Flow/Structure ← 50 Shades + templates                │
│  Layer 3: Causal Reasoning ← D3 Dataset                        │
│  Layer 4: Designer Role ← hand-crafted episode templates        │
│                                                                 │
│  Output: sft_merged_for_grpo/ (both roles bootstrapped)         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: GRPO + ELO Self-Play                                  │
│                                                                 │
│  Consumer (trainable, GRPO)                                     │
│       ↕ episodes                                                │
│  Designer pool (frozen checkpoints, ELO-matched)                │
│       ↕ rewards                                                 │
│  Environment (single source of truth — all rewards here)        │
│                                                                 │
│  Every N rounds: Consumer → checkpoint → joins Designer pool    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Files

| File | Purpose |
|------|---------|
| `stage1_data_prep.py` | Converts all 4 datasets to SFT-ready JSONL |
| `stage1_sft_train.py` | Layered SFT with Unsloth + TRL SFTTrainer |
| `stage2_grpo_selfplay.py` | GRPO training + ELO self-play loop |

---

## Step-by-Step Run Instructions

### Prerequisites
```bash
pip install unsloth trl datasets transformers wandb openenv
# Start DarkGuard env server
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Step 1: Data Prep
```bash
python stage1_data_prep.py
# Outputs: sft_data/layer1_recognition.jsonl
#          sft_data/layer2_structure.jsonl
#          sft_data/layer3_reasoning.jsonl
#          sft_data/layer4_designer.jsonl
#          sft_data/sft_train.jsonl  (merged, shuffled)
#          sft_data/sft_val.jsonl
```

### Step 2: SFT Training
```bash
python stage1_sft_train.py
# Outputs: checkpoints/sft_v0_recognition/
#          checkpoints/sft_v1_structure/
#          checkpoints/sft_v2_reasoning/
#          checkpoints/sft_v3_designer/
#          checkpoints/sft_final/
#          checkpoints/sft_merged_for_grpo/  ← used by Stage 2
```

### Step 3: Self-Play GRPO
```bash
export DARKGUARD_ENV_URL=http://localhost:8000
python stage2_grpo_selfplay.py
# Outputs: checkpoints/selfplay/consumer_round{N}/  (pool grows)
#          checkpoints/selfplay/consumer_final/      (final model)
#          checkpoints/selfplay/reward_history.jsonl
```

---

## Why Qwen3-4B-Thinking

- Already produces `<think>...</think>` chains natively
- GRPO rewards the *outcome* of thinking, not the thinking itself
- `enable_thinking=True` in chat template keeps the thinking block
  in training targets → SFT trains the content of thoughts too
- Temperature 0.6 / top_p 0.9 is Qwen3's recommended sampling config

---

## Reward Function

All rewards come from the environment — no LLM judge:

| Component | Weight | Condition |
|-----------|--------|-----------|
| Goal completion | +0.35 | `env.goal_achieved()` == True |
| Trap detection  | +0.15 | `flags ∩ trap_map / len(trap_map)` |
| Safety outcome  | +0.20 | No harmful state active at terminal |
| False positive penalty | -0.10 | Per wrong flag |
| Efficiency penalty | -0.01 | Per action step |
| Loop penalty | -0.15 | Repeated identical actions > 2 |
| Flag spam | -0.30 | Hard cutoff if > 6 flags |

Reward is clipped to `[-1.0, 1.0]` before GRPO update.

---

## ELO System

- Consumer and each Designer checkpoint have separate ELO ratings
- Starting ELO: 1000 for all
- K-factor: 32 (fast convergence for hackathon)
- Designer selection: consumer ELO + 150 target (productive difficulty band)
- Consumer wins if `mean_reward > 0.40`
- Every 5 rounds: Consumer checkpoint → Designer pool
- New Designer inherits Consumer's ELO as starting point

---

## Anti-Reward-Hacking Checklist

- [x] Rewards from hidden env state — Consumer can't read `trap_map` directly
- [x] False-positive penalty — "flag everything" strategy fails
- [x] Loop detection — repeated actions penalized
- [x] Schema validation on Designer outputs — invalid configs discarded
- [x] KL coefficient (0.04) — Consumer can't drift too far from SFT anchor
- [x] Eval with greedy decoding — separate from training reward
- [x] Hold-out eval dataset — not used in training loop
- [x] Manual rollout audits every 10 rounds (see reward_history.jsonl)

---

## VRAM Requirements

| Stage | Min VRAM | Recommended |
|-------|----------|-------------|
| Data prep | CPU only | CPU |
| SFT (4bit QLoRA) | 12 GB | 16 GB |
| GRPO (4bit QLoRA) | 16 GB | 24 GB |
| Designer inference (4bit) | 8 GB | 12 GB |

For HF Spaces: use A100 (40GB) — runs both Consumer + Designer simultaneously.

---

## What Judges Will See

1. **Before SFT**: Base Qwen3 on a checkout screen → generic response, no flags
2. **After SFT**: Identifies traps, outputs structured `<label>`, `<action>` tags
3. **After GRPO Round 10**: Correctly navigates cancellation maze, flags misdirection
4. **After GRPO Round 30+**: Handles novel trap combinations from higher-ELO Designers
5. **Reward curves**: rising mean reward, rising Consumer ELO, falling false-positive rate
