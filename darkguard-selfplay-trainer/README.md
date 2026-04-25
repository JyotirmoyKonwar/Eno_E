---
title: darkguard-selfplay-trainer
emoji: ⚔️
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
python_version: 3.11
pinned: false
tags:
- gradio
- reinforcement-learning
- self-play
- grpo
- openenv
---

# DarkGuard Self-Play Trainer

Training and monitoring Space for DarkGuard co-evolution:
- **Consumer** learns to detect and avoid dark patterns.
- **Designer** learns to generate harder (but valid) deceptive episodes.

This Space connects remotely to:
- `https://jyo-k-darkguard-openenv.hf.space`

## What this Space does

- Runs an alternating self-play league loop (consumer phase, designer phase, eval phase, snapshot phase).
- Tracks ELO-style ratings for active agents and frozen pools.
- Supports W&B token input and optional logging.
- Provides stop/resume/snapshot controls.
- Exports metrics to CSV and checkpoint metadata to `outputs/`.

## Pragmatic Training Design

This implementation uses a robust **hybrid self-play** design:
- remote environment rewards remain source-of-truth,
- consumer/designer are updated in alternating phases,
- checkpoints are frozen into pools,
- holdout evaluation + rollback guard prevents silent collapse.

`TRL GRPO` dependencies are included and code is structured for rollout/reward composition, but this demo prioritizes reliability on HF Spaces over fragile dual-online GRPO updates.

## Anti-gaming safeguards

- No hidden-label shortcut from environment state.
- Designer reward gated by validity + challenge + novelty.
- Invalid/impossible/leaky episode penalties.
- Holdout evaluation seeds and rollback threshold.
- Historical opponent sampling (ELO-biased + random).
- Frozen pools and promotion through periodic evaluation.

## Run locally

```bash
cd darkguard-selfplay-trainer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD/src
python app.py
```

## Tests

```bash
cd darkguard-selfplay-trainer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD/src
pytest -q
```

## HF Space deployment

1. Create a new **Docker Space**.
2. Push this folder as repo root.
3. Space starts with `python app.py`.
4. In the UI:
   - set env URL to your environment Space,
   - provide W&B token optionally,
   - click **Start Training**.

## Key files

- `app.py` — Gradio entrypoint
- `src/darkguard_trainer/gradio_app.py` — UI
- `src/darkguard_trainer/training.py` — self-play engine
- `src/darkguard_trainer/rollout.py` — remote env rollouts
- `src/darkguard_trainer/rewards.py` — reward routing
- `src/darkguard_trainer/selfplay.py` — phase scheduling and opponent sampling
- `src/darkguard_trainer/evaluation.py` — holdout tournament
- `src/darkguard_trainer/checkpointing.py` — checkpoint + frozen pool metadata
