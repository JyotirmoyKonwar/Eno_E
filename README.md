---
title: DarkGuard
emoji: 🛡️
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
tags:
- openenv
---


# DarkGuard: The Consumer Protection Environment

**DarkGuard** is a real-world OpenEnv environment where an AI agent must help a consumer safely complete everyday digital tasks — free trial signup, product checkout, and subscription cancellation — while detecting and avoiding manipulative interface traps (dark patterns).

## 1. What is DarkGuard?
DarkGuard simulates digital product flows with embedded deceptive design patterns. An agent must complete a stated user goal (like buying a ticket) while avoiding hidden fees, fake urgency, preselected add-ons, and cancellation mazes. Difficulty is determined entirely by **detection complexity**: how many steps it takes to reveal the trap, how much state comparison is required, and how many interacting traps are combined.

## 2. Why it matters
The FTC and various international bodies (like India's CCPA) regulate dark patterns. Agents trained in this environment can protect consumers by performing safe navigation of tricky interfaces, saving users money and avoiding unwanted subscriptions.

## 3. Action Space
The agent has a discrete semantic action space (6 types) mapping to common web interactions:

| Action Type | Description |
|---|---|
| `click` | Advance flow or trigger transition (requires `element_id`) |
| `toggle` | Flip a checkbox or switch (requires `element_id`) |
| `inspect` | Reveal hidden metadata on an element (requires `element_id`) |
| `flag` | Mark an element as suspicious (requires `element_id` and `note`) |
| `go_back` | Return to the previous screen |
| `submit` | Commit current form state |

## 4. Observation Space
At each step, the environment returns a rich JSON observation:

| Field | Type | Description |
|---|---|---|
| `episode_id` | str | Unique identifier for the current episode |
| `task_id` | str | Current active task |
| `step`, `max_steps` | int | Current step and maximum allowed steps |
| `screen_id`, `screen_title`| str | Current screen identifier and title |
| `user_goal` | str | Plain English goal the user wants completed |
| `elements` | list | List of visible `UIElement` objects (id, type, label, state) |
| `event_log` | list | History of actions and events taken |
| `account_state`| dict | Key variables (e.g. `charged: 0`, `subscribed: False`) |
| `step_reward` | float | Reward received on previous step |
| `cumulative_reward` | float | Running total of reward |

## 5. Reward Function
Reward is dense across the trajectory, clipped to `[-0.20, +0.20]` per step:
- `+0.05`: `inspect()` reveals hidden harmful metadata
- `+0.08`: comparing price/state discrepancy across screens
- `+0.12`: taking protective action before irreversible harm (e.g. unchecking auto-renew)
- `+0.06`: goal progress (advancing on the correct path)
- `-0.03`: repeated identical action or no-op
- `-0.05`: submit while a suspicious unflagged trap element is active
- `-0.10`: submit without gathering any evidence via `inspect()`
- `-0.10`: entering a harmful account state
- `-0.08`: explicitly flagging a benign element
- `+0.12`: correctly flagging a real trap (oracle-verified)

### Final Grader Score
S = 0.35P + 0.25D + 0.20G + 0.15E + 0.05X
where:
- P (Prevention): Avoided harmful outcome
- D (Detection): Flagged traps accurately vs false positives
- G (Goal completion): Achieved user's goal
- E (Evidence): Gathered proof via `inspect()` before acting
- X (Efficiency): Not wasting turns

## 6. Tasks
1. **Easy (`easy_safe_signup`)** - Complete a free trial signup without enabling unwanted auto-renewal. One visible trap, no cross-screen comparison needed. Expected baseline: ~0.62.
2. **Medium (`medium_fair_checkout`)** - Purchase a product at the advertised price. Two traps (drip pricing + add-on) revealed only by comparing price across 3-4 screens. Expected baseline: ~0.41.
3. **Hard (`hard_cancel_maze`)** - Cancel a subscription through a multi-screen retention funnel with 3 combined traps and procedural friction. Expected baseline: ~0.22.

## 7. Setup & Usage

### Local Usage
```bash
uv sync
source .venv/bin/activate
# Run standard baseline inference (requires OPENAI_API_KEY)
OPENAI_API_KEY=sk-... python inference.py
```

### Docker
```bash
docker build -t openenv-darkguard .
docker run -p 7860:7860 openenv-darkguard
```

### Validation
```bash
# Verify openenv spec
openenv validate .

# Run pre-submission checks (URL is for local or hosted space)
./validate-submission.sh http://localhost:7860
```

## 8. Baseline Scores
Using `Qwen/Qwen2.5-72B-Instruct` via HTTP API routing: 
- **Easy**: ~0.62
- **Medium**: ~0.41
- **Hard**: ~0.22

## 9. Training Demos

Two training demo scripts are included:

- `scripts/train_selfplay_demo.py`
  - Online rollout training loop over DarkGuard self-play episodes.
  - Logs episode score, cumulative reward, and dual ELO (consumer/designer).
  - Optional Weights & Biases logging with `--use-wandb`.
  - No dataset required.

- `scripts/train_trl_demo.py`
  - Dataset-driven TRL SFT demo.
  - Intentionally does not hardcode dataset paths or sources.
  - Pass `--dataset-path` once your dataset is ready.

### Install training extras
```bash
uv sync --extra train
source .venv/bin/activate
```

### Run self-play demo (+ wandb)
```bash
python scripts/train_selfplay_demo.py --episodes 90 --use-wandb --wandb-project darkguard-arena
```

### Run TRL demo (dataset hook)
```bash
python scripts/train_trl_demo.py --dataset-path /path/to/dataset.jsonl --text-column text --use-wandb
```

If `--dataset-path` is omitted, the TRL script exits cleanly with instructions.

## 10. V2 Changes (DarkGuard-Arena Upgrade)

This section tracks all V2 changes currently implemented in the repo.

### Core Environment Upgrades
- Upgraded consumer action space to 6 actions: `click`, `toggle`, `inspect`, `flag`, `go_back`, `submit`.
- Kept multi-screen conditional flows across all three scenarios, including long-horizon cancellation path.
- Preserved inspect-gated hidden metadata behavior (`element_metadata` only revealed through `inspect`).
- Added stronger event traces and self-play metadata in observations for debugging and demos.

### Oracle + Detection Changes
- Added oracle module (`src/darkguard/oracle.py`) as the trap-verdict interface.
- `flag` now stores oracle-backed fields in trace:
  - `oracle_is_trap`
  - `oracle_confidence`
  - `oracle_reason`
- Current oracle is deterministic and episode-config-backed, with a clean interface for plugging in Eno_E classifier inference.

### Reward / Grading Updates
- Dense rewards now include:
  - reward for correct trap flag (oracle-verified),
  - penalty for incorrect flag,
  - penalty for overconfident submit without prior evidence gathering.
- Existing prevention/detection/evidence shaping remains active.
- Final deterministic rubric remains:
  - `S = 0.35P + 0.25D + 0.20G + 0.15E + 0.05X`

### Self-Play Layer (MVP)
- Added self-play helpers in `src/darkguard/selfplay.py`:
  - designer action generation,
  - runtime episode mutation hooks,
  - ELO update utility.
- Added reset-time self-play controls:
  - `self_play`
  - `designer_actions`
  - `subtlety` (1–5)
  - `role_swap_every`
- Added runtime perturbations:
  - label obscuring,
  - fee discrepancy modifier,
  - friction gate insertion on selected transitions.
- Added ELO tracking in observation metadata:
  - `consumer_elo`, `designer_elo`, roles, subtlety, and designer actions.

### Training + Tooling
- Added optional training dependencies under `train` extras in `pyproject.toml`:
  - `wandb`, `datasets`, `transformers`, `trl`, `accelerate`
- Added training scripts:
  - `scripts/train_selfplay_demo.py` (online self-play rollouts + optional wandb logging)
  - `scripts/train_trl_demo.py` (dataset-driven TRL SFT hook; dataset intentionally not hardcoded)
- Added artifact ignore rules in `.gitignore`:
  - `wandb/`, `runs/`, `runs_smoke/`, `artifacts/`

### Tests Added for V2
- Added tests for:
  - oracle-backed flag reward behavior,
  - submit-without-inspection penalty,
  - self-play metadata presence,
  - friction gate insertion and continuation path.
- Current status: full suite passes in project venv.

### Intentionally Deferred (Not Yet Implemented)
- Full Eno_E classifier inference integration inside oracle (currently interface-ready).
- Full dual-agent policy training loop (consumer and designer as separate trainable policies).
- Automatic curriculum scheduler strictly tied to win-rate thresholds.
- Plotting/export scripts for judge-ready reward + ELO curves (can be added next).
