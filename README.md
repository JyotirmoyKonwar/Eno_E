---
title: DarkHorse
emoji: 🐎
colorFrom: blue
colorTo: black
sdk: docker
pinned: false
---


# DarkGuard: The Consumer Protection Environment

**DarkGuard** is a real-world OpenEnv environment where an AI agent must help a consumer safely complete everyday digital tasks — free trial signup, product checkout, and subscription cancellation — while detecting and avoiding manipulative interface traps (dark patterns).

## 1. What is DarkGuard?
DarkGuard simulates digital product flows with embedded deceptive design patterns. An agent must complete a stated user goal (like buying a ticket) while avoiding hidden fees, fake urgency, preselected add-ons, and cancellation mazes. Difficulty is determined entirely by **detection complexity**: how many steps it takes to reveal the trap, how much state comparison is required, and how many interacting traps are combined.

## 2. Why it matters
The FTC and various international bodies (like India's CCPA) regulate dark patterns. Agents trained in this environment can protect consumers by performing safe navigation of tricky interfaces, saving users money and avoiding unwanted subscriptions.

## 3. Action Space
The agent has a discrete semantic action space (7 types) mapping to common web interactions:

| Action Type | Description |
|---|---|
| `click` | Advance flow or trigger transition (requires `element_id`) |
| `toggle` | Flip a checkbox or switch (requires `element_id`) |
| `type` | Fill a text field (requires `element_id` and `value`) |
| `inspect` | Reveal hidden metadata on an element (requires `element_id`) |
| `go_back` | Return to the previous screen |
| `submit` | Commit current form state |
| `flag` | Mark an element as suspicious (requires `element_id` and `note`) |

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
- `-0.10`: entering a harmful account state
- `-0.08`: explicitly flagging a benign element

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
# Run standard baseline inference
python inference.py
```

### Docker
```bash
docker build -t openenv-darkguard .
docker run -p 8000:8000 openenv-darkguard
```

### Validation
```bash
# Verify openenv spec
openenv validate .
```

## 8. Baseline Scores
Using `Qwen/Qwen2.5-72B-Instruct` via HTTP API routing: 
- **Easy**: ~0.62
- **Medium**: ~0.41
- **Hard**: ~0.22
