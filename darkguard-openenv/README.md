---
title: darkguard-openenv
emoji: 🛡️
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
tags:
- openenv
- reinforcement-learning
- grpo
- fastapi
---

# darkguard-openenv

`darkguard-openenv` is a production-ready OpenEnv-style environment for training a consumer LLM to detect and avoid deceptive UX patterns (dark patterns) across signup, checkout, renewal, and cancellation flows.

This repository intentionally contains **only the environment space**:
- no GRPO trainer
- no Gradio GUI
- no model-serving UI

## Why this environment exists

Dark-pattern safety should be trained against an objective verifier, not free-form heuristics.  
This environment is the source of truth for transitions, hidden traps, and reward computation so external GRPO loops can stay simple and consistent.

## Environment Contract

The environment follows the standard Gym/OpenEnv contract:
- `reset(**kwargs) -> observation`
- `step(action) -> observation`
- `state() -> metadata`

The FastAPI server exposes these as:
- `POST /reset`
- `POST /step`
- `GET /state`

## Task Families

Built-in deterministic task families:
- `easy_safe_signup`
- `medium_fair_checkout`
- `hard_cancel_maze`

Also supported:
- `custom_episode`

For `custom_episode`, pass a strict `episode_config` payload to `reset()`. If absent, built-ins are sampled deterministically by seed.

## Observation and Action Design

### Action
`DarkGuardAction` supports:
- `action_type`: `inspect|click|toggle|flag|go_back|submit`
- `target_id` (optional)
- `flag_category` (optional)
- `notes` (optional)

### Observation
`DarkGuardObservation` includes:
- episode/task/screen identity
- instruction and visible summary
- visible elements (`id`, `type`, `text`, `checked`, `enabled`, `prominence`, `metadata`)
- allowed actions
- step and reward tracking
- done status
- event messages
- reward breakdown (safe for debugging)

### State
`state()` returns operational metadata:
- episode progress
- cumulative reward
- outcome summary
- cumulative reward components

It intentionally excludes hidden oracle labels.

## Reward Design (Verifier-Based)

Reward components:
- progress reward (useful inspection/navigation)
- correct flag reward
- false-positive penalty
- harmful terminal penalty
- safe completion reward
- per-step efficiency penalty
- loop/repetition penalty
- invalid action penalty
- optional evidence bonus (flag/notes aligned with trap evidence)

Per-step reward is clipped to a stable range for GRPO:
- min `-2.0`
- max `2.0`

## Anti-Reward-Hacking Safeguards

Implemented safeguards:
- hidden `trap_map` stays internal to server environment
- public observation/state never return hidden trap labels
- terminal verifier uses objective terminal screen ids
- spam-flagging incurs false-positive costs
- repeated state-action loops are penalized
- malformed actions become safe invalid actions (no crash)
- malformed custom episodes fail strict schema validation
- hard `max_steps` cutoff with consistent done behavior
- no submit shortcut from non-terminal states

## Text Action Parsing

`step()` supports:
1. strict JSON:
```json
{"action_type":"flag","target_id":"hidden_fee","flag_category":"hidden-costs"}
```
2. fallback text:
```text
ACTION: inspect | TARGET: cookie_banner
ACTION: flag | TARGET: hidden_fee | CATEGORY: hidden-costs
```

Parse failures become safe invalid actions and receive a small penalty.

## Run Locally

```bash
cd darkguard-openenv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD/src
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

## Run Tests

```bash
cd darkguard-openenv
export PYTHONPATH=$PWD/src
pytest -q
```

## Smoke Examples

```bash
cd darkguard-openenv
export PYTHONPATH=$PWD/src
python examples/local_smoke.py
python examples/remote_smoke.py
```

## Hugging Face Space Deployment

1. Create a Docker Space named `darkguard-openenv`.
2. Push this repository contents.
3. Space uses `Dockerfile` startup:
   - `uvicorn server.app:app --host 0.0.0.0 --port 7860`
4. Use base URL from Space for remote clients.

`openenv.yaml` metadata is included for environment discovery and documentation.

## Client-Server Separation

Client code (`src/darkguard_openenv/client.py`) communicates strictly over HTTP with base URL.  
No server internals are imported by remote consumers.

## GRPO Integration Pattern (Trainer lives elsewhere)

```python
from darkguard_openenv.client import DarkGuardClient

env = DarkGuardClient("https://<your-space>.hf.space")
obs = env.reset(task_id="custom_episode", seed=123, max_steps=20)

done = False
while not done:
    # your model generates action text or dict
    action = {"action_type": "inspect", "target_id": "some_element"}
    obs = env.step(action)
    reward = obs["reward"]
    done = obs["done"]
    # trainer stores (obs, action, reward, next_obs)
```

This repository deliberately does not include trainer code.
 
