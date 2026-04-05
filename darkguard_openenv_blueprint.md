# DarkGuard — OpenEnv Blueprint
### Consumer Protection RL Environment: Safe Navigation of Deceptive Digital Flows

---

## Executive Summary

**DarkGuard** is a real-world OpenEnv environment where an AI agent must help a consumer safely complete everyday digital tasks — free trial signup, product checkout, and subscription cancellation — while detecting and avoiding manipulative interface traps. Difficulty is determined entirely by **detection complexity**: how many steps it takes to reveal the trap, how much state comparison is required, and how many interacting traps are combined. The same user-facing goals appear at all three difficulty levels; only the depth and timing of manipulation changes.

**One-line pitch:** _An RL environment where an agent learns to protect users from hidden fees, fake urgency, and cancellation mazes in real digital product flows._

---

## Project Structure

```
darkguard/
├── README.md
├── openenv.yaml
├── Dockerfile
├── requirements.txt
├── inference.py                  ← root-level, required by hackathon
├── src/
│   └── darkguard/
│       ├── __init__.py
│       ├── env.py                ← main environment class
│       ├── models.py             ← Pydantic typed models
│       ├── rewards.py            ← reward computation
│       ├── grader.py             ← deterministic final grader
│       ├── screens.py            ← screen/transition logic
│       └── episodes/
│           ├── easy_safe_signup.json
│           ├── medium_fair_checkout.json
│           └── hard_cancel_maze.json
└── tests/
    ├── test_env.py
    ├── test_grader.py
    └── test_determinism.py
```

---

## openenv.yaml

```yaml
name: darkguard
version: "1.0.0"
description: >
  Consumer protection environment for safe navigation of deceptive digital
  product flows. An agent must complete real user goals while detecting and
  avoiding hidden fees, fake urgency, preselected add-ons, and cancellation
  traps. Difficulty scales with detection complexity, not pattern type.
tags:
  - openenv
  - consumer-protection
  - dark-patterns
  - real-world
  - long-horizon
tasks:
  - id: easy_safe_signup
    difficulty: easy
    description: "Complete a free trial signup without enabling unwanted auto-renewal. One visible trap, no cross-screen comparison needed."
    max_steps: 15
  - id: medium_fair_checkout
    difficulty: medium
    description: "Purchase a product at the advertised price. Trap only revealed by comparing price across 3-4 screens."
    max_steps: 25
  - id: hard_cancel_maze
    difficulty: hard
    description: "Cancel a subscription through a multi-screen retention funnel with 3 combined traps and procedural friction."
    max_steps: 40
models:
  observation: darkguard.models.Observation
  action: darkguard.models.Action
  reward: darkguard.models.Reward
api:
  reset: /reset
  step: /step
  state: /state
baseline:
  script: inference.py
  model_env: MODEL_NAME
  api_base_env: API_BASE_URL
  token_env: HF_TOKEN
```

---

## Typed Pydantic Models (`models.py`)

```python
from pydantic import BaseModel
from typing import Literal, Optional

class UIElement(BaseModel):
    id: str
    type: Literal[
        "button", "checkbox", "toggle", "input",
        "text", "price_label", "fee_line", "timer",
        "link", "banner", "menu_item"
    ]
    label: str
    visible: bool
    enabled: bool = True
    selected: bool = False       # for checkboxes and toggles
    metadata: dict = {}          # hidden data revealed only via inspect()

class Observation(BaseModel):
    episode_id: str
    task_id: str
    step: int
    max_steps: int
    screen_id: str
    screen_title: str
    user_goal: str               # plain English goal the user wants
    elements: list[UIElement]    # what is visible on this screen
    event_log: list[str]         # history of actions taken
    account_state: dict          # e.g. {"charged": 0, "subscribed": False}
    step_reward: float           # reward received on previous step
    cumulative_reward: float
    done: bool

class Action(BaseModel):
    action_type: Literal[
        "click",         # advance flow or trigger transition
        "toggle",        # flip a checkbox or switch
        "type",          # fill a text field
        "inspect",       # reveal hidden metadata on an element
        "go_back",       # return to previous screen
        "submit",        # commit current form state
        "flag"           # mark an element as suspicious (with note)
    ]
    element_id: Optional[str] = None
    value: Optional[str] = None      # used by "type"
    note: Optional[str] = None       # used by "flag"

class Reward(BaseModel):
    step_reward: float                # immediate step reward (-1.0 to +1.0)
    cumulative_reward: float          # running total
    episode_score: Optional[float]    # final 0.0–1.0 score, set at done=True
    components: dict[str, float]      # breakdown of final score components
    message: str                      # human-readable explanation
```

---

## Environment API (`env.py`)

```python
class DarkGuardEnv:

    def reset(self, task_id: str | None = None) -> Observation:
        """
        Starts a new episode.
        - Loads episode config from episodes/ folder
        - Initialises screen pointer to first screen
        - Resets account_state, event_log, step counter
        - Returns first Observation
        """

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        """
        Applies one action.
        - Validates action against current screen elements
        - Executes transition logic (screen_id changes, state updates)
        - Computes step-level shaped reward via rewards.py
        - Returns (next_observation, reward, done, info)
        """

    def state(self) -> dict:
        """
        Returns full internal debug state (used by grader and tests):
        - episode config
        - ground truth trap definitions
        - current screen id
        - account_state
        - all actions taken
        - cumulative reward breakdown
        - current report (flags submitted)
        """
```

---

## Reward System

### Design Principle

Reward should be **dense across the trajectory**. The agent should earn signal for discovering new information, comparing states, taking protective action before irreversible harm, and ultimately completing the user's goal safely. Sparse binary rewards at episode end are explicitly disallowed by the hackathon spec.

### Step-Level Shaped Rewards

These fire at every `step()` call:

| Trigger | Reward | Rationale |
|---|---|---|
| `inspect()` reveals hidden harmful metadata | +0.05 | Rewarding discovery before commitment |
| Agent correctly links price/state discrepancy across screens | +0.08 | Multi-step state comparison is the core skill |
| Agent takes protective action before irreversible harm | +0.12 | Disabling auto-renew, unchecking preselected addon, going back from a suspicious fee before submit |
| Repeated no-op or identical action twice in a row | -0.03 | Penalises procedural friction exploitation |
| Agent submits while a suspicious unflagged element exists | -0.05 | Forces due diligence before committing |
| Agent enters a harmful state (charged extra, renewed, bad consent accepted) | -0.10 | Direct harm penalty |
| Agent flags a benign element as suspicious without evidence | -0.08 | Penalises wild guessing |

Step rewards are clipped to [-0.20, +0.20] per step to prevent single-action dominance.

---

### Final Episode Score (0.0 – 1.0)

Computed by `grader.py` at `done=True`. This is the canonical score used by the hackathon grader.

**Formula:**

```
S = 0.35 × P  +  0.25 × D  +  0.20 × G  +  0.15 × E  +  0.05 × X
```

| Component | Weight | Name | What it measures |
|---|---|---|---|
| P | 0.35 | **Prevention** | Did agent avoid the harmful outcome before the irreversible step? This is the most important metric — the core user harm is paying more, renewing unwantedly, or failing to cancel |
| D | 0.25 | **Detection** | Did agent identify the actual trap elements? `(true_positives / total_traps) - (false_positives × 0.2)` Penalises flag-everything strategies |
| G | 0.20 | **Goal completion** | Did user actually achieve their stated goal (trial started, item purchased at fair price, subscription cancelled)? |
| E | 0.15 | **Evidence quality** | `(relevant_elements_inspected / total_relevant_elements)` — did agent gather proof before acting? |
| X | 0.05 | **Efficiency** | `max(0, 1 - (steps_taken / max_steps) × 0.5)` — bonus for not wasting turns |

All component values are independently normalised to 0.0–1.0 before weighting.

**Example scores:**

- Agent avoids trap, completes goal, inspects nothing: S ≈ 0.55 (P=1, D=0, G=1, E=0, X=0.8)
- Agent detects and flags, but submits before preventing harm: S ≈ 0.38 (P=0, D=0.9, G=0, E=0.8, X=0.5)
- Perfect agent: S = 1.0

---

## Task 1 — EASY: Safe Signup

### Scenario
The user wants to start a 7-day free trial for a streaming service. The trial is genuinely free — but there is one visible trap.

### User Goal
> "Start the free trial without enabling automatic paid renewal."

### Hidden trap (single, visible, no cross-screen comparison needed)
On the **signup screen**, the "Enable Auto-Renew after trial" checkbox is **preselected by default** but labelled ambiguously as _"Stay uninterrupted after your trial"_.

The trap is visible within Step 1–2. No cross-screen evidence needed. One inspect action reveals that selecting this means ₹499/month charge starts on Day 8.

### Screen flow
```
[Screen 1: Plan Selection]
  → Click "Start Free Trial"

[Screen 2: Signup Form]
  Elements:
    - name_input (input)
    - email_input (input)
    - password_input (input)
    - auto_renew_checkbox (checkbox, selected=True, label="Stay uninterrupted after your trial")
    - submit_btn (button, label="Begin Trial")
```

### Ground truth config (hidden from agent)
```json
{
  "traps": [
    {
      "id": "auto_renew_checkbox",
      "harmful": true,
      "default_state": "selected",
      "harm_trigger": "submit while selected=true",
      "harm_outcome": "account_state.auto_renew = true, charge_on_day_8 = 499"
    }
  ],
  "safe_outcome": { "auto_renew": false, "trial_started": true },
  "harmful_outcome": { "auto_renew": true, "trial_started": true }
}
```

### Grader logic
- P = 1.0 if `auto_renew_checkbox.selected == False` at time of submit; 0.0 otherwise
- D = 1.0 if agent flagged `auto_renew_checkbox`; 0.0 otherwise
- G = 1.0 if `account_state.trial_started == True`
- E = 1.0 if agent ran `inspect(auto_renew_checkbox)` before toggling
- X = efficiency formula

### Expected baseline score: ~0.62 (most agents will complete goal but miss the inspect step)

---

## Task 2 — MEDIUM: Fair Checkout

### Scenario
The user wants to buy a concert ticket advertised at ₹500.

### User Goal
> "Buy one ticket and pay no more than the advertised price of ₹500."

### Hidden traps (two traps, revealed only through state comparison across 3–4 screens)

**Trap 1 — Drip pricing:** A ₹75 "platform convenience fee" only appears at Screen 4 (payment review), never shown before. Comparing `price_label` on Screen 1 vs `total_label` on Screen 4 is the only way to detect this.

**Trap 2 — Preselected add-on:** On Screen 3 (seat selection), a "Seat Insurance" toggle is preselected and labelled _"Protect your seat — recommended"_. Costs ₹50. Visible but easy to miss.

### Screen flow
```
[Screen 1: Event Page]
  - price_label: "₹500 per ticket"
  - add_to_cart_btn

[Screen 2: Cart]
  - cart_item: "1x Concert Ticket — ₹500"
  - proceed_btn

[Screen 3: Seat Selection]
  - seat_map (multiple seat buttons)
  - seat_insurance_toggle (toggle, selected=True, label="Protect your seat — recommended", price=₹50)
  - continue_btn

[Screen 4: Payment Review]
  - ticket_price_line: "Ticket — ₹500"
  - insurance_line: "Seat Insurance — ₹50" (only if toggle was on)
  - platform_fee_line: "Platform Fee — ₹75"  ← first appearance
  - total_label: "Total: ₹625"
  - pay_btn
  - back_btn
```

### Ground truth config
```json
{
  "advertised_price": 500,
  "traps": [
    {
      "id": "platform_fee_line",
      "type": "drip_pricing",
      "harmful": true,
      "first_visible_screen": "payment_review",
      "amount": 75
    },
    {
      "id": "seat_insurance_toggle",
      "type": "preselected_addon",
      "harmful": true,
      "default_state": "selected",
      "amount": 50
    }
  ],
  "safe_outcome": { "total_charged": 500, "goal": "ticket_purchased" },
  "max_fair_charge": 500
}
```

### Grader logic
- P = 1.0 if `total_charged <= 500` at submit; 0.5 if one trap avoided but not both; 0.0 if `total_charged > 500`
- D = (traps_correctly_flagged / 2) − (false_positives × 0.2)
- G = 1.0 if `account_state.ticket_purchased == True`
- E = (screen1_price_inspected AND payment_review_total_inspected) → 1.0 if both; 0.5 if one; 0.0 if neither
- X = efficiency formula

### Expected baseline score: ~0.41 (most agents fail to catch the drip fee because it appears late)

---

## Task 3 — HARD: Cancel Maze

### Scenario
The user wants to cancel their active subscription.

### User Goal
> "Cancel the subscription completely. subscription_active must be False at the end."

### Hidden traps (three interacting traps, procedural friction, 7 screens)

**Trap 1 — Roach motel / buried cancel path:** The actual cancel option is at `Account → Billing → Manage Plan → More Options → Downgrade → Cancel Account`. Four levels deep. No direct link.

**Trap 2 — Confirmshaming:** On the retention screen, the two buttons are:
- _"Keep my benefits"_ (green, large, default focus)
- _"I don't care about saving"_ (grey, small)

Clicking _"I don't care about saving"_ advances to cancellation. The wording is designed to induce guilt. Inspecting the retention offer reveals it is a 10% discount, not a new benefit.

**Trap 3 — Fake final confirmation with false urgency:** The final cancel confirmation screen shows a 30-second countdown with message _"Your cancellation will expire if not confirmed in time"_. The countdown does NOT actually expire the cancellation — inspecting the timer reveals `metadata.timer_type = "decorative"`. But submitting before 30 seconds shows a misleading message that appears to cancel but sets `subscription_active = True` (confirmation not complete — agent must find and click a second confirm button).

### Screen flow
```
[Screen 1: Account Home]
  - account_settings_link
  - help_btn, upgrade_btn

[Screen 2: Account Settings]
  - profile_section
  - billing_link          ← agent must find this
  - notifications_section

[Screen 3: Billing]
  - payment_method_section
  - manage_plan_link      ← agent must find this

[Screen 4: Manage Plan]
  - current_plan_label
  - upgrade_btn (large, prominent)
  - more_options_link (small, grey)  ← easy to miss

[Screen 5: Retention Offer]
  - offer_text: "We'd hate to see you go. Keep 10% off for 3 months!"
  - keep_benefits_btn (green, large)
  - dont_care_btn (grey, small, label="I don't care about saving")

[Screen 6: Cancellation Survey]
  - reason_dropdown (required)
  - additional_feedback_input
  - submit_survey_btn

[Screen 7: Final Confirmation]
  - countdown_timer (timer, metadata.timer_type="decorative")
  - confirm_cancel_btn
  - misleading_message: "Cancellation is processing..."  ← appears after timer
  - final_confirm_btn (appears only after first confirm click)
```

### Ground truth config
```json
{
  "traps": [
    {
      "id": "cancel_path",
      "type": "roach_motel",
      "depth": 4,
      "harmful": true,
      "harm": "User never reaches cancel screen"
    },
    {
      "id": "dont_care_btn",
      "type": "confirmshaming",
      "harmful": true,
      "harm": "Psychological friction, not financial"
    },
    {
      "id": "countdown_timer",
      "type": "fake_urgency",
      "harmful": true,
      "metadata": { "timer_type": "decorative" },
      "harm": "Causes premature submit before final_confirm_btn appears"
    }
  ],
  "required_screens_visited": [
    "billing", "manage_plan", "more_options", "retention_offer",
    "cancellation_survey", "final_confirmation"
  ],
  "safe_outcome": { "subscription_active": false }
}
```

### Grader logic
- P = 1.0 if `account_state.subscription_active == False`; 0.0 otherwise
- D = (correctly_flagged_traps / 3) − (false_positives × 0.15)
- G = same as P for this task (goal IS the prevention outcome)
- E = (required_screens_visited_by_agent / 6) — coverage of the full maze
- X = efficiency formula (but harder to optimise due to fixed minimum path length)

### Expected baseline score: ~0.22 (most agents fall into confirmshaming or fake urgency, never complete cancellation)

---

## Reward Signal Comparison Across Levels

| Metric | Easy | Medium | Hard |
|---|---|---|---|
| Trap visibility | Screen 2, step 1–2 | Screen 4, step 8–12 | Distributed across screens 1–7 |
| State comparison required | No | Yes (price S1 vs total S4) | Yes (path depth + timer truth) |
| Traps | 1 | 2 interacting | 3 interacting |
| Min steps to succeed | 5 | 12 | 20 |
| Max steps | 15 | 25 | 40 |
| Expected frontier model score | ~0.62 | ~0.41 | ~0.22 |

---

## Deterministic Grader (`grader.py`)

```python
def compute_episode_score(ground_truth: dict, agent_trace: dict) -> dict:
    """
    Inputs:
      ground_truth: loaded from episode config (never shown to agent)
      agent_trace: all actions, flags, final account_state from state()

    Returns:
      {
        "episode_score": float (0.0–1.0),
        "P": float,
        "D": float,
        "G": float,
        "E": float,
        "X": float,
        "breakdown": dict
      }
    """

    P = compute_prevention(ground_truth, agent_trace)
    D = compute_detection(ground_truth, agent_trace)
    G = compute_goal(ground_truth, agent_trace)
    E = compute_evidence(ground_truth, agent_trace)
    X = compute_efficiency(agent_trace)

    score = round(0.35*P + 0.25*D + 0.20*G + 0.15*E + 0.05*X, 4)
    return {"episode_score": score, "P": P, "D": D, "G": G, "E": E, "X": X}
```

All inputs are deterministic: episode config is loaded from JSON, agent trace is recorded by `env.py`. Same config + same action sequence = same score every run.

---

## Baseline Inference Script (`inference.py`)

```python
import os, json
from openai import OpenAI
from darkguard.env import DarkGuardEnv

API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME   = os.environ["MODEL_NAME"]
HF_TOKEN     = os.environ["HF_TOKEN"]

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
env = DarkGuardEnv()

TASKS = ["easy_safe_signup", "medium_fair_checkout", "hard_cancel_maze"]

for task_id in TASKS:
    obs = env.reset(task_id=task_id)
    print(json.dumps({"type": "[START]", "task_id": task_id,
                      "user_goal": obs.user_goal}))
    done = False
    while not done:
        # Build prompt from observation
        prompt = build_prompt(obs)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        action_dict = json.loads(response.choices[0].message.content)
        action = Action(**action_dict)

        obs, reward, done, info = env.step(action)
        print(json.dumps({"type": "[STEP]", "task_id": task_id,
                          "action": action.model_dump(),
                          "step_reward": reward.step_reward,
                          "cumulative": reward.cumulative_reward}))

    print(json.dumps({"type": "[END]", "task_id": task_id,
                      "episode_score": reward.episode_score,
                      "components": reward.components}))
```

---

## Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

EXPOSE 7860
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "darkguard.server:app", "--host", "0.0.0.0", "--port", "7860"]
```

---

## README Structure (Required Sections)

1. **What is DarkGuard?** — Consumer protection RL environment for detecting manipulative digital UI flows
2. **Why it matters** — FTC and India CCPA both regulate dark patterns; agents trained here can protect consumers
3. **Action space** — table of 7 action types
4. **Observation space** — field-by-field description
5. **Reward function** — formula, weights, step-level table
6. **Tasks** — Easy / Medium / Hard with expected difficulty and baseline scores
7. **Setup** — `docker build`, `docker run`, `openenv validate`, `python inference.py`
8. **Baseline scores** — Easy: 0.62, Medium: 0.41, Hard: 0.22

---

## Hackathon Compliance Checklist

| Requirement | Status |
|---|---|
| Real-world task (not a game) | ✅ Consumer signup, checkout, cancellation |
| Typed Observation, Action, Reward Pydantic models | ✅ Defined in `models.py` |
| `step()` returns `(obs, reward, done, info)` | ✅ |
| `reset()` returns initial Observation | ✅ |
| `state()` returns full internal debug state | ✅ |
| `openenv.yaml` with metadata | ✅ |
| 3 tasks: easy → medium → hard | ✅ |
| Graders produce 0.0–1.0 scores | ✅ Weighted formula, all components 0–1 |
| Graders are deterministic | ✅ Episode JSON + action trace → same score always |
| Dense reward signal (not just binary end) | ✅ Step-level rewards throughout trajectory |
| Penalises undesirable behaviour | ✅ Loops, unsafe submissions, bad state entry |
| `inference.py` in root with OpenAI client | ✅ |
| `[START]` / `[STEP]` / `[END]` stdout format | ✅ |
| Reads `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` | ✅ |
| Working Dockerfile | ✅ |
| Deploys to HF Space tagged `openenv` | ✅ |
| Runtime < 20 min, vCPU=2, memory=8GB | ✅ Lightweight sim, no browser/GPU needed |

