---
title: DarkGuard OpenEnv
emoji: 🛡️
colorFrom: indigo
colorTo: red
sdk: docker
pinned: false
---

# DarkGuard OpenEnv

**An OpenEnv-native cybersecurity world where LLM agents learn to defend under uncertainty, pressure, and adversarial adaptation.**

## Overview

DarkGuard OpenEnv is the core environment for our OpenEnv / RL-for-LLMs hackathon project, **DarkGuard**.  
It is designed around a simple but important gap: modern LLMs can explain cybersecurity concepts, but they are still weak at **sequential, evidence-driven defensive decision-making** in noisy, partially observable settings.

Instead of reinventing environment tooling, DarkGuard is built on **OpenEnv philosophy and interfaces**: composable tasks, clear observations/actions, reward-driven learning loops, and compatibility with RL training pipelines.  
This Space provides the environment. The companion training pipeline lives in the trainer Space:

- Trainer Space: https://huggingface.co/spaces/Jyo-K/DarkVader-selfplay-trainer

## The problem

Today’s frontier models can produce impressive cyber advice in one-shot prompts, but real defensive operations are not one-shot.  
Defenders must:

- gather partial evidence over time,
- avoid false positives and costly overreactions,
- manage adversarial deception,
- and commit to actions that have delayed consequences.

This creates a capability gap between **talking about security** and **acting as a reliable defender over a long horizon**.  
DarkGuard turns that gap into a trainable RL environment.

## Why DarkGuard

Cybersecurity is a strong RL testbed because it naturally combines:

- **partial observability** (you rarely see full attacker intent),
- **adversarial pressure** (attackers adapt),
- **high-stakes trade-offs** (missed threats vs operational friction),
- **temporal strategy** (good decisions are sequences, not isolated outputs).

DarkGuard focuses on dark-pattern and adversarial interaction defense as a practical proxy for broader cyber-defense behavior: detect manipulation, inspect signals, and choose safe, robust actions over time.

![DarkGuard Environment Diagram](environment_structure.png)

## Environment design

DarkGuard is framed as an interactive environment where an LLM agent repeatedly observes state and chooses defense actions.

### OpenEnv interaction contract

DarkGuard follows a Gym/OpenEnv-like contract:

- `reset(task_id, seed, max_steps, difficulty, subtlety, episode_config)` initializes a new episode.
- `step(action)` advances the environment by one action.
- `state()` returns compact episode-level summary state.

This Space supports both builtin tasks and `custom_episode` payloads so trainer-generated adversarial scenarios can be injected without changing environment code.

### Observation schema (what the agent sees)

Each step returns an observation with operational context and feedback fields:

- `episode_id`, `task_id`, `screen_id`
- `instruction`, `visible_summary`
- `elements` (UI/control objects, each with `id`, `type`, `text`, `checked`, `enabled`, `prominence`, `metadata`)
- `allowed_actions` (action mask for current state)
- `step_count`, `max_steps`
- `reward` (step reward), `cumulative_reward`
- `done`
- `last_action_result`, `messages`
- `reward_breakdown` (per-component reward terms for this step)

This is intentionally structured for RL rollouts: policy input is explicit, and training diagnostics are visible without hidden side channels.

### Action space (what the agent can do)

Supported action types:

- `inspect`
- `click`
- `toggle`
- `flag`
- `go_back`
- `submit`

Action payload format:

- `action_type` (required)
- `target_id` (required for target-dependent actions like `inspect`, `flag`, `click`, `toggle`)
- `flag_category` (optional evidence label for `flag`)
- `notes` (optional rationale text)

The environment dynamically computes `allowed_actions` from current screen elements and context (for example, `submit` appears only on terminal screens). Invalid format or impossible actions are accepted as input but penalized and reflected in feedback.

### Transition and termination logic

- Screens are task-defined nodes with transitions keyed by target element IDs.
- `click`/`toggle` can trigger transitions when mapped.
- `go_back` is only valid when a back transition exists.
- `submit` is valid only on terminal screens.
- Episode terminates on:
  - reaching a terminal safe/harmful screen, or
  - hitting `max_steps`.

The environment tracks repeated `(screen, action)` patterns to discourage loop exploitation and reward hacking.

### Episode state (`state()`)

The compact state endpoint exposes:

- `task_id`, `screen_id`, `step_count`, `max_steps`
- `cumulative_reward`
- `done`, `outcome_summary`
- recent `messages`
- cumulative `reward_totals` by component

This makes post-episode analysis and judge-facing verification straightforward.

## Reward design

DarkGuard uses a conceptually layered reward structure to encourage practical defender behavior:

- **Progress incentives:** small positive reward for useful interaction (inspection and valid navigation actions).
- **Detection quality:** positive reward for correct trap flags; penalty for false positives.
- **Evidence quality:** bonus when flag category/notes align with actual trap evidence.
- **Terminal outcomes:** strong positive for safe completion, strong negative for harmful completion.
- **Behavior shaping:** efficiency cost per step, extra late-step penalty, repeated-loop penalty, invalid-action penalty.
- **Safety bounds:** step reward is clipped to a stable range to avoid exploding updates during training.

The goal is not “always block everything,” but to learn a policy that is both **secure and operationally sensible**.

## Theme alignment

DarkGuard is designed to map directly to core OpenEnv hackathon themes:

**World Modeling**
- The agent must infer latent risk from incomplete observations and evolving context.

**Multi-Agent Interactions**
- The setting includes adversarial pressure and self-play style evolution through historical opponent pools.

**Self-Improvement**
- The training pipeline supports iterative policy updates against dynamic and historical opponents.

**Long-Horizon Planning**
- Defensive quality emerges from action sequences, not single-turn outputs.

## Results

This Space focuses on environment delivery; training and progress tracking are handled in the trainer Space.

Planned / attached artifacts:

- [Add reward curve image here]
- [Add loss curve image here]
- [Add training GIF here]
- [Add before-vs-after qualitative behavior examples here]
- [Add architecture diagram here]

Qualitative behaviors to highlight in evaluation:

- **Before training:** reactive, brittle, overconfident or inconsistent defensive actions
- **After training:** more deliberate inspection, better threat discrimination, safer completion patterns

## Links and artifacts

- OpenEnv Space (this project): https://huggingface.co/spaces/Jyo-K/DarkVader-openenv
- Trainer Space (pipeline, self-play, checkpoints, metrics): https://huggingface.co/spaces/Jyo-K/DarkVader-selfplay-trainer
- [Add project blog link here]
- [Add demo video link here]
- [Add slides link here]
- [Add repository link here]

This README is intended as the central story + environment entry point, with links to all supporting materials and the full training workflow in the trainer Space.

## Running / using the Space

### For judges and visitors
1. Open the Space UI.
2. Inspect available environment interactions/endpoints (`reset`, `step`, `state`).
3. Run sample episodes and check `allowed_actions`, `last_action_result`, and `reward_breakdown`.
4. Confirm that terminal outcomes map to `outcome_summary` and cumulative reward behavior.
5. Compare with trainer outputs to understand learning progression.

### For contributors
1. Review environment components and OpenEnv interface wiring.
2. Validate observation/action schema consistency.
3. Test rollout behavior for valid, invalid, and adversarial action traces.
4. Connect to the trainer Space for policy iteration and evaluation.

### Minimal action example

```json
{
  "action_type": "flag",
  "target_id": "signup_trap",
  "flag_category": "dark-pattern",
  "notes": "ambiguous consent flow"
}
```

### Minimal observation fields example

```json
{
  "task_id": "custom_episode",
  "screen_id": "start",
  "allowed_actions": ["inspect", "click", "flag"],
  "step_count": 3,
  "max_steps": 25,
  "reward_breakdown": {
    "correct_flag": 0.4,
    "false_positive": 0.0,
    "total": 0.465
  },
  "done": false
}
```

### Notes
- This Space is the **environment layer**.
- The trainer Space contains training loops, self-play management, checkpointing, and experiment tracking.

## Why this project matters

Most non-technical users assume AI cyber-defense is “solved” because models sound confident.  
In reality, confidence is not competence. DarkGuard makes that distinction measurable.

By turning cyber-defense reasoning into an interactive RL environment, we move from:
- static chatbot answers  
to
- testable, improvable defensive behavior over time.

That matters for safer AI systems, better defensive automation, and more realistic evaluation of model capability under real-world pressure.

## Future work

- Expand scenario diversity to broader cyber workflows beyond current task families.
- Increase realism of adversarial adaptation and deception patterns.
- Add richer multi-agent curricula and tournament-style evaluation.
- Improve interpretability tooling for trajectory-level failure analysis.
- Add benchmark suites for cross-policy comparison and reproducibility.
- Package standardized evaluation tracks for future OpenEnv community submissions.