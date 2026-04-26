---
title: DarkGuard Self-Play Trainer
emoji: 📈
colorFrom: gray
colorTo: blue
sdk: gradio
pinned: false
---

# DarkGuard Self-Play Trainer

**A training-first companion Space that demonstrates real RL improvement on the DarkGuard OpenEnv cybersecurity environment.**

## What this Space does

This Space is the **training engine** for DarkGuard.  
While the main environment Space exposes the task world, this trainer Space exists to show the full learning loop end-to-end:

- policy interaction with the environment,
- reward-driven updates,
- self-play/adaptive opponent sampling,
- checkpointing and evaluation,
- and visible evidence that behavior changes over training.

Main environment Space: https://huggingface.co/spaces/Jyo-K/darkguard-openenv  
Trainer Space: https://huggingface.co/spaces/Jyo-K/darkguard-selfplay-trainer

## Training objective

DarkGuard targets a practical gap in LLM capability: models can discuss security, but often fail at **sequential defensive decision-making under uncertainty**.

The objective here is to train policies that improve on:

- safe task completion under adversarial pressure,
- evidence-driven action selection,
- lower invalid/unsafe behavior,
- and stronger multi-step defensive consistency.

This trainer is built to make that objective measurable.

## Why self-play

Static training can overfit quickly in adversarial domains.  
DarkGuard uses self-play and historical-opponent sampling to keep the learning problem moving:

- opponents evolve over time,
- current policies are tested against recent historical snapshots,
- training pressure remains adaptive rather than fixed,
- and Elo-style tracking reflects progress against a changing league.

This makes the learning signal more realistic for security settings where attack/defense patterns co-evolve.

## Training pipeline

The pipeline is designed to stay tightly coupled to OpenEnv rather than building separate custom machinery.

**Core components**
- **Environment:** DarkGuard OpenEnv (OpenEnv-native environment logic)
- **Trainer UI/runtime:** Gradio-based training control and live monitoring
- **RL loop:** policy rollouts + reward aggregation + policy update steps
- **League/self-play:** frozen checkpoint pools + recent-window historical matchups
- **Tracking/artifacts:** metrics CSV, state store, frozen pool registry, checkpoints

**Reward hookup**
- Environment interactions produce trajectory outcomes used to compute training signals.
- Reward terms are aligned with safe completion, action quality, and adversarial robustness.
- Evaluation phases and baseline checks provide additional stability and regression detection.

**Model/framework details**
- RL stack and model configuration are set in trainer config/runtime.
- [Insert training config]
- [Insert model details]
- [Insert RL framework details (TRL / Unsloth / etc.)]

## Reproducibility

This Space is intended to be rerunnable, not just viewable.

To reproduce a run, provide:

1. model/config values,
2. environment endpoint (DarkGuard OpenEnv),
3. training hyperparameters,
4. seed,
5. checkpoint/output directory.

Expected reproducibility artifacts:
- `metrics.csv` with per-round metrics,
- saved checkpoints for both roles,
- frozen league registry,
- resumable state file.

Rerunnable assets:
- [Insert training script link]
- [Insert Colab link]
- [Insert notebook link]

The same links should be referenced from both this README and the main environment README.

## Evidence of training

This section is the proof surface for “real training happened.”

- Reward progression: [Insert reward curve]
- Loss progression: [Insert loss plot]
- Baseline vs trained behavior: [Insert baseline vs trained comparison]
- Qualitative trajectory examples: [Insert before/after episode traces]
- Checkpoint progression summary: [Insert checkpoint table/summary]

Important: this Space is where training evidence is centralized for judging.

## How to inspect results

Judges and contributors can verify the pipeline quickly with this checklist:

1. Confirm trainer is connected to DarkGuard OpenEnv.
2. Inspect training config and seed/setup.
3. Run training and observe live metrics/logs.
4. Check saved artifacts (metrics, checkpoints, registry, state).
5. Compare baseline behavior vs trained behavior.
6. Validate that improvements are reflected in reward/quality trends, not just UI output.

What to verify explicitly:
- what model was trained,
- on what setup,
- against which environment,
- with what reward logic,
- and what improved.

## Linked resources

- Trainer Space: https://huggingface.co/spaces/Jyo-K/darkguard-selfplay-trainer
- Environment Space: https://huggingface.co/spaces/Jyo-K/darkguard-openenv
- [Insert repository link]
- [Insert training script link]
- [Insert Colab link]
- [Insert demo video]
- [Insert slides]

## Notes for judges

This Space is intentionally technical and training-centric.  
It exists to satisfy the hackathon requirement to show **real, inspectable, rerunnable training evidence** tied directly to the environment.

In short: DarkGuard is not just a polished environment demo.  
It is an environment + training system where learning dynamics, artifacts, and improvements can be examined directly.

## Future work

- Expand curriculum diversity and adversarial scenario breadth.
- Add stronger automated evaluation suites and regression gates.
- Improve reproducibility packaging (one-command run + artifact bundles).
- Add richer behavior diagnostics beyond scalar rewards.
- Extend multi-agent league dynamics and tournament evaluation reporting.