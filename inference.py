"""
DarkGuard — OpenEnv Baseline Inference Script
==============================================

Runs a language model agent against all 3 DarkGuard tasks and emits
structured stdout logs in the mandatory [START] / [STEP] / [END] format.

Required environment variables:
    OPENAI_API_KEY  Primary API key (hackathon spec requirement)
    API_BASE_URL    LLM API endpoint  (default: https://router.huggingface.co/v1)
    MODEL_NAME      Model identifier  (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN        HuggingFace token (fallback if OPENAI_API_KEY is unset)

Usage:
    OPENAI_API_KEY=sk-... MODEL_NAME=... python inference.py
"""

import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ── Environment setup ────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("OPENAI_API_KEY", os.getenv("HF_TOKEN", ""))

BENCHMARK    = "darkguard"
TEMPERATURE  = 0.3
MAX_TOKENS   = 256

TASKS = [
    "easy_safe_signup",
    "medium_fair_checkout",
    "hard_cancel_maze",
]

TASK_MAX_STEPS = {
    "easy_safe_signup": 15,
    "medium_fair_checkout": 25,
    "hard_cancel_maze": 40,
}

# ── Logging helpers ──────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f}"
        f" done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps}"
        f" score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── Prompt builder ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are a consumer-protection AI agent navigating digital product flows on behalf of a user.
Your task is to complete the user's goal while detecting and avoiding manipulative interface traps
(dark patterns) such as: pre-selected checkboxes, hidden fees, fake urgency timers, and confusingly
labelled buttons.

DECISION PROCESS:
1. Read the screen elements carefully.
2. Identify any suspicious or unusual elements (pre-selected, ambiguously labelled, hidden fees).
3. Use inspect() on suspicious elements to reveal hidden details before committing.
4. Toggle OFF any pre-selected harmful checkboxes/toggles.
5. Flag suspicious elements with a short explanation.
6. Only submit when you're confident the account state will be safe.

You MUST respond with a single JSON object (no markdown, no prose):
{
  "action_type": "<click|toggle|type|inspect|go_back|submit|flag>",
  "element_id": "<element id or null>",
  "value": "<text value or null>",
  "note": "<reason for flag or null>"
}
""").strip()


def build_user_prompt(obs_data: Dict[str, Any], step: int, history: List[str]) -> str:
    elements_str = json.dumps(obs_data.get("elements", []), indent=2)
    history_str = "\n".join(history[-6:]) if history else "None"
    return textwrap.dedent(f"""
Step {step} | Screen: {obs_data.get('screen_title', '')} (id: {obs_data.get('screen_id', '')})

USER GOAL: {obs_data.get('user_goal', '')}

ACCOUNT STATE:
{json.dumps(obs_data.get('account_state', {}), indent=2)}

VISIBLE ELEMENTS:
{elements_str}

RECENT EVENTS:
{history_str}

Previous step reward: {obs_data.get('step_reward', 0.0):.3f}
Cumulative reward: {obs_data.get('cumulative_reward', 0.0):.3f}

What is your next action? (JSON only)
""").strip()


# ── Agent loop ───────────────────────────────────────────────────────────────

def get_agent_action(
    client: OpenAI,
    obs_data: Dict[str, Any],
    step: int,
    history: List[str],
) -> Dict[str, Any]:
    """Call the LLM and parse its action JSON."""
    prompt = build_user_prompt(obs_data, step, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
            seed=42,
        )
        text = (completion.choices[0].message.content or "{}").strip()
        return json.loads(text)
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True, file=sys.stderr)
        # Fallback: safe default action
        return {"action_type": "go_back", "element_id": None, "value": None, "note": None}


def run_task(client: OpenAI, env, task_id: str) -> Dict[str, Any]:
    """Run a single task episode and return summary."""
    max_steps = TASK_MAX_STEPS[task_id]
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    # Reset with task_id passed as extra kwarg
    obs = env.reset(task_id=task_id)
    obs_data = obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)

    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    done = obs_data.get("done", False)
    error = None

    for step in range(1, max_steps + 1):
        if done:
            break

        action_dict = get_agent_action(client, obs_data, step, history)
        action_str = json.dumps(action_dict, separators=(",", ":"))

        try:
            from src.darkguard.models import DarkGuardAction
            action = DarkGuardAction(**action_dict)
            obs = env.step(action)
            obs_data = obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)
            reward = float(obs_data.get("step_reward", obs_data.get("reward", 0.0)))
            done   = obs_data.get("done", False)
            error  = None
        except Exception as exc:
            reward = 0.0
            done   = False
            error  = str(exc)[:120]
            print(f"[DEBUG] step error: {exc}", flush=True, file=sys.stderr)

        rewards.append(reward)
        steps_taken = step
        log_step(step=step, action=action_str, reward=reward, done=done, error=error)
        history.append(
            f"Step {step}: {action_dict.get('action_type')}({action_dict.get('element_id')}) "
            f"→ reward {reward:+.3f}"
        )

        if done:
            break

    # Final score from metadata
    episode_score = obs_data.get("metadata", {}).get("episode_score") or 0.0
    if episode_score is None:
        episode_score = 0.0
    success = episode_score >= 0.5
    log_end(success=success, steps=steps_taken, score=float(episode_score), rewards=rewards)

    return {
        "task_id": task_id,
        "success": success,
        "steps": steps_taken,
        "episode_score": episode_score,
        "rewards": rewards,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if not HF_TOKEN:
        print("[WARN] OPENAI_API_KEY / HF_TOKEN not set — LLM calls may fail.", file=sys.stderr)

    client = OpenAI(api_key=HF_TOKEN or "dummy", base_url=API_BASE_URL)

    # Import env directly (no Docker needed for the server-side inference mode)
    from src.darkguard.env import DarkGuardEnv
    env = DarkGuardEnv()

    results = []
    try:
        for task_id in TASKS:
            result = run_task(client, env, task_id)
            results.append(result)
    finally:
        # env is in-process, no container — but call close() per OpenEnv spec
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass

    # Summary to stderr (doesn't affect stdout log parsing)
    print("\n=== DarkGuard Baseline Summary ===", file=sys.stderr)
    for r in results:
        print(
            f"  {r['task_id']:30s}  score={r['episode_score']:.3f}  "
            f"success={r['success']}  steps={r['steps']}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
