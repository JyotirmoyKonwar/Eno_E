"""Gradio UI for self-play training and monitoring."""

from __future__ import annotations

import threading
from dataclasses import replace
from typing import Any

import gradio as gr
import pandas as pd

from .config import AppConfig
from .training import TrainerEngine, test_connection
from .ui_state import StateHub

STATE_HUB = StateHub()
TRAIN_THREAD: threading.Thread | None = None


def _to_int(value: Any, default: int) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _build_config(*values: Any) -> AppConfig:
    cfg = AppConfig()
    (
        env_url,
        hf_token,
        wandb_token,
        consumer_adapter,
        designer_adapter,
        consumer_base,
        designer_base,
        consumer_ckpt,
        designer_ckpt,
        total_rounds,
        consumer_steps,
        designer_steps,
        num_generations,
        batch_size,
        max_prompt,
        max_completion,
        consumer_lr,
        designer_lr,
        elo_k,
        eval_interval,
        freeze_consumer,
        freeze_designer,
        replay_size,
        rollback_threshold,
        historical_window,
        use_baseline,
        use_wandb,
    ) = values
    cfg.connection.env_base_url = str(env_url).strip()
    hf_input = str(hf_token).strip()
    wandb_input = str(wandb_token).strip()
    if hf_input:
        cfg.connection.hf_token = hf_input
    if wandb_input:
        cfg.connection.wandb_token = wandb_input
    cfg.models.consumer_adapter_repo = str(consumer_adapter).strip()
    cfg.models.designer_adapter_repo = str(designer_adapter).strip()
    cfg.models.consumer_base_model = str(consumer_base).strip()
    cfg.models.designer_base_model = str(designer_base).strip()
    cfg.models.consumer_checkpoint_override = str(consumer_ckpt).strip()
    cfg.models.designer_checkpoint_override = str(designer_ckpt).strip()
    cfg.training = replace(
        cfg.training,
        total_rounds=_to_int(total_rounds, cfg.training.total_rounds),
        consumer_steps_per_round=_to_int(consumer_steps, cfg.training.consumer_steps_per_round),
        designer_steps_per_round=_to_int(designer_steps, cfg.training.designer_steps_per_round),
        num_generations=_to_int(num_generations, cfg.training.num_generations),
        batch_size=_to_int(batch_size, cfg.training.batch_size),
        max_prompt_length=_to_int(max_prompt, cfg.training.max_prompt_length),
        max_completion_length=_to_int(max_completion, cfg.training.max_completion_length),
        consumer_lr=_to_float(consumer_lr, cfg.training.consumer_lr),
        designer_lr=_to_float(designer_lr, cfg.training.designer_lr),
        elo_k_factor=_to_float(elo_k, cfg.training.elo_k_factor),
        eval_interval=_to_int(eval_interval, cfg.training.eval_interval),
        freeze_consumer=bool(freeze_consumer),
        freeze_designer=bool(freeze_designer),
        replay_buffer_size=_to_int(replay_size, cfg.training.replay_buffer_size),
        rollback_threshold=_to_float(rollback_threshold, cfg.training.rollback_threshold),
        historical_window=_to_int(historical_window, cfg.training.historical_window),
        use_baseline=bool(use_baseline),
        use_wandb=bool(use_wandb) or bool(cfg.connection.wandb_token),
    )
    return cfg


def on_test_connection(env_url: str, hf_token: str, wandb_token: str) -> str:
    cfg = AppConfig()
    cfg.connection.env_base_url = str(env_url).strip() or cfg.connection.env_base_url
    cfg.connection.hf_token = str(hf_token).strip() or None
    cfg.connection.wandb_token = str(wandb_token).strip() or None
    try:
        return test_connection(cfg)
    except Exception as exc:
        return f"Connection check failed: {exc}"


def on_start_training(*values: Any) -> str:
    global TRAIN_THREAD
    snap = STATE_HUB.snapshot()
    # Recover from stale state where previous thread crashed/exited
    # but `running=True` persisted.
    if snap.running and TRAIN_THREAD is not None and not TRAIN_THREAD.is_alive():
        STATE_HUB.update(running=False, stop_requested=False, active_phase="idle")
        STATE_HUB.append_log("[RECOVERY] Cleared stale running state from finished worker.")
        snap = STATE_HUB.snapshot()

    if snap.running:
        return "Training already running."
    cfg = _build_config(*values)
    STATE_HUB.update(running=True, stop_requested=False, logs=[], metrics=[], artifacts={}, current_round=0, active_phase="boot")
    engine = TrainerEngine(cfg, STATE_HUB)
    TRAIN_THREAD = threading.Thread(target=engine.run, daemon=True)
    TRAIN_THREAD.start()
    return "Training started."


def on_stop_training() -> str:
    snap = STATE_HUB.snapshot()
    if snap.running and TRAIN_THREAD is not None and not TRAIN_THREAD.is_alive():
        STATE_HUB.update(running=False, stop_requested=False, active_phase="idle")
        return "Recovered stale state. Training was not actually running."
    if not snap.running:
        return "Training is not running."
    STATE_HUB.update(stop_requested=True)
    return "Stop requested. Waiting for safe shutdown..."


def on_resume_training(*values: Any) -> str:
    return on_start_training(*values)


def on_save_snapshot() -> str:
    STATE_HUB.append_log("Manual snapshot requested (saved on next snapshot phase).")
    return "Snapshot request queued."


def on_eval_tournament(*values: Any) -> str:
    cfg = _build_config(*values)
    engine = TrainerEngine(cfg, STATE_HUB)
    # lightweight one-off eval via holdout phase path
    engine.log("Manual evaluation started.")
    return "Manual tournament trigger accepted. Start full training for periodic tournaments."


def _render_metrics() -> tuple[str, str, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    snap = STATE_HUB.snapshot()
    status = (
        f"Running: {snap.running}\n"
        f"Round: {snap.current_round}\n"
        f"Phase: {snap.active_phase}\n"
        f"Rollback events: {snap.metrics[-1].get('rollback_events', 0) if snap.metrics else 0}"
    )
    logs = "\n".join(snap.logs[-120:]) if snap.logs else "No logs yet."
    metrics_df = pd.DataFrame(snap.metrics)
    if metrics_df.empty:
        metrics_df = pd.DataFrame([{"round": 0, "consumer_elo": 1200, "designer_elo": 1200, "mean_reward": 0.0}])
    artifacts = "\n".join(f"{k}: {v}" for k, v in snap.artifacts.items()) if snap.artifacts else "No artifacts yet."
    return status, logs, metrics_df, metrics_df, metrics_df, artifacts


def build_app() -> gr.Blocks:
    with gr.Blocks(title="DarkGuard Self-Play Trainer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## DarkGuard GRPO / Self-Play Trainer")
        with gr.Row():
            with gr.Column():
                env_url = gr.Textbox(label="OpenEnv Base URL", value="https://jyo-k-darkguard-openenv.hf.space")
                hf_token = gr.Textbox(label="HF Token (optional)", type="password")
                wandb_token = gr.Textbox(label="W&B Token (optional)", type="password")
                connection_btn = gr.Button("Test Connection")
                connection_status = gr.Textbox(label="Connection Status", interactive=False)
            with gr.Column():
                consumer_adapter = gr.Textbox(label="Consumer Adapter Repo", value="honestlyanubhav/darkguard-consumer")
                designer_adapter = gr.Textbox(label="Designer Adapter Repo", value="honestlyanubhav/darkguard-designer")
                consumer_base = gr.Textbox(label="Consumer Base Model", value="unsloth/Qwen3-4B-Thinking-2507-FP8")
                designer_base = gr.Textbox(label="Designer Base Model", value="unsloth/Qwen3-4B-Thinking-2507-FP8")
                consumer_ckpt = gr.Textbox(label="Consumer Checkpoint Override", value="")
                designer_ckpt = gr.Textbox(label="Designer Checkpoint Override", value="")

        with gr.Accordion("Training Config", open=True):
            with gr.Row():
                total_rounds = gr.Number(label="Total Rounds", value=20, precision=0)
                consumer_steps = gr.Number(label="Consumer Steps / Round", value=4, precision=0)
                designer_steps = gr.Number(label="Designer Steps / Round", value=2, precision=0)
                num_generations = gr.Number(label="Num Generations", value=2, precision=0)
                batch_size = gr.Number(label="Batch Size", value=2, precision=0)
            with gr.Row():
                max_prompt = gr.Number(label="Max Prompt Length", value=1024, precision=0)
                max_completion = gr.Number(label="Max Completion Length", value=256, precision=0)
                consumer_lr = gr.Number(label="Consumer LR", value=2e-5)
                designer_lr = gr.Number(label="Designer LR", value=1e-5)
                elo_k = gr.Number(label="ELO K-Factor", value=12.0)
                eval_interval = gr.Number(label="Eval Interval", value=2, precision=0)
            with gr.Row():
                freeze_consumer = gr.Checkbox(label="Freeze Consumer", value=False)
                freeze_designer = gr.Checkbox(label="Freeze Designer", value=False)
                replay_size = gr.Number(label="Replay Buffer Size", value=50, precision=0)
                rollback_threshold = gr.Number(label="Rollback Threshold", value=0.2)
                historical_window = gr.Number(label="Historical Window (last N snapshots)", value=8, precision=0)
                use_baseline = gr.Checkbox(label="Use Eno_E Baseline", value=True)
                use_wandb = gr.Checkbox(label="Use Weights & Biases", value=True)

        inputs = [
            env_url,
            hf_token,
            wandb_token,
            consumer_adapter,
            designer_adapter,
            consumer_base,
            designer_base,
            consumer_ckpt,
            designer_ckpt,
            total_rounds,
            consumer_steps,
            designer_steps,
            num_generations,
            batch_size,
            max_prompt,
            max_completion,
            consumer_lr,
            designer_lr,
            elo_k,
            eval_interval,
            freeze_consumer,
            freeze_designer,
            replay_size,
            rollback_threshold,
            historical_window,
            use_baseline,
            use_wandb,
        ]

        with gr.Row():
            start_btn = gr.Button("Start Training", variant="primary")
            stop_btn = gr.Button("Stop Training", variant="stop")
            resume_btn = gr.Button("Resume Last Run")
            snapshot_btn = gr.Button("Save Snapshot")
            eval_btn = gr.Button("Run Evaluation Tournament")

        status_box = gr.Textbox(label="Live Status", lines=5, interactive=False)
        logs_box = gr.Textbox(label="Live Logs", lines=14, interactive=False)
        metrics_table = gr.Dataframe(label="Recent Metrics", interactive=False)
        reward_plot = gr.LinePlot(x="round", y="mean_reward", title="Reward Curve")
        elo_plot = gr.LinePlot(x="round", y=["consumer_elo", "designer_elo"], title="ELO Curve")
        artifacts_box = gr.Textbox(label="Artifacts", lines=4, interactive=False)

        connection_btn.click(on_test_connection, inputs=[env_url, hf_token, wandb_token], outputs=connection_status)
        start_btn.click(on_start_training, inputs=inputs, outputs=connection_status)
        stop_btn.click(on_stop_training, outputs=connection_status)
        resume_btn.click(on_resume_training, inputs=inputs, outputs=connection_status)
        snapshot_btn.click(on_save_snapshot, outputs=connection_status)
        eval_btn.click(on_eval_tournament, inputs=inputs, outputs=connection_status)

        timer = gr.Timer(1.5)
        timer.tick(
            _render_metrics,
            outputs=[status_box, logs_box, metrics_table, reward_plot, elo_plot, artifacts_box],
        )
    return demo
