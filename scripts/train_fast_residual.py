"""Train the selected single-step force-conditioned Fast residual student."""

from __future__ import annotations

import argparse
import json
import pathlib
import time

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb

from openpi.models import force_encoder
from openpi.models import slow_fast
from openpi.training import fast_dataset
from openpi.training import slow_fast_distillation


def _make_batch(arrays, cache, indices: np.ndarray) -> dict[str, jax.Array]:
    key_positions = cache.row_key_positions[indices]
    return {
        "slow_context": jnp.asarray(cache.context_tokens[key_positions], dtype=jnp.float32),
        "slow_context_mask": jnp.asarray(cache.context_mask[key_positions], dtype=jnp.bool_),
        "force_history": jnp.asarray(arrays.force_history[indices], dtype=jnp.float32),
        "force_history_mask": jnp.asarray(arrays.force_history_mask[indices], dtype=jnp.bool_),
        "state": jnp.asarray(arrays.state[indices], dtype=jnp.float32),
        "reference_action": jnp.asarray(cache.reference_actions[indices], dtype=jnp.float32),
        "time_features": jnp.asarray(cache.time_features[indices], dtype=jnp.float32),
        "target_residual": jnp.asarray(arrays.residual_pose[indices], dtype=jnp.float32),
        "target_full_pose": jnp.asarray(arrays.full_pose[indices], dtype=jnp.float32),
    }


def _loss(model, batch, loss_config, *, train: bool):
    predicted, _, _ = model(
        batch["slow_context"],
        batch["slow_context_mask"],
        batch["force_history"],
        batch["force_history_mask"],
        batch["state"],
        batch["reference_action"],
        batch["time_features"],
        train=train,
    )
    target = slow_fast_distillation.SingleStepTeacherTargets(
        full_action=batch["target_full_pose"],
        nominal_action=batch["reference_action"],
        residual_pose=batch["target_residual"],
    )
    total, parts = slow_fast_distillation.fast_residual_loss(
        predicted,
        batch["reference_action"],
        target,
        loss_config,
    )
    prediction_l2 = jnp.mean(jnp.linalg.norm(predicted, axis=-1))
    target_l2 = jnp.mean(jnp.linalg.norm(batch["target_residual"], axis=-1))
    return total, {
        **parts,
        "prediction_l2": prediction_l2,
        "target_l2": target_l2,
    }


def _save_params(model, output_dir: pathlib.Path, step: int) -> None:
    destination = output_dir / f"step-{step:05d}" / "params"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(destination, {"params": nnx.state(model, nnx.Param)}, force=True)


def _parameter_count(model) -> int:
    return sum(int(np.prod(value.shape)) for value in jax.tree.leaves(nnx.state(model, nnx.Param)))


def _model_config(profile: str) -> slow_fast.FastResidualConfig:
    if profile == "selected":
        return slow_fast.FastResidualConfig()
    if profile == "smoke":
        return slow_fast.FastResidualConfig(
            intent_dim=16,
            width=32,
            mlp_dim=64,
            num_heads=2,
            num_kv_heads=1,
            head_dim=16,
            force_encoder=force_encoder.ForceEncoderConfig(
                type="tcn",
                hidden_dims=(32, 32),
                dilations=(1, 2),
                dropout_rate=0.0,
                sampling_rate_hz=100,
                window_ms=100,
            ),
        )
    raise ValueError(f"Unknown model profile: {profile}")


def _jsonable_args(args) -> dict:
    return {name: str(value) if isinstance(value, pathlib.Path) else value for name, value in vars(args).items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-targets", type=pathlib.Path, required=True)
    parser.add_argument("--val-targets", type=pathlib.Path, required=True)
    parser.add_argument("--train-slow-cache", type=pathlib.Path, required=True)
    parser.add_argument("--val-slow-cache", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--peak-lr", type=float, default=1e-4)
    parser.add_argument("--end-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--save-interval", type=int, default=5_000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-samples", type=int, default=2_048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-project", default="forcevla")
    parser.add_argument("--wandb-name", default="button_press_fast_residual")
    parser.add_argument("--model-profile", choices=("selected", "smoke"), default="selected")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    if min(args.steps, args.batch_size, args.peak_lr, args.end_lr, args.save_interval, args.eval_interval) <= 0:
        raise ValueError("Steps, batch size, learning rates, and intervals must be positive")
    if not 0 <= args.warmup_steps < args.steps:
        raise ValueError("warmup-steps must be in [0, steps)")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_arrays = fast_dataset.load_stage3_fast_arrays(args.train_targets)
    val_arrays = fast_dataset.load_stage3_fast_arrays(args.val_targets)
    train_cache = fast_dataset.load_slow_cache(args.train_slow_cache, expected_rows=len(train_arrays.dataset_indices))
    val_cache = fast_dataset.load_slow_cache(args.val_slow_cache, expected_rows=len(val_arrays.dataset_indices))

    config = _model_config(args.model_profile)
    slow_context_dim = int(train_cache.context_tokens.shape[-1])
    if val_cache.context_tokens.shape[-1] != slow_context_dim:
        raise ValueError("Train and validation Slow contexts have different widths")
    model = slow_fast.FastStudentWithIntentProjector(
        config,
        slow_context_dim=slow_context_dim,
        rngs=nnx.Rngs(args.seed),
    )
    loss_config = slow_fast_distillation.FastDistillationLossConfig(
        residual_weight=1.0,
        reconstruction_weight=args.reconstruction_weight,
    )
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.peak_lr,
        warmup_steps=args.warmup_steps,
        decay_steps=args.steps,
        end_value=args.end_lr,
    )
    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(schedule, weight_decay=args.weight_decay),
        ),
        wrt=nnx.Param,
    )

    @nnx.jit
    def train_step(module, opt, batch):
        def loss_fn(m):
            return _loss(m, batch, loss_config, train=True)

        (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(module)
        grad_norm = optax.global_norm(grads)
        opt.update(grads)
        return {"loss": loss, "grad_norm": grad_norm, **metrics}

    @nnx.jit
    def eval_step(module, batch):
        return _loss(module, batch, loss_config, train=False)

    parameter_count = _parameter_count(model)
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config={
            **_jsonable_args(args),
            "model": {**vars(config), "force_encoder": vars(config.force_encoder)},
            "parameter_count": parameter_count,
        },
    )
    (args.output_dir / "wandb_id.txt").write_text(run.id)
    metadata = {
        "format_version": 1,
        "architecture": "100 Hz causal TCN + pooled Slow V-L context + one Gemma-style layer + 6D residual",
        "fast_output": "single-step normalized 6D pose residual; gripper remains owned by Slow",
        "parameter_count": parameter_count,
        "config": {**vars(config), "force_encoder": vars(config.force_encoder)},
        "loss": vars(loss_config),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"Fast parameters: {parameter_count:,}", flush=True)

    rng = np.random.default_rng(args.seed)
    eval_rng = np.random.default_rng(args.seed + 1)
    fixed_eval_indices = eval_rng.choice(
        len(val_arrays.dataset_indices),
        size=min(args.eval_samples, len(val_arrays.dataset_indices)),
        replace=False,
    )
    start_time = time.monotonic()
    last_metrics = None
    for step in range(args.steps):
        indices = rng.integers(0, len(train_arrays.dataset_indices), size=args.batch_size)
        metrics = train_step(model, optimizer, _make_batch(train_arrays, train_cache, indices))
        if step % 100 == 0:
            last_metrics = {f"train/{name}": float(value) for name, value in jax.device_get(metrics).items()}
            last_metrics["train/learning_rate"] = float(schedule(step))
            last_metrics["train/steps_per_second"] = (step + 1) / max(time.monotonic() - start_time, 1e-9)
            wandb.log(last_metrics, step=step)
            print(
                f"step={step} loss={last_metrics['train/loss']:.6f} "
                f"residual={last_metrics['train/residual_loss']:.6f} "
                f"reconstruction={last_metrics['train/reconstruction_loss']:.6f}",
                flush=True,
            )

        if step % args.eval_interval == 0 or step == args.steps - 1:
            totals = []
            for start in range(0, len(fixed_eval_indices), args.batch_size):
                batch_indices = fixed_eval_indices[start : start + args.batch_size]
                if len(batch_indices) < args.batch_size:
                    batch_indices = np.pad(batch_indices, (0, args.batch_size - len(batch_indices)), mode="edge")
                loss, parts = eval_step(model, _make_batch(val_arrays, val_cache, batch_indices))
                totals.append({"loss": loss, **parts})
            val_metrics = {
                f"val/{name}": float(np.mean([float(jax.device_get(item[name])) for item in totals]))
                for name in totals[0]
            }
            zero_residual_mse = float(np.mean(np.square(val_arrays.residual_pose[fixed_eval_indices])))
            val_metrics["val/zero_residual_mse"] = zero_residual_mse
            wandb.log(val_metrics, step=step)
            print(
                f"val step={step} residual={val_metrics['val/residual_loss']:.6f} "
                f"zero_baseline={zero_residual_mse:.6f}",
                flush=True,
            )
            last_metrics = {**(last_metrics or {}), **val_metrics}

        completed_step = step + 1
        if completed_step % args.save_interval == 0 or completed_step == args.steps:
            _save_params(model, args.output_dir, completed_step)

    summary = {
        **metadata,
        "completed_steps": args.steps,
        "elapsed_seconds": time.monotonic() - start_time,
        "final_metrics": last_metrics,
        "wandb_run_id": run.id,
        "wandb_url": run.url,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    wandb.finish()
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
