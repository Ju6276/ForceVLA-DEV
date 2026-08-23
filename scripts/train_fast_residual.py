"""Train the selected short-chunk force-conditioned Fast residual student."""

from __future__ import annotations

import argparse
import dataclasses
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
from openpi.shared import normalize as normalize_lib
from openpi.training import fast_dataset
from openpi.training import slow_fast_distillation


def _band(values: list[float], name: str) -> tuple[float, float]:
    """Read a `VALUE` or `MIN MAX` argument as an ordered band."""
    if len(values) == 1:
        return (float(values[0]), float(values[0]))
    if len(values) == 2 and values[0] <= values[1]:
        return (float(values[0]), float(values[1]))
    raise ValueError(f"{name} takes one value or an ordered MIN MAX pair, got {values}")


def _ready_row_indices(cache, n_rows: int) -> np.ndarray:
    ready = cache.row_ready
    if ready is None:
        return np.arange(n_rows, dtype=np.int64)
    if len(ready) != n_rows:
        raise ValueError(f"row_ready has {len(ready)} entries but the dataset has {n_rows} rows")
    indices = np.flatnonzero(ready)
    if len(indices) == 0:
        raise ValueError("Slow cache has no rows whose packet would already be ready")
    return indices


def _staleness_target(arrays, cache, indices: np.ndarray, *, pose_dims: int, state_to_action_scale) -> np.ndarray:
    """The drift the reference accumulated, expressed on the current row's base.

    `A_ref` is a delta from the state of the *key* row while `A_null(t)` is a delta
    from the current row's state, so their difference is not the drift: it is missing
    the base-state gap between the two rows. Left out, the target's rotation part is
    almost entirely wrong, since the arm's orientation moves as much between those two
    rows as the reference itself drifts. The gap is converted into action units
    because that is what the head's output is scaled by.
    """
    key_rows = cache.key_dataset_indices[cache.row_key_positions[indices]]
    base_gap = arrays.state[indices, :pose_dims] - arrays.state[key_rows, :pose_dims]
    drift = arrays.null_pose[indices] - cache.reference_actions[indices, :, :pose_dims]
    return (drift + (state_to_action_scale * base_gap)[:, None, :]).astype(np.float32)


def _make_batch(arrays, cache, indices: np.ndarray, *, staleness_target=None) -> dict[str, jax.Array]:
    key_positions = cache.row_key_positions[indices]
    reference_chunk = jnp.asarray(cache.reference_actions[indices], dtype=jnp.float32)
    if staleness_target is not None:
        staleness_target = jnp.asarray(staleness_target, dtype=jnp.float32)
    return {
        "target_staleness": staleness_target,
        "slow_context": jnp.asarray(cache.context_tokens[key_positions], dtype=jnp.float32),
        "slow_context_mask": jnp.asarray(cache.context_mask[key_positions], dtype=jnp.bool_),
        "force_history": jnp.asarray(arrays.force_history[indices], dtype=jnp.float32),
        "force_history_mask": jnp.asarray(arrays.force_history_mask[indices], dtype=jnp.bool_),
        "state": jnp.asarray(arrays.state[indices], dtype=jnp.float32),
        # The student is conditioned on the reference at the current tick only; the
        # whole rollout is needed to score the reconstruction of every emitted step.
        "reference_action": reference_chunk[:, 0],
        "reference_chunk": reference_chunk,
        "time_features": jnp.asarray(cache.time_features[indices], dtype=jnp.float32),
        "target_residual": jnp.asarray(arrays.residual_pose[indices], dtype=jnp.float32),
        "target_full_pose": jnp.asarray(arrays.full_pose[indices], dtype=jnp.float32),
        "target_null_pose": jnp.asarray(arrays.null_pose[indices], dtype=jnp.float32),
    }


def _loss(model, batch, loss_config, *, train: bool):
    predicted, staleness, _, _ = model(
        batch["slow_context"],
        batch["slow_context_mask"],
        batch["force_history"],
        batch["force_history_mask"],
        batch["state"],
        batch["reference_action"],
        batch["time_features"],
        train=train,
    )
    pose_dims = loss_config.pose_dims
    staleness_target = batch["target_staleness"] if staleness is not None else None
    target = slow_fast_distillation.FastChunkTargets(
        full_action=batch["target_full_pose"],
        nominal_action=batch["reference_chunk"],
        residual_pose=batch["target_residual"],
        staleness_pose=staleness_target,
    )
    total, parts = slow_fast_distillation.fast_residual_loss(
        predicted,
        batch["reference_chunk"],
        target,
        loss_config,
        predicted_staleness=staleness,
    )
    prediction_l2 = jnp.mean(jnp.linalg.norm(predicted, axis=-1))
    target_l2 = jnp.mean(jnp.linalg.norm(batch["target_residual"], axis=-1))
    # What the robot actually executes is A_ref + force + staleness. Its error against
    # the Teacher no longer splits into a trained and an untrained half: this is the
    # zero-correction baseline for the staleness head, not an error nobody owns.
    slow_error = jnp.mean(jnp.square(batch["reference_chunk"][..., :pose_dims] - batch["target_null_pose"]))
    metrics = {
        **parts,
        "prediction_l2": prediction_l2,
        "target_l2": target_l2,
        "slow_reference_error": slow_error,
        "deployment_error": parts["reconstruction_loss"],
    }
    if staleness is not None:
        metrics["staleness_prediction_l2"] = jnp.mean(jnp.linalg.norm(staleness, axis=-1))
    return total, metrics


def _save_params(model, output_dir: pathlib.Path, step: int) -> None:
    # TensorStore requires the checkpoint destination to be absolute even
    # when the user supplied a repository-relative output directory.
    destination = (output_dir / f"step-{step:05d}" / "params").resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(destination, {"params": nnx.state(model, nnx.Param)}, force=True)


def _parameter_count(model) -> int:
    return sum(int(np.prod(value.shape)) for value in jax.tree.leaves(nnx.state(model, nnx.Param)))


def _model_config(profile: str, *, chunk_steps: int) -> slow_fast.FastResidualConfig:
    if profile == "selected":
        return slow_fast.FastResidualConfig(chunk_steps=chunk_steps)
    if profile == "smoke":
        return slow_fast.FastResidualConfig(
            chunk_steps=chunk_steps,
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
    parser.add_argument(
        "--norm-stats-dir",
        type=pathlib.Path,
        default=None,
        help=(
            "Directory holding norm_stats.json. Required with the staleness head: its "
            "target spans two different base states, and converting that gap into "
            "action units needs the state and action scales."
        ),
    )
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--peak-lr", type=float, default=1e-4)
    parser.add_argument("--end-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--reconstruction-weight",
        type=float,
        default=0.0,
        help=(
            "Weight on the executed action A_ref + force + staleness. With both heads "
            "supervised on their own targets this term is redundant at the optimum, so "
            "it stays off by default; raise it to let the heads trade errors off."
        ),
    )
    parser.add_argument(
        "--staleness-weight",
        type=float,
        default=1.0,
        help=(
            "Weight on the stale-reference correction A_null(t) - A_ref(t). This is the "
            "term that makes the objective depend on how old the cached context is."
        ),
    )
    parser.add_argument(
        "--no-staleness-head",
        action="store_true",
        help=(
            "Ablation: single-head student, as before the staleness head existed. The "
            "force residual then carries no information about context age, and the "
            "stale-reference error is left entirely unoptimized."
        ),
    )
    parser.add_argument(
        "--chunk-steps",
        type=int,
        default=slow_fast.DEFAULT_FAST_CHUNK_STEPS,
        help=(
            "Residual steps emitted per Fast tick, spaced by the Teacher action period. "
            "More than one lets the fast loop run below the action rate, or stall, without gaps."
        ),
    )
    parser.add_argument(
        "--step-decay",
        type=float,
        default=0.5,
        help="Geometric down-weighting of later chunk steps, which only execute when Fast falls behind.",
    )
    parser.add_argument(
        "--timing-resample-interval",
        type=int,
        default=500,
        help=(
            "Redraw the Slow timing of the training cache every N steps. A cache stores one "
            "random realization of the latency/rate band; redrawing covers the whole band instead. "
            "Requires a full-rate train cache. Pass 0 to train on the stored realization."
        ),
    )
    parser.add_argument(
        "--slow-rate-hz",
        type=float,
        nargs="+",
        default=list(slow_fast.DEFAULT_SLOW_RATE_RANGE_HZ),
        help="Slow update band redrawn each time, as a single value or MIN MAX.",
    )
    parser.add_argument(
        "--slow-latency-ms",
        type=float,
        nargs="+",
        default=[value * 1000.0 for value in slow_fast.DEFAULT_SLOW_LATENCY_RANGE_S],
        help="Slow serving latency band redrawn each time, as a single value or MIN MAX.",
    )
    parser.add_argument("--save-interval", type=int, default=5_000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-samples", type=int, default=2_048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-project", default="forcevla")
    parser.add_argument("--wandb-name", default="button_press_fast_residual")
    parser.add_argument("--model-profile", choices=("selected", "smoke"), default="selected")
    parser.add_argument(
        "--no-reference-token",
        action="store_true",
        help=(
            "Ablation: hide the Slow reference action from Fast. The residual target does not "
            "depend on it, so this measures whether it carries anything the time features do not."
        ),
    )
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    if min(args.steps, args.batch_size, args.peak_lr, args.end_lr, args.save_interval, args.eval_interval) <= 0:
        raise ValueError("Steps, batch size, learning rates, and intervals must be positive")
    if not 0 <= args.warmup_steps < args.steps:
        raise ValueError("warmup-steps must be in [0, steps)")
    if args.chunk_steps <= 0:
        raise ValueError("chunk-steps must be positive")
    if args.timing_resample_interval < 0:
        raise ValueError("timing-resample-interval must be non-negative")
    slow_rate_band = _band(args.slow_rate_hz, "--slow-rate-hz")
    slow_latency_band = tuple(value / 1000.0 for value in _band(args.slow_latency_ms, "--slow-latency-ms"))
    if min(slow_rate_band) <= 0 or min(slow_latency_band) < 0:
        raise ValueError("Slow rates must be positive and latencies non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = _model_config(args.model_profile, chunk_steps=args.chunk_steps)
    if args.no_reference_token:
        config = dataclasses.replace(config, use_reference_token=False)
    if args.no_staleness_head:
        config = dataclasses.replace(config, predict_staleness=False)

    state_to_action_scale = None
    if config.predict_staleness:
        if args.norm_stats_dir is None or not (args.norm_stats_dir / "norm_stats.json").is_file():
            raise ValueError(
                "--norm-stats-dir must point at a directory containing norm_stats.json when the "
                "staleness head is enabled, or pass --no-staleness-head. The target is a gap "
                "between two base states, and without the scales it cannot be put in action units."
            )
        norm_stats = normalize_lib.load(args.norm_stats_dir)
        pose = config.pose_dims
        state_to_action_scale = (
            (np.asarray(norm_stats["state"].std, dtype=np.float64)[:pose] + 1e-6)
            / (np.asarray(norm_stats["actions"].std, dtype=np.float64)[:pose] + 1e-6)
        ).astype(np.float32)

    train_arrays = fast_dataset.load_stage3_fast_arrays(args.train_targets, chunk_steps=config.chunk_steps)
    val_arrays = fast_dataset.load_stage3_fast_arrays(args.val_targets, chunk_steps=config.chunk_steps)
    train_cache = fast_dataset.load_slow_cache(args.train_slow_cache, expected_rows=len(train_arrays.dataset_indices))
    val_cache = fast_dataset.load_slow_cache(args.val_slow_cache, expected_rows=len(val_arrays.dataset_indices))
    for name, cache in (("train", train_cache), ("validation", val_cache)):
        if cache.chunk_steps != config.chunk_steps:
            raise ValueError(
                f"The {name} Slow cache holds {cache.chunk_steps}-step reference rollouts but the "
                f"student emits {config.chunk_steps}; re-extract the cache with a matching --chunk-steps"
            )

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
        staleness_weight=args.staleness_weight,
        step_decay=args.step_decay,
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
        # 2 adds the staleness head, which deployment must add to the reference on top
        # of the force residual. A version-1 run must not be composed as if it had one.
        "format_version": 2,
        "architecture": "100 Hz causal TCN + pooled Slow V-L context + one Gemma-style layer + 9D pose residual",
        "fast_output": (
            f"{config.chunk_steps}-step normalized xyz+6D residual chunk spaced by the Teacher "
            "action period; gripper remains owned by Slow"
            + (
                "; a second force-blind head emits the stale-reference correction and the "
                "executed action is their sum"
                if config.predict_staleness
                else "; single head, no stale-reference correction"
            )
        ),
        "predict_staleness": config.predict_staleness,
        "parameter_count": parameter_count,
        "config": {**vars(config), "force_encoder": vars(config.force_encoder)},
        "loss": vars(loss_config),
        # The band actually trained on. It is not recoverable from the train cache:
        # that cache is extracted full-rate so the timing can be redrawn here, which
        # leaves its own recorded band degenerate. Deployment reads this to check the
        # measured Slow timing against what the student has seen.
        "trained_timing": {
            "slow_rate_range_hz": list(slow_rate_band),
            "slow_latency_range_ms": [value * 1000.0 for value in slow_latency_band],
            "timing_resample_interval": args.timing_resample_interval,
        },
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"Fast parameters: {parameter_count:,}", flush=True)

    train_cache_source = train_cache
    timing_rng = np.random.default_rng(args.seed + 7)

    # The validation cache is deliberately never redrawn: comparable curves need a
    # fixed timing realization, and it is the held-out timing the model never trains on.
    def redraw_train_timing(seed: int):
        try:
            return fast_dataset.resample_slow_cache(
                train_cache_source,
                train_arrays.episode_indices,
                train_arrays.timestamps,
                slow_rate_range_hz=slow_rate_band,
                ready_delay_range_s=slow_latency_band,
                rng=seed,
            )
        except ValueError as error:
            raise ValueError(
                "Redrawing the Slow timing needs a train cache extracted at full rate "
                "(--slow-rate-hz 0); pass --timing-resample-interval 0 to train on the "
                f"realization stored in the cache instead. Underlying error: {error}"
            ) from error

    if args.timing_resample_interval:
        train_cache = redraw_train_timing(int(timing_rng.integers(2**31 - 1)))

    train_pool = _ready_row_indices(train_cache, len(train_arrays.dataset_indices))
    val_pool = _ready_row_indices(val_cache, len(val_arrays.dataset_indices))
    rng = np.random.default_rng(args.seed)
    eval_rng = np.random.default_rng(args.seed + 1)
    fixed_eval_indices = eval_rng.choice(
        val_pool,
        size=min(args.eval_samples, len(val_pool)),
        replace=False,
    )
    def _staleness(arrays, cache, indices):
        if state_to_action_scale is None:
            return None
        return _staleness_target(
            arrays,
            cache,
            indices,
            pose_dims=config.pose_dims,
            state_to_action_scale=state_to_action_scale,
        )

    start_time = time.monotonic()
    last_metrics = None
    for step in range(args.steps):
        if args.timing_resample_interval and step and step % args.timing_resample_interval == 0:
            train_cache = redraw_train_timing(int(timing_rng.integers(2**31 - 1)))
            train_pool = _ready_row_indices(train_cache, len(train_arrays.dataset_indices))
            wandb.log(
                {
                    "timing/ready_row_fraction": float(np.mean(train_cache.row_ready)),
                    "timing/mean_normalized_age": float(np.mean(train_cache.time_features[train_pool, 0])),
                    "timing/saturated_age_fraction": float(
                        np.mean(train_cache.time_features[train_pool, 0] >= 1.0)
                    ),
                    "timing/slow_packets": len(train_cache.key_timestamps),
                },
                step=step,
            )
        indices = rng.choice(train_pool, size=args.batch_size, replace=True)
        metrics = train_step(
            model,
            optimizer,
            _make_batch(
                train_arrays,
                train_cache,
                indices,
                staleness_target=_staleness(train_arrays, train_cache, indices),
            ),
        )
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
            eval_indices = (
                np.asarray(val_pool, dtype=np.int64)
                if step == args.steps - 1
                else fixed_eval_indices
            )
            totals: list[tuple[int, dict]] = []
            for start in range(0, len(eval_indices), args.batch_size):
                batch_indices = eval_indices[start : start + args.batch_size]
                loss, parts = eval_step(
                    model,
                    _make_batch(
                        val_arrays,
                        val_cache,
                        batch_indices,
                        staleness_target=_staleness(val_arrays, val_cache, batch_indices),
                    ),
                )
                totals.append((len(batch_indices), {"loss": loss, **parts}))
            total_examples = sum(count for count, _ in totals)
            val_metrics = {
                f"val/{name}": sum(
                    count * float(jax.device_get(item[name])) for count, item in totals
                )
                / total_examples
                for name in totals[0][1]
            }
            # Both baselines are step-0 only, to match the unweighted step-0 losses.
            zero_residual_mse = float(np.mean(np.square(val_arrays.residual_pose[eval_indices, 0])))
            reference_only_mse = float(
                np.mean(
                    np.square(
                        val_cache.reference_actions[eval_indices, 0, : config.pose_dims]
                        - val_arrays.full_pose[eval_indices, 0]
                    )
                )
            )
            val_metrics["val/zero_residual_mse"] = zero_residual_mse
            val_metrics["val/reference_only_reconstruction_mse"] = reference_only_mse
            val_metrics["val/residual_gain_vs_zero"] = 1.0 - val_metrics["val/residual_loss_step0"] / max(
                zero_residual_mse, 1e-12
            )
            if config.predict_staleness:
                # Emitting zero is what the single-head student effectively did, so this
                # is the baseline the staleness head has to beat to be worth its capacity.
                zero_staleness_mse = float(
                    np.mean(np.square(_staleness(val_arrays, val_cache, eval_indices)[:, 0]))
                )
                val_metrics["val/zero_staleness_mse"] = zero_staleness_mse
                val_metrics["val/staleness_gain_vs_zero"] = 1.0 - val_metrics["val/staleness_loss_step0"] / max(
                    zero_staleness_mse, 1e-12
                )
            val_metrics["val/reconstruction_gain_vs_reference"] = 1.0 - val_metrics[
                "val/reconstruction_loss_step0"
            ] / max(reference_only_mse, 1e-12)
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
