"""Extract timestamped Slow packets consumed by the Fast residual student.

Slow is the frozen Stage-2 Teacher's null path, evaluated only at causal Slow
update rows. Each packet contains its predicted reference chunk plus a compact,
frozen representation of the Slow vision-language prefix.

The Slow update period and the serving latency are drawn from bands rather than
pinned to one operating point. A cache extracted at exactly 10 Hz and exactly
100 ms only describes the machine those numbers were measured on; randomizing
both is what lets one cache, and one student, cover a range of deployments. Key
times are additionally jittered so interpolation alphas are not stuck on the
action-rate grid.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import time
import typing

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi import transforms
from openpi.models import model as model_lib
from openpi.models import pi0_force
from openpi.models import slow_fast
from openpi.policies import rotation_6d as rot
from openpi.shared import nnx_utils
from openpi.training import config as config_lib
from openpi.training import data_loader
from openpi.training import fast_dataset
from openpi.training import weight_loaders


def _atomic_savez(path: pathlib.Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        np.savez(file, **arrays)
    temporary.replace(path)


def _sample_transform(data_config):
    return transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )


def _load_observation_batch(raw_dataset, transform, indices: np.ndarray):
    samples = []
    episodes = []
    timestamps = []
    for index in indices:
        raw = raw_dataset[int(index)]
        episodes.append(int(np.asarray(raw["episode_index"]).item()))
        timestamps.append(float(np.asarray(raw["timestamp"]).item()))
        sample = transform(raw)
        sample.pop("actions")
        samples.append(sample)
    batch = jax.tree.map(lambda *xs: np.stack(xs), *samples)
    observation = model_lib.Observation.from_dict(jax.tree.map(jnp.asarray, batch))
    return observation, np.asarray(episodes, dtype=np.int64), np.asarray(timestamps, dtype=np.float64)


def _band(values: list[float], name: str) -> tuple[float, float]:
    """Read a `VALUE` or `MIN MAX` argument as an ordered band."""
    if len(values) == 1:
        return (float(values[0]), float(values[0]))
    if len(values) == 2 and values[0] <= values[1]:
        return (float(values[0]), float(values[1]))
    raise ValueError(f"{name} takes one value or an ordered MIN MAX pair, got {values}")


def _checkpoint_params_path(checkpoint: pathlib.Path) -> pathlib.Path:
    checkpoint = checkpoint.resolve()
    return checkpoint if checkpoint.name == "params" else checkpoint / "params"


def _slow_sampler(model) -> tuple[typing.Callable, str]:
    """Slow is the frozen Teacher's null path. There is no separately distilled Pi0."""
    if not isinstance(model, pi0_force.Pi0_Guidance):
        raise TypeError(
            "Slow cache extraction requires the Stage-2 Teacher (Pi0_Guidance). "
            "The independently distilled Slow VLA has been removed."
        )
    if model.null_force_token is None:
        raise ValueError("Stage-2 checkpoint is missing null_force_token")
    return model.sample_nominal_actions_and_context, "teacher_null_path"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument(
        "--data-config-name",
        default=None,
        help=(
            "Loader to read rows from, when it differs from the model config. Needed to run a "
            "train-split model config over a held-out split, as the paired extraction does."
        ),
    )
    parser.add_argument("--stage3-dir", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--slow-rate-hz",
        type=float,
        nargs="+",
        default=list(slow_fast.DEFAULT_SLOW_RATE_RANGE_HZ),
        help=(
            "Slow update rate, as a single value or a MIN MAX band each interval is drawn from. "
            "Pass 0 to run Slow on every row; that full-rate cache can then be resampled to any "
            "lower rate offline via scripts/resample_slow_cache.py."
        ),
    )
    parser.add_argument(
        "--action-rate-hz",
        type=float,
        default=30.0,
        help=(
            "Spacing of the Teacher's chunk steps. This is fixed by the data the Teacher was "
            "trained on, not a deployment choice, so it is not randomized."
        ),
    )
    parser.add_argument(
        "--slow-latency-ms",
        type=float,
        nargs="+",
        default=[value * 1000.0 for value in slow_fast.DEFAULT_SLOW_LATENCY_RANGE_S],
        help=(
            "Slow inference time, as a single value or a MIN MAX band drawn per packet. A packet "
            "observed at t is not usable until t + latency, matching serving. Widen the band to "
            "cover every machine the student is meant to run on."
        ),
    )
    parser.add_argument(
        "--chunk-steps",
        type=int,
        default=slow_fast.DEFAULT_FAST_CHUNK_STEPS,
        help="Length of the reference rollout stored per row, matching the student's output chunk.",
    )
    parser.add_argument(
        "--update-jitter-ms",
        type=float,
        default=slow_fast.DEFAULT_UPDATE_JITTER_S * 1000.0,
        help="Jitter Slow update times so interpolation alphas are not stuck at 0/1.",
    )
    parser.add_argument(
        "--context-age-scale-ms",
        type=float,
        default=slow_fast.DEFAULT_CONTEXT_AGE_SCALE_S * 1000.0,
        help="Divisor for the Fast time token's context age. Serving must use the same value.",
    )
    parser.add_argument("--context-tokens", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Flow noise is keyed per dataset row from this seed. Match the Stage-3 seed so that "
            "A_ref and A_null start from the same noise and their gap is not sampling variance."
        ),
    )
    args = parser.parse_args()

    slow_rate_range_hz = _band(args.slow_rate_hz, "--slow-rate-hz")
    latency_range_s = tuple(value / 1000.0 for value in _band(args.slow_latency_ms, "--slow-latency-ms"))
    if min(slow_rate_range_hz) < 0:
        raise ValueError("Slow rate must be positive, or 0 for full-rate extraction")
    if (
        min(
            args.action_rate_hz,
            args.context_age_scale_ms,
            args.context_tokens,
            args.chunk_steps,
            args.batch_size,
            args.num_steps,
        )
        <= 0
    ):
        raise ValueError("All rates, dimensions, batch size, and flow steps must be positive")

    if min(latency_range_s) < 0 or args.update_jitter_ms < 0:
        raise ValueError("Slow latency and jitter must be non-negative")

    stage3 = fast_dataset.load_stage3_fast_arrays(args.stage3_dir, chunk_steps=args.chunk_steps)
    key_rows, _ = fast_dataset.select_slow_update_rows(
        stage3.episode_indices,
        stage3.timestamps,
        period_range_s=(
            fast_dataset.period_range_from_rates(slow_rate_range_hz) if min(slow_rate_range_hz) > 0 else None
        ),
        rng=args.seed,
    )
    key_episodes = stage3.episode_indices[key_rows]
    key_timestamps = fast_dataset.jitter_key_timestamps(
        stage3.timestamps[key_rows],
        jitter_s=args.update_jitter_ms / 1000.0,
        rng=args.seed + 1,
    )

    config = dataclasses.replace(config_lib.get_config(args.config_name), batch_size=args.batch_size, num_workers=0)
    source_config = config if args.data_config_name is None else config_lib.get_config(args.data_config_name)
    data_config = source_config.data.create(source_config.assets_dirs, config.model)
    raw_dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    if len(raw_dataset) != len(stage3.dataset_indices):
        raise ValueError(f"Slow dataset has {len(raw_dataset)} rows but Stage-3 has {len(stage3.dataset_indices)}")
    transform = _sample_transform(data_config)

    model_shape = nnx.eval_shape(config.model.create, jax.random.key(0))
    reference_state = nnx.state(model_shape).to_pure_dict()
    params = weight_loaders.CheckpointWeightLoader(
        str(_checkpoint_params_path(args.checkpoint)), missing_regex=r"a^"
    ).load(reference_state)
    model = config.model.load(params, remove_extra_params=False)
    sampler, slow_source = _slow_sampler(model)
    sample = nnx_utils.module_jit(sampler)
    print(f"Slow reference source: {slow_source}", flush=True)

    context_parts = []
    context_mask_parts = []
    action_parts = []
    start_time = time.monotonic()
    for batch_start in range(0, len(key_rows), args.batch_size):
        valid_rows = key_rows[batch_start : batch_start + args.batch_size]
        valid_count = len(valid_rows)
        padded_rows = np.pad(valid_rows, (0, args.batch_size - valid_count), mode="edge")
        observation, episodes, timestamps = _load_observation_batch(raw_dataset, transform, padded_rows)
        np.testing.assert_array_equal(episodes[:valid_count], key_episodes[batch_start : batch_start + valid_count])
        np.testing.assert_allclose(
            timestamps[:valid_count],
            stage3.timestamps[valid_rows],
            atol=1e-6,
            rtol=0,
        )
        noise = model_lib.row_keyed_noise(
            args.seed,
            stage3.dataset_indices[padded_rows],
            action_horizon=config.model.action_horizon,
            action_dim=config.model.action_dim,
        )
        actions, context, context_mask = sample(
            jax.random.key(args.seed), observation, num_steps=args.num_steps, noise=noise
        )
        pooled, pooled_mask = fast_dataset.pool_context_tokens(
            context,
            context_mask,
            num_tokens=args.context_tokens,
        )
        context_parts.append(np.asarray(pooled[:valid_count], dtype=np.float16))
        context_mask_parts.append(np.asarray(pooled_mask[:valid_count], dtype=np.bool_))
        action_parts.append(np.asarray(actions[:valid_count, :, : rot.ROBOT_DIMS], dtype=np.float32))
        completed = min(batch_start + valid_count, len(key_rows))
        if completed % (args.batch_size * 25) == 0 or completed == len(key_rows):
            rate = completed / max(time.monotonic() - start_time, 1e-9)
            print(f"Slow cache: {completed}/{len(key_rows)} packets ({rate:.2f}/s)", flush=True)

    context_tokens = np.concatenate(context_parts)
    context_mask = np.concatenate(context_mask_parts)
    action_chunks = np.concatenate(action_parts)
    key_ready_delays = fast_dataset.sample_ready_delays(
        len(key_rows),
        delay_range_s=latency_range_s,
        rng=args.seed + 2,
    )
    ready_mapping = fast_dataset.assign_ready_packets(
        stage3.episode_indices,
        stage3.timestamps,
        key_rows,
        key_timestamps,
        ready_delay_s=key_ready_delays,
    )
    row_ready = ready_mapping >= 0
    safe_mapping = np.where(row_ready, ready_mapping, 0)
    reference_actions, time_features = fast_dataset.build_reference_rollout(
        action_chunks,
        safe_mapping,
        stage3.timestamps,
        key_timestamps,
        action_period_s=1.0 / args.action_rate_hz,
        context_age_scale_s=args.context_age_scale_ms / 1000.0,
        chunk_steps=args.chunk_steps,
    )
    reference_actions = np.where(row_ready[:, None, None], reference_actions, 0)
    time_features = np.where(row_ready[:, None], time_features, 0)
    row_key_positions = safe_mapping
    arrays = {
        "key_dataset_indices": stage3.dataset_indices[key_rows],
        "key_episode_indices": key_episodes,
        "key_timestamps": key_timestamps,
        "context_tokens": context_tokens,
        "context_mask": context_mask,
        "action_chunks": action_chunks,
        "row_key_positions": row_key_positions,
        "reference_actions": reference_actions,
        "time_features": time_features,
        "context_age_scale_s": np.float64(args.context_age_scale_ms / 1000.0),
        "action_period_s": np.float64(1.0 / args.action_rate_hz),
        "key_ready_delays": key_ready_delays,
        "row_ready": row_ready,
        # Kept alongside the jittered times so that a later resample redraws
        # jitter from the lattice instead of stacking a second draw on top.
        "key_grid_timestamps": stage3.timestamps[key_rows],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_savez(args.output, arrays)
    null_pose = stage3.full_pose - stage3.residual_pose
    summary = {
        "format_version": 1,
        "config_name": args.config_name,
        "data_config_name": args.data_config_name or args.config_name,
        "checkpoint": str(args.checkpoint.resolve()),
        "stage3_dir": str(args.stage3_dir.resolve()),
        "rows": len(stage3.dataset_indices),
        "slow_packets": len(key_rows),
        "slow_rate_range_hz": list(slow_rate_range_hz),
        "action_rate_hz": args.action_rate_hz,
        "chunk_steps": args.chunk_steps,
        "context_age_scale_ms": args.context_age_scale_ms,
        "pooled_context_shape": list(context_tokens.shape),
        "action_chunk_shape": list(action_chunks.shape),
        "reference_rollout_shape": list(reference_actions.shape),
        "slow_latency_range_ms": [value * 1000.0 for value in latency_range_s],
        "update_jitter_ms": args.update_jitter_ms,
        "ready_row_fraction": float(np.mean(row_ready)),
        "elapsed_seconds": time.monotonic() - start_time,
        "slow_reference_source": slow_source,
        "chunk_head_vs_teacher_null_at_key_rows_mse": float(
            np.mean(np.square(action_chunks[:, : args.chunk_steps, : rot.POSE_DIMS] - null_pose[key_rows]))
        ),
        "reference_vs_teacher_null_on_ready_rows_mse": float(
            np.mean(np.square(reference_actions[row_ready][..., : rot.POSE_DIMS] - null_pose[row_ready]))
        )
        if row_ready.any()
        else None,
    }
    if row_ready.any():
        ages = stage3.timestamps[row_ready] - key_timestamps[ready_mapping[row_ready]]
        normalized_age, alpha = time_features[row_ready, 0], time_features[row_ready, 1]
        summary["time_features"] = {
            "context_age_s": {
                "min": float(ages.min()),
                "p50": float(np.percentile(ages, 50)),
                "max": float(ages.max()),
            },
            # The age token is clipped at 1.0. Any saturation means the student is blind
            # to staleness beyond the scale, which is the failure this cache exists to avoid.
            "saturated_age_fraction": float(np.mean(normalized_age >= 1.0)),
            "alpha_interior_fraction": float(np.mean((alpha > 1e-3) & (alpha < 1 - 1e-3))),
            "age_alpha_correlation": float(np.corrcoef(normalized_age, alpha)[0, 1]),
            "chunk_indices_used": sorted(set(np.floor(ages * args.action_rate_hz).astype(int).tolist())),
        }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
