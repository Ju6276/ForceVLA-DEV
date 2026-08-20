"""Extract timestamped Slow packets consumed by the Fast residual student.

The force-free Slow VLA is evaluated only at causal Slow update rows (10 Hz by
default).  Each packet contains its predicted reference chunk plus a compact,
frozen representation of the Slow vision-language prefix.  References are
then sampled at every dataset timestamp using pose interpolation and gripper
zero-order hold.
"""

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

from openpi import transforms
from openpi.models import model as model_lib
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


def _checkpoint_params_path(checkpoint: pathlib.Path) -> pathlib.Path:
    checkpoint = checkpoint.resolve()
    return checkpoint if checkpoint.name == "params" else checkpoint / "params"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--stage3-dir", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--slow-rate-hz", type=float, default=10.0)
    parser.add_argument("--action-rate-hz", type=float, default=30.0)
    parser.add_argument("--context-age-scale-ms", type=float, default=100.0)
    parser.add_argument("--context-tokens", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if min(
        args.slow_rate_hz,
        args.action_rate_hz,
        args.context_age_scale_ms,
        args.context_tokens,
        args.batch_size,
        args.num_steps,
    ) <= 0:
        raise ValueError("All rates, dimensions, batch size, and flow steps must be positive")

    stage3 = fast_dataset.load_stage3_fast_arrays(args.stage3_dir)
    key_rows, row_key_positions = fast_dataset.select_slow_update_rows(
        stage3.episode_indices,
        stage3.timestamps,
        update_period_s=1.0 / args.slow_rate_hz,
    )
    key_episodes = stage3.episode_indices[key_rows]
    key_timestamps = stage3.timestamps[key_rows]

    config = dataclasses.replace(config_lib.get_config(args.config_name), batch_size=args.batch_size, num_workers=0)
    data_config = config.data.create(config.assets_dirs, config.model)
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
    sample = nnx_utils.module_jit(model.sample_actions_and_context)

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
            timestamps[:valid_count], key_timestamps[batch_start : batch_start + valid_count], atol=1e-6, rtol=0
        )
        rng = jax.random.fold_in(jax.random.key(args.seed), batch_start // args.batch_size)
        actions, context, context_mask = sample(rng, observation, num_steps=args.num_steps)
        pooled, pooled_mask = fast_dataset.pool_context_tokens(
            context,
            context_mask,
            num_tokens=args.context_tokens,
        )
        context_parts.append(np.asarray(pooled[:valid_count], dtype=np.float16))
        context_mask_parts.append(np.asarray(pooled_mask[:valid_count], dtype=np.bool_))
        action_parts.append(np.asarray(actions[:valid_count, :, :7], dtype=np.float32))
        completed = min(batch_start + valid_count, len(key_rows))
        if completed % (args.batch_size * 25) == 0 or completed == len(key_rows):
            rate = completed / max(time.monotonic() - start_time, 1e-9)
            print(f"Slow cache: {completed}/{len(key_rows)} packets ({rate:.2f}/s)", flush=True)

    context_tokens = np.concatenate(context_parts)
    context_mask = np.concatenate(context_mask_parts)
    action_chunks = np.concatenate(action_parts)
    reference_actions, time_features = fast_dataset.build_reference_rollout(
        action_chunks,
        row_key_positions,
        stage3.timestamps,
        key_timestamps,
        action_period_s=1.0 / args.action_rate_hz,
        context_age_scale_s=args.context_age_scale_ms / 1000.0,
    )
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
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_savez(args.output, arrays)
    summary = {
        "format_version": 1,
        "config_name": args.config_name,
        "checkpoint": str(args.checkpoint.resolve()),
        "stage3_dir": str(args.stage3_dir.resolve()),
        "rows": len(stage3.dataset_indices),
        "slow_packets": len(key_rows),
        "slow_rate_hz": args.slow_rate_hz,
        "action_rate_hz": args.action_rate_hz,
        "context_age_scale_ms": args.context_age_scale_ms,
        "pooled_context_shape": list(context_tokens.shape),
        "action_chunk_shape": list(action_chunks.shape),
        "reference_vs_teacher_null_first_pose_mse": float(
            np.mean(np.square(reference_actions[:, :6] - (stage3.full_pose - stage3.residual_pose)))
        ),
        "elapsed_seconds": time.monotonic() - start_time,
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
