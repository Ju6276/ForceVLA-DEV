"""Extract resumable full/null ForceVLA targets for slow-fast distillation.

The extractor runs one Stage-2 Teacher checkpoint twice from a shared
vision-language prefix and the exact same flow noise. Outputs are written in
small normalized-space NPZ shards so a long extraction can be resumed without
recomputing completed rows.
"""

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
from openpi.policies import rotation_6d as rot
from openpi.shared import nnx_utils
from openpi.training import config as config_lib
from openpi.training import data_loader
from openpi.training import fast_dataset
from openpi.training import weight_loaders

FORMAT_VERSION = 2


def _atomic_write_json(path: pathlib.Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as file:
        json.dump(value, file, indent=2)
    temporary.replace(path)


def _atomic_savez(path: pathlib.Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **arrays)
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


def _load_batch(raw_dataset, transform, indices: np.ndarray) -> tuple[model_lib.Observation, dict[str, np.ndarray]]:
    samples = []
    metadata = {"episode_indices": [], "frame_indices": [], "timestamps": []}
    for index in indices:
        raw = raw_dataset[int(index)]
        metadata["episode_indices"].append(int(np.asarray(raw["episode_index"]).item()))
        metadata["frame_indices"].append(int(np.asarray(raw["frame_index"]).item()))
        metadata["timestamps"].append(float(np.asarray(raw["timestamp"]).item()))
        samples.append(transform(raw))

    batch = jax.tree.map(lambda *xs: np.stack(xs), *samples)
    expert_actions = np.asarray(batch.pop("actions"), dtype=np.float32)
    observation = model_lib.Observation.from_dict(jax.tree.map(jnp.asarray, batch))
    arrays = {
        "episode_indices": np.asarray(metadata["episode_indices"], dtype=np.int64),
        "frame_indices": np.asarray(metadata["frame_indices"], dtype=np.int64),
        "timestamps": np.asarray(metadata["timestamps"], dtype=np.float64),
        "normalized_state": np.asarray(observation.state, dtype=np.float32),
        "normalized_expert_actions": expert_actions,
    }
    if observation.force_history is not None:
        arrays["normalized_force_history"] = np.asarray(observation.force_history, dtype=np.float32)
        arrays["force_history_mask"] = np.asarray(observation.force_history_mask, dtype=np.bool_)
    return observation, arrays


def _shard_path(shard_dir: pathlib.Path, start: int, stop: int) -> pathlib.Path:
    return shard_dir / f"rows_{start:06d}_{stop - 1:06d}.npz"


def _validate_existing_shard(path: pathlib.Path, start: int, stop: int) -> None:
    with np.load(path, allow_pickle=False) as shard:
        expected = np.arange(start, stop, dtype=np.int64)
        np.testing.assert_array_equal(shard["dataset_indices"], expected)
        required = {
            "episode_indices",
            "frame_indices",
            "timestamps",
            "normalized_state",
            "normalized_expert_actions",
            "normalized_full_actions",
            "normalized_null_actions",
            "normalized_pose_residual",
            "normalized_force_history",
            "force_history_mask",
        }
        missing = required.difference(shard.files)
        if missing:
            raise ValueError(f"Existing shard {path} is missing arrays: {sorted(missing)}")
        if shard["normalized_pose_residual"].shape[-1] != rot.POSE_DIMS:
            raise ValueError(f"Existing shard {path} has a pose residual that is not {rot.POSE_DIMS}D")


def _completed_rows(shard_dir: pathlib.Path, ranges: list[tuple[int, int]]) -> int:
    return sum(stop - start for start, stop in ranges if _shard_path(shard_dir, start, stop).is_file())


def _write_manifest(
    output_dir: pathlib.Path,
    *,
    args,
    dataset_size: int,
    extraction_size: int,
    ranges: list[tuple[int, int]],
    elapsed_seconds: float,
) -> None:
    shard_dir = output_dir / "shards"
    completed = _completed_rows(shard_dir, ranges)
    manifest = {
        "format_version": FORMAT_VERSION,
        "space": "ForceVLA normalized action space",
        "residual_definition": "normalized_full_actions[..., :9] - normalized_null_actions[..., :9]",
        "target_context_mode": "cached_slow_packet" if args.context_cache is not None else "current_observation",
        "context_cache": None if args.context_cache is None else str(args.context_cache.resolve()),
        "flow_noise_key": "context_dataset_index" if args.context_cache is not None else "current_dataset_index",
        "config_name": args.config_name,
        "data_config_name": args.data_config_name or args.config_name,
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_size": dataset_size,
        "extraction_size": extraction_size,
        "completed_rows": completed,
        "complete": completed == extraction_size,
        "batch_size": args.batch_size,
        "shard_size": args.shard_size,
        "num_flow_steps": args.num_steps,
        "seed": args.seed,
        "elapsed_seconds_this_run": elapsed_seconds,
        "shards": [
            {
                "start": start,
                "stop": stop,
                "path": str(_shard_path(shard_dir, start, stop).relative_to(output_dir)),
                "complete": _shard_path(shard_dir, start, stop).is_file(),
            }
            for start, stop in ranges
        ],
    }
    _atomic_write_json(output_dir / "manifest.json", manifest)


def _validate_resume_manifest(output_dir: pathlib.Path, *, args, extraction_size: int) -> None:
    path = output_dir / "manifest.json"
    if not path.is_file():
        return
    with path.open() as file:
        existing = json.load(file)
    expected = {
        "format_version": FORMAT_VERSION,
        "config_name": args.config_name,
        "data_config_name": args.data_config_name or args.config_name,
        "checkpoint": str(args.checkpoint.resolve()),
        "extraction_size": extraction_size,
        "batch_size": args.batch_size,
        "shard_size": args.shard_size,
        "num_flow_steps": args.num_steps,
        "seed": args.seed,
        "target_context_mode": "cached_slow_packet" if args.context_cache is not None else "current_observation",
        "context_cache": None if args.context_cache is None else str(args.context_cache.resolve()),
    }
    mismatches = {
        name: {"existing": existing.get(name), "requested": value}
        for name, value in expected.items()
        if existing.get(name) != value
    }
    if mismatches:
        raise ValueError(
            f"Refusing to mix incompatible extraction settings in {output_dir}: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )


def _summarize(output_dir: pathlib.Path, ranges: list[tuple[int, int]], extraction_size: int) -> dict:
    dataset_indices = []
    episode_indices = []
    frame_indices = []
    timestamps = []
    residual_pose_l2 = []
    residual_first_pose_l2 = []
    # The model emits `action_dim` values but only the leading ROBOT_DIMS drive the
    # robot; averaging the padded tail in dilutes every error toward zero and makes
    # the force gain look smaller than it is.
    groups = {
        "all": slice(0, rot.ROBOT_DIMS),
        "xyz": slice(0, rot.XYZ_DIMS),
        "rotation_6d": slice(rot.XYZ_DIMS, rot.POSE_DIMS),
        "gripper": slice(rot.POSE_DIMS, rot.ROBOT_DIMS),
    }
    squared_error = {name: {"full": 0.0, "null": 0.0, "count": 0} for name in groups}
    mask_valid = 0
    mask_count = 0
    for start, stop in ranges:
        path = _shard_path(output_dir / "shards", start, stop)
        if not path.is_file():
            continue
        _validate_existing_shard(path, start, stop)
        with np.load(path, allow_pickle=False) as shard:
            dataset_indices.append(shard["dataset_indices"])
            episode_indices.append(shard["episode_indices"])
            frame_indices.append(shard["frame_indices"])
            timestamps.append(shard["timestamps"])
            residual = shard["normalized_pose_residual"]
            residual_pose_l2.append(np.linalg.norm(residual, axis=-1))
            residual_first_pose_l2.append(np.linalg.norm(residual[:, 0], axis=-1))
            expert = shard["normalized_expert_actions"]
            full = shard["normalized_full_actions"]
            null = shard["normalized_null_actions"]
            for name, dims in groups.items():
                target = expert[..., dims]
                squared_error[name]["full"] += float(np.sum(np.square(full[..., dims] - target), dtype=np.float64))
                squared_error[name]["null"] += float(np.sum(np.square(null[..., dims] - target), dtype=np.float64))
                squared_error[name]["count"] += target.size
            mask = shard["force_history_mask"]
            mask_valid += int(np.count_nonzero(mask))
            mask_count += mask.size

    combined_indices = np.concatenate(dataset_indices) if dataset_indices else np.empty(0, dtype=np.int64)
    if len(combined_indices):
        np.testing.assert_array_equal(combined_indices, np.arange(len(combined_indices), dtype=np.int64))
    residual_all = np.concatenate(residual_pose_l2) if residual_pose_l2 else np.empty(0)
    residual_first = np.concatenate(residual_first_pose_l2) if residual_first_pose_l2 else np.empty(0)
    combined_episodes = np.concatenate(episode_indices) if episode_indices else np.empty(0, dtype=np.int64)
    combined_frames = np.concatenate(frame_indices) if frame_indices else np.empty(0, dtype=np.int64)
    combined_timestamps = np.concatenate(timestamps) if timestamps else np.empty(0)
    within_episode = combined_episodes[1:] == combined_episodes[:-1]
    if np.any(np.diff(combined_frames)[within_episode] < 0):
        raise ValueError("Frame indices are not monotonic within an episode")
    if np.any(np.diff(combined_timestamps)[within_episode] < -1e-9):
        raise ValueError("Timestamps are not monotonic within an episode")

    def distribution(values: np.ndarray) -> dict[str, float] | None:
        if not len(values):
            return None
        return {
            "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }

    summary = {
        "rows": len(combined_indices),
        "expected_rows": extraction_size,
        "complete": len(combined_indices) == extraction_size,
        "episodes": len(np.unique(combined_episodes)),
        # Reported per group over the robot dimensions only; "all" is xyz+6D+gripper.
        "normalized_full_vs_expert_mse": {
            name: sums["full"] / sums["count"] if sums["count"] else None for name, sums in squared_error.items()
        },
        "normalized_null_vs_expert_mse": {
            name: sums["null"] / sums["count"] if sums["count"] else None for name, sums in squared_error.items()
        },
        "normalized_pose_residual_l2_all_horizon": distribution(residual_all),
        "normalized_pose_residual_l2_first_action": distribution(residual_first),
        "force_history_valid_fraction": mask_valid / mask_count if mask_count else None,
    }
    _atomic_write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="forcevla_button_temporal_stage2_null_bc")
    parser.add_argument("--data-config-name", default=None)
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        default=pathlib.Path("checkpoints/forcevla_button_temporal_stage2_null_bc/button_press_stage2_null_bc/9999"),
    )
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--context-cache",
        type=pathlib.Path,
        default=None,
        help=(
            "Optional fixed Slow cache defining one active packet k for every current row t. "
            "When set, Teacher full/null queries use V_k,L from that packet and current S_t/F_t. "
            "Targets are then bound to this exact schedule and must not be trained with cache resampling."
        ),
    )
    args = parser.parse_args()

    if args.batch_size <= 0 or args.shard_size <= 0:
        raise ValueError("batch-size and shard-size must be positive")
    if args.shard_size % args.batch_size:
        raise ValueError("shard-size must be divisible by batch-size")
    if args.num_steps <= 0:
        raise ValueError("num-steps must be positive")

    config = config_lib.get_config(args.config_name)
    config = dataclasses.replace(config, batch_size=args.batch_size, num_workers=0)
    source_config = config if args.data_config_name is None else config_lib.get_config(args.data_config_name)
    data_config = source_config.data.create(source_config.assets_dirs, config.model)
    raw_dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    transform = _sample_transform(data_config)
    dataset_size = len(raw_dataset)
    extraction_size = dataset_size if args.max_samples is None else min(dataset_size, args.max_samples)
    if extraction_size <= 0:
        raise ValueError("Extraction set is empty")

    context_cache = None
    context_dataset_indices = None
    context_row_ready = None
    if args.context_cache is not None:
        context_cache = fast_dataset.load_slow_cache(args.context_cache, expected_rows=dataset_size)
        context_row_ready = (
            np.ones(dataset_size, dtype=np.bool_)
            if context_cache.row_ready is None
            else np.asarray(context_cache.row_ready, dtype=np.bool_)
        )
        context_dataset_indices = np.arange(dataset_size, dtype=np.int64)
        ready = np.flatnonzero(context_row_ready)
        context_dataset_indices[ready] = context_cache.key_dataset_indices[
            context_cache.row_key_positions[ready]
        ]
        if np.any(context_dataset_indices < 0) or np.any(context_dataset_indices >= dataset_size):
            raise ValueError("Context cache refers to dataset rows outside the extraction dataset")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _validate_resume_manifest(args.output_dir, args=args, extraction_size=extraction_size)
    shard_dir = args.output_dir / "shards"
    shard_dir.mkdir(exist_ok=True)
    ranges = [
        (start, min(start + args.shard_size, extraction_size)) for start in range(0, extraction_size, args.shard_size)
    ]
    for start, stop in ranges:
        existing = _shard_path(shard_dir, start, stop)
        if existing.is_file():
            _validate_existing_shard(existing, start, stop)
    missing_ranges = [(start, stop) for start, stop in ranges if not _shard_path(shard_dir, start, stop).is_file()]
    _write_manifest(
        args.output_dir,
        args=args,
        dataset_size=dataset_size,
        extraction_size=extraction_size,
        ranges=ranges,
        elapsed_seconds=0.0,
    )
    if not missing_ranges:
        print(json.dumps(_summarize(args.output_dir, ranges, extraction_size), indent=2))
        return

    checkpoint = args.checkpoint.resolve()
    model_shape = nnx.eval_shape(config.model.create, jax.random.key(0))
    reference_state = nnx.state(model_shape).to_pure_dict()
    params = weight_loaders.CheckpointWeightLoader(str(checkpoint / "params"), missing_regex=r"a^").load(
        reference_state
    )
    model = config.model.load(params, remove_extra_params=False)
    sample_paired = nnx_utils.module_jit(
        model.sample_paired_actions_and_context
        if context_cache is None
        else model.sample_paired_actions_from_cached_context
    )
    run_start = time.monotonic()
    processed_this_run = 0

    for shard_number, (shard_start, shard_stop) in enumerate(missing_ranges, start=1):
        shard_parts: dict[str, list[np.ndarray]] = {}
        shard_start_time = time.monotonic()
        for batch_start in range(shard_start, shard_stop, args.batch_size):
            valid_stop = min(batch_start + args.batch_size, shard_stop)
            valid_count = valid_stop - batch_start
            batch_indices = np.arange(batch_start, valid_stop, dtype=np.int64)
            if valid_count < args.batch_size:
                batch_indices = np.pad(batch_indices, (0, args.batch_size - valid_count), mode="edge")
            observation, batch_arrays = _load_batch(raw_dataset, transform, batch_indices)
            context_observation = None
            noise_indices = batch_indices
            if context_dataset_indices is not None:
                context_indices = context_dataset_indices[batch_indices]
                context_observation, context_arrays = _load_batch(raw_dataset, transform, context_indices)
                np.testing.assert_array_equal(
                    context_arrays["episode_indices"], batch_arrays["episode_indices"]
                )
                batch_arrays["context_dataset_indices"] = context_indices
                batch_arrays["context_timestamps"] = context_arrays["timestamps"]
                batch_arrays["context_row_ready"] = context_row_ready[batch_indices]
                # Tie all rows using one Slow packet to the same flow-noise draw.
                # This removes sampling variation between A_ref,k and the cached-context
                # query while full/null still share identical noise.
                noise_indices = context_indices
            noise = model_lib.row_keyed_noise(
                args.seed,
                noise_indices,
                action_horizon=config.model.action_horizon,
                action_dim=config.model.action_dim,
            )
            if context_observation is None:
                full, null, _, _ = sample_paired(
                    jax.random.key(args.seed), observation, num_steps=args.num_steps, noise=noise
                )
            else:
                full, null, _, _ = sample_paired(
                    jax.random.key(args.seed),
                    context_observation,
                    observation,
                    num_steps=args.num_steps,
                    noise=noise,
                )
            full = np.asarray(full[:valid_count], dtype=np.float32)
            null = np.asarray(null[:valid_count], dtype=np.float32)
            batch_arrays.update(
                dataset_indices=np.arange(batch_start, valid_stop, dtype=np.int64),
                normalized_full_actions=full,
                normalized_null_actions=null,
                normalized_pose_residual=full[..., : rot.POSE_DIMS] - null[..., : rot.POSE_DIMS],
            )
            for name, value in batch_arrays.items():
                shard_parts.setdefault(name, []).append(np.asarray(value[:valid_count]))
            processed_this_run += valid_count

        shard_arrays = {name: np.concatenate(parts) for name, parts in shard_parts.items()}
        shard_path = _shard_path(shard_dir, shard_start, shard_stop)
        _atomic_savez(shard_path, shard_arrays)
        _validate_existing_shard(shard_path, shard_start, shard_stop)
        elapsed = time.monotonic() - run_start
        _write_manifest(
            args.output_dir,
            args=args,
            dataset_size=dataset_size,
            extraction_size=extraction_size,
            ranges=ranges,
            elapsed_seconds=elapsed,
        )
        rate = processed_this_run / elapsed
        remaining = extraction_size - _completed_rows(shard_dir, ranges)
        print(
            f"[{shard_number}/{len(missing_ranges)}] {shard_path.name}: "
            f"{shard_stop - shard_start} rows in {time.monotonic() - shard_start_time:.1f}s; "
            f"run rate={rate:.2f} rows/s, remaining ETA={remaining / max(rate, 1e-9) / 60:.1f} min",
            flush=True,
        )

    summary = _summarize(args.output_dir, ranges, extraction_size)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
