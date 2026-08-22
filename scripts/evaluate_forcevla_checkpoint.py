"""Evaluate one ForceVLA checkpoint on a held-out split with fixed flow noise.

This evaluator is shared by the instantaneous and temporal Stage-1 Teachers.
Every split-local row gets deterministic row-keyed flow noise, so differences
between separately evaluated checkpoints are model differences rather than
sampling variance.
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
from openpi.policies import rotation_6d as rot
from openpi.shared import nnx_utils
from openpi.training import config as config_lib
from openpi.training import data_loader
from openpi.training import weight_loaders


def _sample_transform(data_config):
    return transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )


def _load_batch(raw_dataset, transform, indices: np.ndarray):
    samples = []
    metadata = {"episode_indices": [], "frame_indices": [], "timestamps": []}
    for index in indices:
        raw = raw_dataset[int(index)]
        metadata["episode_indices"].append(int(np.asarray(raw["episode_index"]).item()))
        metadata["frame_indices"].append(int(np.asarray(raw["frame_index"]).item()))
        metadata["timestamps"].append(float(np.asarray(raw["timestamp"]).item()))
        samples.append(transform(raw))
    batch = jax.tree.map(lambda *xs: np.stack(xs), *samples)
    expert = np.asarray(batch.pop("actions"), dtype=np.float32)
    observation = model_lib.Observation.from_dict(jax.tree.map(jnp.asarray, batch))
    return observation, expert, {name: np.asarray(value) for name, value in metadata.items()}


def _checkpoint_params_path(checkpoint: pathlib.Path) -> pathlib.Path:
    checkpoint = checkpoint.resolve()
    return checkpoint if checkpoint.name == "params" else checkpoint / "params"


def _to_physical_actions(normalized_state, normalized_actions, data_config) -> np.ndarray:
    output_stats = {key: value for key, value in data_config.norm_stats.items() if key in {"state", "actions"}}
    output_transform = transforms.compose(
        [
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(output_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ]
    )
    converted = []
    for state, actions in zip(normalized_state, normalized_actions, strict=True):
        converted.append(np.asarray(output_transform({"state": state, "actions": actions})["actions"]))
    return np.stack(converted).astype(np.float32)


def physical_action_error_summary(prediction: np.ndarray, expert: np.ndarray) -> dict[str, float]:
    """Report translation, SO(3) rotation, and gripper without mixing units."""
    prediction = np.asarray(prediction, dtype=np.float64)
    expert = np.asarray(expert, dtype=np.float64)
    if prediction.shape != expert.shape or prediction.ndim != 3 or prediction.shape[-1] < 7:
        raise ValueError(f"Expected matching physical actions [N,H,>=7], got {prediction.shape} and {expert.shape}")
    translation = prediction[..., :3] - expert[..., :3]
    angle = rot.geodesic_angle(
        rot.rpy_to_matrix(prediction[..., 3:6]),
        rot.rpy_to_matrix(expert[..., 3:6]),
    )
    gripper = prediction[..., 6] - expert[..., 6]

    def summarize(xyz: np.ndarray, rotation: np.ndarray, grip: np.ndarray) -> dict[str, float]:
        return {
            "translation_rmse_m": float(np.sqrt(np.mean(np.sum(np.square(xyz), axis=-1)))),
            "rotation_geodesic_rmse_rad": float(np.sqrt(np.mean(np.square(rotation)))),
            "gripper_rmse": float(np.sqrt(np.mean(np.square(grip)))),
            "gripper_mae": float(np.mean(np.abs(grip))),
        }

    return {
        "all_horizon": summarize(translation, angle, gripper),
        "first_action": summarize(translation[:, 0], angle[:, 0], gripper[:, 0]),
    }


def _normalized_action_error_summary(prediction: np.ndarray, expert: np.ndarray) -> dict[str, float]:
    groups = {
        "robot_10d": slice(0, rot.ROBOT_DIMS),
        "xyz": slice(0, rot.XYZ_DIMS),
        "rotation_6d": slice(rot.XYZ_DIMS, rot.POSE_DIMS),
        "gripper": slice(rot.POSE_DIMS, rot.ROBOT_DIMS),
    }
    return {name: float(np.mean(np.square(prediction[..., dims] - expert[..., dims]))) for name, dims in groups.items()}


def _latest_wrench_and_contact(
    episode_indices: np.ndarray,
    timestamps: np.ndarray,
    *,
    sidecar_dir: pathlib.Path,
    baseline_rows: int,
    threshold_n: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    wrench = np.zeros((len(timestamps), 6), dtype=np.float32)
    valid = np.zeros(len(timestamps), dtype=bool)
    for row, (episode_value, timestamp) in enumerate(zip(episode_indices, timestamps, strict=True)):
        episode = int(episode_value)
        if episode not in cache:
            path = sidecar_dir / f"episode_{episode:06d}.npz"
            if not path.exists():
                raise FileNotFoundError(f"Missing contact-label sidecar: {path}")
            with np.load(path, allow_pickle=False) as values:
                cache[episode] = (
                    np.asarray(values["force"], dtype=np.float32),
                    np.asarray(values["timestamps"], dtype=np.float64),
                )
        force, force_timestamps = cache[episode]
        index = int(np.searchsorted(force_timestamps, timestamp + 1e-9, side="right") - 1)
        if index >= 0:
            wrench[row] = force[index]
            valid[row] = True
    baseline = np.zeros_like(wrench)
    for episode_value in np.unique(episode_indices):
        rows = (episode_indices == episode_value) & valid
        valid_indices = np.flatnonzero(rows)
        if len(valid_indices) == 0:
            continue
        opening = valid_indices[:baseline_rows]
        baseline[episode_indices == episode_value] = np.median(wrench[opening], axis=0)
    magnitude = np.linalg.norm((wrench - baseline)[:, :3], axis=-1)
    contact = valid & (magnitude >= threshold_n)
    summary = {
        "criterion": "latest causal wrench minus per-episode opening median",
        "threshold_n": threshold_n,
        "baseline_rows": baseline_rows,
        "valid_rows": int(np.count_nonzero(valid)),
        "contact_rows": int(np.count_nonzero(contact)),
        "free_space_rows": int(np.count_nonzero(valid & ~contact)),
        "invalid_rows": int(np.count_nonzero(~valid)),
        "baseline_corrected_force_n": {
            "mean": float(np.mean(magnitude[valid])) if np.any(valid) else None,
            "p95": float(np.percentile(magnitude[valid], 95)) if np.any(valid) else None,
            "max": float(np.max(magnitude[valid])) if np.any(valid) else None,
        },
    }
    return contact, valid, wrench, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True, help="Model/checkpoint architecture config.")
    parser.add_argument("--data-config-name", required=True, help="Held-out data split config.")
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Uniform subset for smoke; omit to evaluate every held-out row.",
    )
    parser.add_argument("--contact-sidecar-dir", type=pathlib.Path, default=None)
    parser.add_argument("--contact-threshold-n", type=float, default=5.0)
    parser.add_argument("--baseline-rows", type=int, default=15)
    args = parser.parse_args()
    if min(args.batch_size, args.num_steps, args.contact_threshold_n, args.baseline_rows) <= 0:
        raise ValueError("Batch size, flow steps, contact threshold, and baseline rows must be positive")

    config = dataclasses.replace(config_lib.get_config(args.config_name), batch_size=args.batch_size, num_workers=0)
    source_config = config_lib.get_config(args.data_config_name)
    data_config = source_config.data.create(source_config.assets_dirs, config.model)
    raw_dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    transform = _sample_transform(data_config)
    if args.max_samples is None:
        selected_indices = np.arange(len(raw_dataset), dtype=np.int64)
    else:
        if not 0 < args.max_samples <= len(raw_dataset):
            raise ValueError(f"max-samples must be in [1, {len(raw_dataset)}]")
        selected_indices = np.linspace(0, len(raw_dataset) - 1, args.max_samples, dtype=np.int64)

    model_shape = nnx.eval_shape(config.model.create, jax.random.key(0))
    reference_state = nnx.state(model_shape).to_pure_dict()
    params = weight_loaders.CheckpointWeightLoader(
        str(_checkpoint_params_path(args.checkpoint)), missing_regex=r"a^"
    ).load(reference_state)
    model = config.model.load(params, remove_extra_params=False)
    sample_actions = nnx_utils.module_jit(model.sample_actions)

    predictions = []
    experts = []
    states = []
    metadata_parts: dict[str, list[np.ndarray]] = {
        "episode_indices": [],
        "frame_indices": [],
        "timestamps": [],
    }
    started = time.monotonic()
    for start in range(0, len(selected_indices), args.batch_size):
        valid_indices = selected_indices[start : start + args.batch_size]
        valid_count = len(valid_indices)
        padded_indices = np.pad(valid_indices, (0, args.batch_size - valid_count), mode="edge")
        observation, expert, metadata = _load_batch(raw_dataset, transform, padded_indices)
        noise = model_lib.row_keyed_noise(
            args.seed,
            padded_indices,
            action_horizon=config.model.action_horizon,
            action_dim=config.model.action_dim,
        )
        prediction = sample_actions(
            jax.random.key(args.seed),
            observation,
            num_steps=args.num_steps,
            noise=noise,
        )
        predictions.append(np.asarray(prediction[:valid_count], dtype=np.float32))
        experts.append(expert[:valid_count])
        states.append(np.asarray(observation.state[:valid_count], dtype=np.float32))
        for name, parts in metadata_parts.items():
            parts.append(metadata[name][:valid_count])
        completed = min(start + valid_count, len(selected_indices))
        if completed % 256 == 0 or completed == len(selected_indices):
            rate = completed / max(time.monotonic() - started, 1e-9)
            print(f"Evaluation: {completed}/{len(selected_indices)} rows ({rate:.2f}/s)", flush=True)

    normalized_prediction = np.concatenate(predictions)
    normalized_expert = np.concatenate(experts)
    normalized_state = np.concatenate(states)
    metadata = {name: np.concatenate(parts) for name, parts in metadata_parts.items()}
    physical_prediction = _to_physical_actions(normalized_state, normalized_prediction, data_config)
    physical_expert = _to_physical_actions(normalized_state, normalized_expert, data_config)

    selectors: dict[str, np.ndarray] = {"all": np.ones(len(selected_indices), dtype=bool)}
    contact_summary = None
    wrench = None
    if args.contact_sidecar_dir is not None:
        contact, valid_wrench, wrench, contact_summary = _latest_wrench_and_contact(
            metadata["episode_indices"],
            metadata["timestamps"],
            sidecar_dir=args.contact_sidecar_dir,
            baseline_rows=args.baseline_rows,
            threshold_n=args.contact_threshold_n,
        )
        if np.any(contact):
            selectors["contact"] = contact
        free_space = valid_wrench & ~contact
        if np.any(free_space):
            selectors["free_space"] = free_space

    summary = {
        "config_name": args.config_name,
        "data_config_name": args.data_config_name,
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_size": len(raw_dataset),
        "evaluated_rows": len(selected_indices),
        "seed": args.seed,
        "num_flow_steps": args.num_steps,
        "noise_scheme": "row_keyed_noise(split-local dataset index)",
        "force_encoder_type": config.model.force_encoder.type,
        "contact": contact_summary,
        "strata": {
            name: {
                "rows": int(np.count_nonzero(selector)),
                "normalized_action_mse": _normalized_action_error_summary(
                    normalized_prediction[selector], normalized_expert[selector]
                ),
                "physical_action_error": physical_action_error_summary(
                    physical_prediction[selector], physical_expert[selector]
                ),
            }
            for name, selector in selectors.items()
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    saved = {
        "dataset_indices": selected_indices,
        **metadata,
        "normalized_prediction": normalized_prediction,
        "normalized_expert": normalized_expert,
        "physical_prediction": physical_prediction,
        "physical_expert": physical_expert,
    }
    if wrench is not None:
        saved["latest_wrench"] = wrench
        saved["contact"] = selectors.get("contact", np.zeros(len(selected_indices), dtype=bool))
    np.savez_compressed(args.output_dir / "predictions.npz", **saved)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
