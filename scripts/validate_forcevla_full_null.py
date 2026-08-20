"""Paired validation of the selected full/null ForceVLA Teacher.

Both conditions use the same observations and flow-sampling RNG keys. Metrics
are reported in the dataset's physical action space with pose and gripper kept
separate.
"""

import argparse
import dataclasses
import json
import pathlib

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi import transforms
from openpi.models import model as model_lib
from openpi.shared import nnx_utils
from openpi.training import config as config_lib
from openpi.training import data_loader
from openpi.training import weight_loaders


def _stack_samples(samples: list[dict]) -> tuple[model_lib.Observation, np.ndarray]:
    batch = jax.tree.map(lambda *xs: np.stack(xs), *samples)
    actions = np.asarray(batch.pop("actions"))
    batch = jax.tree.map(jnp.asarray, batch)
    return model_lib.Observation.from_dict(batch), actions


def _to_absolute_actions(
    normalized_state: np.ndarray,
    normalized_actions: np.ndarray,
    output_transform: transforms.DataTransformFn,
) -> np.ndarray:
    converted = []
    for state, actions in zip(normalized_state, normalized_actions, strict=True):
        output = output_transform({"state": state, "actions": actions})
        converted.append(np.asarray(output["actions"], dtype=np.float32))
    return np.stack(converted)


def _angle_difference(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    difference = lhs - rhs
    return np.arctan2(np.sin(difference), np.cos(difference))


def _pose_difference(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    difference = lhs[..., :6] - rhs[..., :6]
    difference[..., 3:6] = _angle_difference(lhs[..., 3:6], rhs[..., 3:6])
    return difference


def _action_error_metrics(prediction: np.ndarray, expert: np.ndarray) -> dict[str, float]:
    error = prediction - expert
    error[..., 3:6] = _angle_difference(prediction[..., 3:6], expert[..., 3:6])

    def metrics_for(values: np.ndarray, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}/pose_mae": float(np.mean(np.abs(values[..., :6]))),
            f"{prefix}/xyz_l2_mean": float(np.mean(np.linalg.norm(values[..., :3], axis=-1))),
            f"{prefix}/rpy_l2_mean": float(np.mean(np.linalg.norm(values[..., 3:6], axis=-1))),
            f"{prefix}/gripper_mae": float(np.mean(np.abs(values[..., 6]))),
        }

    return {
        **metrics_for(error, "all_horizon"),
        **metrics_for(error[..., 0, :], "first_action"),
    }


def _residual_metrics(full: np.ndarray, null: np.ndarray) -> dict[str, float]:
    residual = _pose_difference(full, null)

    def distribution(values: np.ndarray, prefix: str) -> dict[str, float]:
        xyz = np.linalg.norm(values[..., :3], axis=-1)
        rpy = np.linalg.norm(values[..., 3:6], axis=-1)
        pose = np.linalg.norm(values, axis=-1)
        return {
            f"{prefix}/pose_l2_mean": float(np.mean(pose)),
            f"{prefix}/pose_l2_p50": float(np.percentile(pose, 50)),
            f"{prefix}/pose_l2_p95": float(np.percentile(pose, 95)),
            f"{prefix}/pose_l2_max": float(np.max(pose)),
            f"{prefix}/xyz_l2_mean": float(np.mean(xyz)),
            f"{prefix}/rpy_l2_mean": float(np.mean(rpy)),
        }

    return {
        **distribution(residual, "all_horizon"),
        **distribution(residual[..., 0, :], "first_action"),
        "all_horizon/gripper_abs_difference_mean": float(np.mean(np.abs(full[..., 6] - null[..., 6]))),
        "first_action/gripper_abs_difference_mean": float(np.mean(np.abs(full[..., 0, 6] - null[..., 0, 6]))),
    }


def _save_plot(output_path: pathlib.Path, full: np.ndarray, null: np.ndarray, expert: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    horizon = np.arange(full.shape[1])
    full_error = _pose_difference(full, expert)
    null_error = _pose_difference(null, expert)
    residual = _pose_difference(full, null)
    curves = (
        (
            np.mean(np.linalg.norm(full_error[..., :3], axis=-1), axis=0),
            np.mean(np.linalg.norm(null_error[..., :3], axis=-1), axis=0),
            "XYZ error vs expert (m)",
        ),
        (
            np.mean(np.linalg.norm(full_error[..., 3:6], axis=-1), axis=0),
            np.mean(np.linalg.norm(null_error[..., 3:6], axis=-1), axis=0),
            "Wrapped RPY error vs expert (rad)",
        ),
        (
            np.mean(np.abs(full[..., 6] - expert[..., 6]), axis=0),
            np.mean(np.abs(null[..., 6] - expert[..., 6]), axis=0),
            "Gripper error vs expert",
        ),
        (
            np.mean(np.linalg.norm(residual[..., :3], axis=-1), axis=0),
            np.mean(np.linalg.norm(residual[..., 3:6], axis=-1), axis=0),
            "Full-null residual",
        ),
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for index, (first, second, title) in enumerate(curves):
        axis = axes.flat[index]
        if index < 3:
            axis.plot(horizon, first, label="full", color="tab:blue")
            axis.plot(horizon, second, label="null", color="tab:orange")
        else:
            axis.plot(horizon, first, label="XYZ (m)", color="tab:green")
            axis.plot(horizon, second, label="wrapped RPY (rad)", color="tab:red")
        axis.set_title(title)
        axis.set_xlabel("action horizon step")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Paired ForceVLA full/null validation")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="forcevla_button_temporal_stage2_null_bc")
    parser.add_argument(
        "--data-config-name",
        default=None,
        help="Optional config whose dataset/split is used while retaining the model and Stage 2 settings from --config-name.",
    )
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        default=pathlib.Path(
            "checkpoints/forcevla_button_temporal_stage2_null_bc/button_press_stage2_null_bc/9999"
        ),
    )
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument(
        "--index-start",
        type=int,
        default=None,
        help="Select --num-samples consecutive split-local rows starting here instead of uniform rows.",
    )
    parser.add_argument(
        "--selection",
        type=pathlib.Path,
        default=None,
        help="Optional NPZ with dataset_indices and regime_labels; overrides uniform --num-samples selection.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/forcevla_full_null_validation"),
    )
    args = parser.parse_args()

    config = config_lib.get_config(args.config_name)
    config = dataclasses.replace(config, batch_size=args.batch_size, num_workers=0)
    data_source_config = config if args.data_config_name is None else config_lib.get_config(args.data_config_name)
    data_config = data_source_config.data.create(data_source_config.assets_dirs, config.model)
    raw_dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = data_loader.transform_dataset(raw_dataset, data_config)
    regime_labels = None
    if args.selection is not None:
        selection = np.load(args.selection)
        indices = np.asarray(selection["dataset_indices"], dtype=np.int64)
        regime_labels = selection["regime_labels"].astype(str)
        if len(indices) != len(regime_labels):
            raise ValueError("selection dataset_indices and regime_labels must have equal length")
    elif args.index_start is not None:
        if args.index_start < 0 or args.index_start + args.num_samples > len(dataset):
            raise ValueError(
                f"Consecutive range [{args.index_start}, {args.index_start + args.num_samples}) "
                f"must fit dataset size {len(dataset)}"
            )
        indices = np.arange(args.index_start, args.index_start + args.num_samples, dtype=np.int64)
    else:
        if args.num_samples <= 0 or args.num_samples > len(dataset):
            raise ValueError(f"num_samples must be in [1, {len(dataset)}], got {args.num_samples}")
        # Uniform indices cover the full training trajectory instead of
        # evaluating only the first contiguous segment.
        indices = np.linspace(0, len(dataset) - 1, args.num_samples, dtype=np.int64)
    batches = []
    normalized_expert = []
    normalized_state = []
    normalized_force_history = []
    force_history_mask = []
    valid_counts = []
    for start in range(0, len(indices), args.batch_size):
        batch_indices = indices[start : start + args.batch_size]
        valid_count = len(batch_indices)
        valid_counts.append(valid_count)
        if valid_count < args.batch_size:
            batch_indices = np.pad(batch_indices, (0, args.batch_size - valid_count), mode="edge")
        observation, expert = _stack_samples([dataset[int(index)] for index in batch_indices])
        batches.append(observation)
        normalized_expert.append(expert[:valid_count])
        normalized_state.append(np.asarray(observation.state[:valid_count]))
        if observation.force_history is not None:
            normalized_force_history.append(np.asarray(observation.force_history[:valid_count]))
            force_history_mask.append(np.asarray(observation.force_history_mask[:valid_count]))

    checkpoint = args.checkpoint.resolve()
    # Use the same canonical path merge as training. Orbax restores list-like
    # TCN block indices as string keys, while the NNX reference tree uses ints.
    model_shape = nnx.eval_shape(config.model.create, jax.random.key(0))
    reference_state = nnx.state(model_shape).to_pure_dict()
    params = weight_loaders.CheckpointWeightLoader(str(checkpoint / "params"), missing_regex=r"a^").load(
        reference_state
    )
    predictions: dict[str, np.ndarray] = {}
    for condition in ("full", "null"):
        model_config = dataclasses.replace(config.model, force_condition=condition)
        # The loader has already intersected and canonicalized the full tree.
        # Re-running Orbax intersect_trees here stringifies NNX list indices.
        model = model_config.load(params, remove_extra_params=False)
        sample_actions = nnx_utils.module_jit(model.sample_actions)
        condition_predictions = []
        for batch_index, (observation, valid_count) in enumerate(zip(batches, valid_counts, strict=True)):
            # Reusing the exact key for full and null makes this a paired test:
            # both samplers start from the same flow noise.
            sample_key = jax.random.fold_in(jax.random.key(args.seed), batch_index)
            action = sample_actions(sample_key, observation, num_steps=args.num_steps)
            condition_predictions.append(np.asarray(action[:valid_count]))
        predictions[condition] = np.concatenate(condition_predictions)

    normalized_state_array = np.concatenate(normalized_state)
    normalized_expert_array = np.concatenate(normalized_expert)
    # Temporal input statistics also contain force_history, but model outputs
    # intentionally contain only state and actions. Select output-side stats so
    # strict unnormalization does not ask for an input-only modality.
    output_norm_stats = {key: value for key, value in data_config.norm_stats.items() if key in {"state", "actions"}}
    output_transform = transforms.compose(
        [
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(output_norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ]
    )
    expert = _to_absolute_actions(normalized_state_array, normalized_expert_array, output_transform)
    full = _to_absolute_actions(normalized_state_array, predictions["full"], output_transform)
    null = _to_absolute_actions(normalized_state_array, predictions["null"], output_transform)
    summary = {
        "config_name": args.config_name,
        "data_config_name": args.data_config_name or args.config_name,
        "checkpoint": str(checkpoint),
        "dataset_size": len(dataset),
        "num_samples": len(indices),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "num_flow_steps": args.num_steps,
        "full_vs_expert": _action_error_metrics(full, expert),
        "null_vs_expert": _action_error_metrics(null, expert),
        "full_vs_null": _residual_metrics(full, null),
    }
    if regime_labels is not None:
        regimes = sorted({label for combined in regime_labels for label in combined.split(",")})
        summary["regimes"] = {}
        for regime in regimes:
            positions = np.flatnonzero([regime in combined.split(",") for combined in regime_labels])
            regime_summary = {
                "sample_count": len(positions),
                "full_vs_expert": _action_error_metrics(full[positions], expert[positions]),
                "null_vs_expert": _action_error_metrics(null[positions], expert[positions]),
                "full_vs_null": _residual_metrics(full[positions], null[positions]),
            }
            summary["regimes"][regime] = regime_summary

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.json").open("w") as file:
        json.dump(summary, file, indent=2)
    saved_arrays = {
        "dataset_indices": indices,
        "episode_indices": np.asarray([raw_dataset[int(index)]["episode_index"] for index in indices], dtype=np.int64),
        "frame_indices": np.asarray([raw_dataset[int(index)]["frame_index"] for index in indices], dtype=np.int64),
        "timestamps": np.asarray([raw_dataset[int(index)]["timestamp"] for index in indices], dtype=np.float64),
        "expert_actions": expert,
        "full_actions": full,
        "null_actions": null,
        "pose_residual": _pose_difference(full, null),
        "normalized_state": normalized_state_array,
        "normalized_expert_actions": normalized_expert_array,
        "normalized_full_actions": predictions["full"],
        "normalized_null_actions": predictions["null"],
        "normalized_pose_residual": predictions["full"][..., :6] - predictions["null"][..., :6],
    }
    if normalized_force_history:
        saved_arrays["normalized_force_history"] = np.concatenate(normalized_force_history)
        saved_arrays["force_history_mask"] = np.concatenate(force_history_mask)
    if regime_labels is not None:
        saved_arrays["regime_labels"] = regime_labels
    np.savez_compressed(
        args.output_dir / "paired_actions.npz",
        **saved_arrays,
    )
    _save_plot(args.output_dir / "mean_action_chunks.png", full, null, expert)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
