"""Evaluate a trained Fast student and test whether it uses force history.

Beyond the Teacher-referenced residual metrics, this reports how every policy
variant compares to the ground-truth expert action, and splits the held-out set
into contact and free-space rows so that contact behaviour is not diluted by the
much larger number of free-space samples.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib

from flax import nnx
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.models import slow_fast
from openpi.policies import rotation_6d as rot
from openpi.shared import normalize as normalize_lib
from openpi.training import fast_dataset

TRANSLATION_DIMS = slice(0, 3)
ROTATION_6D_DIMS = slice(3, 9)
POSE_DIM_NAMES = ("x", "y", "z", "r6d_0", "r6d_1", "r6d_2", "r6d_3", "r6d_4", "r6d_5")


def to_absolute_pose(delta: np.ndarray, base_state: np.ndarray, action_stats, state_stats) -> np.ndarray:
    """Rebase a normalized delta action onto its base state, in metres and radians.

    This is the offline copy of `slow_fast_runtime.to_absolute_command`: unnormalize
    first, then add the *physical* state, because `DeltaActions` runs before
    `Normalize` in the training chain. Two things depend on getting this right.
    The Slow reference is a delta from the state its packet was conditioned on while
    the Teacher and expert are deltas from the current row's state, so subtracting
    the deltas directly compares quantities with different origins. And the 6D
    rotation columns only describe a rotation once the state is added back; a delta
    on its own is a near-zero vector that `sixd_to_matrix` will happily
    Gram-Schmidt into an arbitrary rotation.
    """
    physical = fast_dataset.denormalize(delta, action_stats, dims=rot.POSE_DIMS)
    base = fast_dataset.denormalize(base_state[:, : rot.POSE_DIMS], state_stats, dims=rot.POSE_DIMS)
    return (physical + base).astype(np.float32)


def _load_model(checkpoint: pathlib.Path, *, chunk_steps: int):
    config = slow_fast.FastResidualConfig(chunk_steps=chunk_steps)
    model = slow_fast.FastStudentWithIntentProjector(config, slow_context_dim=2048, rngs=nnx.Rngs(0))
    params = model_lib.restore_params(checkpoint)
    state = nnx.state(model, nnx.Param)
    state.replace_by_pure_dict(params)
    nnx.update(model, state)
    return model


def _predict_all(model, arrays, cache, *, batch_size: int, seed: int) -> dict[str, np.ndarray]:
    @nnx.jit
    def infer(module, context, context_mask, force, force_mask, state, reference, time_features):
        residual, _, _ = module(
            context,
            context_mask,
            force,
            force_mask,
            state,
            reference,
            time_features,
            train=False,
        )
        return residual

    permutation = np.random.default_rng(seed).permutation(len(arrays.dataset_indices))
    predictions: dict[str, list[np.ndarray]] = {"full": [], "zero_force": [], "shuffled_force": []}
    for start in range(0, len(arrays.dataset_indices), batch_size):
        indices = np.arange(start, min(start + batch_size, len(arrays.dataset_indices)))
        key_positions = cache.row_key_positions[indices]
        common = (
            jnp.asarray(cache.context_tokens[key_positions], dtype=jnp.float32),
            jnp.asarray(cache.context_mask[key_positions], dtype=jnp.bool_),
        )
        force = jnp.asarray(arrays.force_history[indices], dtype=jnp.float32)
        force_mask = jnp.asarray(arrays.force_history_mask[indices], dtype=jnp.bool_)
        suffix = (
            jnp.asarray(arrays.state[indices], dtype=jnp.float32),
            jnp.asarray(cache.reference_actions[indices, 0], dtype=jnp.float32),
            jnp.asarray(cache.time_features[indices], dtype=jnp.float32),
        )
        predictions["full"].append(np.asarray(infer(model, *common, force, force_mask, *suffix)))
        predictions["zero_force"].append(np.asarray(infer(model, *common, jnp.zeros_like(force), force_mask, *suffix)))
        shuffled = permutation[indices]
        predictions["shuffled_force"].append(
            np.asarray(
                infer(
                    model,
                    *common,
                    jnp.asarray(arrays.force_history[shuffled], dtype=jnp.float32),
                    jnp.asarray(arrays.force_history_mask[shuffled], dtype=jnp.bool_),
                    *suffix,
                )
            )
        )
    return {name: np.concatenate(parts) for name, parts in predictions.items()}


def physical_pose_error_summary(prediction: np.ndarray, target: np.ndarray) -> dict:
    """Report physical translation and rotation without mixing their units.

    The 6D rotation coordinates are useful diagnostics but are dimensionless; an
    aggregate MSE over xyz metres and those six coordinates has no physical unit.
    Rotation quality is therefore ranked only by the SO(3) geodesic angle.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    error = prediction - target
    geodesic = rot.geodesic_angle(
        rot.sixd_to_matrix(prediction[:, ROTATION_6D_DIMS]),
        rot.sixd_to_matrix(target[:, ROTATION_6D_DIMS]),
    )
    return {
        "translation_rmse_m": float(np.sqrt(np.mean(np.sum(np.square(error[:, TRANSLATION_DIMS]), axis=-1)))),
        "rotation_geodesic_rmse_rad": float(np.sqrt(np.mean(np.square(geodesic)))),
        "rotation_6d_coordinate_rmse": float(np.sqrt(np.mean(np.square(error[:, ROTATION_6D_DIMS])))),
        "per_dim_coordinate_rmse": {
            name: float(np.sqrt(np.mean(np.square(error[:, index])))) for index, name in enumerate(POSE_DIM_NAMES)
        },
    }


def _stratum_metrics(
    poses: dict[str, np.ndarray],
    physical_predictions: dict[str, np.ndarray],
    normalized_predictions: dict[str, np.ndarray],
    normalized_residual_target: np.ndarray,
) -> dict:
    """Compare every variant against both the Teacher and the true expert."""
    reference = poses["reference"]
    metrics: dict = {
        "rows": len(reference),
        "residual_vs_teacher_deviation": {
            "space": "normalized pose residual",
            "zero_residual_baseline_mse_normalized": float(np.mean(np.square(normalized_residual_target))),
        },
        "action_vs_teacher_full": {
            "slow_reference_only": physical_pose_error_summary(reference, poses["teacher_full"])
        },
        "action_vs_expert": {
            "slow_reference_only": physical_pose_error_summary(reference, poses["expert"]),
            "teacher_full": physical_pose_error_summary(poses["teacher_full"], poses["expert"]),
            "teacher_null": physical_pose_error_summary(poses["teacher_null"], poses["expert"]),
        },
    }
    baseline = metrics["residual_vs_teacher_deviation"]["zero_residual_baseline_mse_normalized"]
    for name, normalized_prediction in normalized_predictions.items():
        commanded = reference + physical_predictions[name]
        metrics["residual_vs_teacher_deviation"][name] = {
            "mse_normalized": float(np.mean(np.square(normalized_prediction - normalized_residual_target))),
            "predicted_l2_mean_normalized": float(np.mean(np.linalg.norm(normalized_prediction, axis=-1))),
        }
        metrics["residual_vs_teacher_deviation"][name]["gain_vs_zero_residual"] = 1.0 - metrics[
            "residual_vs_teacher_deviation"
        ][name]["mse_normalized"] / max(baseline, 1e-12)
        metrics["action_vs_teacher_full"][f"slow_plus_{name}"] = physical_pose_error_summary(
            commanded, poses["teacher_full"]
        )
        metrics["action_vs_expert"][f"slow_plus_{name}"] = physical_pose_error_summary(commanded, poses["expert"])

    physical_error_keys = ("translation_rmse_m", "rotation_geodesic_rmse_rad")
    for group in ("action_vs_expert", "action_vs_teacher_full"):
        section = metrics[group]
        section["fast_gain_vs_slow_only"] = {
            key: 1.0 - section["slow_plus_full"][key] / max(section["slow_reference_only"][key], 1e-12)
            for key in physical_error_keys
        }
    # Do not collapse this into one physical MSE: xyz is measured in metres while
    # 6D rotation coordinates are dimensionless. The Slow and total terms are
    # physical pose summaries; the Fast term is the normalized loss it actually
    # optimizes.
    metrics["deployment_error_decomposition"] = {
        "slow_reference_vs_teacher_null_physical": physical_pose_error_summary(reference, poses["teacher_null"]),
        "fast_residual_vs_teacher_deviation_mse_normalized": metrics["residual_vs_teacher_deviation"]["full"][
            "mse_normalized"
        ],
        "composed_vs_teacher_full_physical": metrics["action_vs_teacher_full"]["slow_plus_full"],
    }
    metrics["action_vs_expert"]["teacher_force_gain"] = {
        key: 1.0
        - metrics["action_vs_expert"]["teacher_full"][key]
        / max(metrics["action_vs_expert"]["teacher_null"][key], 1e-12)
        for key in physical_error_keys
    }
    return metrics


def _print_report(metrics: dict) -> None:
    print("\n=== Fast residual evaluation ===")
    per_step = metrics["per_step_residual_mse_normalized"]["full"]
    print(f"  emitted chunk steps={metrics['chunk_steps']}")
    print("  per-step residual mse (normalized): " + "  ".join(f"k{k}={v:.6f}" for k, v in enumerate(per_step)))
    for stratum, values in metrics["strata"].items():
        print(f"\n[{stratum}]  rows={values['rows']}")
        print("  action error vs ground-truth expert:")
        for name in ("slow_reference_only", "slow_plus_full", "teacher_null", "teacher_full"):
            item = values["action_vs_expert"][name]
            per_dim = "  ".join(f"{key}={value:.5f}" for key, value in item["per_dim_coordinate_rmse"].items())
            print(
                f"    {name:<22} translation={item['translation_rmse_m']:.5f} m  "
                f"rotation={item['rotation_geodesic_rmse_rad']:.5f} rad  "
                f"r6d_coord={item['rotation_6d_coordinate_rmse']:.5f}"
            )
            print(f"      per-dim coordinate rmse: {per_dim}")
        for label, key in (
            ("fast vs slow-only", "fast_gain_vs_slow_only"),
            ("teacher force gain", "teacher_force_gain"),
        ):
            gain = values["action_vs_expert"][key]
            print(
                f"    {label:<20} translation={gain['translation_rmse_m'] * 100:+.2f}%  "
                f"rotation={gain['rotation_geodesic_rmse_rad'] * 100:+.2f}%"
            )
        residual = values["residual_vs_teacher_deviation"]
        print(
            "  Teacher deviation reproduction, normalized "
            f"(zero baseline={residual['zero_residual_baseline_mse_normalized']:.6f}):"
        )
        for name in ("full", "zero_force", "shuffled_force"):
            print(
                f"    {name:<22} mse={residual[name]['mse_normalized']:.6f}  "
                f"gain={residual[name]['gain_vs_zero_residual']:.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=pathlib.Path, required=True)
    parser.add_argument("--slow-cache", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument(
        "--norm-stats-dir",
        type=pathlib.Path,
        default=pathlib.Path("assets/forcevla_button_temporal_100hz/panda_button_press_temporal_100hz_train56"),
        help="Required directory holding norm_stats.json for absolute-pose and contact evaluation.",
    )
    parser.add_argument(
        "--contact-threshold-n",
        type=float,
        default=5.0,
        help="Baseline-corrected linear force magnitude above which a row counts as in contact.",
    )
    parser.add_argument(
        "--baseline-rows",
        type=int,
        default=15,
        help="Opening rows per episode used to estimate the resting wrench offset.",
    )
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.contact_threshold_n <= 0:
        raise ValueError("contact-threshold-n must be positive")

    cache = fast_dataset.load_slow_cache(args.slow_cache)
    arrays = fast_dataset.load_stage3_fast_arrays(args.targets, chunk_steps=cache.chunk_steps)
    if len(cache.row_key_positions) != len(arrays.dataset_indices):
        raise ValueError(
            f"Slow cache covers {len(cache.row_key_positions)} rows but Stage-3 has {len(arrays.dataset_indices)}"
        )
    if args.norm_stats_dir is None or not (args.norm_stats_dir / "norm_stats.json").is_file():
        raise ValueError(
            f"--norm-stats-dir must point at a directory containing norm_stats.json (got {args.norm_stats_dir}). "
            "Absolute poses cannot be reconstructed without it, and comparing raw normalized deltas would mix "
            "the Slow packet's base state with the current row's."
        )
    norm_stats = normalize_lib.load(args.norm_stats_dir)

    # Estimate the resting wrench from the true opening rows before removing rows
    # whose Slow packet was not ready. Computing it after that subset silently moves
    # the baseline 50--300 ms into the episode, potentially into contact.
    wrench_by_row = fast_dataset.latest_physical_wrench(
        arrays.force_history, arrays.force_history_mask, norm_stats["force_history"]
    )
    baseline_by_row = fast_dataset.episode_baseline_wrench(
        wrench_by_row, arrays.episode_indices, num_rows=args.baseline_rows
    )
    ready = np.flatnonzero(cache.row_ready) if cache.row_ready is not None else np.arange(len(arrays.dataset_indices))
    if len(ready) == 0:
        raise ValueError("Slow cache has no rows whose packet would already be ready")
    # Kept before the row subset: the Slow reference is a delta from the state of
    # its *key* row, which is generally not one of the rows being scored.
    state_by_row = np.asarray(arrays.state)
    arrays = dataclasses.replace(
        arrays,
        **{field.name: getattr(arrays, field.name)[ready] for field in dataclasses.fields(arrays)},
    )
    cache = dataclasses.replace(
        cache,
        reference_actions=cache.reference_actions[ready],
        time_features=cache.time_features[ready],
        row_key_positions=cache.row_key_positions[ready],
        row_ready=np.ones(len(ready), dtype=bool) if cache.row_ready is not None else None,
    )
    model = _load_model(args.checkpoint.resolve(), chunk_steps=cache.chunk_steps)
    chunk_predictions = _predict_all(model, arrays, cache, batch_size=args.batch_size, seed=args.seed)
    # Every step of the emitted chunk is scored, but the detailed pose report is on
    # step 0: that is the one that executes whenever Fast keeps up with the action rate.
    per_step_residual_mse = {
        name: [
            float(np.mean(np.square(value[:, step] - arrays.residual_pose[:, step])))
            for step in range(cache.chunk_steps)
        ]
        for name, value in chunk_predictions.items()
    }
    normalized_predictions = {name: value[:, 0] for name, value in chunk_predictions.items()}

    # Each delta is scored against the state it was produced from.
    base_state = {
        "reference": state_by_row[cache.key_dataset_indices[cache.row_key_positions]],
        "teacher_full": arrays.state,
        "teacher_null": arrays.state,
        "expert": arrays.state,
    }
    deltas = {
        "reference": cache.reference_actions[:, 0, : rot.POSE_DIMS].astype(np.float32),
        "teacher_full": arrays.full_pose[:, 0],
        "teacher_null": arrays.null_pose[:, 0],
        "expert": arrays.expert_pose[:, 0],
    }
    normalized_residual_target = arrays.residual_pose[:, 0]
    contact_summary: dict = {"available": False}
    strata: dict[str, np.ndarray] = {}

    # Reporting in metres and radians is what makes the numbers actionable;
    # normalized MSE cannot tell whether an error is safe on hardware.
    action_stats = norm_stats["actions"]
    poses = {
        name: to_absolute_pose(value, base_state[name], action_stats, norm_stats["state"])
        for name, value in deltas.items()
    }
    # The residual is a difference of two deltas sharing one base, so it needs
    # only the scale, not the offset.
    scale = (np.asarray(action_stats.std, dtype=np.float64)[: rot.POSE_DIMS] + 1e-6).astype(np.float32)
    physical_predictions = {name: value * scale for name, value in normalized_predictions.items()}

    wrench = wrench_by_row[ready]
    baseline = baseline_by_row[ready]
    magnitude = np.linalg.norm((wrench - baseline)[:, :3], axis=-1)
    in_contact = magnitude >= args.contact_threshold_n
    contact_summary = {
        "available": True,
        "criterion": "linear force deviation from the per-episode resting wrench",
        "threshold_n": args.contact_threshold_n,
        "baseline_rows": args.baseline_rows,
        "contact_rows": int(np.count_nonzero(in_contact)),
        "free_space_rows": int(np.count_nonzero(~in_contact)),
        "raw_force_magnitude_n": {
            "mean": float(np.mean(np.linalg.norm(wrench[:, :3], axis=-1))),
            "p50": float(np.percentile(np.linalg.norm(wrench[:, :3], axis=-1), 50)),
        },
        "baseline_corrected_magnitude_n": {
            "mean": float(np.mean(magnitude)),
            "p50": float(np.percentile(magnitude, 50)),
            "p95": float(np.percentile(magnitude, 95)),
            "max": float(np.max(magnitude)),
        },
    }
    if contact_summary["contact_rows"]:
        strata["contact"] = in_contact
    if contact_summary["free_space_rows"]:
        strata["free_space"] = ~in_contact

    metrics = {
        "rows": len(normalized_residual_target),
        "action_units": {"translation": "metres", "rotation": "radians_geodesic"},
        "residual_mse_space": "normalized_pose_residual",
        "chunk_steps": cache.chunk_steps,
        "per_step_residual_mse_normalized": per_step_residual_mse,
        "norm_stats_dir": str(args.norm_stats_dir),
        "contact": contact_summary,
        "strata": {
            "all": _stratum_metrics(
                poses,
                physical_predictions,
                normalized_predictions,
                normalized_residual_target,
            ),
            **{
                name: _stratum_metrics(
                    {key: value[selector] for key, value in poses.items()},
                    {key: value[selector] for key, value in physical_predictions.items()},
                    {key: value[selector] for key, value in normalized_predictions.items()},
                    normalized_residual_target[selector],
                )
                for name, selector in strata.items()
            },
        },
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    _print_report(metrics)


if __name__ == "__main__":
    main()
