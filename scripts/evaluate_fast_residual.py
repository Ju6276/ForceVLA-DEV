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


def _load_model(checkpoint: pathlib.Path, *, chunk_steps: int, force_blind_staleness: bool = True):
    """Rebuild the student, taking the head layout from the checkpoint itself.

    Whether a run has a staleness head is not something the caller should have to
    remember: a single-head checkpoint loaded into a two-head module would silently
    leave the staleness projection at its zero initialization and report the run as
    if it had simply learned nothing there.
    """
    params = model_lib.restore_params(checkpoint)
    predicts_staleness = "staleness_head" in params.get("fast_student", {})
    config = slow_fast.FastResidualConfig(
        chunk_steps=chunk_steps,
        predict_staleness=predicts_staleness,
        force_blind_staleness=force_blind_staleness,
    )
    model = slow_fast.FastStudentWithIntentProjector(config, slow_context_dim=2048, rngs=nnx.Rngs(0))
    state = nnx.state(model, nnx.Param)
    state.replace_by_pure_dict(params)
    nnx.update(model, state)
    return model, predicts_staleness


def _run_metadata(checkpoint: pathlib.Path) -> dict:
    """Read the training run's metadata from whichever ancestor directory holds it.

    Two of the recorded choices change what a forward pass means rather than what
    shape it has, so a checkpoint loaded without them runs silently and wrongly:
    `force_blind_staleness` decides whether the staleness head gets its own masked
    pass, and `analytic_rebase` decides whether the base-state gap is part of what
    the head emits or is supplied in closed form afterwards.
    """
    for directory in checkpoint.parents:
        metadata = directory / "metadata.json"
        if metadata.is_file():
            return json.loads(metadata.read_text())
    return {}


def _predict_all(model, arrays, cache, *, batch_size: int, seed: int):
    """Run the student under the force ablations, keeping both heads separate.

    The staleness head is collected under the same ablations even though it cannot
    see force. Its outputs must then be bit-identical across them, which is the
    cheapest available check that the force token really is masked out of it.
    """

    @nnx.jit
    def infer(module, context, context_mask, force, force_mask, state, reference, time_features):
        residual, staleness, _, _ = module(
            context,
            context_mask,
            force,
            force_mask,
            state,
            reference,
            time_features,
            train=False,
        )
        return residual, staleness

    permutation = np.random.default_rng(seed).permutation(len(arrays.dataset_indices))
    predictions: dict[str, list[np.ndarray]] = {"full": [], "zero_force": [], "shuffled_force": []}
    staleness_parts: dict[str, list[np.ndarray]] = {"full": [], "zero_force": [], "shuffled_force": []}
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
        shuffled = permutation[indices]
        variants = {
            "full": (force, force_mask),
            "zero_force": (jnp.zeros_like(force), force_mask),
            "shuffled_force": (
                jnp.asarray(arrays.force_history[shuffled], dtype=jnp.float32),
                jnp.asarray(arrays.force_history_mask[shuffled], dtype=jnp.bool_),
            ),
        }
        for name, (variant_force, variant_mask) in variants.items():
            residual, staleness = infer(model, *common, variant_force, variant_mask, *suffix)
            predictions[name].append(np.asarray(residual))
            if staleness is not None:
                staleness_parts[name].append(np.asarray(staleness))
    residuals = {name: np.concatenate(parts) for name, parts in predictions.items()}
    staleness = (
        {name: np.concatenate(parts) for name, parts in staleness_parts.items()}
        if staleness_parts["full"]
        else None
    )
    return residuals, staleness


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
    *,
    normalized_staleness: np.ndarray | None = None,
    physical_staleness: np.ndarray | None = None,
    normalized_staleness_target: np.ndarray | None = None,
) -> dict:
    """Compare every variant against both the Teacher and the true expert.

    The staleness correction enters every commanded action, since it is what the robot
    executes, but it is scored on its own target. Folding it into the force numbers
    would make the force ablations look better than they are: it improves the command
    without knowing anything about contact.
    """
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
    if normalized_staleness is not None:
        staleness_baseline = float(np.mean(np.square(normalized_staleness_target)))
        staleness_mse = float(np.mean(np.square(normalized_staleness - normalized_staleness_target)))
        metrics["staleness_vs_reference_drift"] = {
            "space": "normalized pose residual",
            "zero_staleness_baseline_mse_normalized": staleness_baseline,
            "mse_normalized": staleness_mse,
            "predicted_l2_mean_normalized": float(np.mean(np.linalg.norm(normalized_staleness, axis=-1))),
            "gain_vs_zero_staleness": 1.0 - staleness_mse / max(staleness_baseline, 1e-12),
        }
    baseline = metrics["residual_vs_teacher_deviation"]["zero_residual_baseline_mse_normalized"]
    for name, normalized_prediction in normalized_predictions.items():
        commanded = reference + physical_predictions[name]
        if physical_staleness is not None:
            commanded = commanded + physical_staleness
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
    if normalized_staleness is not None:
        metrics["deployment_error_decomposition"]["fast_staleness_vs_reference_drift_mse_normalized"] = metrics[
            "staleness_vs_reference_drift"
        ]["mse_normalized"]
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
        drift = values.get("staleness_vs_reference_drift")
        if drift is not None:
            print(
                "  Stale-reference drift recovery, normalized "
                f"(zero baseline={drift['zero_staleness_baseline_mse_normalized']:.6f}): "
                f"mse={drift['mse_normalized']:.6f}  gain={drift['gain_vs_zero_staleness']:.4f}"
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
    metadata = _run_metadata(args.checkpoint.resolve())
    model, predicts_staleness = _load_model(
        args.checkpoint.resolve(),
        chunk_steps=cache.chunk_steps,
        force_blind_staleness=bool(metadata.get("config", {}).get("force_blind_staleness", True)),
    )
    metrics = evaluate(
        model,
        arrays,
        cache,
        norm_stats,
        wrench_by_row,
        baseline_by_row,
        predicts_staleness=predicts_staleness,
        analytic_rebase=bool(metadata.get("analytic_rebase", False)),
        contact_threshold_n=args.contact_threshold_n,
        baseline_rows=args.baseline_rows,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    metrics["norm_stats_dir"] = str(args.norm_stats_dir)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    _print_report(metrics)


def evaluate(
    model,
    arrays,
    cache,
    norm_stats,
    wrench_by_row,
    baseline_by_row,
    *,
    predicts_staleness: bool,
    analytic_rebase: bool = False,
    contact_threshold_n: float,
    baseline_rows: int,
    batch_size: int = 256,
    seed: int = 0,
) -> dict:
    """Score one timing realization of the Slow cache.

    Split out of `main` so a timing sweep can reuse it: the whole point of the sweep
    is that every operating point is scored by exactly this code, and a second copy
    of the metric pipeline would make the curve incomparable to the single-point run.
    """
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
    chunk_predictions, chunk_staleness = _predict_all(model, arrays, cache, batch_size=batch_size, seed=seed)
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
    normalized_staleness = None if chunk_staleness is None else chunk_staleness["full"][:, 0]
    force_blindness_deviation = (
        None
        if chunk_staleness is None
        else max(
            float(np.max(np.abs(chunk_staleness[name] - chunk_staleness["full"])))
            for name in ("zero_force", "shuffled_force")
        )
    )

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
    normalized_staleness_target = None
    if normalized_staleness is not None:
        # The reference is a delta from its own key row's state, so the drift is not
        # `A_null - A_ref`: that difference is missing the base-state gap between the
        # two rows, which on the rotation coordinates is as large as the drift itself.
        state_scale = (np.asarray(norm_stats["state"].std, dtype=np.float64)[: rot.POSE_DIMS] + 1e-6).astype(
            np.float32
        )
        base_gap = arrays.state[:, : rot.POSE_DIMS] - state_by_row[cache.key_dataset_indices[cache.row_key_positions]][
            :, : rot.POSE_DIMS
        ]
        rebase = (state_scale / scale) * base_gap
        normalized_staleness_target = (
            arrays.null_pose[:, 0] - cache.reference_actions[:, 0, : rot.POSE_DIMS].astype(np.float32) + rebase
        )
        if analytic_rebase:
            # This run's head was trained on the drift alone, so the gap is supplied
            # here to put its numbers on the same target as every other run's. This is
            # an offline ablation only: nothing in `openpi.serving` reads the flag or
            # adds this term, so such a checkpoint is not deployable as it stands.
            normalized_staleness = normalized_staleness + rebase
    physical_staleness = None if normalized_staleness is None else normalized_staleness * scale

    wrench = wrench_by_row[ready]
    baseline = baseline_by_row[ready]
    magnitude = np.linalg.norm((wrench - baseline)[:, :3], axis=-1)
    in_contact = magnitude >= contact_threshold_n
    contact_summary = {
        "available": True,
        "criterion": "linear force deviation from the per-episode resting wrench",
        "threshold_n": contact_threshold_n,
        "baseline_rows": baseline_rows,
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
        "predicts_staleness": predicts_staleness,
        "analytic_rebase": analytic_rebase,
        # Exactly zero for a force-blind run, whose staleness head is evaluated under
        # force ablations it structurally cannot see; anything else means the force
        # token is reaching it. For the force-sighted ablation the reverse holds, and a
        # zero here would mean the checkpoint was loaded into the wrong architecture.
        "staleness_force_blindness_max_deviation": force_blindness_deviation,
        "contact": contact_summary,
        "strata": {
            "all": _stratum_metrics(
                poses,
                physical_predictions,
                normalized_predictions,
                normalized_residual_target,
                normalized_staleness=normalized_staleness,
                physical_staleness=physical_staleness,
                normalized_staleness_target=normalized_staleness_target,
            ),
            **{
                name: _stratum_metrics(
                    {key: value[selector] for key, value in poses.items()},
                    {key: value[selector] for key, value in physical_predictions.items()},
                    {key: value[selector] for key, value in normalized_predictions.items()},
                    normalized_residual_target[selector],
                    normalized_staleness=None if normalized_staleness is None else normalized_staleness[selector],
                    physical_staleness=None if physical_staleness is None else physical_staleness[selector],
                    normalized_staleness_target=(
                        None if normalized_staleness_target is None else normalized_staleness_target[selector]
                    ),
                )
                for name, selector in strata.items()
            },
        },
    }
    return metrics


if __name__ == "__main__":
    main()
