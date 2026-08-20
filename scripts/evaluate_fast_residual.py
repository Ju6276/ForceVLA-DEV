"""Evaluate a trained Fast student and test whether it uses force history.

Beyond the Teacher-referenced residual metrics, this reports how every policy
variant compares to the ground-truth expert action, and splits the held-out set
into contact and free-space rows so that contact behaviour is not diluted by the
much larger number of free-space samples.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from flax import nnx
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.models import slow_fast
from openpi.shared import normalize as normalize_lib
from openpi.training import fast_dataset

TRANSLATION_DIMS = slice(0, 3)
ROTATION_DIMS = slice(3, 6)
POSE_DIM_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")


def _pose_difference(prediction: np.ndarray, target: np.ndarray, *, physical: bool) -> np.ndarray:
    """Subtract two poses, wrapping rotation dimensions into (-pi, pi].

    A few episodes record roll near -2*pi instead of 0, so a plain subtraction
    would report a full revolution as a huge error.
    """
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    if physical:
        rotation = error[:, ROTATION_DIMS]
        error[:, ROTATION_DIMS] = (rotation + np.pi) % (2 * np.pi) - np.pi
    return error


def _load_model(checkpoint: pathlib.Path):
    config = slow_fast.FastResidualConfig()
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
            jnp.asarray(cache.reference_actions[indices], dtype=jnp.float32),
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


def _pose_error_summary(prediction: np.ndarray, target: np.ndarray, *, physical: bool) -> dict:
    error = _pose_difference(prediction, target, physical=physical)
    return {
        "mse": float(np.mean(np.square(error))),
        "translation_rmse": float(np.sqrt(np.mean(np.sum(np.square(error[:, TRANSLATION_DIMS]), axis=-1)))),
        "rotation_rmse": float(np.sqrt(np.mean(np.sum(np.square(error[:, ROTATION_DIMS]), axis=-1)))),
        "per_dim_rmse": {
            name: float(np.sqrt(np.mean(np.square(error[:, index])))) for index, name in enumerate(POSE_DIM_NAMES)
        },
    }


def _stratum_metrics(
    poses: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    residual_target,
    *,
    physical: bool,
) -> dict:
    """Compare every variant against both the Teacher and the true expert."""
    reference = poses["reference"]
    metrics: dict = {
        "rows": len(reference),
        "residual_vs_teacher_deviation": {"zero_residual_baseline_mse": float(np.mean(np.square(residual_target)))},
        "action_vs_teacher_full": {
            "slow_reference_only": _pose_error_summary(reference, poses["teacher_full"], physical=physical)
        },
        "action_vs_expert": {
            "slow_reference_only": _pose_error_summary(reference, poses["expert"], physical=physical),
            "teacher_full": _pose_error_summary(poses["teacher_full"], poses["expert"], physical=physical),
            "teacher_null": _pose_error_summary(poses["teacher_null"], poses["expert"], physical=physical),
        },
    }
    baseline = metrics["residual_vs_teacher_deviation"]["zero_residual_baseline_mse"]
    for name, prediction in predictions.items():
        commanded = reference + prediction
        metrics["residual_vs_teacher_deviation"][name] = {
            "mse": float(np.mean(np.square(prediction - residual_target))),
            "predicted_l2_mean": float(np.mean(np.linalg.norm(prediction, axis=-1))),
        }
        metrics["residual_vs_teacher_deviation"][name]["gain_vs_zero_residual"] = 1.0 - metrics[
            "residual_vs_teacher_deviation"
        ][name]["mse"] / max(baseline, 1e-12)
        metrics["action_vs_teacher_full"][f"slow_plus_{name}"] = _pose_error_summary(
            commanded, poses["teacher_full"], physical=physical
        )
        metrics["action_vs_expert"][f"slow_plus_{name}"] = _pose_error_summary(
            commanded, poses["expert"], physical=physical
        )

    # Translation and rotation are reported separately: an inflated rotation
    # error would otherwise hide a genuine translation improvement.
    for group, baseline_name in (("action_vs_expert", "expert"), ("action_vs_teacher_full", "teacher_full")):
        del baseline_name
        section = metrics[group]
        section["fast_gain_vs_slow_only"] = {
            key: 1.0 - section["slow_plus_full"][key] / max(section["slow_reference_only"][key], 1e-12)
            for key in ("mse", "translation_rmse", "rotation_rmse")
        }
    # The commanded action's deviation from the Teacher splits exactly, as vectors,
    # into a force-agnostic Slow term and the Fast term the loss optimizes:
    #   (reference + delta) - A_full = (reference - A_null) + (delta - residual_target)
    # Only the second term is trained, so the first bounds what Fast can achieve.
    # The two mean squares do not add up to the total: the terms share A_null, so
    # there is a cross term. Compare their magnitudes; read `total_mse` for the sum.
    metrics["deployment_error_decomposition"] = {
        "slow_term_mse": float(np.mean(np.square(reference[..., : poses["teacher_null"].shape[-1]] - poses["teacher_null"]))),
        "fast_term_mse": metrics["residual_vs_teacher_deviation"]["full"]["mse"],
        "total_mse": metrics["action_vs_teacher_full"]["slow_plus_full"]["mse"],
    }
    metrics["action_vs_expert"]["teacher_force_gain"] = {
        key: 1.0
        - metrics["action_vs_expert"]["teacher_full"][key]
        / max(metrics["action_vs_expert"]["teacher_null"][key], 1e-12)
        for key in ("mse", "translation_rmse", "rotation_rmse")
    }
    return metrics


def _print_report(metrics: dict) -> None:
    print("\n=== Fast residual evaluation ===")
    for stratum, values in metrics["strata"].items():
        print(f"\n[{stratum}]  rows={values['rows']}")
        physical = metrics["units"] == "physical"
        unit = "m / rad" if physical else "normalized"
        print(f"  action error vs ground-truth expert ({unit}):")
        for name in ("slow_reference_only", "slow_plus_full", "teacher_null", "teacher_full"):
            item = values["action_vs_expert"][name]
            per_dim = "  ".join(f"{key}={value:.5f}" for key, value in item["per_dim_rmse"].items())
            print(
                f"    {name:<22} mse={item['mse']:.6f}  "
                f"trans_rmse={item['translation_rmse']:.5f}  rot_rmse={item['rotation_rmse']:.5f}"
            )
            print(f"      per-dim rmse: {per_dim}")
        for label, key in (
            ("fast vs slow-only", "fast_gain_vs_slow_only"),
            ("teacher force gain", "teacher_force_gain"),
        ):
            gain = values["action_vs_expert"][key]
            print(
                f"    {label:<20} mse={gain['mse'] * 100:+.2f}%  "
                f"translation={gain['translation_rmse'] * 100:+.2f}%  rotation={gain['rotation_rmse'] * 100:+.2f}%"
            )
        residual = values["residual_vs_teacher_deviation"]
        print(f"  Teacher deviation reproduction (zero baseline={residual['zero_residual_baseline_mse']:.6f}):")
        for name in ("full", "zero_force", "shuffled_force"):
            print(f"    {name:<22} mse={residual[name]['mse']:.6f}  gain={residual[name]['gain_vs_zero_residual']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=pathlib.Path, required=True)
    parser.add_argument("--slow-cache", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument(
        "--norm-stats-dir",
        type=pathlib.Path,
        default=pathlib.Path("assets/forcevla_button_temporal_100hz/panda_button_press_temporal_100hz_train56"),
        help="Directory holding norm_stats.json. Enables physical units and contact stratification.",
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

    arrays = fast_dataset.load_stage3_fast_arrays(args.targets)
    cache = fast_dataset.load_slow_cache(args.slow_cache, expected_rows=len(arrays.dataset_indices))
    model = _load_model(args.checkpoint.resolve())
    predictions = _predict_all(model, arrays, cache, batch_size=args.batch_size, seed=args.seed)

    poses = {
        "reference": cache.reference_actions[:, :6].astype(np.float32),
        "teacher_full": arrays.full_pose,
        "teacher_null": arrays.null_pose,
        "expert": arrays.expert_pose,
    }
    residual_target = arrays.residual_pose
    units = "normalized"
    contact_summary: dict = {"available": False}
    strata: dict[str, np.ndarray] = {}

    norm_stats = None
    if args.norm_stats_dir is not None and (args.norm_stats_dir / "norm_stats.json").is_file():
        norm_stats = normalize_lib.load(args.norm_stats_dir)

    if norm_stats is not None:
        # Reporting in metres and radians is what makes the numbers actionable;
        # normalized MSE cannot tell whether an error is safe on hardware.
        action_stats = norm_stats["actions"]
        poses = {name: fast_dataset.denormalize(value, action_stats, dims=6) for name, value in poses.items()}
        scale = (np.asarray(action_stats.std, dtype=np.float64)[:6] + 1e-6).astype(np.float32)
        predictions = {name: value * scale for name, value in predictions.items()}
        residual_target = residual_target * scale
        units = "physical"

        wrench = fast_dataset.latest_physical_wrench(
            arrays.force_history, arrays.force_history_mask, norm_stats["force_history"]
        )
        baseline = fast_dataset.episode_baseline_wrench(wrench, arrays.episode_indices, num_rows=args.baseline_rows)
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
        "rows": len(residual_target),
        "units": units,
        "norm_stats_dir": str(args.norm_stats_dir) if norm_stats is not None else None,
        "contact": contact_summary,
        "strata": {
            "all": _stratum_metrics(poses, predictions, residual_target, physical=units == "physical"),
            **{
                name: _stratum_metrics(
                    {key: value[selector] for key, value in poses.items()},
                    {key: value[selector] for key, value in predictions.items()},
                    residual_target[selector],
                    physical=units == "physical",
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
