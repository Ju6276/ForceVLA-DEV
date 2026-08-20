"""Evaluate a trained Fast student and test whether it uses force history."""

from __future__ import annotations

import argparse
import json
import pathlib

from flax import nnx
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.models import slow_fast
from openpi.training import fast_dataset


def _load_model(checkpoint: pathlib.Path):
    config = slow_fast.FastResidualConfig()
    model = slow_fast.FastStudentWithIntentProjector(config, slow_context_dim=2048, rngs=nnx.Rngs(0))
    params = model_lib.restore_params(checkpoint)
    state = nnx.state(model, nnx.Param)
    state.replace_by_pure_dict(params)
    nnx.update(model, state)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=pathlib.Path, required=True)
    parser.add_argument("--slow-cache", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    arrays = fast_dataset.load_stage3_fast_arrays(args.targets)
    cache = fast_dataset.load_slow_cache(args.slow_cache, expected_rows=len(arrays.dataset_indices))
    model = _load_model(args.checkpoint.resolve())

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

    permutation = np.random.default_rng(args.seed).permutation(len(arrays.dataset_indices))
    predictions = {"full": [], "zero_force": [], "shuffled_force": []}
    for start in range(0, len(arrays.dataset_indices), args.batch_size):
        indices = np.arange(start, min(start + args.batch_size, len(arrays.dataset_indices)))
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
        shuffled_indices = permutation[indices]
        predictions["shuffled_force"].append(
            np.asarray(
                infer(
                    model,
                    *common,
                    jnp.asarray(arrays.force_history[shuffled_indices], dtype=jnp.float32),
                    jnp.asarray(arrays.force_history_mask[shuffled_indices], dtype=jnp.bool_),
                    *suffix,
                )
            )
        )

    predictions = {name: np.concatenate(parts) for name, parts in predictions.items()}
    target = arrays.residual_pose
    reference = cache.reference_actions[:, :6]
    metrics = {
        "rows": len(target),
        "zero_residual_mse": float(np.mean(np.square(target))),
        "reference_only_reconstruction_mse": float(np.mean(np.square(reference - arrays.full_pose))),
    }
    for name, prediction in predictions.items():
        metrics[f"{name}_residual_mse"] = float(np.mean(np.square(prediction - target)))
        metrics[f"{name}_reconstruction_mse"] = float(
            np.mean(np.square(reference + prediction - arrays.full_pose))
        )
        metrics[f"{name}_predicted_l2_mean"] = float(np.mean(np.linalg.norm(prediction, axis=-1)))
    for name in ("zero_force", "shuffled_force"):
        metrics[f"{name}_prediction_change_l2_mean"] = float(
            np.mean(np.linalg.norm(predictions[name] - predictions["full"], axis=-1))
        )
        metrics[f"{name}_mse_increase_vs_full"] = (
            metrics[f"{name}_residual_mse"] / metrics["full_residual_mse"] - 1.0
        )
    metrics["full_gain_vs_zero_residual"] = 1.0 - metrics["full_residual_mse"] / metrics["zero_residual_mse"]
    metrics["full_reconstruction_gain_vs_reference"] = (
        1.0 - metrics["full_reconstruction_mse"] / metrics["reference_only_reconstruction_mse"]
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
