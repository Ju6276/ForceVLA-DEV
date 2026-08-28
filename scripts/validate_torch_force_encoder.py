"""Validate JAX -> PyTorch conversion of a trained Temporal ForceVLA TCN."""

from __future__ import annotations

import argparse
import json
import pathlib

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch

from openpi.models import force_encoder
from openpi.models import model as model_lib
from openpi.models_pytorch import force_encoder as torch_force_encoder


def _select_param_tree(template: dict, source: dict) -> dict:
    """Select checkpoint leaves present in an NNX Param-state template.

    Orbax checkpoints may also contain non-parameter state (for example the
    dropout RNG counter).  Block indices additionally round-trip through
    Orbax as strings, whereas a freshly created NNX state uses integer keys.
    """
    selected = {}
    for key, value in template.items():
        source_key = key if key in source else str(key)
        source_value = source[source_key]
        selected[key] = (
            _select_param_tree(value, source_value) if isinstance(value, dict) else source_value
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    params = model_lib.restore_params(args.checkpoint / "params", restore_type=np.ndarray)
    source = params["temporal_force_encoder"]
    config = force_encoder.ForceEncoderConfig(
        type="tcn", sampling_rate_hz=100, window_ms=100, max_sample_age_ms=12
    )
    jax_model = force_encoder.TemporalTCNForceEncoder(config, output_dim=2048, rngs=nnx.Rngs(0))
    jax_state = nnx.state(jax_model, nnx.Param)
    jax_state.replace_by_pure_dict(_select_param_tree(jax_state.to_pure_dict(), source))
    nnx.update(jax_model, jax_state)

    torch_model = torch_force_encoder.TemporalTCNForceEncoder(config, output_dim=2048)
    torch_force_encoder.load_jax_params(torch_model, source)
    torch_model.eval()

    generator = np.random.default_rng(args.seed)
    history = generator.normal(size=(3, config.max_history_samples, 6)).astype(np.float32)
    mask = np.ones((3, config.max_history_samples), dtype=np.bool_)
    mask[1, :3] = False
    mask[2, :7] = False
    jax_output = np.asarray(jax_model(jnp.asarray(history), jnp.asarray(mask), train=False))
    with torch.inference_mode():
        torch_output = torch_model(torch.from_numpy(history), torch.from_numpy(mask)).numpy()

    difference = np.abs(jax_output - torch_output)
    rmse = float(np.sqrt(np.mean((jax_output - torch_output) ** 2)))
    reference_rms = float(np.sqrt(np.mean(jax_output**2)))
    relative_rmse = rmse / max(reference_rms, np.finfo(np.float32).eps)
    cosine = np.sum(jax_output * torch_output, axis=-1) / (
        np.linalg.norm(jax_output, axis=-1) * np.linalg.norm(torch_output, axis=-1)
    )
    # CPU convolution is suitable for a strict conversion check. XLA GPU and
    # PyTorch CUDA use different FP32 convolution kernels, so the accumulated
    # error through eight wide convolutions is slightly larger.
    backend = jax.default_backend()
    parity_passed = (
        bool(np.allclose(jax_output, torch_output, rtol=1e-4, atol=1e-4))
        if backend == "cpu"
        else relative_rmse < 1.5e-3 and float(cosine.min()) > 0.999999
    )
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "jax_backend": backend,
        "shape": list(jax_output.shape),
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "rmse": rmse,
        "reference_rms": reference_rms,
        "relative_rmse": relative_rmse,
        "min_cosine_similarity": float(cosine.min()),
        "allclose_rtol_1e-4_atol_1e-4": bool(np.allclose(jax_output, torch_output, rtol=1e-4, atol=1e-4)),
        "numerical_parity_passed": parity_passed,
    }
    print(json.dumps(result, indent=2))
    if not parity_passed:
        raise SystemExit("Converted PyTorch TCN does not reproduce the JAX encoder")


if __name__ == "__main__":
    main()
