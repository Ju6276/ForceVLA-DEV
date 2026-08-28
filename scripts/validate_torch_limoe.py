"""Compare the trained JAX and PyTorch LIMoE inference paths."""

from __future__ import annotations

import argparse
import json
import pathlib

import jax.numpy as jnp
import numpy as np
import torch

from openpi.models import limoe_simple
from openpi.models import model as model_lib
from openpi.models_pytorch import limoe as torch_limoe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    args = parser.parse_args()
    params = model_lib.restore_params(args.checkpoint / "params", restore_type=np.ndarray)["limoe"]
    inputs = np.random.default_rng(5).normal(size=(1, 817, 2048)).astype(np.float32)

    jax_module = limoe_simple.LIMoEBlock(mlp_dim=2048, num_experts=4, num_top_k=1, num_heads=8, out_dim=1024)
    jax_output = np.asarray(jax_module.apply({"params": params}, jnp.asarray(inputs), True)[0])
    torch_module = torch_limoe.load_jax_params(torch_limoe.BatchOneLIMoE(), params).eval()
    with torch.inference_mode():
        torch_output = torch_module(torch.from_numpy(inputs)).numpy()

    diff = jax_output - torch_output
    rmse = float(np.sqrt(np.mean(diff**2)))
    reference_rms = float(np.sqrt(np.mean(jax_output**2)))
    cosine = np.sum(jax_output * torch_output, axis=-1) / (
        np.linalg.norm(jax_output, axis=-1) * np.linalg.norm(torch_output, axis=-1)
    )
    result = {
        "max_abs_error": float(np.max(np.abs(diff))),
        "relative_rmse": rmse / reference_rms,
        "min_token_cosine_similarity": float(np.min(cosine)),
    }
    result["passed"] = result["relative_rmse"] < 2e-4 and result["min_token_cosine_similarity"] > 0.99999
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("PyTorch LIMoE does not reproduce JAX LIMoE")


if __name__ == "__main__":
    main()
