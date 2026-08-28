"""Audit dense LoRA folding on a real ForceVLA Orbax checkpoint."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from openpi.models import model as model_lib
from openpi.models_pytorch import lora_merge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    params = model_lib.restore_params(args.checkpoint / "params", restore_type=np.ndarray)
    source = params["PaliGemma"]["llm"]
    merged, merged_paths = lora_merge.merge_openpi_gemma_lora(source)
    remaining = lora_merge.find_lora_paths(merged)

    # Independently check representative attention and MLP contractions.  This
    # catches both an incorrect rank axis and an incorrect scale.
    rng = np.random.default_rng(args.seed)
    checks = []
    examples = (
        (source["layers"]["attn"]["q_einsum"], merged["layers"]["attn"]["q_einsum"]["w"]),
        (source["layers"]["attn"]["kv_einsum_1"], merged["layers"]["attn"]["kv_einsum_1"]["w"]),
    )
    for original, dense in examples:
        layer = 0
        base = np.asarray(original["w"][layer], dtype=np.float32)
        a = np.asarray(original["lora_a"][layer], dtype=np.float32)
        b = np.asarray(original["lora_b"][layer], dtype=np.float32)
        x = rng.normal(size=base.shape[:-2] + (3, base.shape[-2])).astype(np.float32)
        expected = np.matmul(x, base) + np.matmul(np.matmul(x, a), b)
        actual = np.matmul(x, np.asarray(dense[layer]))
        checks.append(float(np.max(np.abs(expected - actual))))

    for name in ("mlp", "mlp_1"):
        original = source["layers"][name]
        dense = merged["layers"][name]["linear"]
        layer = 0
        base = np.asarray(original["linear"][layer], dtype=np.float32)
        a = np.asarray(original["linear_lora_a"][layer], dtype=np.float32)
        b = np.asarray(original["linear_lora_b"][layer], dtype=np.float32)
        x = rng.normal(size=(3, base.shape[-2])).astype(np.float32)
        expected = x @ base + (x @ a) @ b
        actual = x @ np.asarray(dense[layer])
        checks.append(float(np.max(np.abs(expected - actual))))

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "merged_dense_arrays": len(merged_paths),
        "merged_paths": merged_paths,
        "remaining_lora_paths": remaining,
        "max_contraction_error": max(checks),
        "passed": not remaining and max(checks) < 2e-3,
    }
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("LoRA merge validation failed")


if __name__ == "__main__":
    main()
