"""Utilities for folding OpenPI JAX LoRA weights into dense kernels.

The official JAX-to-PyTorch converter understands dense Gemma kernels.  A
ForceVLA LoRA checkpoint additionally stores low-rank A/B leaves, so those
updates must be folded into the dense arrays before applying the ordinary
layout conversion.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
from typing import Any

import numpy as np


def merge_low_rank(base: Any, a: Any, b: Any, *, scale: float) -> np.ndarray:
    """Return ``base + scale * (a @ b)`` over the final contraction axes."""
    base_array = np.asarray(base, dtype=np.float32)
    a_array = np.asarray(a, dtype=np.float32)
    b_array = np.asarray(b, dtype=np.float32)
    update = np.matmul(a_array, b_array)
    if update.shape != base_array.shape:
        raise ValueError(f"LoRA update {update.shape} does not match base kernel {base_array.shape}")
    return base_array + np.float32(scale) * update


def merge_openpi_gemma_lora(llm: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Fold all LoRA leaves in an OpenPI Gemma parameter subtree.

    OpenPI's Gemma LoRA variants use alpha=rank for attention projections, so
    their alpha/rank factor is one.  ``FeedForward`` applies A@B directly and
    likewise has an effective scale of one.
    """
    result = copy.deepcopy(dict(llm))
    layers = result["layers"]
    merged: list[str] = []

    attn = layers["attn"]
    for name, values in attn.items():
        if not isinstance(values, dict) or "w" not in values:
            continue
        if "lora_a" not in values and "lora_b" not in values:
            continue
        if "lora_a" not in values or "lora_b" not in values:
            raise ValueError(f"Incomplete LoRA pair for layers/attn/{name}")
        values["w"] = merge_low_rank(values["w"], values.pop("lora_a"), values.pop("lora_b"), scale=1.0)
        merged.append(f"layers/attn/{name}/w")

    for name, values in layers.items():
        if not name.startswith("mlp") or not isinstance(values, dict):
            continue
        for base_name, a_name, b_name in (
            ("gating_einsum", "gating_einsum_lora_a", "gating_einsum_lora_b"),
            ("linear", "linear_lora_a", "linear_lora_b"),
        ):
            present = (a_name in values, b_name in values)
            if not any(present):
                continue
            if not all(present):
                raise ValueError(f"Incomplete LoRA pair for layers/{name}/{base_name}")
            values[base_name] = merge_low_rank(
                values[base_name], values.pop(a_name), values.pop(b_name), scale=1.0
            )
            merged.append(f"layers/{name}/{base_name}")

    return result, merged


def find_lora_paths(tree: Mapping[str, Any], prefix: str = "") -> list[str]:
    paths: list[str] = []
    for key, value in tree.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if "lora" in str(key):
            paths.append(path)
        if isinstance(value, Mapping):
            paths.extend(find_lora_paths(value, path))
    return paths
