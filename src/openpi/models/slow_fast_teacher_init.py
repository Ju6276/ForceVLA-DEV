"""Initialize the Fast student from the unified ForceVLA Teacher.

Only modules with a genuine one-to-one correspondence are transferred:

* the causal TCN stem and temporal blocks (the Teacher's 1024->2048 output
  projection is deliberately excluded), and
* one 1024-wide Action Expert Gemma block.

The Teacher Action Expert is LoRA-tuned while the Fast block is dense.  Its
LoRA deltas therefore have to be merged into the copied dense weights; copying
only ``w`` would silently discard the task-specific part of the Teacher.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax.numpy as jnp
import numpy as np

from openpi.models import slow_fast


@dataclasses.dataclass(frozen=True)
class TeacherInitReport:
    teacher_layer: int
    tcn_arrays: int
    gemma_arrays: int
    merged_lora_arrays: int
    copied_parameters: int
    action_expert_lora_scale: float


def _child(tree: dict, key: str | int):
    if key in tree:
        return tree[key]
    alternate = str(key) if isinstance(key, int) else int(key) if key.isdigit() else None
    if alternate is not None and alternate in tree:
        return tree[alternate]
    raise KeyError(f"Teacher checkpoint is missing key {key!r}; available keys: {tuple(tree)}")


def _layer(array: Any, index: int, *, name: str) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim == 0 or not 0 <= index < value.shape[0]:
        raise ValueError(f"Teacher {name} has shape {value.shape}; cannot select layer {index}")
    return value[index]


def _copy(target, source: Any, *, name: str) -> int:
    value = np.asarray(source)
    if tuple(value.shape) != tuple(target.value.shape):
        raise ValueError(f"Teacher {name} shape {value.shape} does not match Fast shape {target.value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"Teacher {name} contains non-finite values")
    target.value = jnp.asarray(value, dtype=target.value.dtype)
    return int(value.size)


def _merge_lora(base: Any, a: Any | None, b: Any | None, *, scale: float, name: str) -> tuple[np.ndarray, bool]:
    base_value = np.asarray(base, dtype=np.float32)
    if a is None and b is None:
        return base_value, False
    if a is None or b is None:
        raise ValueError(f"Teacher {name} has only one of the two LoRA factors")
    a_value = np.asarray(a, dtype=np.float32)
    b_value = np.asarray(b, dtype=np.float32)
    try:
        delta = np.matmul(a_value, b_value)
    except ValueError as error:
        raise ValueError(f"Teacher {name} LoRA shapes {a_value.shape} and {b_value.shape} cannot be merged") from error
    if delta.shape != base_value.shape:
        raise ValueError(f"Teacher {name} LoRA delta shape {delta.shape} does not match base {base_value.shape}")
    return base_value + delta * scale, True


def _copy_tcn(target, source: dict) -> tuple[int, int]:
    arrays = 0
    parameters = 0

    def copy_param(target_param, source_tree, key, name):
        nonlocal arrays, parameters
        parameters += _copy(target_param, _child(source_tree, key), name=name)
        arrays += 1

    source_stem = _child(source, "stem")
    copy_param(target.stem.kernel, source_stem, "kernel", "temporal_force_encoder/stem/kernel")
    copy_param(target.stem.bias, source_stem, "bias", "temporal_force_encoder/stem/bias")

    source_blocks = _child(source, "blocks")
    if len(source_blocks) != len(target.blocks):
        raise ValueError(f"Teacher has {len(source_blocks)} TCN blocks but Fast has {len(target.blocks)}")
    for index, target_block in enumerate(target.blocks):
        source_block = _child(source_blocks, index)
        for module_name in ("conv1", "conv2"):
            target_module = getattr(target_block, module_name)
            source_module = _child(source_block, module_name)
            copy_param(target_module.kernel, source_module, "kernel", f"TCN block {index}/{module_name}/kernel")
            copy_param(target_module.bias, source_module, "bias", f"TCN block {index}/{module_name}/bias")
        for module_name in ("norm1", "norm2"):
            target_module = getattr(target_block, module_name)
            source_module = _child(source_block, module_name)
            copy_param(target_module.scale, source_module, "scale", f"TCN block {index}/{module_name}/scale")
            copy_param(target_module.bias, source_module, "bias", f"TCN block {index}/{module_name}/bias")
        if target_block.residual is not None:
            source_residual = _child(source_block, "residual")
            copy_param(target_block.residual.kernel, source_residual, "kernel", f"TCN block {index}/residual/kernel")
            copy_param(target_block.residual.bias, source_residual, "bias", f"TCN block {index}/residual/bias")
    return arrays, parameters


def _copy_gemma_block(target, teacher_layers: dict, *, layer_index: int) -> tuple[int, int, int, float]:
    arrays = 0
    parameters = 0
    merged_arrays = 0
    # ForceVLA's gemma_300m_lora configuration has rank=alpha=32, hence scale=1.
    # Compute it from the checkpoint rank and the repository-fixed alpha so a
    # malformed or differently ranked checkpoint cannot be merged silently.
    action_attn = {
        "attn_vec_einsum": _child(_child(teacher_layers, "attn"), "attn_vec_einsum_1"),
        "kv_einsum": _child(_child(teacher_layers, "attn"), "kv_einsum_1"),
        "q_einsum": _child(_child(teacher_layers, "attn"), "q_einsum_1"),
    }
    first_lora = action_attn["q_einsum"].get("lora_a")
    lora_rank = 32 if first_lora is None else int(np.asarray(first_lora).shape[-1])
    lora_scale = 32.0 / lora_rank

    for name, source in action_attn.items():
        base = _layer(_child(source, "w"), layer_index, name=f"Action Expert layer/{name}/w")
        a = None if "lora_a" not in source else _layer(source["lora_a"], layer_index, name=f"{name}/lora_a")
        b = None if "lora_b" not in source else _layer(source["lora_b"], layer_index, name=f"{name}/lora_b")
        merged, used_lora = _merge_lora(base, a, b, scale=lora_scale, name=f"Action Expert/{name}")
        target_param = target.attn[name]["w"]
        parameters += _copy(target_param, merged, name=f"Action Expert/{name}")
        arrays += 1
        merged_arrays += int(used_lora)

    source_mlp = _child(teacher_layers, "mlp_1")
    for name in ("gating_einsum", "linear"):
        base = _layer(_child(source_mlp, name), layer_index, name=f"Action Expert layer/mlp/{name}")
        a_key, b_key = f"{name}_lora_a", f"{name}_lora_b"
        a = None if a_key not in source_mlp else _layer(source_mlp[a_key], layer_index, name=a_key)
        b = None if b_key not in source_mlp else _layer(source_mlp[b_key], layer_index, name=b_key)
        # openpi.models.lora.FeedForward applies A@B without an extra scale.
        merged, used_lora = _merge_lora(base, a, b, scale=1.0, name=f"Action Expert/mlp/{name}")
        parameters += _copy(target.mlp[name], merged, name=f"Action Expert/mlp/{name}")
        arrays += 1
        merged_arrays += int(used_lora)

    for source_name, target_name in (
        ("pre_attention_norm_1", "pre_attention_norm"),
        ("pre_ffw_norm_1", "pre_ffw_norm"),
    ):
        source = _child(_child(teacher_layers, source_name), "scale")
        value = _layer(source, layer_index, name=f"Action Expert layer/{source_name}/scale")
        parameters += _copy(getattr(target, target_name)["scale"], value, name=f"Action Expert/{target_name}/scale")
        arrays += 1

    return arrays, parameters, merged_arrays, lora_scale


def initialize_fast_from_forcevla_teacher(
    model: slow_fast.FastStudentWithIntentProjector,
    teacher_params: dict,
    *,
    layer_index: int = 0,
) -> TeacherInitReport:
    """Transfer the compatible TCN and Action Expert layer into ``model``.

    All Fast-specific projections, intent queries, residual queries, and output
    heads remain at their normal random/zero-safe initialization.
    """
    if model.fast_student.config.decoder_type != "flash_gemma":
        raise ValueError("Teacher initialization requires decoder_type='flash_gemma'")
    if layer_index < 0:
        raise ValueError("teacher layer index must be non-negative")
    if "params" in teacher_params and "PaliGemma" not in teacher_params:
        teacher_params = teacher_params["params"]

    teacher_tcn = _child(teacher_params, "temporal_force_encoder")
    tcn_arrays, tcn_parameters = _copy_tcn(model.fast_student.force_encoder, teacher_tcn)

    teacher_layers = _child(_child(_child(teacher_params, "PaliGemma"), "llm"), "layers")
    gemma_arrays, gemma_parameters, merged_arrays, lora_scale = _copy_gemma_block(
        model.fast_student.decoder.block,
        teacher_layers,
        layer_index=layer_index,
    )
    return TeacherInitReport(
        teacher_layer=layer_index,
        tcn_arrays=tcn_arrays,
        gemma_arrays=gemma_arrays,
        merged_lora_arrays=merged_arrays,
        copied_parameters=tcn_parameters + gemma_parameters,
        action_expert_lora_scale=lora_scale,
    )
