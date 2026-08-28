import copy

from flax import nnx
import numpy as np
import pytest

from openpi.models import force_encoder
from openpi.models import slow_fast
from openpi.models import slow_fast_teacher_init


def _array_like(param, value):
    return np.full(param.value.shape, value, dtype=np.float32)


def _layers(shape, first, second):
    return np.stack(
        [np.full(shape, first, dtype=np.float32), np.full(shape, second, dtype=np.float32)],
        axis=0,
    )


def test_teacher_init_copies_tcn_and_merges_action_expert_lora_only():
    config = slow_fast.FastResidualConfig(
        reference_dim=7,
        state_dim=8,
        pose_dims=6,
        chunk_steps=3,
        intent_dim=16,
        width=16,
        mlp_dim=32,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        decoder_type="flash_gemma",
        force_encoder=force_encoder.ForceEncoderConfig(
            type="tcn",
            hidden_dims=(16,),
            dilations=(1,),
            dropout_rate=0.0,
            sampling_rate_hz=100,
            window_ms=20,
        ),
    )
    model = slow_fast.FastStudentWithIntentProjector(config, slow_context_dim=16, rngs=nnx.Rngs(0))
    fast = model.fast_student
    untouched_state_kernel = np.asarray(fast.state_proj.kernel.value).copy()

    source_block = fast.force_encoder.blocks[0]
    teacher_tcn = {
        "stem": {
            "kernel": _array_like(fast.force_encoder.stem.kernel, 1.0),
            "bias": _array_like(fast.force_encoder.stem.bias, 2.0),
        },
        "blocks": {
            "0": {
                "conv1": {
                    "kernel": _array_like(source_block.conv1.kernel, 3.0),
                    "bias": _array_like(source_block.conv1.bias, 4.0),
                },
                "conv2": {
                    "kernel": _array_like(source_block.conv2.kernel, 5.0),
                    "bias": _array_like(source_block.conv2.bias, 6.0),
                },
                "norm1": {
                    "scale": _array_like(source_block.norm1.scale, 7.0),
                    "bias": _array_like(source_block.norm1.bias, 8.0),
                },
                "norm2": {
                    "scale": _array_like(source_block.norm2.scale, 9.0),
                    "bias": _array_like(source_block.norm2.bias, 10.0),
                },
            }
        },
        # The Teacher-only 1024->2048 projection must not be consumed.
        "output_proj": {"kernel": np.zeros((16, 32)), "bias": np.zeros((32,))},
    }

    target_block = fast.decoder.block
    rank = 2
    attention = {}
    expected_attention = {}
    for name in ("attn_vec_einsum", "kv_einsum", "q_einsum"):
        shape = target_block.attn[name]["w"].value.shape
        a_shape = (*shape[:-1], rank)
        b_shape = (*shape[:-2], rank, shape[-1])
        base = _layers(shape, 1.0, 2.0)
        a = _layers(a_shape, 0.1, 0.2)
        b = _layers(b_shape, 0.3, 0.4)
        attention[f"{name}_1"] = {"w": base, "lora_a": a, "lora_b": b}
        expected_attention[name] = base[1] + np.matmul(a[1], b[1]) * (32.0 / rank)

    mlp = {}
    expected_mlp = {}
    for name in ("gating_einsum", "linear"):
        shape = target_block.mlp[name].value.shape
        a_shape = (*shape[:-1], rank)
        b_shape = (*shape[:-2], rank, shape[-1])
        base = _layers(shape, 3.0, 4.0)
        a = _layers(a_shape, 0.2, 0.3)
        b = _layers(b_shape, 0.4, 0.5)
        mlp[name] = base
        mlp[f"{name}_lora_a"] = a
        mlp[f"{name}_lora_b"] = b
        expected_mlp[name] = base[1] + np.matmul(a[1], b[1])

    teacher = {
        "temporal_force_encoder": teacher_tcn,
        "PaliGemma": {
            "llm": {
                "layers": {
                    "attn": attention,
                    "mlp_1": mlp,
                    "pre_attention_norm_1": {"scale": _layers((config.width,), 5.0, 6.0)},
                    "pre_ffw_norm_1": {"scale": _layers((config.width,), 7.0, 8.0)},
                }
            }
        },
    }

    report = slow_fast_teacher_init.initialize_fast_from_forcevla_teacher(model, copy.deepcopy(teacher), layer_index=1)

    np.testing.assert_array_equal(fast.force_encoder.stem.kernel.value, teacher_tcn["stem"]["kernel"])
    np.testing.assert_array_equal(source_block.conv2.kernel.value, teacher_tcn["blocks"]["0"]["conv2"]["kernel"])
    for name, expected in expected_attention.items():
        np.testing.assert_allclose(target_block.attn[name]["w"].value, expected, rtol=1e-6)
    for name, expected in expected_mlp.items():
        np.testing.assert_allclose(target_block.mlp[name].value, expected, rtol=1e-6)
    np.testing.assert_array_equal(target_block.pre_attention_norm["scale"].value, 6.0)
    np.testing.assert_array_equal(target_block.pre_ffw_norm["scale"].value, 8.0)

    # Fast-only conditions and safe zero output heads are intentionally untouched.
    np.testing.assert_array_equal(fast.state_proj.kernel.value, untouched_state_kernel)
    np.testing.assert_array_equal(fast.residual_head.kernel.value, 0.0)
    assert report.teacher_layer == 1
    assert report.tcn_arrays == 10
    assert report.gemma_arrays == 7
    assert report.merged_lora_arrays == 5
    assert report.action_expert_lora_scale == 16.0


def test_teacher_init_rejects_non_gemma_student():
    config = slow_fast.FastResidualConfig(
        intent_dim=16,
        width=16,
        mlp_dim=32,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        force_encoder=force_encoder.ForceEncoderConfig(
            type="tcn", hidden_dims=(16,), dilations=(1,), sampling_rate_hz=100, window_ms=20
        ),
    )
    model = slow_fast.FastStudentWithIntentProjector(config, slow_context_dim=16, rngs=nnx.Rngs(0))
    with pytest.raises(ValueError, match="flash_gemma"):
        slow_fast_teacher_init.initialize_fast_from_forcevla_teacher(model, {})
