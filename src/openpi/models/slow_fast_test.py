import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import force_encoder
from openpi.models import slow_fast


def test_selected_fast_defaults_use_7d_state_and_100hz_force():
    config = slow_fast.FastResidualConfig()

    assert config.state_dim == 7
    assert config.reference_dim == 7
    assert config.pose_dims == 6
    assert config.force_encoder.sampling_rate_hz == 100
    assert config.force_encoder.window_ms == 100
    assert config.force_encoder.max_history_samples == 10


def _config(*, predict_gate: bool = False):
    return slow_fast.FastResidualConfig(
        reference_dim=7,
        state_dim=8,
        pose_dims=6,
        intent_dim=16,
        width=32,
        mlp_dim=64,
        num_heads=2,
        num_kv_heads=1,
        head_dim=16,
        predict_gate=predict_gate,
        force_encoder=force_encoder.ForceEncoderConfig(
            type="tcn",
            hidden_dims=(32, 32),
            dilations=(1, 2),
            dropout_rate=0.0,
            sampling_rate_hz=100,
            window_ms=40,
        ),
    )


def test_fast_residual_student_shapes_and_zero_safe_initialization():
    model = slow_fast.FastForceResidualStudent(_config(), rngs=nnx.Rngs(0))
    residual, gate = model(
        jnp.ones((2, 4, 6)),
        jnp.ones((2, 4), dtype=jnp.bool_),
        jnp.ones((2, 2, 16)),
        jnp.ones((2, 8)),
        jnp.ones((2, 7)),
        jnp.zeros((2, 2)),
    )
    assert residual.shape == (2, 6)
    assert gate.shape == (2,)
    np.testing.assert_array_equal(residual, 0)
    np.testing.assert_array_equal(gate, 1)


def test_fast_residual_student_optional_gate_starts_at_half():
    model = slow_fast.FastForceResidualStudent(_config(predict_gate=True), rngs=nnx.Rngs(0))
    residual, gate = model(
        jnp.ones((1, 4, 6)),
        jnp.ones((1, 4), dtype=jnp.bool_),
        jnp.ones((1, 1, 16)),
        jnp.ones((1, 8)),
        jnp.ones((1, 7)),
        jnp.zeros((1, 2)),
    )
    np.testing.assert_array_equal(residual, 0)
    np.testing.assert_allclose(gate, 0.5)


def test_intent_projector_masks_padding_and_handles_all_padding():
    model = slow_fast.IntentProjector(4, 8, num_tokens=2, rngs=nnx.Rngs(1))
    hidden = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
    mask = jnp.array([[True, True, False], [False, False, False]])
    projected = model(hidden, mask)
    assert projected.shape == (2, 2, 8)
    assert np.isfinite(projected).all()
    np.testing.assert_array_equal(projected[1], 0)


def test_slow_adapter_returns_nominal_chunk_and_compressed_context():
    adapter = slow_fast.SlowNominalStudentAdapter(context_dim=16, intent_dim=8, num_intent_tokens=2, rngs=nnx.Rngs(2))
    actions = jnp.ones((2, 4, 7))
    output_actions, intent = adapter(
        actions,
        jnp.ones((2, 5, 16)),
        jnp.ones((2, 5), dtype=jnp.bool_),
    )
    np.testing.assert_array_equal(output_actions, actions)
    assert intent.shape == (2, 2, 8)


def test_fast_config_rejects_width_mismatch():
    with pytest.raises(ValueError, match="TCN hidden width"):
        slow_fast.FastResidualConfig(
            width=64,
            force_encoder=force_encoder.ForceEncoderConfig(type="tcn", hidden_dims=(32,), dilations=(1,)),
        )


def test_fast_student_requires_one_current_reference_and_time_features():
    model = slow_fast.FastForceResidualStudent(_config(), rngs=nnx.Rngs(3))
    with pytest.raises(ValueError, match="current reference action"):
        model(
            jnp.ones((1, 4, 6)),
            jnp.ones((1, 4), dtype=jnp.bool_),
            jnp.ones((1, 2, 16)),
            jnp.ones((1, 8)),
            jnp.ones((1, 4, 7)),
            jnp.zeros((1, 2)),
        )


def test_fast_student_single_step_loss_has_output_head_gradients():
    model = slow_fast.FastForceResidualStudent(_config(), rngs=nnx.Rngs(4))

    def loss_fn(candidate):
        residual, _ = candidate(
            jnp.ones((2, 4, 6)),
            jnp.ones((2, 4), dtype=jnp.bool_),
            jnp.ones((2, 2, 16)),
            jnp.ones((2, 8)),
            jnp.ones((2, 7)),
            jnp.zeros((2, 2)),
        )
        return jnp.mean(jnp.square(residual - 1))

    loss, gradients = nnx.value_and_grad(loss_fn)(model)
    assert np.isfinite(loss)
    assert jax.numpy.any(gradients["residual_head"]["kernel"].value != 0)


def test_stage5_head_projects_slow_context_and_predicts_one_step():
    model = slow_fast.FastStudentWithIntentProjector(
        _config(), slow_context_dim=24, num_intent_tokens=2, rngs=nnx.Rngs(5)
    )
    residual, gate, intent = model(
        jnp.ones((2, 5, 24)),
        jnp.ones((2, 5), dtype=jnp.bool_),
        jnp.ones((2, 4, 6)),
        jnp.ones((2, 4), dtype=jnp.bool_),
        jnp.ones((2, 8)),
        jnp.ones((2, 7)),
        jnp.zeros((2, 2)),
    )
    assert residual.shape == (2, 6)
    assert gate.shape == (2,)
    assert intent.shape == (2, 2, 16)
