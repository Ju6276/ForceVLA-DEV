from flax import nnx
import jax.numpy as jnp
import numpy as np

from openpi.models import force_encoder


def test_history_sample_count_uses_physical_time():
    assert force_encoder.ForceEncoderConfig(sampling_rate_hz=200, window_ms=100).max_history_samples == 20
    assert force_encoder.ForceEncoderConfig(sampling_rate_hz=300, window_ms=25).max_history_samples == 8
    assert force_encoder.ForceEncoderConfig(sampling_rate_hz=30, window_ms=100).max_history_samples == 3


def test_masked_pooling_ignores_padding():
    history = jnp.asarray([[[99.0] * 6, [1.0] * 6, [3.0] * 6]])
    mask = jnp.asarray([[False, True, True]])
    np.testing.assert_allclose(force_encoder.pool_force_history(history, mask, "avg_pool"), 2.0)
    np.testing.assert_allclose(force_encoder.pool_force_history(history, mask, "max_pool"), 3.0)


def test_tcn_is_causal_and_preserves_sequence_length():
    config = force_encoder.ForceEncoderConfig(
        type="tcn", hidden_dims=(8, 8), dilations=(1, 2), aggregation="last", dropout_rate=0.0
    )
    encoder = force_encoder.TemporalTCNForceEncoder(config, output_dim=16, rngs=nnx.Rngs(0))
    first = jnp.zeros((1, 12, 6))
    changed_future = first.at[:, 8:].set(100)

    # Inspect a block directly: changing samples after index 7 cannot change its
    # hidden representation at index 7.
    stem_a = encoder.stem(first)
    stem_b = encoder.stem(changed_future)
    out_a = encoder.blocks[0](stem_a)
    out_b = encoder.blocks[0](stem_b)
    assert out_a.shape == stem_a.shape
    np.testing.assert_allclose(out_a[:, :8], out_b[:, :8], rtol=1e-5, atol=1e-5)
