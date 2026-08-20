from flax import nnx
import jax
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import pi0_fast
from openpi.shared import download
from openpi.shared import nnx_utils


def test_pi0_model():
    key = jax.random.key(0)
    config = pi0.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_lora_model():
    key = jax.random.key(0)
    config = pi0.Pi0Config(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_fast_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)


def test_pi0_fast_lora_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)

    lora_filter = nnx_utils.PathRegex(".*lora.*")
    model_state = nnx.state(model)

    lora_state_elems = list(model_state.filter(lora_filter))
    assert len(lora_state_elems) > 0


def test_row_keyed_noise_is_independent_of_batching_and_order():
    rows = np.arange(37)
    expected = np.asarray(_model.row_keyed_noise(0, rows, action_horizon=4, action_dim=7))

    for batch_size in (1, 5, 16):
        batched = np.concatenate(
            [
                np.asarray(_model.row_keyed_noise(0, rows[i : i + batch_size], action_horizon=4, action_dim=7))
                for i in range(0, len(rows), batch_size)
            ]
        )
        np.testing.assert_array_equal(batched, expected)

    reversed_rows = np.asarray(_model.row_keyed_noise(0, rows[::-1], action_horizon=4, action_dim=7))
    np.testing.assert_array_equal(reversed_rows, expected[::-1])

    other_seed = np.asarray(_model.row_keyed_noise(1, rows, action_horizon=4, action_dim=7))
    assert not np.allclose(other_seed, expected)
    assert not np.allclose(expected[0], expected[1])


def test_resolve_sample_noise_prefers_supplied_noise():
    supplied = np.asarray(_model.row_keyed_noise(0, np.arange(3), action_horizon=4, action_dim=7))
    np.testing.assert_array_equal(
        np.asarray(_model.resolve_sample_noise(jax.random.key(9), supplied, shape=(3, 4, 7))), supplied
    )
    drawn = _model.resolve_sample_noise(jax.random.key(0), None, shape=(3, 4, 7))
    assert drawn.shape == (3, 4, 7)
    with pytest.raises(ValueError, match="Expected noise of shape"):
        _model.resolve_sample_noise(jax.random.key(0), supplied, shape=(4, 4, 7))


@pytest.mark.manual
def test_model_restore():
    key = jax.random.key(0)
    config = pi0.Pi0Config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
