from types import SimpleNamespace

from flax import nnx
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0_force
from openpi.training import config as training_config
from openpi.training import weight_loaders


def test_null_condition_requires_null_token():
    with pytest.raises(ValueError, match="enable_null_force_token"):
        pi0_force.Pi0_GuidanceConfig(force_condition="null")


def test_null_force_token_is_broadcast_across_batch():
    model = SimpleNamespace(
        force_condition="null",
        null_force_token=nnx.Param(jnp.arange(4, dtype=jnp.float32)),
    )
    obs = SimpleNamespace(state=jnp.zeros((3, 13), dtype=jnp.float32))

    encoded = pi0_force.Pi0_Guidance.encode_force(model, obs)

    assert encoded.shape == (3, 4)
    np.testing.assert_array_equal(encoded, np.tile(np.arange(4, dtype=np.float32), (3, 1)))


def test_stage2a_filter_trains_only_null_force_token():
    class ToyModel(nnx.Module):
        def __init__(self):
            self.null_force_token = nnx.Param(jnp.zeros(4))
            self.existing_teacher_weight = nnx.Param(jnp.ones(4))
            self.dropout = nnx.Dropout(0.1, rngs=nnx.Rngs(0))

    state = nnx.state(ToyModel())
    config = training_config.get_config("forcevla_usb_temporal_stage2a_null")
    trainable_state = state.filter(config.trainable_filter)
    frozen_state = state.filter(config.freeze_filter)

    assert set(trainable_state) == {"null_force_token"}
    assert set(frozen_state) == {"existing_teacher_weight"}


def test_stage1_checkpoint_can_initialize_new_null_token():
    loaded = {
        "teacher": {"weight": np.ones((2,), dtype=np.float32)},
        # Orbax restores list indices as strings, while the initialized NNX
        # parameter tree uses integer keys.
        "temporal_force_encoder": {"blocks": {"0": {"kernel": np.ones((2,), dtype=np.float32)}}},
    }
    initialized = {
        "teacher": {"weight": np.zeros((2,), dtype=np.float32)},
        "temporal_force_encoder": {"blocks": {0: {"kernel": np.zeros((2,), dtype=np.float32)}}},
        "null_force_token": np.zeros((4,), dtype=np.float32),
    }

    merged = weight_loaders._merge_params(
        loaded,
        initialized,
        missing_regex=".*null_force_token.*|.*temporal_force_encoder.*",
    )

    np.testing.assert_array_equal(merged["teacher"]["weight"], loaded["teacher"]["weight"])
    np.testing.assert_array_equal(merged["null_force_token"], initialized["null_force_token"])
    np.testing.assert_array_equal(
        merged["temporal_force_encoder"]["blocks"][0]["kernel"],
        loaded["temporal_force_encoder"]["blocks"]["0"]["kernel"],
    )
