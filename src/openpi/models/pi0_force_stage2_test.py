from types import SimpleNamespace

from flax import nnx
import jax
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


def test_native_instantaneous_uses_only_the_current_causal_slot_and_original_projection():
    model = SimpleNamespace(
        force_condition="full",
        force_encoder_config=SimpleNamespace(type="native_instantaneous"),
        force_in_proj=lambda force: force * 2,
        temporal_force_encoder=None,
    )
    history = jnp.asarray(
        [
            [[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
            [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
        ],
        dtype=jnp.float32,
    )
    observation = SimpleNamespace(
        state=jnp.zeros((2, 32), dtype=jnp.float32),
        force_history=history,
        # The second row has valid history, but no fresh sample in the current
        # timestamp slot. It must not silently fall back to an older force.
        force_history_mask=jnp.asarray([[True, True, True], [True, True, False]]),
    )

    encoded = pi0_force.Pi0_Guidance.encode_force(model, observation)

    np.testing.assert_array_equal(encoded[0], np.full(6, 6, dtype=np.float32))
    np.testing.assert_array_equal(encoded[1], np.zeros(6, dtype=np.float32))


def test_stage1_checkpoint_can_initialize_selected_stage2_parameters():
    loaded = {
        "teacher": {"weight": np.ones((2,), dtype=np.float32)},
        # Orbax restores list indices as strings, while NNX uses integers.
        "temporal_force_encoder": {"blocks": {"0": {"kernel": np.ones((2,), dtype=np.float32)}}},
    }
    initialized = {
        "teacher": {"weight": np.zeros((2,), dtype=np.float32)},
        "temporal_force_encoder": {"blocks": {0: {"kernel": np.zeros((2,), dtype=np.float32)}}},
        "null_force_token": np.zeros((4,), dtype=np.float32),
        "nominal_adapter_in": {"kernel": np.zeros((2, 2), dtype=np.float32)},
        "nominal_adapter_out": {"kernel": np.zeros((2, 2), dtype=np.float32)},
    }

    merged = weight_loaders._merge_params(  # noqa: SLF001
        loaded,
        initialized,
        missing_regex=".*null_force_token.*|.*nominal_adapter_(in|out).*|.*temporal_force_encoder.*",
    )

    np.testing.assert_array_equal(merged["teacher"]["weight"], loaded["teacher"]["weight"])
    np.testing.assert_array_equal(merged["null_force_token"], initialized["null_force_token"])
    np.testing.assert_array_equal(
        merged["temporal_force_encoder"]["blocks"][0]["kernel"],
        loaded["temporal_force_encoder"]["blocks"]["0"]["kernel"],
    )


def test_checkpoint_merge_preserves_bias_free_none_leaf():
    loaded = {"adapter": {"kernel": np.ones((2, 2), dtype=np.float32), "bias": None}}
    initialized = {"adapter": {"kernel": np.zeros((2, 2), dtype=np.float32), "bias": None}}

    merged = weight_loaders._merge_params(loaded, initialized, missing_regex=r"a^")  # noqa: SLF001

    assert merged["adapter"]["bias"] is None
    np.testing.assert_array_equal(merged["adapter"]["kernel"], loaded["adapter"]["kernel"])


def test_selected_button_stage2_uses_original_bc_and_only_trains_null_parameters():
    class ToyModel(nnx.Module):
        def __init__(self):
            self.null_force_token = nnx.Param(jnp.zeros(4))
            self.nominal_adapter_in = nnx.Linear(4, 2, rngs=nnx.Rngs(0))
            self.nominal_adapter_out = nnx.Linear(2, 4, rngs=nnx.Rngs(1))
            self.existing_teacher_weight = nnx.Param(jnp.ones(4))

    config = training_config.get_config("forcevla_button_temporal_stage2_null_bc")
    trainable_state = nnx.state(ToyModel()).filter(config.trainable_filter)

    assert config.model.force_condition == "null"
    assert config.model.enable_null_force_token
    assert config.model.enable_nominal_adapter
    assert config.model.nominal_adapter_rank == 32
    assert config.lr_schedule.warmup_steps == 500
    assert config.lr_schedule.decay_steps == config.num_train_steps == 10_000
    assert set(trainable_state) == {"nominal_adapter_in", "nominal_adapter_out", "null_force_token"}


def test_null_adapter_is_bypassed_for_full_force():
    class ToyModel:
        action_out_proj = staticmethod(lambda hidden: hidden)
        nominal_adapter_in = staticmethod(lambda hidden: hidden)
        nominal_adapter_out = staticmethod(lambda hidden: jnp.ones_like(hidden[..., :2]))
        nominal_adapter_pose_dims = 2

    hidden = jnp.zeros((1, 2, 3), dtype=jnp.float32)

    full = pi0_force.Pi0_Guidance._project_action_velocity(ToyModel(), hidden, "full")  # noqa: SLF001
    null = pi0_force.Pi0_Guidance._project_action_velocity(ToyModel(), hidden, "null")  # noqa: SLF001

    np.testing.assert_array_equal(full, np.zeros((1, 2, 3), dtype=np.float32))
    np.testing.assert_array_equal(null[..., :2], np.ones((1, 2, 2), dtype=np.float32))
    np.testing.assert_array_equal(null[..., 2:], np.zeros((1, 2, 1), dtype=np.float32))


def test_nominal_context_sampling_forwards_the_row_keyed_noise():
    """Slow can only replace the Stage-4 distillation if it reproduces A_null exactly.

    The paired extraction and the Slow cache must therefore reach the flow sampler
    with the same noise and the same null conditioning.
    """
    config = pi0_force.Pi0_GuidanceConfig(enable_null_force_token=True, force_condition="null")
    noise = jnp.arange(config.action_horizon * config.action_dim, dtype=jnp.float32).reshape(
        1, config.action_horizon, config.action_dim
    )
    recorded = {}

    class Stub:
        null_force_token = nnx.Param(jnp.zeros((4,), dtype=jnp.float32))

        def _prepare_action_prefix(self, observation):
            return "tokens", "mask", "prefix", "cache"

        def _sample_actions_with_prefix(self, rng, observation, **kwargs):
            recorded.update(kwargs)
            return "actions"

    actions, prefix, mask = pi0_force.Pi0_Guidance.sample_nominal_actions_and_context(
        Stub(), jax.random.key(0), config.fake_obs(batch_size=1), num_steps=4, noise=noise
    )

    assert (actions, prefix, mask) == ("actions", "prefix", "mask")
    assert recorded["force_condition"] == "null"
    np.testing.assert_array_equal(recorded["noise"], noise)


def test_standard_sampling_forwards_caller_supplied_noise():
    config = pi0_force.Pi0_GuidanceConfig()
    noise = jnp.arange(config.action_horizon * config.action_dim, dtype=jnp.float32).reshape(
        1, config.action_horizon, config.action_dim
    )
    recorded = {}

    class Stub:
        force_condition = "full"
        null_force_token = None
        sample_actions_for_force_condition = pi0_force.Pi0_Guidance.sample_actions_for_force_condition

        def _prepare_action_prefix(self, observation):
            return "tokens", "mask", "prefix", "cache"

        def _sample_actions_with_prefix(self, rng, observation, **kwargs):
            recorded.update(kwargs)
            return "actions"

    actions = pi0_force.Pi0_Guidance.sample_actions(
        Stub(), jax.random.key(0), config.fake_obs(batch_size=1), num_steps=4, noise=noise
    )

    assert actions == "actions"
    assert recorded["force_condition"] == "full"
    np.testing.assert_array_equal(recorded["noise"], noise)


def test_button_instantaneous_baseline_matches_temporal_experiment_contract():
    instantaneous = training_config.get_config("forcevla_button_instantaneous")
    temporal = training_config.get_config("forcevla_button_temporal_100hz")
    instantaneous_val = training_config.get_config("forcevla_button_instantaneous_val")
    temporal_val = training_config.get_config("forcevla_button_temporal_100hz_val")

    assert instantaneous.model.force_encoder.type == "instantaneous"
    assert temporal.model.force_encoder.type == "tcn"
    assert instantaneous.data.native_force_sidecar is None
    assert temporal.data.native_force_sidecar is not None
    assert instantaneous.data.base_config.episodes == temporal.data.base_config.episodes
    assert instantaneous_val.data.base_config.episodes == temporal_val.data.base_config.episodes
    assert instantaneous.num_train_steps == temporal.num_train_steps == 40_000
    assert instantaneous.save_interval == temporal.save_interval == 20_000
    assert instantaneous.batch_size == temporal.batch_size == 4
    assert instantaneous.model.action_dim == temporal.model.action_dim
    assert instantaneous.model.action_horizon == temporal.model.action_horizon


def test_button_native_instantaneous_is_a_history_only_control_for_temporal():
    native = training_config.get_config("forcevla_button_native_instantaneous_100hz")
    native_val = training_config.get_config("forcevla_button_native_instantaneous_100hz_val")
    temporal = training_config.get_config("forcevla_button_temporal_100hz")
    temporal_val = training_config.get_config("forcevla_button_temporal_100hz_val")

    assert native.model.force_encoder.type == "native_instantaneous"
    assert native.model.force_encoder.max_history_samples == temporal.model.force_encoder.max_history_samples == 10
    assert native.model.force_encoder.sampling_rate_hz == temporal.model.force_encoder.sampling_rate_hz == 100
    assert native.model.force_encoder.window_ms == temporal.model.force_encoder.window_ms == 100
    assert native.model.force_encoder.max_sample_age_ms == temporal.model.force_encoder.max_sample_age_ms == 12
    assert native.data.native_force_sidecar == temporal.data.native_force_sidecar
    assert native_val.data.native_force_sidecar == temporal_val.data.native_force_sidecar
    assert native.data.base_config.episodes == temporal.data.base_config.episodes
    assert native_val.data.base_config.episodes == temporal_val.data.base_config.episodes
    assert native.data.assets == temporal_val.data.assets
    assert native_val.data.assets == temporal_val.data.assets
    assert native.num_train_steps == temporal.num_train_steps == 40_000
    assert native.save_interval == temporal.save_interval == 20_000
    assert native.keep_period == temporal.keep_period == 20_000
    assert native.batch_size == temporal.batch_size == 4
