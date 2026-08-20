import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.training import slow_fast_distillation


class _FakePairedTeacher:
    def sample_paired_actions_and_context(self, rng, observation, *, num_steps):
        del observation, num_steps
        shared = jax.random.normal(rng, (1, 2, 7))
        context = jnp.ones((1, 3, 8))
        mask = jnp.ones((1, 3), dtype=jnp.bool_)
        return shared + 1, shared, context, mask


def test_paired_targets_define_pose_residual_and_preserve_nominal():
    null = jnp.zeros((2, 3, 7))
    full = null.at[..., :6].set(2)
    targets = slow_fast_distillation.make_paired_teacher_targets(full, null)
    np.testing.assert_array_equal(targets.nominal_actions, null)
    np.testing.assert_array_equal(targets.residual_pose, 2)


def test_fast_loss_reconstructs_full_and_ignores_gripper_residual():
    null = jnp.zeros((1, 2, 7)).at[..., 6].set(0.75)
    full = null.at[..., :6].set(0.25)
    targets = slow_fast_distillation.make_paired_teacher_targets(full, null)
    current = slow_fast_distillation.select_single_step_targets(targets)
    total, metrics = slow_fast_distillation.fast_residual_loss(
        current.residual_pose,
        null[:, 0],
        current,
        slow_fast_distillation.FastDistillationLossConfig(),
    )
    np.testing.assert_allclose(total, 0)
    np.testing.assert_allclose(metrics["reconstruction_loss"], 0)

def test_paired_targets_require_matching_shapes():
    with pytest.raises(ValueError, match="matched"):
        slow_fast_distillation.make_paired_teacher_targets(jnp.zeros((1, 2, 7)), jnp.zeros((1, 3, 7)))


def test_single_step_target_uses_only_timestamp_aligned_chunk_entry():
    null = jnp.zeros((1, 3, 7))
    full = null.at[:, 0, :6].set(1).at[:, 1, :6].set(9)
    current = slow_fast_distillation.select_single_step_targets(
        slow_fast_distillation.make_paired_teacher_targets(full, null)
    )
    assert current.full_action.shape == (1, 7)
    assert current.residual_pose.shape == (1, 6)
    np.testing.assert_array_equal(current.residual_pose, 1)


def test_paired_sampler_returns_one_shared_context_and_force_only_difference():
    targets, context, mask = slow_fast_distillation.sample_paired_teacher_actions(
        _FakePairedTeacher(), jax.random.key(0), object(), num_steps=2
    )
    np.testing.assert_allclose(targets.residual_pose, 1, atol=2e-7)
    assert context.shape == (1, 3, 8)
    assert mask.shape == (1, 3)
