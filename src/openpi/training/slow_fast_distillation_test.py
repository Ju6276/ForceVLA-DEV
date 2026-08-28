import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.training import slow_fast_distillation


class _FakePairedTeacher:
    def sample_paired_actions_and_context(self, rng, observation, *, num_steps):
        del observation, num_steps
        shared = jax.random.normal(rng, (1, 2, 10))
        context = jnp.ones((1, 3, 8))
        mask = jnp.ones((1, 3), dtype=jnp.bool_)
        return shared + 1, shared, context, mask


def test_paired_targets_define_pose_residual_and_preserve_nominal():
    null = jnp.zeros((2, 3, 10))
    full = null.at[..., :9].set(2)
    targets = slow_fast_distillation.make_paired_teacher_targets(full, null)
    np.testing.assert_array_equal(targets.nominal_actions, null)
    np.testing.assert_array_equal(targets.residual_pose, 2)


def test_fast_loss_reconstructs_full_and_ignores_gripper_residual():
    null = jnp.zeros((1, 2, 10)).at[..., 9].set(0.75)
    full = null.at[..., :9].set(0.25)
    targets = slow_fast_distillation.make_paired_teacher_targets(full, null)
    current = slow_fast_distillation.select_chunk_targets(targets, chunk_steps=2)
    total, metrics = slow_fast_distillation.fast_residual_loss(
        current.residual_pose,
        null,
        current,
        slow_fast_distillation.FastDistillationLossConfig(),
    )
    np.testing.assert_allclose(total, 0)
    np.testing.assert_allclose(metrics["reconstruction_loss"], 0)


def test_fast_loss_weights_the_executed_step_above_the_lookahead():
    targets = slow_fast_distillation.select_chunk_targets(
        slow_fast_distillation.make_paired_teacher_targets(jnp.zeros((1, 2, 10)), jnp.zeros((1, 2, 10))),
        chunk_steps=2,
    )
    config = slow_fast_distillation.FastDistillationLossConfig(reconstruction_weight=0.0, step_decay=0.5)
    reference = jnp.zeros((1, 2, 10))
    step0_wrong = jnp.zeros((1, 2, 9)).at[:, 0].set(1.0)
    step1_wrong = jnp.zeros((1, 2, 9)).at[:, 1].set(1.0)
    first, _ = slow_fast_distillation.fast_residual_loss(step0_wrong, reference, targets, config)
    second, _ = slow_fast_distillation.fast_residual_loss(step1_wrong, reference, targets, config)
    assert float(first) == pytest.approx(2.0 * float(second))
    # Normalized weights keep the loss scale independent of the chunk length.
    np.testing.assert_allclose(np.sum(np.asarray(config.step_weights(5))), 1.0, atol=1e-6)


def test_two_heads_summing_to_the_stale_gap_reconstruct_the_full_action():
    """A stale reference is recovered exactly by the two targets together.

    `A_full - A_ref` splits into the force deviation and the drift the reference
    accumulated, so a student that nails both targets lands on the Teacher even
    though neither target on its own describes what it must add to the reference.
    """
    reference = jnp.full((1, 2, 10), 0.3)
    null = jnp.zeros((1, 2, 10)).at[..., 9].set(0.75)
    full = null.at[..., :9].set(0.25)
    chunk = slow_fast_distillation.select_chunk_targets(
        slow_fast_distillation.make_paired_teacher_targets(full, null), chunk_steps=2
    )
    staleness_target = null[..., :9] - reference[..., :9]
    targets = dataclasses.replace(chunk, staleness_pose=staleness_target)

    total, metrics = slow_fast_distillation.fast_residual_loss(
        targets.residual_pose,
        reference,
        targets,
        slow_fast_distillation.FastDistillationLossConfig(),
        predicted_staleness=staleness_target,
    )
    np.testing.assert_allclose(total, 0, atol=1e-12)
    np.testing.assert_allclose(metrics["reconstruction_loss"], 0, atol=1e-12)
    np.testing.assert_allclose(metrics["staleness_loss"], 0, atol=1e-12)

    # Dropping the staleness correction leaves exactly the reference drift behind.
    _, single_head = slow_fast_distillation.fast_residual_loss(
        targets.residual_pose,
        reference,
        chunk,
        slow_fast_distillation.FastDistillationLossConfig(),
    )
    np.testing.assert_allclose(
        single_head["reconstruction_loss"], float(jnp.mean(jnp.square(staleness_target))), rtol=1e-6
    )
    assert "staleness_loss" not in single_head


def test_staleness_prediction_and_target_must_be_supplied_together():
    chunk = slow_fast_distillation.select_chunk_targets(
        slow_fast_distillation.make_paired_teacher_targets(jnp.zeros((1, 2, 10)), jnp.zeros((1, 2, 10))),
        chunk_steps=2,
    )
    with pytest.raises(ValueError, match="staleness target"):
        slow_fast_distillation.fast_residual_loss(
            chunk.residual_pose,
            jnp.zeros((1, 2, 10)),
            chunk,
            slow_fast_distillation.FastDistillationLossConfig(),
            predicted_staleness=jnp.zeros((1, 2, 9)),
        )


def test_single_head_total_target_is_exact_sum_of_two_head_targets():
    force = jnp.full((2, 3, 9), 0.25)
    stale = jnp.full((2, 3, 9), -0.10)
    targets = slow_fast_distillation.FastChunkTargets(
        full_action=jnp.zeros((2, 3, 9)),
        nominal_action=jnp.zeros((2, 3, 10)),
        residual_pose=force,
        staleness_pose=stale,
    )
    np.testing.assert_allclose(slow_fast_distillation.single_head_total_target(targets), 0.15, atol=1e-7)


def test_single_head_total_target_requires_staleness_supervision():
    targets = slow_fast_distillation.FastChunkTargets(
        full_action=jnp.zeros((1, 2, 9)),
        nominal_action=jnp.zeros((1, 2, 10)),
        residual_pose=jnp.zeros((1, 2, 9)),
    )
    with pytest.raises(ValueError, match="requires a staleness target"):
        slow_fast_distillation.single_head_total_target(targets)


def test_paired_targets_require_matching_shapes():
    with pytest.raises(ValueError, match="matched"):
        slow_fast_distillation.make_paired_teacher_targets(jnp.zeros((1, 2, 7)), jnp.zeros((1, 3, 7)))


def test_chunk_targets_take_the_leading_teacher_steps():
    null = jnp.zeros((1, 3, 10))
    full = null.at[:, 0, :9].set(1).at[:, 1, :9].set(9)
    current = slow_fast_distillation.select_chunk_targets(
        slow_fast_distillation.make_paired_teacher_targets(full, null), chunk_steps=2
    )
    assert current.full_action.shape == (1, 2, 10)
    assert current.residual_pose.shape == (1, 2, 9)
    np.testing.assert_array_equal(current.residual_pose[:, 0], 1)
    np.testing.assert_array_equal(current.residual_pose[:, 1], 9)


def test_chunk_targets_cannot_exceed_the_teacher_horizon():
    targets = slow_fast_distillation.make_paired_teacher_targets(jnp.zeros((1, 3, 10)), jnp.zeros((1, 3, 10)))
    with pytest.raises(ValueError, match="chunk_steps must be in"):
        slow_fast_distillation.select_chunk_targets(targets, chunk_steps=4)


def test_paired_sampler_returns_one_shared_context_and_force_only_difference():
    targets, context, mask = slow_fast_distillation.sample_paired_teacher_actions(
        _FakePairedTeacher(), jax.random.key(0), object(), num_steps=2
    )
    np.testing.assert_allclose(targets.residual_pose, 1, atol=2e-7)
    assert context.shape == (1, 3, 8)
    assert mask.shape == (1, 3)
