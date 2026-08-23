"""The composition chain must reconstruct the Teacher exactly when both heads are oracles.

Nothing here is learned. Feeding the two heads their own ground-truth targets and walking
the real deployment path has to land on the Teacher's full-force action in absolute
coordinates. That identity crosses every conversion the Fast path depends on: the
reference is a delta from the Slow observation's state, the staleness target carries a
base correction between two different states, the force residual is a difference of two
deltas sharing one base, and `to_absolute_command` unnormalizes before rebasing. Each of
those can be individually wrong while every shape check passes, and a learned model would
hide the error inside its own residual.
"""

from __future__ import annotations

import pathlib
import sys
import types

import numpy as np
import pytest

from openpi.serving import slow_fast_runtime

# The point of this test is to exercise the *shipped* target definition rather than a
# copy of it, and the training entry point is a script rather than an installed module.
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from train_fast_residual import _staleness_target  # noqa: E402

POSE_DIMS = 9
ACTION_DIMS = 10
CHUNK_STEPS = 3


def _fixture(seed: int = 0):
    rng = np.random.default_rng(seed)
    action_mean = rng.normal(size=ACTION_DIMS).astype(np.float32)
    action_std = rng.uniform(0.5, 2.0, size=ACTION_DIMS).astype(np.float32)
    state_mean = rng.normal(size=ACTION_DIMS).astype(np.float32)
    state_std = rng.uniform(0.5, 2.0, size=ACTION_DIMS).astype(np.float32)

    # Row 0 is the key row the Slow packet was produced from, row 1 is the row being
    # scored. They must differ, since a zero base gap is exactly the case that cannot
    # tell a correct base correction from a missing one.
    state_normalized = rng.normal(size=(2, ACTION_DIMS)).astype(np.float32)
    arrays = types.SimpleNamespace(
        state=state_normalized,
        null_pose=rng.normal(size=(2, CHUNK_STEPS, POSE_DIMS)).astype(np.float32),
        full_pose=rng.normal(size=(2, CHUNK_STEPS, POSE_DIMS)).astype(np.float32),
    )
    cache = types.SimpleNamespace(
        key_dataset_indices=np.array([0], dtype=np.int64),
        row_key_positions=np.array([0, 0], dtype=np.int64),
        reference_actions=rng.normal(size=(2, CHUNK_STEPS, ACTION_DIMS)).astype(np.float32),
    )
    scale = ((state_std[:POSE_DIMS] + 1e-6) / (action_std[:POSE_DIMS] + 1e-6)).astype(np.float32)
    return arrays, cache, action_mean, action_std, state_mean, state_std, scale


def _physical_state(normalized, state_mean, state_std):
    return normalized * state_std + state_mean


def _packet(reference_chunk, anchor_physical):
    return slow_fast_runtime.SlowPacket(
        observation_timestamp=0.0,
        ready_timestamp=0.1,
        reference_start_timestamp=0.0,
        intent_tokens=np.zeros((1, 4), dtype=np.float32),
        intent_mask=np.ones(1, dtype=bool),
        reference_actions=np.asarray(reference_chunk, dtype=np.float32),
        action_period_s=0.1,
        version=0,
        state_at_observation=np.asarray(anchor_physical, dtype=np.float32),
    )


def _oracle_command(*, include_base_correction: bool):
    arrays, cache, action_mean, action_std, state_mean, state_std, scale = _fixture()
    scored = np.array([1], dtype=np.int64)

    force = arrays.full_pose[scored] - arrays.null_pose[scored]
    if include_base_correction:
        stale = _staleness_target(arrays, cache, scored, pose_dims=POSE_DIMS, state_to_action_scale=scale)
    else:
        # The shortcut the README used to imply: subtract the two deltas and stop.
        stale = arrays.null_pose[scored] - cache.reference_actions[scored, :, :POSE_DIMS]

    reference_chunk = cache.reference_actions[1]
    anchor = _physical_state(arrays.state[0], state_mean, state_std)
    composed = slow_fast_runtime.compose_reference_residual(
        reference_chunk,
        force[0],
        staleness_pose=stale[0],
        pose_dims=POSE_DIMS,
    )
    command = slow_fast_runtime.to_absolute_command(
        composed,
        _packet(reference_chunk, anchor),
        lambda action: action * action_std + action_mean,
        delta_dims=POSE_DIMS,
    )

    current = _physical_state(arrays.state[1], state_mean, state_std)
    teacher_full = arrays.full_pose[1] * action_std[:POSE_DIMS] + action_mean[:POSE_DIMS] + current[:POSE_DIMS]
    reference_gripper_physical = (
        reference_chunk[:, POSE_DIMS:] * action_std[POSE_DIMS:] + action_mean[POSE_DIMS:]
    )
    return command, teacher_full, reference_gripper_physical


def test_oracle_heads_reconstruct_the_teacher_full_action_in_absolute_coordinates():
    command, teacher_full, reference_chunk = _oracle_command(include_base_correction=True)
    np.testing.assert_allclose(command[:, :POSE_DIMS], teacher_full, rtol=0, atol=2e-5)


def test_the_gripper_stays_whatever_slow_commanded():
    """Only the pose dimensions are corrected, so the identity above is pose-only."""
    command, _, reference_gripper_physical = _oracle_command(include_base_correction=True)
    # The gripper column is unnormalized, but never touched by either correction or
    # by the DeltaActions inverse.
    np.testing.assert_allclose(command[:, POSE_DIMS:], reference_gripper_physical, rtol=0, atol=1e-7)


def test_dropping_the_base_correction_breaks_the_reconstruction():
    """This is what makes the test above meaningful rather than tautological.

    `A_ref` is a delta from the key row's state and `A_null` from the current row's, so
    their plain difference is short by the base gap. Without this case a target that
    silently omitted the gap would still pass.
    """
    command, teacher_full, _ = _oracle_command(include_base_correction=False)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(command[:, :POSE_DIMS], teacher_full, rtol=0, atol=2e-5)
