import numpy as np

from openpi import transforms
from openpi.policies import rotation_6d as rot
from openpi.serving import slow_fast_runtime

from . import evaluate_fast_residual


def _stats(rng, width):
    return transforms.NormStats(mean=rng.normal(size=width), std=rng.uniform(0.5, 2.0, size=width))


def test_offline_rebase_matches_the_serving_absolute_command():
    """The offline report and the robot must agree on what the commanded pose is."""
    rng = np.random.default_rng(0)
    action_stats = _stats(rng, rot.ROBOT_DIMS)
    state_stats = _stats(rng, rot.ROBOT_DIMS)
    normalized_delta = rng.normal(size=(4, rot.POSE_DIMS)).astype(np.float32)
    normalized_state = rng.normal(size=(4, rot.ROBOT_DIMS)).astype(np.float32)

    offline = evaluate_fast_residual.to_absolute_pose(normalized_delta, normalized_state, action_stats, state_stats)

    unnormalize = transforms.Unnormalize({"actions": action_stats})
    for row in range(4):
        physical_state = np.asarray(
            transforms.Unnormalize({"state": state_stats})({"state": normalized_state[row]})["state"],
            dtype=np.float32,
        )
        padded = np.concatenate([normalized_delta[row], np.zeros(rot.ROBOT_DIMS - rot.POSE_DIMS, dtype=np.float32)])
        served = slow_fast_runtime.to_absolute_command(
            padded[None],
            slow_fast_runtime.SlowPacket(
                observation_timestamp=0.0,
                ready_timestamp=0.0,
                reference_start_timestamp=0.0,
                intent_tokens=np.zeros((1, 4), dtype=np.float32),
                reference_actions=np.zeros((1, rot.ROBOT_DIMS), dtype=np.float32),
                action_period_s=0.1,
                version=0,
                state_at_observation=physical_state,
                intent_mask=np.ones(1, dtype=bool),
            ),
            lambda action: unnormalize({"actions": action})["actions"],
            delta_dims=rot.POSE_DIMS,
        )
        np.testing.assert_allclose(offline[row], served[0, : rot.POSE_DIMS], rtol=1e-4, atol=1e-5)


def test_rebased_rotation_is_a_usable_rotation_but_the_raw_delta_is_not():
    """Feeding a delta to sixd_to_matrix silently yields an unrelated rotation."""
    rng = np.random.default_rng(1)
    action_stats = _stats(rng, rot.ROBOT_DIMS)
    state_stats = transforms.NormStats(mean=np.zeros(rot.ROBOT_DIMS), std=np.ones(rot.ROBOT_DIMS))
    # A state whose rotation columns are an honest 6D encoding of identity.
    normalized_state = np.zeros((3, rot.ROBOT_DIMS), dtype=np.float32)
    normalized_state[:, rot.XYZ_DIMS : rot.POSE_DIMS] = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    tiny_delta = (rng.normal(size=(3, rot.POSE_DIMS)) * 1e-3).astype(np.float32)

    absolute = evaluate_fast_residual.to_absolute_pose(tiny_delta, normalized_state, action_stats, state_stats)
    columns = np.linalg.norm(absolute[:, rot.XYZ_DIMS : rot.XYZ_DIMS + 3], axis=-1)
    assert np.all(columns > 0.5), "A rebased pose must carry the state's rotation, not just the correction"

    raw_columns = np.linalg.norm(tiny_delta[:, rot.XYZ_DIMS : rot.XYZ_DIMS + 3], axis=-1)
    assert np.all(raw_columns < 0.1), "The delta alone has no rotation magnitude to normalize"
    # Gram-Schmidt happily returns a valid rotation matrix for that near-zero
    # vector, which is exactly why the old code produced plausible-looking numbers.
    assert np.allclose(np.linalg.det(rot.sixd_to_matrix(tiny_delta[:, rot.XYZ_DIMS : rot.POSE_DIMS])), 1.0, atol=1e-4)


def test_physical_pose_report_never_mixes_metres_with_6d_coordinates():
    target = np.zeros((1, rot.POSE_DIMS), dtype=np.float32)
    target[:, rot.XYZ_DIMS :] = rot.rpy_to_6d(np.zeros((1, 3), dtype=np.float32))
    prediction = target.copy()
    prediction[:, 0] = 0.1
    prediction[:, rot.XYZ_DIMS :] = rot.rpy_to_6d(np.array([[0.0, 0.0, 0.2]], dtype=np.float32))

    summary = evaluate_fast_residual.physical_pose_error_summary(prediction, target)

    assert "mse" not in summary
    np.testing.assert_allclose(summary["translation_rmse_m"], 0.1, atol=1e-6)
    np.testing.assert_allclose(summary["rotation_geodesic_rmse_rad"], 0.2, atol=1e-6)
