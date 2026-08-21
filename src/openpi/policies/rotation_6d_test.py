import numpy as np

from openpi.policies import rotation_6d as rot


def test_plus_pi_and_minus_pi_map_to_the_same_6d():
    plus = rot.rpy_to_6d(np.array([[np.pi, 0.0, 0.0]]))
    minus = rot.rpy_to_6d(np.array([[-np.pi, 0.0, 0.0]]))
    np.testing.assert_allclose(plus, minus, atol=1e-6)


def test_roundtrip_preserves_the_rotation():
    rpy = np.array([[3.13, 0.02, -0.04], [-3.13, -0.01, 0.03], [0.4, 1.1, -0.7]])
    recovered = rot.sixd_to_matrix(rot.rpy_to_6d(rpy))
    np.testing.assert_allclose(rot.geodesic_angle(rot.rpy_to_matrix(rpy), recovered), 0.0, atol=1e-6)


def test_restored_rpy_is_the_same_pose():
    rpy = np.array([[np.pi, 0.02, -0.04]])
    restored = rot.sixd_to_rpy(rot.rpy_to_6d(rpy))
    np.testing.assert_allclose(
        rot.geodesic_angle(rot.rpy_to_matrix(rpy), rot.rpy_to_matrix(restored)), 0.0, atol=1e-6
    )


def test_converted_state_layout_keeps_xyz_gripper_and_force():
    state = np.arange(13, dtype=np.float32)[None, :]
    converted = rot.convert_state(state)
    assert converted.shape == (1, 16)
    np.testing.assert_allclose(converted[0, :3], state[0, :3])
    np.testing.assert_allclose(converted[0, 9], state[0, 6])
    np.testing.assert_allclose(converted[0, 10:16], state[0, 7:13])


def test_fake_2pi_delta_disappears_in_6d():
    state = np.array([[0, 0, 0, np.pi - 0.01, 0, 0, 0.1, *([0.0] * 6)]], dtype=np.float32)
    action = np.array([[0, 0, 0, -np.pi + 0.01, 0, 0, 0.1]], dtype=np.float32)
    naive = float(action[0, 3] - state[0, 3])
    sixd = float(np.linalg.norm(rot.convert_action(action)[0, 3:9] - rot.convert_state(state)[0, 3:9]))
    assert abs(naive) > 6.0
    assert sixd < 0.05


def test_wrapping_euler_trajectory_is_continuous_in_6d():
    rolls = np.concatenate(
        [np.linspace(np.pi - 0.05, np.pi, 6, endpoint=False), np.linspace(-np.pi, -np.pi + 0.05, 6)]
    )
    rpy = np.stack([rolls, np.zeros_like(rolls), np.zeros_like(rolls)], axis=-1)
    sixd = rot.rpy_to_6d(rpy)
    step = np.linalg.norm(np.diff(sixd, axis=0), axis=-1)
    geodesic = rot.geodesic_angle(rot.rpy_to_matrix(rpy[:-1]), rot.rpy_to_matrix(rpy[1:]))
    naive = np.abs(np.diff(rolls))
    assert float(naive.max()) > 6.0
    assert float(step.max()) < 0.05
    assert float(np.degrees(geodesic.max())) < 1.0
    recovered = rot.actions_6d_to_rpy(
        np.concatenate([np.zeros((len(sixd), 3)), sixd, np.zeros((len(sixd), 1))], axis=-1)
    )
    np.testing.assert_allclose(
        rot.geodesic_angle(rot.rpy_to_matrix(rpy), rot.rpy_to_matrix(recovered[:, 3:6])),
        0.0,
        atol=1e-5,
    )
