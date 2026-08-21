import numpy as np

from openpi.models import model
from openpi.policies import forcevla_policy
from openpi.policies import rotation_6d as rot


def test_aligned_state_history_preserves_current_state_and_extracts_force():
    states = np.arange(3 * 13, dtype=np.float32).reshape(3, 13)
    transform = forcevla_policy.Forcevla_inputs(
        action_dim=32,
        model_type=model.ModelType.PI0,
        use_force_history=True,
        force_history_from_state=True,
    )
    result = transform(
        {
            "state": states,
            "image": np.zeros((8, 8, 3), dtype=np.uint8),
            "wrist_image": np.zeros((8, 8, 3), dtype=np.uint8),
        }
    )

    converted = rot.convert_state(states[-1])
    np.testing.assert_array_equal(result["state"][: converted.shape[-1]], converted)
    np.testing.assert_array_equal(result["force_history"], states[:, 7:13])
    np.testing.assert_array_equal(result["force_history_mask"], np.ones(3, dtype=np.bool_))


def test_force_free_slow_keeps_only_robot_state():
    state = np.arange(13, dtype=np.float32)
    transform = forcevla_policy.Forcevla_inputs(
        action_dim=32,
        model_type=model.ModelType.PI0,
        robot_state_dims=10,
    )
    result = transform(
        {
            "state": state,
            "image": np.zeros((8, 8, 3), dtype=np.uint8),
            "wrist_image": np.zeros((8, 8, 3), dtype=np.uint8),
        }
    )

    converted = rot.convert_state(state)[:10]
    np.testing.assert_array_equal(result["state"][:10], converted)
    np.testing.assert_array_equal(result["state"][10:], np.zeros(22, dtype=np.float32))


def test_outputs_restore_rpy_for_the_robot():
    sixd = rot.convert_action(np.array([[0.1, 0.2, 0.3, 3.13, 0.02, -0.04, 0.8]], dtype=np.float32))
    restored = forcevla_policy.Forcevla_outputs()({"actions": sixd})
    geodesic = rot.geodesic_angle(
        rot.rpy_to_matrix(np.array([[3.13, 0.02, -0.04]])),
        rot.rpy_to_matrix(restored["actions"][:, 3:6]),
    )
    np.testing.assert_allclose(restored["actions"][0, :3], [0.1, 0.2, 0.3], atol=1e-5)
    np.testing.assert_allclose(geodesic, 0.0, atol=1e-5)
    np.testing.assert_allclose(restored["actions"][0, 6], 0.8, atol=1e-5)
