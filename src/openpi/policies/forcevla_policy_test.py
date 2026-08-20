import numpy as np

from openpi.models import model
from openpi.policies import forcevla_policy


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

    np.testing.assert_array_equal(result["state"][:13], states[-1])
    np.testing.assert_array_equal(result["force_history"], states[:, 7:13])
    np.testing.assert_array_equal(result["force_history_mask"], np.ones(3, dtype=np.bool_))


def test_force_free_slow_keeps_only_robot_state():
    state = np.arange(13, dtype=np.float32)
    transform = forcevla_policy.Forcevla_inputs(
        action_dim=32,
        model_type=model.ModelType.PI0,
        robot_state_dims=7,
    )
    result = transform(
        {
            "state": state,
            "image": np.zeros((8, 8, 3), dtype=np.uint8),
            "wrist_image": np.zeros((8, 8, 3), dtype=np.uint8),
        }
    )

    np.testing.assert_array_equal(result["state"][:7], state[:7])
    np.testing.assert_array_equal(result["state"][7:], np.zeros(25, dtype=np.float32))
