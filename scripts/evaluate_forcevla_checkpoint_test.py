import numpy as np

from . import evaluate_forcevla_checkpoint


def test_physical_action_report_separates_translation_rotation_and_gripper():
    expert = np.zeros((1, 2, 7), dtype=np.float32)
    prediction = expert.copy()
    prediction[..., 0] = 0.1
    prediction[..., 5] = 0.2
    prediction[..., 6] = 0.3

    summary = evaluate_forcevla_checkpoint.physical_action_error_summary(prediction, expert)

    assert set(summary) == {"all_horizon", "first_action"}
    for horizon in summary.values():
        np.testing.assert_allclose(horizon["translation_rmse_m"], 0.1, atol=1e-6)
        np.testing.assert_allclose(horizon["rotation_geodesic_rmse_rad"], 0.2, atol=1e-6)
        np.testing.assert_allclose(horizon["gripper_rmse"], 0.3, atol=1e-6)
        np.testing.assert_allclose(horizon["gripper_mae"], 0.3, atol=1e-6)
