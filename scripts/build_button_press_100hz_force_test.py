import numpy as np

from scripts.build_button_press_100hz_force import causal_force_history


def test_causal_history_never_selects_future_sample() -> None:
    observation_times = np.array([0.1])
    force_times = np.array([0.001, 0.015, 0.025, 0.035, 0.045, 0.055, 0.065, 0.075, 0.085, 0.095, 0.101])
    force_values = np.repeat(force_times[:, None], 6, axis=1)

    result = causal_force_history(
        observation_times,
        force_times,
        force_values,
        output_rate_hz=100,
        window_ms=100,
        max_sample_age_ms=12,
    )

    np.testing.assert_allclose(result["grid_timestamps"][0], np.arange(0.01, 0.101, 0.01))
    assert result["force_history_mask"].all()
    assert np.all(result["source_timestamps"] <= result["grid_timestamps"])
    assert result["source_timestamps"][0, -1] == 0.095
    assert not np.any(result["source_timestamps"] == 0.101)


def test_causal_history_masks_stale_and_incomplete_slots() -> None:
    result = causal_force_history(
        np.array([0.1]),
        np.array([0.001, 0.095]),
        np.array([[1.0] * 6, [2.0] * 6]),
        output_rate_hz=100,
        window_ms=100,
        max_sample_age_ms=12,
    )

    np.testing.assert_array_equal(
        result["force_history_mask"][0], [True, False, False, False, False, False, False, False, False, True]
    )
    np.testing.assert_array_equal(result["force_history"][0, 1:9], 0.0)
