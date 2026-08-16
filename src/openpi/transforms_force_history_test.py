import numpy as np
import pytest

from openpi import transforms


def test_timestamp_force_history_is_causal_and_left_padded():
    transform = transforms.TimestampAlignedForceHistory(
        force_key="force",
        force_timestamps_key="force_time",
        observation_timestamp_key="time",
        window_ms=101,
        max_samples=5,
    )
    force = np.arange(30, dtype=np.float32).reshape(5, 6)
    result = transform(
        {
            "force": force,
            "force_time": np.asarray([0.80, 0.90, 0.95, 1.00, 1.01]),
            "time": np.asarray(1.0),
        }
    )
    np.testing.assert_array_equal(result["force_history"][-3:], force[1:4])
    np.testing.assert_array_equal(result["force_history_mask"], [False, False, True, True, True])
    assert not np.any(np.all(result["force_history"] == force[4], axis=-1))


def test_timestamp_force_history_rejects_unsorted_timestamps():
    transform = transforms.TimestampAlignedForceHistory("force", "force_time", "time", 100, 5)
    with pytest.raises(ValueError, match="monotonically"):
        transform(
            {
                "force": np.zeros((2, 6), dtype=np.float32),
                "force_time": np.asarray([1.0, 0.9]),
                "time": np.asarray(1.0),
            }
        )
