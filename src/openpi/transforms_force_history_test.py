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


@pytest.mark.parametrize(("sampling_rate_hz", "expected_samples"), [(30, 3), (200, 20)])
def test_timestamp_force_history_excludes_exact_left_boundary(sampling_rate_hz, expected_samples):
    target_time = 1.0
    # Include the exact t - 100 ms endpoint, which must be excluded from
    # the half-open interval (t - 100 ms, t].
    timestamps = target_time - np.arange(expected_samples, -1, -1) / sampling_rate_hz
    force = np.arange(timestamps.size * 6, dtype=np.float32).reshape(timestamps.size, 6)
    transform = transforms.TimestampAlignedForceHistory(
        "force", "force_time", "time", window_ms=100, max_samples=expected_samples
    )

    result = transform({"force": force, "force_time": timestamps, "time": np.asarray(target_time)})

    np.testing.assert_array_equal(result["force_history"], force[1:])
    np.testing.assert_array_equal(result["force_history_mask"], np.ones(expected_samples, dtype=np.bool_))
