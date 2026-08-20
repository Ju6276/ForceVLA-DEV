import numpy as np
import pytest

from openpi.serving import slow_fast_runtime


def _packet(timestamp=1.0, version=0, ready_timestamp=None):
    ready_timestamp = timestamp if ready_timestamp is None else ready_timestamp
    return slow_fast_runtime.SlowPacket(
        observation_timestamp=timestamp,
        ready_timestamp=ready_timestamp,
        reference_start_timestamp=timestamp,
        intent_tokens=np.ones((2, 8), dtype=np.float32),
        reference_actions=np.array(
            [
                [0, 0, 0, 0, 0, 0, 0.25],
                [1, 1, 1, 1, 1, 1, 0.50],
                [2, 2, 2, 2, 2, 2, 0.75],
            ],
            dtype=np.float32,
        ),
        action_period_s=0.1,
        version=version,
    )


def test_atomic_cache_and_interpolated_reference():
    cache = slow_fast_runtime.SlowReferenceCache()
    cache.update(_packet())
    packet = cache.snapshot()
    np.testing.assert_allclose(
        slow_fast_runtime.sample_reference(packet, 1.05),
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.25],
    )


def test_composition_changes_pose_but_not_gripper():
    reference = np.array([1, 2, 3, 4, 5, 6, 0.8], dtype=np.float32)
    result = slow_fast_runtime.compose_reference_residual(
        reference,
        np.ones(6),
        gate=0.5,
        residual_limit=0.25,
    )
    np.testing.assert_allclose(result[:6], reference[:6] + 0.125)
    assert result[6] == reference[6]


def test_cache_rejects_old_packet_and_stale_reference():
    cache = slow_fast_runtime.SlowReferenceCache()
    cache.update(_packet(version=1))
    with pytest.raises(ValueError, match="monotonic"):
        cache.update(_packet(timestamp=0.9, version=2))
    with pytest.raises(slow_fast_runtime.StaleSlowReferenceError):
        slow_fast_runtime.sample_reference(cache.snapshot(), 2.0, max_staleness_s=0.1)


def test_sampling_compensates_slow_latency_and_builds_time_features():
    packet = _packet(timestamp=1.0, ready_timestamp=1.12)
    # When inference finishes at 1.12, action zero and part of action one are
    # already stale; sampling uses the reference's original time base.
    np.testing.assert_allclose(
        slow_fast_runtime.sample_reference(packet, 1.15),
        [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 0.5],
    )
    np.testing.assert_allclose(
        slow_fast_runtime.reference_time_features(packet, 1.15, context_age_scale_s=0.5),
        [0.75, 0.3],
    )


def test_packet_cannot_be_consumed_before_ready():
    with pytest.raises(ValueError, match="before it is ready"):
        slow_fast_runtime.sample_reference(_packet(ready_timestamp=1.1), 1.05)
