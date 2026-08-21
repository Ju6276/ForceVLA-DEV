import numpy as np
import pytest

from openpi.serving import slow_fast_runtime
from openpi.training import fast_dataset


def _packet(timestamp=1.0, version=0, ready_timestamp=None, context_age_scale_s=None):
    ready_timestamp = timestamp if ready_timestamp is None else ready_timestamp
    extra = {} if context_age_scale_s is None else {"context_age_scale_s": context_age_scale_s}
    return slow_fast_runtime.SlowPacket(
        **extra,
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
        [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.25]],
    )


def test_sampled_chunk_advances_one_action_period_per_step():
    packet = _packet()
    chunk = slow_fast_runtime.sample_reference(packet, 1.05, steps=3)
    assert chunk.shape == (3, 7)
    np.testing.assert_allclose(chunk[:, 0], [0.5, 1.5, 2.0])
    # The last step runs past the stored horizon and holds the final waypoint
    # rather than extrapolating.
    np.testing.assert_allclose(chunk[2], packet.reference_actions[2])
    # The gripper is held, not interpolated, at every step.
    np.testing.assert_allclose(chunk[:, 6], [0.25, 0.50, 0.75])


def test_composition_changes_pose_but_not_gripper():
    reference = np.array([[1, 2, 3, 4, 5, 6, 0.8], [2, 3, 4, 5, 6, 7, 0.9]], dtype=np.float32)
    result = slow_fast_runtime.compose_reference_residual(
        reference,
        np.ones((2, 6)),
        gate=0.5,
        pose_dims=6,
        residual_limit=0.25,
    )
    np.testing.assert_allclose(result[:, :6], reference[:, :6] + 0.125)
    np.testing.assert_array_equal(result[:, 6], reference[:, 6])


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
        [[1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 0.5]],
    )
    # 150 ms of age over a 500 ms scale, and halfway between waypoints 1 and 2.
    np.testing.assert_allclose(
        slow_fast_runtime.reference_time_features(packet, 1.15, context_age_scale_s=0.5),
        [0.3, 0.5],
    )


def test_time_features_default_to_the_scale_carried_by_the_packet():
    packet = _packet(timestamp=1.0, ready_timestamp=1.12, context_age_scale_s=0.5)
    np.testing.assert_allclose(
        slow_fast_runtime.reference_time_features(packet, 1.15), [0.3, 0.5]
    )
    # An explicit argument rescales only the age; alpha is not a normalized quantity.
    np.testing.assert_allclose(
        slow_fast_runtime.reference_time_features(packet, 1.15, context_age_scale_s=0.25), [0.6, 0.5]
    )


def test_serving_time_features_match_the_offline_cache():
    # The student is trained on the offline rollout and deployed on the runtime, so a
    # disagreement here silently feeds it a token it never saw.
    packet = _packet(timestamp=1.0, ready_timestamp=1.12, context_age_scale_s=0.5)
    _, offline = fast_dataset.build_reference_rollout(
        packet.reference_actions[None],
        np.zeros(1, dtype=np.int64),
        np.array([1.15]),
        np.array([packet.observation_timestamp]),
        action_period_s=packet.action_period_s,
        context_age_scale_s=packet.context_age_scale_s,
        chunk_steps=1,
    )
    np.testing.assert_allclose(
        slow_fast_runtime.reference_time_features(packet, 1.15), offline[0], atol=1e-6
    )


def test_serving_reference_chunk_matches_the_offline_rollout():
    packet = _packet(timestamp=1.0, ready_timestamp=1.0)
    offline, _ = fast_dataset.build_reference_rollout(
        packet.reference_actions[None],
        np.zeros(1, dtype=np.int64),
        np.array([1.05]),
        np.array([packet.observation_timestamp]),
        action_period_s=packet.action_period_s,
        context_age_scale_s=0.5,
        chunk_steps=3,
    )
    np.testing.assert_allclose(
        slow_fast_runtime.sample_reference(packet, 1.05, steps=3), offline[0], atol=1e-6
    )


def test_packet_rejects_a_non_positive_context_age_scale():
    with pytest.raises(ValueError, match="context_age_scale_s must be positive"):
        _packet(context_age_scale_s=0.0)


def test_packet_cannot_be_consumed_before_ready():
    with pytest.raises(ValueError, match="before it is ready"):
        slow_fast_runtime.sample_reference(_packet(ready_timestamp=1.1), 1.05)
