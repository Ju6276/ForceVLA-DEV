import numpy as np
import pytest

from openpi import transforms
from openpi.serving import slow_fast_loop
from openpi.serving import slow_fast_runtime


def _history_transform(max_samples: int = 4, rate_hz: float = 100.0):
    return transforms.TimestampAlignedForceHistory(
        force_key="force",
        force_timestamps_key="force_timestamps",
        observation_timestamp_key="timestamp",
        window_ms=1000.0 * max_samples / rate_hz,
        max_samples=max_samples,
        sampling_rate_hz=rate_hz,
        max_sample_age_ms=12.0,
    )


def _buffer(max_samples: int = 4, rate_hz: float = 100.0):
    buffer = slow_fast_loop.ForceStreamBuffer(_history_transform(max_samples, rate_hz))
    for step in range(20):
        buffer.append(step / rate_hz, np.full((6,), float(step), dtype=np.float32))
    return buffer


def test_force_window_matches_the_training_transform_exactly():
    """Serving must not reimplement the causal grid; skew there is invisible."""
    rate_hz, max_samples = 100.0, 4
    buffer = _buffer(max_samples, rate_hz)
    timestamp = 19 / rate_hz

    history, mask = buffer.window(timestamp)

    expected = _history_transform(max_samples, rate_hz)(
        {
            "force": np.stack([np.full((6,), float(step), dtype=np.float32) for step in range(20)]),
            "force_timestamps": np.arange(20, dtype=np.float64) / rate_hz,
            "timestamp": np.float64(timestamp),
        }
    )
    np.testing.assert_array_equal(history, expected["force_history"])
    np.testing.assert_array_equal(mask, expected["force_history_mask"])
    assert mask.all()


def test_force_buffer_rejects_out_of_order_and_malformed_samples():
    buffer = slow_fast_loop.ForceStreamBuffer(_history_transform())
    buffer.append(1.0, np.zeros(6))
    with pytest.raises(ValueError, match="monotonically nondecreasing"):
        buffer.append(0.5, np.zeros(6))
    with pytest.raises(ValueError, match="finite 6D wrench"):
        buffer.append(2.0, np.zeros(3))


def _packet(*, state_at_observation, chunk=None, version=0):
    chunk = np.zeros((5, 7), dtype=np.float32) if chunk is None else np.asarray(chunk, dtype=np.float32)
    return slow_fast_runtime.SlowPacket(
        observation_timestamp=1.0,
        ready_timestamp=1.0,
        reference_start_timestamp=1.0,
        intent_tokens=np.zeros((2, 3), dtype=np.float32),
        intent_mask=np.ones((2,), dtype=np.bool_),
        reference_actions=chunk,
        action_period_s=0.1,
        version=version,
        context_age_scale_s=0.3,
        state_at_observation=np.asarray(state_at_observation, dtype=np.float32),
    )


def test_absolute_command_rebases_on_the_state_slow_saw_not_the_current_one():
    """The chunk is a delta from the Slow observation, so the arm moving since then
    must not shift the command."""
    packet = _packet(state_at_observation=np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.5]))
    composed = np.array([0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.9], dtype=np.float32)

    command = slow_fast_runtime.to_absolute_command(composed, packet, lambda x: x)

    # xyz/rpy are rebased onto the Slow-time pose; the gripper is already absolute.
    np.testing.assert_allclose(command[:3], [1.1, 2.1, 3.1])
    np.testing.assert_allclose(command[3:6], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(command[6], 0.9)


def test_absolute_command_refuses_a_packet_without_its_conditioning_state():
    packet = slow_fast_runtime.SlowPacket(
        observation_timestamp=1.0,
        ready_timestamp=1.0,
        reference_start_timestamp=1.0,
        intent_tokens=np.zeros((2, 3), dtype=np.float32),
        reference_actions=np.zeros((5, 7), dtype=np.float32),
        action_period_s=0.1,
        version=0,
    )
    with pytest.raises(ValueError, match="state the Slow packet was conditioned on"):
        slow_fast_runtime.to_absolute_command(np.zeros(7), packet, lambda x: x)


def test_absolute_command_undoes_normalization_before_rebasing():
    packet = _packet(state_at_observation=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    composed = np.ones(7, dtype=np.float32)

    command = slow_fast_runtime.to_absolute_command(composed, packet, lambda x: 2.0 * x)

    # The delta is scaled by the action stats first, then the pose is added once.
    np.testing.assert_allclose(command[0], 3.0)
    np.testing.assert_allclose(command[1], 2.0)


def _controller(cache, *, residual=None, residual_limit=None):
    residual = np.full((6,), 0.05, dtype=np.float32) if residual is None else residual
    return slow_fast_loop.SlowFastController(
        cache=cache,
        force_buffer=_buffer(),
        predict_residual=lambda **_: (residual, 1.0),
        normalize_state=lambda state: state,
        unnormalize_action=lambda action: action,
        config=slow_fast_loop.SlowFastConfig(
            action_period_s=0.1, residual_limit=residual_limit, max_staleness_s=0.25
        ),
    )


def test_controller_produces_an_absolute_command_from_reference_plus_residual():
    cache = slow_fast_runtime.SlowReferenceCache()
    chunk = np.tile(np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32), (5, 1))
    cache.update(_packet(state_at_observation=np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), chunk=chunk))

    result = _controller(cache).step(1.05, np.zeros(7, dtype=np.float32))

    # 0.2 reference + 0.05 residual, rebased onto x = 10.
    np.testing.assert_allclose(result["command"][0], 10.25, atol=1e-6)
    # The gripper stays whatever Slow commanded.
    np.testing.assert_allclose(result["command"][6], 1.0)
    assert result["packet_version"] == 0
    np.testing.assert_allclose(result["context_age_s"], 0.05, atol=1e-9)


def test_controller_clips_the_residual_to_the_configured_limit():
    cache = slow_fast_runtime.SlowReferenceCache()
    cache.update(_packet(state_at_observation=np.zeros(7)))

    result = _controller(cache, residual=np.full((6,), 5.0), residual_limit=0.01).step(1.0, np.zeros(7))

    np.testing.assert_allclose(result["command"][:6], np.full((6,), 0.01), atol=1e-6)


def test_controller_refuses_to_extrapolate_a_stale_reference():
    cache = slow_fast_runtime.SlowReferenceCache()
    cache.update(_packet(state_at_observation=np.zeros(7)))
    controller = _controller(cache)

    with pytest.raises(slow_fast_runtime.StaleSlowReferenceError):
        controller.step(1.0 + 0.4 + 0.26, np.zeros(7))


def test_controller_reports_a_missing_reference_before_the_first_packet():
    controller = _controller(slow_fast_runtime.SlowReferenceCache())

    with pytest.raises(slow_fast_runtime.MissingSlowReferenceError):
        controller.step(1.0, np.zeros(7))


def test_slow_worker_stamps_readiness_after_inference_and_bumps_the_version():
    cache = slow_fast_runtime.SlowReferenceCache()
    clock = iter([1.5, 2.5])
    worker = slow_fast_loop.SlowWorker(
        observe=lambda: (1.0, "obs", np.arange(7, dtype=np.float32)),
        infer=lambda _: (
            np.zeros((5, 7), dtype=np.float32),
            np.zeros((2, 3), dtype=np.float32),
            np.ones((2,), dtype=np.bool_),
        ),
        cache=cache,
        config=slow_fast_loop.SlowFastConfig(action_period_s=0.1),
        context_age_scale_s=0.3,
        period_s=0.1,
        clock=lambda: next(clock),
    )

    first = worker.publish_once()
    second = worker.publish_once()

    # Readiness trails the observation by the inference time, which is exactly the
    # latency the training-time context age never modelled.
    assert first.observation_timestamp == 1.0
    assert first.ready_timestamp == 1.5
    assert first.reference_start_timestamp == 1.0
    assert (first.version, second.version) == (0, 1)
    np.testing.assert_array_equal(cache.snapshot().state_at_observation, np.arange(7))
