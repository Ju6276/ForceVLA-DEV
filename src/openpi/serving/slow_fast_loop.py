"""End-to-end asynchronous Slow/Fast control loop.

Slow runs in its own thread at a low rate and publishes atomic packets; Fast runs in
the caller's real-time thread at the force rate and consumes whichever packet is
current. This module owns the three things that are easy to get subtly wrong once the
two rates are decoupled:

- the force window is built by the same transform the dataset used, so the deployed
  history cannot drift from the trained one;
- the composed action is returned to physical units by inverting `Normalize` and then
  `DeltaActions`, rebasing the delta dimensions onto the state the Slow packet was
  conditioned on rather than the state at execution time;
- a reference that has gone stale raises instead of being silently extrapolated.

The robot interface is injected, so this module has no dependency on a particular arm.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from typing import Callable

import numpy as np

from openpi import transforms
from openpi.serving import slow_fast_runtime


class ForceStreamBuffer:
    """Thread-safe ring buffer of the native-rate wrench stream."""

    def __init__(self, history_transform: transforms.TimestampAlignedForceHistory, *, capacity: int = 8192):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._transform = history_transform
        self._capacity = int(capacity)
        self._lock = threading.Lock()
        self._times: list[float] = []
        self._wrenches: list[np.ndarray] = []

    def append(self, timestamp: float, wrench) -> None:
        sample = np.asarray(wrench, dtype=np.float32)
        if sample.shape != (6,) or not np.all(np.isfinite(sample)):
            raise ValueError(f"Expected a finite 6D wrench, got {np.shape(wrench)}")
        with self._lock:
            if self._times and timestamp < self._times[-1]:
                raise ValueError("Wrench timestamps must be monotonically nondecreasing")
            self._times.append(float(timestamp))
            self._wrenches.append(sample)
            if len(self._times) > self._capacity:
                del self._times[: len(self._times) - self._capacity]
                del self._wrenches[: len(self._wrenches) - self._capacity]

    def window(self, timestamp: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the causal `[N,6]` history and its validity mask at `timestamp`."""
        with self._lock:
            if not self._times:
                raise slow_fast_runtime.MissingSlowReferenceError("No wrench samples have been recorded yet")
            times = np.asarray(self._times, dtype=np.float64)
            forces = np.stack(self._wrenches)
        result = self._transform(
            {
                self._transform.force_key: forces,
                self._transform.force_timestamps_key: times,
                self._transform.observation_timestamp_key: np.float64(timestamp),
            }
        )
        return (
            np.asarray(result["force_history"], dtype=np.float32),
            np.asarray(result["force_history_mask"], dtype=np.bool_),
        )


@dataclasses.dataclass(frozen=True)
class SlowFastConfig:
    """Timing and safety limits for the composed controller."""

    action_period_s: float
    pose_dims: int = 6
    # ForceVLA makes xyz and rpy relative to the state; the gripper stays absolute.
    delta_dims: int = 6
    max_staleness_s: float = 0.25
    # Per-dimension cap on the Fast correction, in normalized action units. Leaving
    # this unset lets an unbounded residual reach the arm.
    residual_limit: np.ndarray | float | None = None

    def __post_init__(self) -> None:
        if self.action_period_s <= 0 or self.max_staleness_s < 0:
            raise ValueError("action_period_s must be positive and max_staleness_s non-negative")
        if not 0 < self.delta_dims <= self.pose_dims:
            raise ValueError("delta_dims must be positive and at most pose_dims")


class SlowWorker:
    """Runs the Slow VLA in its own thread and publishes atomic packets."""

    def __init__(
        self,
        *,
        observe: Callable[[], tuple[float, object, np.ndarray]],
        infer: Callable[[object], tuple[np.ndarray, np.ndarray, np.ndarray]],
        cache: slow_fast_runtime.SlowReferenceCache,
        config: SlowFastConfig,
        context_age_scale_s: float,
        period_s: float,
        clock: Callable[[], float] = time.monotonic,
    ):
        if period_s <= 0:
            raise ValueError("period_s must be positive")
        self._observe = observe
        self._infer = infer
        self._cache = cache
        self._config = config
        self._context_age_scale_s = context_age_scale_s
        self._period_s = period_s
        self._clock = clock
        self._version = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def publish_once(self) -> slow_fast_runtime.SlowPacket:
        """Run one Slow update and install it. Exposed so tests need no thread."""
        observation_timestamp, observation, state = self._observe()
        chunk, context, context_mask = self._infer(observation)
        ready_timestamp = self._clock()
        packet = slow_fast_runtime.SlowPacket(
            observation_timestamp=observation_timestamp,
            ready_timestamp=ready_timestamp,
            # The chunk predicts the future from the observation, so its first entry
            # belongs at the observation instant even though it lands late.
            reference_start_timestamp=observation_timestamp,
            intent_tokens=np.asarray(context),
            intent_mask=np.asarray(context_mask, dtype=np.bool_),
            reference_actions=np.asarray(chunk, dtype=np.float32),
            action_period_s=self._config.action_period_s,
            version=self._version,
            context_age_scale_s=self._context_age_scale_s,
            state_at_observation=np.asarray(state, dtype=np.float32),
        )
        self._cache.update(packet)
        self._version += 1
        return packet

    def _run(self) -> None:
        while not self._stop.is_set():
            started = self._clock()
            self.publish_once()
            self._stop.wait(max(0.0, self._period_s - (self._clock() - started)))

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("SlowWorker is already running")
        self._thread = threading.Thread(target=self._run, name="slow-vla", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None


class SlowFastController:
    """Composes the current Slow reference with one Fast force correction."""

    def __init__(
        self,
        *,
        cache: slow_fast_runtime.SlowReferenceCache,
        force_buffer: ForceStreamBuffer,
        predict_residual: Callable[..., tuple[np.ndarray, float]],
        normalize_state: Callable[[np.ndarray], np.ndarray],
        unnormalize_action: Callable[[np.ndarray], np.ndarray],
        config: SlowFastConfig,
    ):
        self._cache = cache
        self._force_buffer = force_buffer
        self._predict_residual = predict_residual
        self._normalize_state = normalize_state
        self._unnormalize_action = unnormalize_action
        self._config = config

    def step(self, timestamp: float, state) -> dict:
        """Produce one absolute command, in physical units, for the given instant.

        Raises `MissingSlowReferenceError` before the first packet arrives and
        `StaleSlowReferenceError` once the current chunk has run out; the caller is
        expected to hold position or abort rather than extrapolate.
        """
        packet = self._cache.snapshot()
        config = self._config
        reference = slow_fast_runtime.sample_reference(
            packet, timestamp, max_staleness_s=config.max_staleness_s
        )
        time_features = slow_fast_runtime.reference_time_features(packet, timestamp)
        force_history, force_mask = self._force_buffer.window(timestamp)
        residual, gate = self._predict_residual(
            intent_tokens=packet.intent_tokens,
            intent_mask=packet.intent_mask,
            force_history=force_history,
            force_history_mask=force_mask,
            state=np.asarray(self._normalize_state(np.asarray(state, dtype=np.float32)), dtype=np.float32),
            reference_action=reference,
            time_features=time_features,
        )
        composed = slow_fast_runtime.compose_reference_residual(
            reference,
            residual,
            gate=gate,
            pose_dims=config.pose_dims,
            residual_limit=config.residual_limit,
        )
        command = slow_fast_runtime.to_absolute_command(
            composed, packet, self._unnormalize_action, delta_dims=config.delta_dims
        )
        return {
            "command": command,
            "reference_normalized": reference,
            "residual_normalized": np.asarray(residual, dtype=np.float32),
            "gate": float(gate),
            "packet_version": packet.version,
            "context_age_s": timestamp - packet.observation_timestamp,
            "valid_force_samples": int(np.count_nonzero(force_mask)),
        }
