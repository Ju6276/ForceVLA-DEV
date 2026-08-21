"""End-to-end asynchronous Slow/Fast control loop.

Slow and Fast inference run in independent workers and publish atomic packets. The
actuator thread only samples the current Fast command chunk, so neither model can
block its timing. This module owns the things that are easy to get subtly wrong once
the rates are decoupled:

- the force window is built by the same transform the dataset used, so the deployed
  history cannot drift from the trained one;
- the composed action is returned to physical units by inverting `Normalize` and then
  `DeltaActions`, rebasing the delta dimensions onto the state the Slow packet was
  conditioned on rather than the state at execution time;
- a reference that has gone stale raises instead of being silently extrapolated.
- Fast inference latency advances the actuator into the matching command-chunk step
  instead of executing an already stale step zero.

The robot interface is injected, so this module has no dependency on a particular arm.
"""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import threading
import time

import numpy as np

from openpi import transforms
from openpi.policies import rotation_6d as rot
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
    # Fast is trained on xyz + continuous 6D rotation + gripper. Keeping this
    # explicit makes a raw 7D xyz+rpy+gripper state fail at the controller
    # boundary instead of much later inside the model.
    state_dims: int = rot.ROBOT_DIMS
    pose_dims: int = 9
    # ForceVLA makes xyz and 6D rotation relative to the state; the gripper stays absolute.
    delta_dims: int = 9
    # Length of the chunk Fast emits per tick. The caller may execute only the first
    # entry and tick again, or walk further into the chunk when Fast falls behind, so
    # the Fast call rate does not have to be fixed at training time.
    chunk_steps: int = 5
    max_staleness_s: float = 0.25
    # Per-dimension cap on the Fast correction, in normalized action units. Leaving
    # this unset lets an unbounded residual reach the arm.
    residual_limit: np.ndarray | float | None = None

    def __post_init__(self) -> None:
        if self.action_period_s <= 0 or self.max_staleness_s < 0:
            raise ValueError("action_period_s must be positive and max_staleness_s non-negative")
        if self.chunk_steps <= 0 or self.state_dims <= 0:
            raise ValueError("chunk_steps and state_dims must be positive")
        if self.pose_dims >= self.state_dims:
            raise ValueError("pose_dims must leave at least one Slow-owned action dimension")
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
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (self._config.state_dims,) or not np.all(np.isfinite(state)):
            raise ValueError(
                f"Slow observe() must return a finite {self._config.state_dims}D model-layout state; "
                "convert raw xyz+rpy+gripper with slow_fast_deploy.convert_robot_state()"
            )
        chunk, context, context_mask = self._infer(observation)
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[-1] != self._config.state_dims:
            raise ValueError(
                f"Slow infer() must return [H,{self._config.state_dims}] model-layout actions, got {chunk.shape}"
            )
        ready_timestamp = self._clock()
        packet = slow_fast_runtime.SlowPacket(
            observation_timestamp=observation_timestamp,
            ready_timestamp=ready_timestamp,
            # The chunk predicts the future from the observation, so its first entry
            # belongs at the observation instant even though it lands late.
            reference_start_timestamp=observation_timestamp,
            intent_tokens=np.asarray(context),
            intent_mask=np.asarray(context_mask, dtype=np.bool_),
            reference_actions=chunk,
            action_period_s=self._config.action_period_s,
            version=self._version,
            context_age_scale_s=self._context_age_scale_s,
            state_at_observation=state,
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
        normalize_force_history: Callable[[np.ndarray], np.ndarray],
        unnormalize_action: Callable[[np.ndarray], np.ndarray],
        config: SlowFastConfig,
    ):
        self._cache = cache
        self._force_buffer = force_buffer
        self._predict_residual = predict_residual
        self._normalize_state = normalize_state
        # Required rather than defaulted: the student was trained on normalized
        # wrench, and raw newtons are off by both an offset and an order of
        # magnitude without failing any shape check.
        self._normalize_force_history = normalize_force_history
        self._unnormalize_action = unnormalize_action
        self._config = config

    def step(self, timestamp: float, state) -> dict:
        """Produce a short chunk of absolute commands, in physical units, from one tick.

        `command[k]` is meant for `timestamp + k * action_period_s`. Executing only
        `command[0]` and ticking again is the nominal case; the later entries are what
        the caller falls back on when Fast cannot be called at the action rate, which
        is what keeps the loop from depending on a rate fixed at training time.

        Raises `MissingSlowReferenceError` before the first packet arrives and
        `StaleSlowReferenceError` once the current chunk has run out; the caller is
        expected to hold position or abort rather than extrapolate.
        """
        if not np.isfinite(timestamp):
            raise ValueError("Fast controller timestamp must be finite")
        packet = self._cache.snapshot()
        config = self._config
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (config.state_dims,) or not np.all(np.isfinite(state)):
            raise ValueError(
                f"Fast controller requires a finite {config.state_dims}D xyz+6D+gripper state, got {state.shape}; "
                "convert the latest raw robot state with slow_fast_deploy.convert_robot_state()"
            )
        if packet.reference_actions.shape[-1] != config.state_dims:
            raise ValueError(
                f"Slow packet actions are {packet.reference_actions.shape[-1]}D but Fast expects {config.state_dims}D"
            )
        if packet.intent_mask is None:
            # Pooling leaves empty context bins invalid, so substituting an
            # all-valid mask would feed the student padding as if it were context.
            raise ValueError("The Slow packet carries no intent mask; the Fast student requires one")
        reference = slow_fast_runtime.sample_reference(
            packet, timestamp, steps=config.chunk_steps, max_staleness_s=config.max_staleness_s
        )
        time_features = slow_fast_runtime.reference_time_features(packet, timestamp)
        force_history, force_mask = self._force_buffer.window(timestamp)
        # Match the training order: normalize the zero-filled window, then let the
        # TCN apply force_history_mask before its stem. Do not invent a second,
        # deployment-only normalization or imputation rule here.
        raw_shape = force_history.shape
        force_history = np.asarray(self._normalize_force_history(force_history), dtype=np.float32)
        if force_history.shape != raw_shape or not np.all(np.isfinite(force_history)):
            raise ValueError(
                f"Normalized force window must remain finite with shape {raw_shape}, got {force_history.shape}"
            )
        normalized_state = np.asarray(self._normalize_state(state), dtype=np.float32)
        if normalized_state.shape != state.shape or not np.all(np.isfinite(normalized_state)):
            raise ValueError(
                f"Normalized robot state must remain finite with shape {state.shape}, got {normalized_state.shape}"
            )
        residual, gate = self._predict_residual(
            intent_tokens=packet.intent_tokens,
            intent_mask=packet.intent_mask,
            force_history=force_history,
            force_history_mask=force_mask,
            state=normalized_state,
            reference_action=reference[0],
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
        if command.shape[-1] >= rot.ROBOT_DIMS:
            command = rot.actions_6d_to_rpy(command)
        return {
            "command": command,
            "command_period_s": config.action_period_s,
            "reference_normalized": reference,
            "residual_normalized": np.asarray(residual, dtype=np.float32),
            "gate": float(gate),
            "packet_version": packet.version,
            "context_age_s": timestamp - packet.observation_timestamp,
            "valid_force_samples": int(np.count_nonzero(force_mask)),
        }


class MissingFastCommandError(RuntimeError):
    """Raised before the asynchronous Fast worker has published a command."""


class StaleFastCommandError(RuntimeError):
    """Raised once every command in the latest Fast chunk has expired."""


class StaleFastObservationError(RuntimeError):
    """Raised when `observe` returns a timestamp no newer than the published chunk.

    A momentarily repeated or backwards robot timestamp is a driver hiccup, not a
    reason to take Fast down for the rest of the run, so the worker treats this as
    transient and simply skips the cycle.
    """


@dataclasses.dataclass(frozen=True)
class FastCommandPacket:
    """One timestamped command chunk published by asynchronous Fast inference."""

    start_timestamp: float
    ready_timestamp: float
    commands: np.ndarray
    command_period_s: float
    version: int
    slow_packet_version: int

    def __post_init__(self) -> None:
        commands = np.array(self.commands, dtype=np.float32, copy=True)
        timestamps = np.asarray([self.start_timestamp, self.ready_timestamp], dtype=np.float64)
        if commands.ndim != 2 or not commands.shape[0] or not commands.shape[1]:
            raise ValueError(f"Fast commands must be a non-empty [K,A] array, got {commands.shape}")
        if not np.all(np.isfinite(commands)) or not np.all(np.isfinite(timestamps)):
            raise ValueError("Fast command packet must contain only finite values")
        if not np.isfinite(self.command_period_s) or self.command_period_s <= 0:
            raise ValueError("Fast command period must be finite and positive")
        if self.version < 0 or self.slow_packet_version < 0:
            raise ValueError("Fast command period and packet versions must be valid")
        if self.ready_timestamp < self.start_timestamp:
            raise ValueError(
                "Fast ready_timestamp precedes its observation timestamp; sensor timestamps and the worker clock "
                "must share one monotonic clock domain"
            )
        expires_at = self.start_timestamp + commands.shape[0] * self.command_period_s
        if self.ready_timestamp >= expires_at:
            raise StaleFastCommandError(
                f"Fast inference finished after its entire {commands.shape[0]}-step chunk had expired"
            )
        commands.setflags(write=False)
        object.__setattr__(self, "commands", commands)

    @property
    def expires_at(self) -> float:
        """End of the final command's one-period validity interval."""
        return self.start_timestamp + self.commands.shape[0] * self.command_period_s


class FastCommandCache:
    """Atomic Fast-command handoff sampled independently by the actuator loop."""

    def __init__(self):
        self._lock = threading.Lock()
        self._packet: FastCommandPacket | None = None

    def update(self, packet: FastCommandPacket) -> None:
        with self._lock:
            if self._packet is not None and (
                packet.start_timestamp < self._packet.start_timestamp or packet.version <= self._packet.version
            ):
                raise ValueError("Fast command packets must have monotonic timestamps and increasing versions")
            self._packet = packet

    def snapshot(self) -> FastCommandPacket:
        with self._lock:
            if self._packet is None:
                raise MissingFastCommandError("The Fast worker has not produced a command chunk yet")
            return self._packet

    def sample(self, timestamp: float) -> dict:
        """Return the command whose validity interval contains `timestamp`.

        If Fast inference took more than one action period this automatically skips
        the already-stale leading commands. It never holds the final command beyond
        its own period; the robot caller should hold safely or abort on expiry.
        """
        packet = self.snapshot()
        if not np.isfinite(timestamp):
            raise ValueError("Actuator timestamp must be finite")
        # The actuator reads its clock before it samples, so a packet installed in
        # between carries a ready time a few microseconds ahead of it. That is a
        # read-ordering artefact, not an error, and raising on it would crash the
        # real-time loop on a benign race. A gap beyond one command period cannot be
        # explained that way and does mean the two clocks disagree.
        if packet.ready_timestamp - timestamp > packet.command_period_s:
            raise ValueError(
                f"The actuator clock is {packet.ready_timestamp - timestamp:.3f}s behind the Fast worker's; "
                "sensor timestamps and the worker clock must share one monotonic clock domain"
            )
        # Selecting from the later of the two never returns a command whose interval
        # had already closed by the time inference produced it.
        elapsed = max(timestamp, packet.ready_timestamp) - packet.start_timestamp
        position = max(0.0, elapsed) / packet.command_period_s
        command_index = int(np.floor(position + 1e-9))
        if command_index >= packet.commands.shape[0]:
            raise StaleFastCommandError(f"Fast command chunk expired by {timestamp - packet.expires_at:.3f}s")
        return {
            "command": packet.commands[command_index].copy(),
            "command_index": command_index,
            "fast_packet_version": packet.version,
            "slow_packet_version": packet.slow_packet_version,
            "command_age_s": timestamp - packet.start_timestamp,
            "expires_at": packet.expires_at,
        }


class FastWorker:
    """Runs Fast inference off the actuator thread and publishes command chunks."""

    def __init__(
        self,
        *,
        observe: Callable[[], tuple[float, np.ndarray]],
        controller: SlowFastController,
        command_cache: FastCommandCache,
        period_s: float,
        clock: Callable[[], float] = time.monotonic,
    ):
        if period_s <= 0:
            raise ValueError("period_s must be positive")
        self._observe = observe
        self._controller = controller
        self._command_cache = command_cache
        self._period_s = period_s
        self._clock = clock
        self._version = 0
        self._published_start: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error_lock = threading.Lock()
        self._error: Exception | None = None

    def publish_once(self) -> FastCommandPacket:
        timestamp, state = self._observe()
        timestamp = float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("Fast observe() must return a finite timestamp")
        # Checked before inference rather than at the cache, so a repeated robot
        # timestamp costs nothing and a backwards one cannot reach the strict
        # monotonicity invariant the cache uses to catch real programming errors.
        if self._published_start is not None and timestamp <= self._published_start:
            raise StaleFastObservationError(
                f"observe() returned t={timestamp:.6f}, which does not advance on the published "
                f"t={self._published_start:.6f}; there is nothing new to compute"
            )
        result = self._controller.step(timestamp, state)
        packet = FastCommandPacket(
            start_timestamp=timestamp,
            ready_timestamp=float(self._clock()),
            commands=np.asarray(result["command"], dtype=np.float32),
            command_period_s=float(result["command_period_s"]),
            version=self._version,
            slow_packet_version=int(result["packet_version"]),
        )
        self._command_cache.update(packet)
        self._published_start = timestamp
        self._version += 1
        return packet

    def _record_error(self, error: Exception) -> None:
        with self._error_lock:
            self._error = error

    def raise_if_failed(self) -> None:
        with self._error_lock:
            error = self._error
        if error is not None:
            raise RuntimeError("The asynchronous Fast worker stopped after an inference error") from error

    def _run(self) -> None:
        while not self._stop.is_set():
            started = self._clock()
            try:
                self.publish_once()
            except (
                slow_fast_runtime.MissingSlowReferenceError,
                slow_fast_runtime.StaleSlowReferenceError,
                StaleFastCommandError,
                StaleFastObservationError,
            ):
                # Slow startup and replacement are transient; keep trying rather
                # than killing Fast. A one-off over-budget Fast call is handled the
                # same way: no stale chunk is published, and the actuator applies
                # its hold/abort policy until the next usable packet.
                pass
            except Exception as error:  # Background failures must be observable.
                self._record_error(error)
                self._stop.set()
                return
            self._stop.wait(max(0.0, self._period_s - (self._clock() - started)))

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("FastWorker is already running")
        self._thread = threading.Thread(target=self._run, name="fast-residual", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        self.raise_if_failed()
