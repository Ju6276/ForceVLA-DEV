"""Thread-safe timestamped handoff between slow nominal and fast residual policies."""

from __future__ import annotations

import dataclasses
import threading

import numpy as np

from openpi.models.slow_fast import DEFAULT_CONTEXT_AGE_SCALE_S


class MissingSlowReferenceError(RuntimeError):
    pass


class StaleSlowReferenceError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class SlowPacket:
    """One atomic Slow-VLA update consumed repeatedly by the fast loop."""

    observation_timestamp: float
    ready_timestamp: float
    reference_start_timestamp: float
    intent_tokens: np.ndarray
    reference_actions: np.ndarray
    action_period_s: float
    version: int
    # The unnormalized robot state this chunk was conditioned on. ForceVLA trains on
    # delta actions, so the whole chunk is expressed relative to the pose at the Slow
    # observation. Undoing that with the pose at execution time instead would bias
    # every command by however far the arm travelled while Slow was thinking.
    state_at_observation: np.ndarray | None = None
    # Validity of `intent_tokens`. The Fast student takes a context mask during
    # training, so serving has to carry one rather than assume every pooled bin is
    # populated.
    intent_mask: np.ndarray | None = None
    # Copied from the Slow cache the Fast student was trained on. Keeping it on the
    # packet stops the serving loop from silently scaling the time token differently
    # than training did.
    context_age_scale_s: float = DEFAULT_CONTEXT_AGE_SCALE_S

    def __post_init__(self) -> None:
        intent = np.asarray(self.intent_tokens)
        reference = np.asarray(self.reference_actions)
        if intent.ndim != 2 or reference.ndim != 2:
            raise ValueError("intent_tokens and reference_actions must be [K,D] and [H,A]")
        timestamps = np.asarray([self.observation_timestamp, self.ready_timestamp, self.reference_start_timestamp])
        if not np.all(np.isfinite(timestamps)) or self.action_period_s <= 0:
            raise ValueError("Slow packet timestamps/action period must be valid")
        if not np.isfinite(self.context_age_scale_s) or self.context_age_scale_s <= 0:
            raise ValueError("context_age_scale_s must be positive")
        if self.ready_timestamp < self.observation_timestamp:
            raise ValueError("ready_timestamp cannot precede observation_timestamp")
        if self.version < 0:
            raise ValueError("Slow packet version must be non-negative")
        if self.state_at_observation is not None:
            state = np.asarray(self.state_at_observation)
            if state.ndim != 1 or not np.all(np.isfinite(state)):
                raise ValueError("state_at_observation must be a finite 1D robot state")
        if self.intent_mask is not None and np.asarray(self.intent_mask).shape != intent.shape[:1]:
            raise ValueError(f"intent_mask must be [{intent.shape[0]}], got {np.shape(self.intent_mask)}")

    @property
    def horizon_end(self) -> float:
        return self.reference_start_timestamp + (self.reference_actions.shape[0] - 1) * self.action_period_s


class SlowReferenceCache:
    """Double-buffer-style atomic cache; readers never mix intent and actions."""

    def __init__(self):
        self._lock = threading.Lock()
        self._packet: SlowPacket | None = None

    def update(self, packet: SlowPacket) -> None:
        with self._lock:
            if self._packet is not None and (
                packet.observation_timestamp < self._packet.observation_timestamp
                or packet.version <= self._packet.version
            ):
                raise ValueError("Slow packets must have monotonic timestamps and strictly increasing versions")
            self._packet = packet

    def snapshot(self) -> SlowPacket:
        with self._lock:
            if self._packet is None:
                raise MissingSlowReferenceError("The slow VLA has not produced a reference packet yet")
            return self._packet


def sample_reference(
    packet: SlowPacket,
    timestamp: float,
    *,
    max_staleness_s: float = 0.25,
    discrete_dims: tuple[int, ...] = (6,),
) -> np.ndarray:
    """Sample the current reference while skipping actions made stale by Slow latency.

    Continuous action coordinates are linearly interpolated in model action
    space.  Discrete dimensions such as the gripper use zero-order hold.
    """
    if timestamp < packet.ready_timestamp:
        raise ValueError("Fast loop cannot consume a Slow packet before it is ready")
    if timestamp > packet.horizon_end + max_staleness_s:
        raise StaleSlowReferenceError(f"Slow reference expired by {timestamp - packet.horizon_end:.3f}s")
    position = max(0.0, (timestamp - packet.reference_start_timestamp) / packet.action_period_s)
    left = min(int(np.floor(position)), packet.reference_actions.shape[0] - 1)
    right = min(left + 1, packet.reference_actions.shape[0] - 1)
    alpha = float(np.clip(position - left, 0.0, 1.0))
    result = (1.0 - alpha) * packet.reference_actions[left] + alpha * packet.reference_actions[right]
    for dim in discrete_dims:
        if not 0 <= dim < result.shape[0]:
            raise ValueError(f"Discrete action dimension {dim} is out of range")
        result[dim] = packet.reference_actions[left, dim]
    return result


def reference_time_features(
    packet: SlowPacket,
    timestamp: float,
    *,
    context_age_scale_s: float | None = None,
) -> np.ndarray:
    """Return normalized [chunk_phase, visual-context age] for the Fast time token.

    The scale defaults to the one carried by the packet, which comes from the
    Slow cache the student was trained on. Override it only to probe a
    different scaling than training used.
    """
    if context_age_scale_s is None:
        context_age_scale_s = packet.context_age_scale_s
    elif context_age_scale_s <= 0:
        raise ValueError("context_age_scale_s must be positive")
    if timestamp < packet.ready_timestamp:
        raise ValueError("Fast loop cannot consume a Slow packet before it is ready")
    duration = max(packet.horizon_end - packet.reference_start_timestamp, packet.action_period_s)
    phase = np.clip((timestamp - packet.reference_start_timestamp) / duration, 0.0, 1.0)
    context_age = np.clip(
        (timestamp - packet.observation_timestamp) / context_age_scale_s,
        0.0,
        1.0,
    )
    return np.asarray([phase, context_age], dtype=np.float32)


def compose_reference_residual(
    reference_action,
    residual_pose,
    *,
    gate: float = 1.0,
    pose_dims: int = 6,
    residual_limit=None,
) -> np.ndarray:
    """Add only pose residuals; gripper and padding remain Slow-owned."""
    reference = np.asarray(reference_action)
    residual = np.asarray(residual_pose)
    if reference.ndim != 1 or residual.shape != (pose_dims,) or reference.shape[0] < pose_dims:
        raise ValueError("Expected reference [A] and residual [pose_dims]")
    correction = residual
    if residual_limit is not None:
        limit = np.broadcast_to(np.asarray(residual_limit), (pose_dims,))
        if np.any(limit < 0):
            raise ValueError("residual_limit must be non-negative")
        correction = np.clip(correction, -limit, limit)
    result = reference.copy()
    result[:pose_dims] += float(np.clip(gate, 0.0, 1.0)) * correction
    return result


def to_absolute_command(
    composed_action,
    packet: SlowPacket,
    unnormalize,
    *,
    delta_dims: int = 6,
) -> np.ndarray:
    """Turn a normalized composed action into an absolute command for the robot.

    This inverts the two training-side transforms in order: `Normalize`, then
    `DeltaActions`. The delta dimensions are rebased onto the state the Slow packet
    was conditioned on rather than the state at execution time; the gripper is
    absolute already and is left untouched.
    """
    if packet.state_at_observation is None:
        raise ValueError("Undoing delta actions needs the state the Slow packet was conditioned on")
    action = np.asarray(composed_action, dtype=np.float32)
    if action.ndim != 1 or action.shape[0] < delta_dims:
        raise ValueError(f"Expected a 1D action with at least {delta_dims} dimensions, got {action.shape}")
    physical = np.asarray(unnormalize(action), dtype=np.float32)
    if physical.shape != action.shape:
        raise ValueError(f"Unnormalize changed the action shape from {action.shape} to {physical.shape}")
    state = np.asarray(packet.state_at_observation, dtype=np.float32)
    if state.shape[0] < delta_dims:
        raise ValueError(f"state_at_observation needs at least {delta_dims} dimensions, got {state.shape}")
    physical[:delta_dims] += state[:delta_dims]
    return physical
