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
            # Undoing DeltaActions subtracts state and action dimension by dimension, so
            # the two must share a layout. Rejecting a mismatch catches the deployment
            # mistake of passing the raw xyz+rpy+gripper+force state: it is long enough
            # to index but its dimension 3 is roll, where the model expects 6D column 0.
            if state.shape[0] != reference.shape[-1]:
                raise ValueError(
                    f"state_at_observation is {state.shape[0]}D but the action space is "
                    f"{reference.shape[-1]}D; pass the converted xyz+6D+gripper state, not the raw one"
                )
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
    steps: int = 1,
    max_staleness_s: float = 0.25,
    discrete_dims: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Sample `steps` consecutive references starting at `timestamp`.

    Step `k` lands `k` action periods after the tick, matching the spacing of the
    residual chunk Fast emits. Continuous action coordinates are linearly
    interpolated in model action space; discrete dimensions such as the gripper use
    zero-order hold.

    `discrete_dims` defaults to the final action dimension, which is where the
    offline rollout holds the gripper. Hardcoding an index instead silently breaks
    whenever the pose layout changes: index 6 is the gripper under xyz+rpy but a
    rotation component under xyz+6D.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    if timestamp < packet.ready_timestamp:
        raise ValueError("Fast loop cannot consume a Slow packet before it is ready")
    if timestamp > packet.horizon_end + max_staleness_s:
        raise StaleSlowReferenceError(f"Slow reference expired by {timestamp - packet.horizon_end:.3f}s")
    base = max(0.0, (timestamp - packet.reference_start_timestamp) / packet.action_period_s)
    horizon = packet.reference_actions.shape[0]
    position = base + np.arange(steps, dtype=np.float64)
    left = np.minimum(np.floor(position).astype(np.int64), horizon - 1)
    right = np.minimum(left + 1, horizon - 1)
    alpha = np.clip(position - left, 0.0, 1.0)[:, None]
    result = (1.0 - alpha) * packet.reference_actions[left] + alpha * packet.reference_actions[right]
    if discrete_dims is None:
        discrete_dims = (result.shape[1] - 1,)
    for dim in discrete_dims:
        if not 0 <= dim < result.shape[1]:
            raise ValueError(f"Discrete action dimension {dim} is out of range")
        result[:, dim] = packet.reference_actions[left, dim]
    return result


def reference_time_features(
    packet: SlowPacket,
    timestamp: float,
    *,
    context_age_scale_s: float | None = None,
) -> np.ndarray:
    """Return the Fast time token, `[context age / scale, interpolation alpha]`.

    Chunk phase is deliberately absent. Phase and age are the same elapsed time
    divided by two constants, so pairing them gives the student one dimension of
    information dressed up as two. Alpha is the fractional position between the two
    Teacher waypoints being interpolated, which age does not determine.

    The scale defaults to the one carried by the packet, which comes from the Slow
    cache the student was trained on. Override it only to probe a different scaling
    than training used.
    """
    if context_age_scale_s is None:
        context_age_scale_s = packet.context_age_scale_s
    elif context_age_scale_s <= 0:
        raise ValueError("context_age_scale_s must be positive")
    if timestamp < packet.ready_timestamp:
        raise ValueError("Fast loop cannot consume a Slow packet before it is ready")
    context_age = np.clip(
        (timestamp - packet.observation_timestamp) / context_age_scale_s,
        0.0,
        1.0,
    )
    position = max(0.0, (timestamp - packet.reference_start_timestamp) / packet.action_period_s)
    alpha = np.clip(position - min(np.floor(position), packet.reference_actions.shape[0] - 1), 0.0, 1.0)
    return np.asarray([context_age, alpha], dtype=np.float32)


def compose_reference_residual(
    reference_actions,
    residual_pose,
    *,
    gate: float = 1.0,
    pose_dims: int = 9,
    residual_limit=None,
) -> np.ndarray:
    """Add only pose residuals; gripper and padding remain Slow-owned."""
    reference = np.asarray(reference_actions)
    residual = np.asarray(residual_pose)
    if reference.ndim != 2 or residual.ndim != 2 or reference.shape[-1] < pose_dims:
        raise ValueError("Expected reference [K,A] and residual [K,pose_dims]")
    if residual.shape != (reference.shape[0], pose_dims):
        raise ValueError(f"Expected residual {(reference.shape[0], pose_dims)}, got {residual.shape}")
    correction = residual
    if residual_limit is not None:
        limit = np.broadcast_to(np.asarray(residual_limit), (pose_dims,))
        if np.any(limit < 0):
            raise ValueError("residual_limit must be non-negative")
        correction = np.clip(correction, -limit, limit)
    result = reference.copy()
    result[:, :pose_dims] += float(np.clip(gate, 0.0, 1.0)) * correction
    return result


def to_absolute_command(
    composed_actions,
    packet: SlowPacket,
    unnormalize,
    *,
    delta_dims: int = 9,
) -> np.ndarray:
    """Turn a normalized composed chunk into absolute commands for the robot.

    This inverts the two training-side transforms in the same order the training
    output chain uses: `Normalize`, then `DeltaActions`. The result is still in the
    model's xyz+6D+gripper space; converting the rotation back to the robot's Euler
    angles is the caller's last step, after the delta has been applied. Doing it in
    that order is the point of the 6D encoding, since the addition happens where
    there is no branch cut.

    The delta dimensions are rebased onto the state the Slow packet was conditioned
    on rather than the state at execution time; the gripper is absolute already and
    is left untouched.
    """
    if packet.state_at_observation is None:
        raise ValueError("Undoing delta actions needs the state the Slow packet was conditioned on")
    actions = np.asarray(composed_actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[-1] < delta_dims:
        raise ValueError(f"Expected a [K,A] chunk with at least {delta_dims} dimensions, got {actions.shape}")
    state = np.asarray(packet.state_at_observation, dtype=np.float32)
    if state.shape[0] != actions.shape[-1]:
        raise ValueError(
            f"state_at_observation is {state.shape[0]}D but the composed action is {actions.shape[-1]}D; "
            "both must be in the model's action layout for the delta to be undone dimension by dimension"
        )
    physical = np.stack([np.asarray(unnormalize(action), dtype=np.float32) for action in actions])
    if physical.shape != actions.shape:
        raise ValueError(f"Unnormalize changed the action shape from {actions.shape} to {physical.shape}")
    physical[:, :delta_dims] += state[:delta_dims]
    return physical
