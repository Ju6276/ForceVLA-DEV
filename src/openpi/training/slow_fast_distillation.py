"""Paired Teacher targets and losses for slow/fast policy distillation."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class PairedTeacherTargets:
    full_actions: jnp.ndarray
    nominal_actions: jnp.ndarray
    residual_pose: jnp.ndarray


@dataclasses.dataclass(frozen=True)
class FastChunkTargets:
    """Teacher targets for the short chunk Fast emits at one observation timestamp.

    `staleness_pose` is the only field that depends on which Slow packet the fast loop
    happens to be riding: the part of the deployed error that exists purely because the
    cached context is older than the force and state it is being combined with. It is
    `A_null(t) - A_ref(t) + (sigma_state/sigma_action) * (S_t - S_key)`. The trailing
    term is not optional. `A_ref` is a delta from the state of the row the Slow packet
    was keyed on while `A_null(t)` is a delta from the current row's state, so without
    it the difference is measured between two different origins and its rotation part
    is almost entirely wrong.
    """

    full_action: jnp.ndarray
    nominal_action: jnp.ndarray
    residual_pose: jnp.ndarray
    staleness_pose: jnp.ndarray | None = None


@dataclasses.dataclass(frozen=True)
class FastDistillationLossConfig:
    reference_dim: int = 10
    state_dim: int = 10
    pose_dims: int = 9
    residual_weight: float = 1.0
    reconstruction_weight: float = 1.0
    # Weight on the stale-reference correction. This is the term that makes training
    # match asynchronous deployment: without it nothing in the objective depends on how
    # old the cached context is, and the time features are free to go unused.
    staleness_weight: float = 1.0
    # Later steps of the emitted chunk only execute when Fast stalls or runs slower
    # than the action rate, so they are supervised but down-weighted relative to the
    # step that runs in the common case.
    step_decay: float = 0.5

    def __post_init__(self) -> None:
        if self.pose_dims <= 0:
            raise ValueError("pose_dims must be positive")
        if min(self.residual_weight, self.reconstruction_weight, self.staleness_weight) < 0:
            raise ValueError("Distillation loss weights must be non-negative")
        if not 0 < self.step_decay <= 1:
            raise ValueError("step_decay must be in (0, 1]")

    def step_weights(self, chunk_steps: int) -> jnp.ndarray:
        """Normalized per-step weights, so the loss scale does not depend on chunk length."""
        weights = self.step_decay ** jnp.arange(chunk_steps, dtype=jnp.float32)
        return weights / jnp.sum(weights)


def make_paired_teacher_targets(full_actions, null_actions, *, pose_dims: int = 9) -> PairedTeacherTargets:
    """Define nominal and Teacher-specific force-induced deviation."""
    if full_actions.shape != null_actions.shape or full_actions.ndim != 3:
        raise ValueError(f"Expected matched [B,H,D] Teacher actions, got {full_actions.shape} and {null_actions.shape}")
    if not 0 < pose_dims <= full_actions.shape[-1]:
        raise ValueError("pose_dims must fit inside the Teacher action dimension")
    return PairedTeacherTargets(
        full_actions=full_actions,
        nominal_actions=null_actions,
        residual_pose=full_actions[..., :pose_dims] - null_actions[..., :pose_dims],
    )


def select_chunk_targets(targets: PairedTeacherTargets, *, chunk_steps: int) -> FastChunkTargets:
    """Take the leading `chunk_steps` of the Teacher chunk as Fast's supervision.

    This is causal even though the later steps lie in the future: the Teacher
    produced its whole chunk from the single observation at this row, so step `k`
    is the Teacher's own answer to "what does the force I can see now imply for
    `k` action periods from now". No future force or state enters the label.
    """
    horizon = targets.full_actions.shape[1]
    if not 0 < chunk_steps <= horizon:
        raise ValueError(f"chunk_steps must be in (0, {horizon}], got {chunk_steps}")
    return FastChunkTargets(
        full_action=targets.full_actions[:, :chunk_steps],
        nominal_action=targets.nominal_actions[:, :chunk_steps],
        residual_pose=targets.residual_pose[:, :chunk_steps],
    )


def sample_paired_teacher_actions(model, rng, observation, *, num_steps: int = 10, pose_dims: int = 9):
    """Run full/null with identical diffusion noise and sampling settings."""
    if hasattr(model, "sample_paired_actions_and_context"):
        full, null, prefix_context, prefix_mask = model.sample_paired_actions_and_context(
            rng, observation, num_steps=num_steps
        )
    else:
        full = model.sample_actions_for_force_condition(rng, observation, force_condition="full", num_steps=num_steps)
        null = model.sample_actions_for_force_condition(rng, observation, force_condition="null", num_steps=num_steps)
        prefix_context = None
        prefix_mask = None
    return make_paired_teacher_targets(full, null, pose_dims=pose_dims), prefix_context, prefix_mask


def slow_nominal_loss(predicted_actions, targets: PairedTeacherTargets, *, physical_dims: int = 10):
    if predicted_actions.shape != targets.nominal_actions.shape:
        raise ValueError("Slow prediction and nominal Teacher target must have identical shapes")
    if not 0 < physical_dims <= predicted_actions.shape[-1]:
        raise ValueError("physical_dims must fit inside the action dimension")
    return jnp.mean(jnp.square(predicted_actions[..., :physical_dims] - targets.nominal_actions[..., :physical_dims]))


def fast_residual_loss(
    predicted_residual,
    slow_reference,
    targets: FastChunkTargets,
    config: FastDistillationLossConfig,
    *,
    predicted_staleness=None,
):
    """Supervise each correction on its own target, and their sum on the executed action.

    The reconstruction term is what ties the two heads to what the robot actually runs,
    `A_ref(t) + force + staleness`. It is redundant only if both heads were exact; with
    a weight on it the heads can trade off their individual errors so the composition
    lands closer than either target alone would force it to.
    """
    expected = targets.residual_pose.shape
    if len(expected) != 3:
        raise ValueError(f"Expected chunk residual [B,K,D], got {expected}")
    if predicted_residual.shape != expected:
        raise ValueError(f"Expected residual prediction {expected}, got {predicted_residual.shape}")
    if (
        slow_reference.shape[:2] != expected[:2]
        or slow_reference.ndim != 3
        or slow_reference.shape[-1] < config.pose_dims
    ):
        raise ValueError("Slow reference must be [B,K,A] and contain all pose dimensions")
    if (predicted_staleness is None) != (targets.staleness_pose is None):
        raise ValueError("A staleness prediction requires a staleness target, and vice versa")
    if predicted_staleness is not None:
        if predicted_staleness.shape != expected:
            raise ValueError(f"Expected staleness prediction {expected}, got {predicted_staleness.shape}")
        if targets.staleness_pose.shape != expected:
            raise ValueError(f"Expected staleness target {expected}, got {targets.staleness_pose.shape}")

    weights = config.step_weights(expected[1])[None, :, None]
    scale = expected[0] * expected[2]
    residual_error = jnp.square(predicted_residual - targets.residual_pose)
    residual_loss = jnp.sum(residual_error * weights) / scale

    correction = predicted_residual if predicted_staleness is None else predicted_residual + predicted_staleness
    reconstructed = slow_reference[..., : config.pose_dims] + correction
    reconstruction_error = jnp.square(reconstructed - targets.full_action[..., : config.pose_dims])
    reconstruction_loss = jnp.sum(reconstruction_error * weights) / scale

    total = config.residual_weight * residual_loss + config.reconstruction_weight * reconstruction_loss
    metrics = {
        "residual_loss": residual_loss,
        "reconstruction_loss": reconstruction_loss,
        # The step actually executed in steady state, reported unweighted so it is
        # comparable to the old single-step numbers and to the zero-residual baseline.
        "residual_loss_step0": jnp.mean(residual_error[:, 0]),
        "reconstruction_loss_step0": jnp.mean(reconstruction_error[:, 0]),
    }
    if predicted_staleness is not None:
        staleness_error = jnp.square(predicted_staleness - targets.staleness_pose)
        staleness_loss = jnp.sum(staleness_error * weights) / scale
        total = total + config.staleness_weight * staleness_loss
        metrics["staleness_loss"] = staleness_loss
        metrics["staleness_loss_step0"] = jnp.mean(staleness_error[:, 0])
    return total, metrics
