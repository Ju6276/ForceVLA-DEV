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
class SingleStepTeacherTargets:
    """Teacher targets at one real observation/action timestamp."""

    full_action: jnp.ndarray
    nominal_action: jnp.ndarray
    residual_pose: jnp.ndarray


@dataclasses.dataclass(frozen=True)
class FastDistillationLossConfig:
    pose_dims: int = 6
    residual_weight: float = 1.0
    reconstruction_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.pose_dims <= 0:
            raise ValueError("pose_dims must be positive")
        if min(self.residual_weight, self.reconstruction_weight) < 0:
            raise ValueError("Distillation loss weights must be non-negative")


def make_paired_teacher_targets(full_actions, null_actions, *, pose_dims: int = 6) -> PairedTeacherTargets:
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


def select_single_step_targets(targets: PairedTeacherTargets, *, step_index: int = 0) -> SingleStepTeacherTargets:
    """Select the action aligned to the current Fast tick from Teacher chunks.

    For the current 30 Hz action-supervised dataset this should normally be
    index zero.  Future chunk entries are not treated as high-rate labels:
    their force/state observations have not happened yet.
    """
    horizon = targets.full_actions.shape[1]
    if not 0 <= step_index < horizon:
        raise ValueError(f"step_index must be in [0, {horizon}), got {step_index}")
    return SingleStepTeacherTargets(
        full_action=targets.full_actions[:, step_index],
        nominal_action=targets.nominal_actions[:, step_index],
        residual_pose=targets.residual_pose[:, step_index],
    )


def sample_paired_teacher_actions(model, rng, observation, *, num_steps: int = 10, pose_dims: int = 6):
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


def slow_nominal_loss(predicted_actions, targets: PairedTeacherTargets, *, physical_dims: int = 7):
    if predicted_actions.shape != targets.nominal_actions.shape:
        raise ValueError("Slow prediction and nominal Teacher target must have identical shapes")
    if not 0 < physical_dims <= predicted_actions.shape[-1]:
        raise ValueError("physical_dims must fit inside the action dimension")
    return jnp.mean(jnp.square(predicted_actions[..., :physical_dims] - targets.nominal_actions[..., :physical_dims]))


def fast_residual_loss(
    predicted_residual,
    slow_reference,
    targets: SingleStepTeacherTargets,
    config: FastDistillationLossConfig,
):
    """Preserve residual specialization while also reconstructing full actions."""
    expected = targets.residual_pose.shape
    if len(expected) != 2:
        raise ValueError(f"Expected single-step residual [B,D], got {expected}")
    if predicted_residual.shape != expected:
        raise ValueError(f"Expected residual prediction {expected}, got {predicted_residual.shape}")
    if (
        slow_reference.shape[0] != expected[0]
        or slow_reference.ndim != 2
        or slow_reference.shape[-1] < config.pose_dims
    ):
        raise ValueError("Slow reference must be [B,A] and contain all pose dimensions")
    residual_loss = jnp.mean(jnp.square(predicted_residual - targets.residual_pose))
    reconstructed = slow_reference[..., : config.pose_dims] + predicted_residual
    reconstruction_loss = jnp.mean(jnp.square(reconstructed - targets.full_action[..., : config.pose_dims]))
    total = config.residual_weight * residual_loss + config.reconstruction_weight * reconstruction_loss
    return total, {
        "residual_loss": residual_loss,
        "reconstruction_loss": reconstruction_loss,
    }
