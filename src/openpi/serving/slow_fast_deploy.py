"""Bridge from a live Slow VLA call to the packet the Fast student was trained on.

There is no offline cache at deployment. `scripts/extract_slow_cache.py` writes an
npz so that Fast *trains* on the same packet structure it will meet at runtime; on
the robot the packets come from actually running the Teacher's null path in the
Slow thread. This module owns the three post-processing steps that the offline
extraction applies and that a hand-written deployment loop silently gets wrong:

- the action chunk is truncated to the model's xyz+6D+gripper dimensions;
- the vision-language prefix is mean-pooled into the same number of bins;
- the robot state is rewritten into 6D before it is used to undo `DeltaActions`;
- the state and the force window are normalized with the training statistics.

The shapes are read from the artifact Fast was trained against, so they cannot
drift away from it.
"""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import json
import pathlib

import numpy as np

from openpi.policies import rotation_6d as rot
from openpi.serving import slow_fast_loop
from openpi.training import fast_dataset


@dataclasses.dataclass(frozen=True)
class DeploymentContract:
    """Everything the runtime must copy from the Slow cache Fast was trained on."""

    context_tokens: int
    action_dims: int
    chunk_steps: int
    action_period_s: float
    context_age_scale_s: float
    # None when the trained band is unknown, which is the normal state of a train
    # cache: it is extracted full-rate so training can redraw the timing itself.
    slow_rate_range_hz: tuple[float, float] | None
    slow_latency_range_s: tuple[float, float] | None
    # Whether the Fast run emits a stale-reference correction alongside the force
    # residual. None when no Fast run was supplied, so it could not be read.
    predicts_staleness: bool | None = None

    def slow_fast_config(self, **overrides) -> slow_fast_loop.SlowFastConfig:
        settings = {
            "action_period_s": self.action_period_s,
            "state_dims": self.action_dims,
            "pose_dims": rot.POSE_DIMS,
            "delta_dims": rot.POSE_DIMS,
            "chunk_steps": self.chunk_steps,
        }
        return slow_fast_loop.SlowFastConfig(**{**settings, **overrides})

    def check_measured_timing(self, *, slow_rate_hz: float, slow_latency_s: float) -> list[str]:
        """Report the measured Slow timing that falls outside the trained band.

        Outside the band the context-age token saturates or leaves the range the
        student ever saw, so the correction it applies is extrapolation.
        """
        if self.slow_rate_range_hz is None or self.slow_latency_range_s is None:
            raise ValueError(
                "This contract carries no trained timing band. A full-rate train cache "
                "records none, because the band lives in the Fast training run that "
                "redraws the timing. Pass fast_run= to load_contract()."
            )
        problems = []
        low, high = self.slow_rate_range_hz
        if not low <= slow_rate_hz <= high:
            problems.append(f"measured Slow rate {slow_rate_hz:.2f} Hz is outside the trained band {low}-{high} Hz")
        low, high = self.slow_latency_range_s
        if not low <= slow_latency_s <= high:
            problems.append(
                f"measured Slow latency {slow_latency_s * 1000:.0f} ms is outside the trained band "
                f"{low * 1000:.0f}-{high * 1000:.0f} ms"
            )
        return problems


def _band(values, scale: float = 1.0) -> tuple[float, float] | None:
    """Read a recorded band, treating a degenerate or absent one as unknown."""
    if not values:
        return None
    low, high = float(values[0]) * scale, float(values[1]) * scale
    return None if low <= 0 and high <= 0 else (low, high)


def load_contract(slow_cache: str | pathlib.Path, *, fast_run: str | pathlib.Path | None = None) -> DeploymentContract:
    """Read the deployment contract from the summary written beside a Slow cache.

    The summary is used rather than the npz because a full-rate cache holds every
    pooled context and is far too large to load on the robot just to read shapes.

    Pass `fast_run` (the Fast training output directory) to take the timing band
    from the run that redrew it. Without it the band comes from the cache, which is
    only meaningful for a cache extracted at a fixed rate.
    """
    path = pathlib.Path(slow_cache).with_suffix(".json")
    if not path.is_file():
        raise FileNotFoundError(
            f"No Slow-cache summary at {path}. It is written next to the npz by "
            "extract_slow_cache.py and resample_slow_cache.py, and carries the shapes "
            "and timing the Fast student was trained on."
        )
    summary = json.loads(path.read_text())
    try:
        _, context_tokens, _ = summary["pooled_context_shape"]
        timing = summary
        predicts_staleness = None
        if fast_run is not None:
            run = pathlib.Path(fast_run)
            metadata = json.loads((run if run.suffix == ".json" else run / "metadata.json").read_text())
            timing = metadata.get("trained_timing")
            if not timing:
                raise ValueError(f"{run} records no trained_timing; it predates timing randomization")
            # Absent in format_version 1, where there was only one head.
            predicts_staleness = bool(metadata.get("predict_staleness", False))
        contract = DeploymentContract(
            predicts_staleness=predicts_staleness,
            context_tokens=int(context_tokens),
            action_dims=int(summary["action_chunk_shape"][-1]),
            chunk_steps=int(summary["chunk_steps"]),
            action_period_s=1.0 / float(summary["action_rate_hz"]),
            context_age_scale_s=float(summary["context_age_scale_ms"]) / 1000.0,
            slow_rate_range_hz=_band(timing.get("slow_rate_range_hz")),
            slow_latency_range_s=_band(timing.get("slow_latency_range_ms"), scale=1e-3),
        )
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise ValueError(f"{path} is not a Slow-cache summary this runtime understands: {error}") from error
    if contract.action_dims != rot.ROBOT_DIMS:
        raise ValueError(
            f"The Slow cache stores {contract.action_dims}D actions but the robot interface expects "
            f"{rot.ROBOT_DIMS}D xyz+6D+gripper"
        )
    return contract


@dataclasses.dataclass(frozen=True)
class SlowPacketBuilder:
    """Post-process one raw Teacher null-path call exactly as the extraction does."""

    contract: DeploymentContract

    def __call__(self, actions, prefix_context, prefix_mask):
        """Map `sample_nominal_actions_and_context` output to `SlowWorker.infer` output.

        `actions` is `[1, H, D]`, `prefix_context` is `[1, S, W]` and `prefix_mask`
        is `[1, S]`, i.e. the batch-of-one the Teacher returns.
        """
        chunk = np.asarray(actions, dtype=np.float32)
        context = np.asarray(prefix_context)
        mask = np.asarray(prefix_mask, dtype=np.bool_)
        if chunk.ndim != 3 or context.ndim != 3 or mask.ndim != 2:
            raise ValueError(
                f"Expected batched Teacher output [1,H,D], [1,S,W] and [1,S], got "
                f"{chunk.shape}, {context.shape} and {mask.shape}"
            )
        if chunk.shape[0] != 1 or context.shape[0] != 1:
            raise ValueError("The Slow thread runs one observation at a time")
        if chunk.shape[-1] < self.contract.action_dims:
            raise ValueError(
                f"The Teacher produced {chunk.shape[-1]}D actions but the contract needs {self.contract.action_dims}D"
            )
        pooled, pooled_mask = fast_dataset.pool_context_tokens(context, mask, num_tokens=self.contract.context_tokens)
        return (
            chunk[0, :, : self.contract.action_dims],
            np.asarray(pooled[0], dtype=np.float32),
            np.asarray(pooled_mask[0], dtype=np.bool_),
        )


def load_normalizers(norm_stats) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    """Build the deployment normalizers from the norm stats the model was trained with.

    `state` and `force_history` are both normalized inside the training transform
    pipeline, so the runtime has to apply the same affine map before the student
    sees either of them. Reproducing `transforms.Normalize` here rather than
    reusing it keeps the runtime free of the tree-structured data dict.
    """
    missing = [key for key in ("state", "force_history", "actions") if key not in norm_stats]
    if missing:
        raise ValueError(f"The norm stats are missing {missing}, which the runtime normalizes")

    def affine(key: str, *, invert: bool) -> Callable[[np.ndarray], np.ndarray]:
        mean = np.asarray(norm_stats[key].mean, dtype=np.float32)
        std = np.asarray(norm_stats[key].std, dtype=np.float32)

        def apply(value) -> np.ndarray:
            array = np.asarray(value, dtype=np.float32)
            width = array.shape[-1]
            if width > mean.shape[-1]:
                raise ValueError(f"{key} is {width}D but its norm stats only cover {mean.shape[-1]}D")
            # The stats are stored at the padded model width while the runtime works
            # at the robot width. Normalization is elementwise, so the leading slice
            # of the padded map is the same affine transform.
            offset, scale = mean[:width], std[:width] + 1e-6
            return array * scale + offset if invert else (array - offset) / scale

        return apply

    return {
        "normalize_state": affine("state", invert=False),
        "normalize_force_history": affine("force_history", invert=False),
        "unnormalize_action": affine("actions", invert=True),
    }


def convert_robot_state(raw_state) -> np.ndarray:
    """Rewrite the robot's xyz+rpy+gripper[+force] state into the model's 6D layout.

    The Slow packet carries this state to undo `DeltaActions`, which adds it to the
    action dimension by dimension. Handing over the raw state instead lines roll up
    with the first 6D column, which is wrong but numerically unremarkable.
    """
    converted = np.asarray(rot.convert_state(np.asarray(raw_state, dtype=np.float32)), dtype=np.float32)
    return converted[: rot.ROBOT_DIMS]
