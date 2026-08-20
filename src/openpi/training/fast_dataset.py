"""Offline arrays for the selected Slow-to-Fast distillation pipeline."""

from __future__ import annotations

import dataclasses
import itertools
import json
import pathlib

import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class Stage3FastArrays:
    dataset_indices: np.ndarray
    episode_indices: np.ndarray
    timestamps: np.ndarray
    state: np.ndarray
    force_history: np.ndarray
    force_history_mask: np.ndarray
    full_pose: np.ndarray
    residual_pose: np.ndarray


@dataclasses.dataclass(frozen=True)
class SlowCache:
    key_dataset_indices: np.ndarray
    key_episode_indices: np.ndarray
    key_timestamps: np.ndarray
    context_tokens: np.ndarray
    context_mask: np.ndarray
    action_chunks: np.ndarray
    row_key_positions: np.ndarray
    reference_actions: np.ndarray
    time_features: np.ndarray


def load_stage3_fast_arrays(target_dir: str | pathlib.Path) -> Stage3FastArrays:
    """Load only the single-step fields needed by Fast from Stage-3 shards."""
    root = pathlib.Path(target_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest.get("complete", False):
        raise ValueError(f"Stage-3 target extraction is incomplete: {root}")

    fields: dict[str, list[np.ndarray]] = {
        "dataset_indices": [],
        "episode_indices": [],
        "timestamps": [],
        "state": [],
        "force_history": [],
        "force_history_mask": [],
        "full_pose": [],
        "residual_pose": [],
    }
    next_index = 0
    for shard_info in manifest["shards"]:
        if not shard_info.get("complete", False):
            raise ValueError(f"Incomplete Stage-3 shard: {shard_info}")
        with np.load(root / shard_info["path"], allow_pickle=False) as shard:
            indices = np.asarray(shard["dataset_indices"], dtype=np.int64)
            stop = next_index + len(indices)
            np.testing.assert_array_equal(indices, np.arange(next_index, stop, dtype=np.int64))
            fields["dataset_indices"].append(indices)
            fields["episode_indices"].append(np.asarray(shard["episode_indices"], dtype=np.int64))
            fields["timestamps"].append(np.asarray(shard["timestamps"], dtype=np.float64))
            fields["state"].append(np.asarray(shard["normalized_state"][:, :7], dtype=np.float32))
            fields["force_history"].append(np.asarray(shard["normalized_force_history"], dtype=np.float32))
            fields["force_history_mask"].append(np.asarray(shard["force_history_mask"], dtype=np.bool_))
            fields["full_pose"].append(np.asarray(shard["normalized_full_actions"][:, 0, :6], dtype=np.float32))
            fields["residual_pose"].append(
                np.asarray(shard["normalized_pose_residual"][:, 0, :6], dtype=np.float32)
            )
        next_index = stop

    expected = int(manifest["extraction_size"])
    if next_index != expected:
        raise ValueError(f"Expected {expected} Stage-3 rows, loaded {next_index}")
    arrays = {key: np.concatenate(parts, axis=0) for key, parts in fields.items()}
    return Stage3FastArrays(**arrays)


def select_slow_update_rows(
    episode_indices: np.ndarray,
    timestamps: np.ndarray,
    *,
    update_period_s: float,
    tolerance_s: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Select causal Slow updates and map every row to its latest Slow packet."""
    episodes = np.asarray(episode_indices, dtype=np.int64)
    times = np.asarray(timestamps, dtype=np.float64)
    if episodes.ndim != 1 or times.shape != episodes.shape or len(times) == 0:
        raise ValueError("episode_indices and timestamps must be non-empty 1D arrays")
    if update_period_s <= 0 or tolerance_s < 0:
        raise ValueError("Slow update period must be positive and tolerance non-negative")

    key_rows = []
    row_key_positions = np.empty(len(times), dtype=np.int64)
    current_episode = None
    latest_key_time = None
    key_position = -1
    previous_time = None
    for row, (episode, timestamp) in enumerate(zip(episodes, times, strict=True)):
        if episode != current_episode:
            current_episode = int(episode)
            previous_time = None
            latest_key_time = None
        elif previous_time is not None and timestamp + tolerance_s < previous_time:
            raise ValueError(f"Timestamps are not monotonic inside episode {episode}")

        if latest_key_time is None or timestamp + tolerance_s >= latest_key_time + update_period_s:
            key_rows.append(row)
            key_position += 1
            latest_key_time = float(timestamp)
        row_key_positions[row] = key_position
        previous_time = float(timestamp)
    return np.asarray(key_rows, dtype=np.int64), row_key_positions


def pool_context_tokens(hidden, mask, *, num_tokens: int):
    """Deterministically compress a long Slow V-L prefix into contiguous bins."""
    if hidden.ndim != 3 or mask.shape != hidden.shape[:-1]:
        raise ValueError(f"Expected hidden [B,S,D] and mask [B,S], got {hidden.shape} and {mask.shape}")
    if not 0 < num_tokens <= hidden.shape[1]:
        raise ValueError("num_tokens must be in [1, prefix_length]")
    boundaries = np.linspace(0, hidden.shape[1], num_tokens + 1, dtype=np.int64)
    pooled = []
    pooled_mask = []
    for left, right in itertools.pairwise(boundaries):
        segment_mask = mask[:, left:right]
        weights = segment_mask[..., None].astype(hidden.dtype)
        value = jnp.sum(hidden[:, left:right] * weights, axis=1) / jnp.maximum(jnp.sum(weights, axis=1), 1)
        valid = jnp.any(segment_mask, axis=1)
        pooled.append(jnp.where(valid[:, None], value, 0))
        pooled_mask.append(valid)
    return jnp.stack(pooled, axis=1), jnp.stack(pooled_mask, axis=1)


def build_reference_rollout(
    action_chunks: np.ndarray,
    row_key_positions: np.ndarray,
    row_timestamps: np.ndarray,
    key_timestamps: np.ndarray,
    *,
    action_period_s: float,
    context_age_scale_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample cached Slow chunks at row timestamps and return [phase, age]."""
    chunks = np.asarray(action_chunks, dtype=np.float32)
    mapping = np.asarray(row_key_positions, dtype=np.int64)
    row_times = np.asarray(row_timestamps, dtype=np.float64)
    key_times = np.asarray(key_timestamps, dtype=np.float64)
    if chunks.ndim != 3 or chunks.shape[-1] < 7:
        raise ValueError(f"Expected Slow chunks [K,H,>=7], got {chunks.shape}")
    if mapping.ndim != 1 or row_times.shape != mapping.shape:
        raise ValueError("row_key_positions and row_timestamps must be matching 1D arrays")
    if len(key_times) != len(chunks) or np.any(mapping < 0) or np.any(mapping >= len(chunks)):
        raise ValueError("Slow key arrays and row mapping are inconsistent")
    if action_period_s <= 0 or context_age_scale_s <= 0:
        raise ValueError("Action period and context-age scale must be positive")

    age = np.maximum(row_times - key_times[mapping], 0.0)
    position = age / action_period_s
    left = np.minimum(np.floor(position).astype(np.int64), chunks.shape[1] - 1)
    right = np.minimum(left + 1, chunks.shape[1] - 1)
    alpha = np.clip(position - left, 0.0, 1.0).astype(np.float32)
    left_action = chunks[mapping, left, :7]
    right_action = chunks[mapping, right, :7]
    reference = (1.0 - alpha[:, None]) * left_action + alpha[:, None] * right_action
    # The gripper is not a continuous pose coordinate.
    reference[:, 6] = left_action[:, 6]

    horizon_duration = max((chunks.shape[1] - 1) * action_period_s, action_period_s)
    phase = np.clip(age / horizon_duration, 0.0, 1.0)
    normalized_age = np.clip(age / context_age_scale_s, 0.0, 1.0)
    time_features = np.stack([phase, normalized_age], axis=-1).astype(np.float32)
    return reference.astype(np.float32), time_features


def load_slow_cache(path: str | pathlib.Path, *, expected_rows: int | None = None) -> SlowCache:
    with np.load(path, allow_pickle=False) as cache:
        result = SlowCache(**{field.name: np.asarray(cache[field.name]) for field in dataclasses.fields(SlowCache)})
    row_count = len(result.row_key_positions)
    if expected_rows is not None and row_count != expected_rows:
        raise ValueError(f"Expected {expected_rows} Slow-cache rows, got {row_count}")
    key_count = len(result.action_chunks)
    if any(
        len(array) != key_count
        for array in (
            result.key_dataset_indices,
            result.key_episode_indices,
            result.key_timestamps,
            result.context_tokens,
            result.context_mask,
        )
    ):
        raise ValueError("All Slow packet arrays must have the same number of keys")
    if len(result.reference_actions) != row_count or len(result.time_features) != row_count:
        raise ValueError("Slow row mapping, references, and time features must have the same number of rows")
    if np.any(result.row_key_positions < 0) or np.any(result.row_key_positions >= key_count):
        raise ValueError("Slow row mapping contains an invalid key position")
    return result
