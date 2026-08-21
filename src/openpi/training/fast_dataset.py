"""Offline arrays for the selected Slow-to-Fast distillation pipeline."""

from __future__ import annotations

import dataclasses
import itertools
import json
import pathlib

import jax.numpy as jnp
import numpy as np

from openpi.models import slow_fast
from openpi.policies import rotation_6d as rot


@dataclasses.dataclass(frozen=True)
class Stage3FastArrays:
    """Stage-3 rows reduced to what Fast consumes.

    The pose fields are `[N, chunk_steps, pose_dims]`. Keeping more than the first
    step is not a future-force leak: the Teacher produced its entire chunk from the
    single observation at the row, so step `k` is already "what the force at this
    row implies for `k` action periods later".
    """

    dataset_indices: np.ndarray
    episode_indices: np.ndarray
    timestamps: np.ndarray
    state: np.ndarray
    force_history: np.ndarray
    force_history_mask: np.ndarray
    full_pose: np.ndarray
    residual_pose: np.ndarray
    expert_pose: np.ndarray

    @property
    def null_pose(self) -> np.ndarray:
        return self.full_pose - self.residual_pose

    @property
    def chunk_steps(self) -> int:
        return int(self.residual_pose.shape[1])


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
    # Carried with the cache so that serving can reproduce the exact time token
    # scaling the student was trained on.
    context_age_scale_s: float = slow_fast.DEFAULT_CONTEXT_AGE_SCALE_S
    action_period_s: float = 1.0 / 30.0
    # One serving latency per Slow packet. Latency is randomized over a band at
    # extraction, so a single scalar cannot describe the cache.
    key_ready_delays: np.ndarray | None = None
    row_ready: np.ndarray | None = None

    @property
    def chunk_steps(self) -> int:
        return int(self.reference_actions.shape[1])


def load_stage3_fast_arrays(
    target_dir: str | pathlib.Path,
    *,
    chunk_steps: int = slow_fast.DEFAULT_FAST_CHUNK_STEPS,
) -> Stage3FastArrays:
    """Load only the short-chunk fields needed by Fast from Stage-3 shards."""
    if chunk_steps <= 0:
        raise ValueError("chunk_steps must be positive")
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
        "expert_pose": [],
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
            fields["state"].append(np.asarray(shard["normalized_state"][:, : rot.ROBOT_DIMS], dtype=np.float32))
            fields["force_history"].append(np.asarray(shard["normalized_force_history"], dtype=np.float32))
            fields["force_history_mask"].append(np.asarray(shard["force_history_mask"], dtype=np.bool_))
            for name, key in (
                ("full_pose", "normalized_full_actions"),
                ("residual_pose", "normalized_pose_residual"),
                ("expert_pose", "normalized_expert_actions"),
            ):
                horizon = shard[key].shape[1]
                if horizon < chunk_steps:
                    raise ValueError(f"{key} has a {horizon}-step horizon, shorter than chunk_steps={chunk_steps}")
                fields[name].append(np.asarray(shard[key][:, :chunk_steps, : rot.POSE_DIMS], dtype=np.float32))
        next_index = stop

    expected = int(manifest["extraction_size"])
    if next_index != expected:
        raise ValueError(f"Expected {expected} Stage-3 rows, loaded {next_index}")
    arrays = {key: np.concatenate(parts, axis=0) for key, parts in fields.items()}
    return Stage3FastArrays(**arrays)


def period_range_from_rates(rate_range_hz: tuple[float, float] | float | None) -> tuple[float, float] | None:
    """Convert a Slow rate band in Hz into the period band the selector expects."""
    if rate_range_hz is None:
        return None
    low, high = (rate_range_hz, rate_range_hz) if np.isscalar(rate_range_hz) else rate_range_hz
    if min(low, high) <= 0:
        raise ValueError("Slow rates must be positive")
    return (1.0 / max(low, high), 1.0 / min(low, high))


def select_slow_update_rows(
    episode_indices: np.ndarray,
    timestamps: np.ndarray,
    *,
    period_range_s: tuple[float, float] | None,
    tolerance_s: float = 1e-9,
    rng=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select causal Slow updates and map every row to its latest Slow packet.

    A `None` range means every row is its own Slow update. Because the selection
    is greedy from the first row of each episode, the key rows of any slower
    period are a subset of those full-rate rows, so one full-rate extraction can
    be resampled to any lower Slow rate without rerunning the model.

    Each interval between keys is drawn uniformly from `period_range_s` rather
    than fixed. A degenerate range pins the Slow rate; a wide one is what makes a
    single cache cover machines whose Slow loop does not run at the rate the cache
    was extracted on.
    """
    episodes = np.asarray(episode_indices, dtype=np.int64)
    times = np.asarray(timestamps, dtype=np.float64)
    if episodes.ndim != 1 or times.shape != episodes.shape or len(times) == 0:
        raise ValueError("episode_indices and timestamps must be non-empty 1D arrays")
    if period_range_s is None:
        return np.arange(len(times), dtype=np.int64), np.arange(len(times), dtype=np.int64)
    period_low, period_high = (float(value) for value in period_range_s)
    if period_low <= 0 or period_high < period_low or tolerance_s < 0:
        raise ValueError("Slow update periods must be positive, ordered, and the tolerance non-negative")

    sampler = None if period_low == period_high else np.random.default_rng(rng)
    next_period = period_low if sampler is None else float(sampler.uniform(period_low, period_high))
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
            next_period = period_low if sampler is None else float(sampler.uniform(period_low, period_high))
        elif previous_time is not None and timestamp + tolerance_s < previous_time:
            raise ValueError(f"Timestamps are not monotonic inside episode {episode}")

        if latest_key_time is None or timestamp + tolerance_s >= latest_key_time + next_period:
            key_rows.append(row)
            key_position += 1
            latest_key_time = float(timestamp)
            if sampler is not None:
                next_period = float(sampler.uniform(period_low, period_high))
        row_key_positions[row] = key_position
        previous_time = float(timestamp)
    return np.asarray(key_rows, dtype=np.int64), row_key_positions


def jitter_key_timestamps(
    key_timestamps: np.ndarray,
    *,
    jitter_s: float,
    rng=None,
) -> np.ndarray:
    """Offset Slow key times off the action-rate grid.

    Dataset rows sit on an exact 30 Hz lattice. Selecting a different row still
    leaves `age / action_period` an integer, so the interpolation alpha stays in
    `{0, 1}`. A few milliseconds of timestamp jitter is what actually trains the
    linear-interpolation branch.
    """
    times = np.asarray(key_timestamps, dtype=np.float64)
    if jitter_s < 0:
        raise ValueError("jitter_s must be non-negative")
    if jitter_s == 0 or len(times) == 0:
        return times
    return times + np.random.default_rng(rng).uniform(-jitter_s, jitter_s, size=len(times))


def sample_ready_delays(
    num_keys: int,
    *,
    delay_range_s: tuple[float, float] | float,
    rng=None,
) -> np.ndarray:
    """Draw one Slow serving latency per packet from a band.

    A single measured latency only describes the machine it was measured on. Drawing
    from a band is what lets the student see, during training, the range of staleness
    it will actually meet across deployments.
    """
    low, high = (delay_range_s, delay_range_s) if np.isscalar(delay_range_s) else delay_range_s
    low, high = float(low), float(high)
    if low < 0 or high < low:
        raise ValueError("delay_range_s must be non-negative and ordered")
    if num_keys < 0:
        raise ValueError("num_keys must be non-negative")
    if low == high:
        return np.full(num_keys, low, dtype=np.float64)
    return np.random.default_rng(rng).uniform(low, high, size=num_keys)


def assign_ready_packets(
    episode_indices: np.ndarray,
    timestamps: np.ndarray,
    key_rows: np.ndarray,
    key_timestamps: np.ndarray,
    *,
    ready_delay_s: np.ndarray | float,
) -> np.ndarray:
    """Map each row to the freshest Slow packet that would already be ready.

    Serving refuses to consume a packet before `observation_timestamp + latency`.
    Training has to use the same rule, otherwise context age starts at 0 in
    training and at the true latency on the robot.

    With a per-packet latency the arrival order need not follow the observation
    order: a slow packet can land after a fresher one has already been installed,
    and serving would discard it. Restricting the search to packets that are not
    superseded on arrival reproduces that.
    """
    episodes = np.asarray(episode_indices, dtype=np.int64)
    times = np.asarray(timestamps, dtype=np.float64)
    keys = np.asarray(key_rows, dtype=np.int64)
    key_times = np.asarray(key_timestamps, dtype=np.float64)
    delays = np.broadcast_to(np.asarray(ready_delay_s, dtype=np.float64), key_times.shape)
    if np.any(delays < 0):
        raise ValueError("ready_delay_s must be non-negative")
    mapping = np.full(len(times), -1, dtype=np.int64)
    if len(keys) == 0:
        return mapping
    key_eps = episodes[keys]
    ready_times = key_times + delays
    for episode in np.unique(episodes):
        rows = np.flatnonzero(episodes == episode)
        key_idx = np.flatnonzero(key_eps == episode)
        if len(key_idx) == 0:
            continue
        episode_ready = ready_times[key_idx]
        # Keep only packets that arrive before every later observation does, so the
        # surviving arrival times are strictly increasing and searchsorted is valid.
        suffix_min = np.minimum.accumulate(episode_ready[::-1])[::-1]
        usable = np.ones(len(episode_ready), dtype=bool)
        usable[:-1] = episode_ready[:-1] < suffix_min[1:]
        key_idx = key_idx[usable]
        episode_ready = episode_ready[usable]
        chosen = np.searchsorted(episode_ready, times[rows], side="right") - 1
        mapping[rows] = np.where(chosen >= 0, key_idx[np.maximum(chosen, 0)], -1)
    return mapping


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
    chunk_steps: int = slow_fast.DEFAULT_FAST_CHUNK_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample cached Slow chunks over a short horizon at each row timestamp.

    Returns the reference rollout `[N, chunk_steps, A]` and the time features
    `[N, 2] = [context age / scale, interpolation alpha]`.

    The two features are deliberately not "phase and age". Both of those are the
    same scalar age divided by a different constant, so they are perfectly
    collinear inside the operating band and the time token degenerates to one
    dimension. Alpha is the fractional position between two Teacher waypoints; it
    is `age mod action_period` rescaled, so it carries information age does not.
    """
    chunks = np.asarray(action_chunks, dtype=np.float32)
    mapping = np.asarray(row_key_positions, dtype=np.int64)
    row_times = np.asarray(row_timestamps, dtype=np.float64)
    key_times = np.asarray(key_timestamps, dtype=np.float64)
    if chunks.ndim != 3 or chunks.shape[-1] < 2:
        raise ValueError(f"Expected Slow chunks [K,H,>=2], got {chunks.shape}")
    if mapping.ndim != 1 or row_times.shape != mapping.shape:
        raise ValueError("row_key_positions and row_timestamps must be matching 1D arrays")
    if len(key_times) != len(chunks) or np.any(mapping < 0) or np.any(mapping >= len(chunks)):
        raise ValueError("Slow key arrays and row mapping are inconsistent")
    if action_period_s <= 0 or context_age_scale_s <= 0:
        raise ValueError("Action period and context-age scale must be positive")
    if chunk_steps <= 0:
        raise ValueError("chunk_steps must be positive")

    age = np.maximum(row_times - key_times[mapping], 0.0)
    # Step k of the emitted chunk lands k Teacher action periods after the row.
    position = age[:, None] / action_period_s + np.arange(chunk_steps, dtype=np.float64)[None, :]
    left = np.minimum(np.floor(position).astype(np.int64), chunks.shape[1] - 1)
    right = np.minimum(left + 1, chunks.shape[1] - 1)
    alpha = np.clip(position - left, 0.0, 1.0).astype(np.float32)
    rows = mapping[:, None]
    left_action = chunks[rows, left]
    right_action = chunks[rows, right]
    reference = (1.0 - alpha[..., None]) * left_action + alpha[..., None] * right_action
    gripper = chunks.shape[-1] - 1
    reference[..., gripper] = left_action[..., gripper]

    normalized_age = np.clip(age / context_age_scale_s, 0.0, 1.0)
    time_features = np.stack([normalized_age, alpha[:, 0]], axis=-1).astype(np.float32)
    return reference.astype(np.float32), time_features


def resample_slow_cache(
    cache: SlowCache,
    episode_indices: np.ndarray,
    timestamps: np.ndarray,
    *,
    slow_rate_range_hz: tuple[float, float] | float,
    jitter_s: float = slow_fast.DEFAULT_UPDATE_JITTER_S,
    ready_delay_range_s: tuple[float, float] | float | None = None,
    rng=None,
) -> SlowCache:
    """Derive a lower-rate Slow cache from a higher-rate one without rerunning the model.

    Only valid when every key row of the target rate is already present in
    `cache`, which holds whenever `cache` was extracted at full rate.

    Jitter is on by default. Key rows can only be selected from the dataset lattice,
    so without it every age is an exact multiple of the action period and the
    interpolation alpha collapses to zero, leaving that branch untrained. Randomizing
    the Slow *period* does not fix this on its own.

    Leaving `ready_delay_range_s` unset reuses the latencies already drawn for the
    selected packets, so resampling only changes the Slow rate.
    """
    key_rows, _ = select_slow_update_rows(
        episode_indices,
        timestamps,
        period_range_s=period_range_from_rates(slow_rate_range_hz),
        rng=rng,
    )
    if len(timestamps) != len(cache.row_key_positions):
        raise ValueError(
            f"Cache covers {len(cache.row_key_positions)} rows but {len(timestamps)} were supplied"
        )

    row_to_packet = np.full(len(timestamps), -1, dtype=np.int64)
    row_to_packet[cache.key_dataset_indices] = np.arange(len(cache.key_dataset_indices), dtype=np.int64)
    selection = row_to_packet[key_rows]
    if np.any(selection < 0):
        missing = int(np.sum(selection < 0))
        raise ValueError(
            f"{missing} of {len(key_rows)} target key rows are absent from the cache; "
            "resampling requires a cache extracted at a higher Slow rate"
        )

    key_timestamps = jitter_key_timestamps(
        cache.key_timestamps[selection],
        jitter_s=jitter_s,
        rng=None if rng is None else np.random.default_rng(rng).integers(2**31 - 1),
    )
    if ready_delay_range_s is not None:
        key_ready_delays = sample_ready_delays(
            len(selection),
            delay_range_s=ready_delay_range_s,
            rng=None if rng is None else np.random.default_rng(rng).integers(2**31 - 1),
        )
    elif cache.key_ready_delays is not None:
        key_ready_delays = np.asarray(cache.key_ready_delays, dtype=np.float64)[selection]
    else:
        key_ready_delays = np.zeros(len(selection), dtype=np.float64)
    ready_mapping = assign_ready_packets(
        episode_indices,
        timestamps,
        key_rows,
        key_timestamps,
        ready_delay_s=key_ready_delays,
    )
    row_ready = ready_mapping >= 0
    safe_mapping = np.where(row_ready, ready_mapping, 0)
    reference_actions, time_features = build_reference_rollout(
        cache.action_chunks[selection],
        safe_mapping,
        timestamps,
        key_timestamps,
        action_period_s=cache.action_period_s,
        context_age_scale_s=cache.context_age_scale_s,
        chunk_steps=cache.chunk_steps,
    )
    reference_actions = np.where(row_ready[:, None, None], reference_actions, 0)
    time_features = np.where(row_ready[:, None], time_features, 0)
    return dataclasses.replace(
        cache,
        key_dataset_indices=cache.key_dataset_indices[selection],
        key_episode_indices=cache.key_episode_indices[selection],
        key_timestamps=key_timestamps,
        context_tokens=cache.context_tokens[selection],
        context_mask=cache.context_mask[selection],
        action_chunks=cache.action_chunks[selection],
        row_key_positions=safe_mapping,
        reference_actions=reference_actions,
        time_features=time_features,
        key_ready_delays=key_ready_delays,
        row_ready=row_ready,
    )


def denormalize(values: np.ndarray, stats, *, dims: int | None = None) -> np.ndarray:
    """Invert the z-score normalization applied by `transforms.Normalize`."""
    mean = np.asarray(stats.mean, dtype=np.float64)
    std = np.asarray(stats.std, dtype=np.float64)
    if dims is not None:
        mean, std = mean[:dims], std[:dims]
    if values.shape[-1] != mean.shape[-1]:
        raise ValueError(f"Cannot denormalize {values.shape} with {mean.shape[-1]}D statistics")
    return (np.asarray(values, dtype=np.float64) * (std + 1e-6) + mean).astype(np.float32)


def latest_physical_wrench(force_history: np.ndarray, force_history_mask: np.ndarray, stats) -> np.ndarray:
    """Return the most recent valid wrench of each row in physical units.

    Rows whose window contains no valid sample yield a zero wrench, which the
    caller can safely treat as free space.
    """
    history = np.asarray(force_history)
    mask = np.asarray(force_history_mask, dtype=bool)
    if history.ndim != 3 or history.shape[-1] != 6 or mask.shape != history.shape[:-1]:
        raise ValueError(f"Expected force history [B,N,6] and mask [B,N], got {history.shape} and {mask.shape}")
    physical = denormalize(history.reshape(-1, 6), stats).reshape(history.shape)
    # Slots are left padded, so the last valid index is the freshest sample.
    slots = np.arange(history.shape[1])[None, :]
    latest = np.max(np.where(mask, slots, -1), axis=1)
    result = physical[np.arange(len(physical)), np.maximum(latest, 0)]
    return np.where((latest >= 0)[:, None], result, 0.0).astype(np.float32)


def episode_baseline_wrench(wrench: np.ndarray, episode_indices: np.ndarray, *, num_rows: int = 15) -> np.ndarray:
    """Estimate each episode's resting wrench from its opening rows.

    Raw wrench readings carry a large tool-weight and sensor bias offset, so an
    absolute magnitude threshold cannot separate contact from free space.  The
    median over the first rows of an episode approximates that offset.
    """
    values = np.asarray(wrench, dtype=np.float64)
    episodes = np.asarray(episode_indices, dtype=np.int64)
    if values.ndim != 2 or episodes.shape != values.shape[:1]:
        raise ValueError(f"Expected wrench [B,D] and episodes [B], got {values.shape} and {episodes.shape}")
    if num_rows <= 0:
        raise ValueError("num_rows must be positive")
    baseline = np.empty_like(values)
    for episode in np.unique(episodes):
        selector = episodes == episode
        opening = values[selector][:num_rows]
        baseline[selector] = np.median(opening, axis=0)
    return baseline.astype(np.float32)


def load_slow_cache(path: str | pathlib.Path, *, expected_rows: int | None = None) -> SlowCache:
    with np.load(path, allow_pickle=False) as cache:
        arrays = {
            field.name: np.asarray(cache[field.name])
            for field in dataclasses.fields(SlowCache)
            if field.default is dataclasses.MISSING
        }
        # Caches written before the timing contract was recorded fall back to the
        # defaults, which are the values those caches were generated with.
        scalars = {
            name: float(cache[name])
            for name in ("context_age_scale_s", "action_period_s")
            if name in cache.files
        }
        extra = {}
        if "row_ready" in cache.files:
            extra["row_ready"] = np.asarray(cache["row_ready"], dtype=np.bool_)
        if "key_ready_delays" in cache.files:
            extra["key_ready_delays"] = np.asarray(cache["key_ready_delays"], dtype=np.float64)
        elif "ready_delay_s" in cache.files:
            extra["key_ready_delays"] = np.full(
                len(cache["key_timestamps"]), float(cache["ready_delay_s"]), dtype=np.float64
            )
        result = SlowCache(**arrays, **scalars, **extra)
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
    if result.key_ready_delays is not None and len(result.key_ready_delays) != key_count:
        raise ValueError("key_ready_delays must have one latency per Slow packet")
    if result.reference_actions.ndim != 3:
        raise ValueError(f"reference_actions must be [rows, chunk_steps, A], got {result.reference_actions.shape}")
    if len(result.reference_actions) != row_count or len(result.time_features) != row_count:
        raise ValueError("Slow row mapping, references, and time features must have the same number of rows")
    if np.any(result.row_key_positions < 0) or np.any(result.row_key_positions >= key_count):
        raise ValueError("Slow row mapping contains an invalid key position")
    if result.row_ready is not None and len(result.row_ready) != row_count:
        raise ValueError("row_ready must have one flag per dataset row")
    return result
