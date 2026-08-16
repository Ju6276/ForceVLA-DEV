"""Dataset adapter for native-rate timestamped force sidecars."""

from collections.abc import Mapping
import functools
from pathlib import Path
from typing import Any

import numpy as np


class NativeForceSidecarDataset:
    """Attach one independently timestamped wrench stream to each LeRobot item.

    The wrapped dataset remains indexed at its original RGB/action rate. Each episode
    has one NPZ sidecar containing the native-rate wrench stream, avoiding duplication
    of the same force samples in every low-rate row. A later transform extracts the
    causal physical-time window for the current observation timestamp.
    """

    def __init__(
        self,
        dataset,
        *,
        data_dir: str | Path,
        file_pattern: str = "episode_{episode_index:06d}.npz",
        force_array_key: str = "force",
        timestamp_array_key: str = "timestamps",
        episode_index_key: str = "episode_index",
        output_force_key: str = "observation.force",
        output_timestamps_key: str = "observation.force_timestamps",
        cache_size: int = 8,
    ):
        if cache_size < 1:
            raise ValueError("Native force sidecar cache_size must be positive")
        self._dataset = dataset
        self._data_dir = Path(data_dir)
        self._file_pattern = file_pattern
        self._force_array_key = force_array_key
        self._timestamp_array_key = timestamp_array_key
        self._episode_index_key = episode_index_key
        self._output_force_key = output_force_key
        self._output_timestamps_key = output_timestamps_key
        self._load_episode = functools.lru_cache(maxsize=cache_size)(self._load_episode_uncached)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index) -> dict[str, Any]:
        item = dict(self._dataset[index])
        if self._episode_index_key not in item:
            raise KeyError(f"Dataset item is missing {self._episode_index_key!r}")
        episode_index = int(np.asarray(item[self._episode_index_key]).item())
        force, timestamps = self._load_episode(episode_index)
        item[self._output_force_key] = force
        item[self._output_timestamps_key] = timestamps
        return item

    def _load_episode_uncached(self, episode_index: int) -> tuple[np.ndarray, np.ndarray]:
        path = self._data_dir / self._file_pattern.format(episode_index=episode_index)
        if not path.is_file():
            raise FileNotFoundError(f"Native force sidecar not found: {path}")
        with np.load(path, allow_pickle=False) as episode:
            self._validate_keys(path, episode)
            force = np.asarray(episode[self._force_array_key], dtype=np.float32)
            timestamps = np.asarray(episode[self._timestamp_array_key], dtype=np.float64)
        if force.ndim != 2 or force.shape[-1] != 6:
            raise ValueError(f"Expected {path} force array [N, 6], got {force.shape}")
        if timestamps.ndim != 1 or timestamps.shape[0] != force.shape[0]:
            raise ValueError(f"Expected {path} timestamps [N] matching force, got {timestamps.shape}")
        if not np.all(np.isfinite(force)) or not np.all(np.isfinite(timestamps)):
            raise ValueError(f"Native force sidecar contains non-finite values: {path}")
        if np.any(np.diff(timestamps) < 0):
            raise ValueError(f"Native force timestamps must be sorted: {path}")
        return force, timestamps

    def _validate_keys(self, path: Path, episode: Mapping[str, Any]) -> None:
        missing = {self._force_array_key, self._timestamp_array_key}.difference(episode)
        if missing:
            raise KeyError(f"Native force sidecar {path} is missing arrays: {sorted(missing)}")
