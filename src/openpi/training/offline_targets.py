"""Datasets backed by sharded offline Teacher action targets."""

import json
import pathlib

import numpy as np


class OfflineActionTargetDataset:
    """Replace a transformed dataset's action with one aligned Teacher target.

    The paired-target extractor writes split-local ``dataset_indices``. This
    wrapper validates that those indices are complete and ordered before using
    the requested array as the training action. Targets are held in host memory
    so shuffled training does not repeatedly decompress NPZ shards.
    """

    def __init__(self, dataset, target_dir: str | pathlib.Path, *, target_key: str):
        self._dataset = dataset
        self._target_dir = pathlib.Path(target_dir).resolve()
        self._target_key = target_key
        self._targets = self._load_targets()
        if len(self._targets) != len(dataset):
            raise ValueError(
                f"Offline targets contain {len(self._targets)} rows but the dataset contains {len(dataset)}"
            )

    def _load_targets(self) -> np.ndarray:
        manifest_path = self._target_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Offline target manifest not found: {manifest_path}")
        with manifest_path.open() as file:
            manifest = json.load(file)
        if not manifest.get("complete", False):
            raise ValueError(f"Offline target extraction is incomplete: {manifest_path}")

        expected_rows = int(manifest["extraction_size"])
        targets = None
        next_index = 0
        for shard_info in manifest["shards"]:
            if not shard_info.get("complete", False):
                raise ValueError(f"Manifest lists an incomplete shard: {shard_info}")
            path = self._target_dir / shard_info["path"]
            with np.load(path, allow_pickle=False) as shard:
                if self._target_key not in shard:
                    raise KeyError(f"Offline target shard {path} has no {self._target_key!r} array")
                indices = np.asarray(shard["dataset_indices"], dtype=np.int64)
                values = np.asarray(shard[self._target_key], dtype=np.float32)
            stop = next_index + len(indices)
            np.testing.assert_array_equal(indices, np.arange(next_index, stop, dtype=np.int64))
            if len(values) != len(indices):
                raise ValueError(f"Target rows do not match dataset_indices in {path}")
            if targets is None:
                targets = np.empty((expected_rows, *values.shape[1:]), dtype=np.float32)
            elif values.shape[1:] != targets.shape[1:]:
                raise ValueError(f"Inconsistent target shape in {path}: {values.shape[1:]} vs {targets.shape[1:]}")
            targets[next_index:stop] = values
            next_index = stop

        if targets is None or next_index != expected_rows:
            raise ValueError(f"Expected {expected_rows} offline targets, loaded {next_index}")
        return targets

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index) -> dict:
        position = int(index)
        sample = dict(self._dataset[position])
        sample["actions"] = self._targets[position]
        return sample
