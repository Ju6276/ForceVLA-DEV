import json

import numpy as np
import pytest

from openpi.training import offline_targets


class _Dataset:
    def __len__(self):
        return 3

    def __getitem__(self, index):
        return {"state": np.asarray([index]), "actions": np.asarray([-1.0])}


def _write_targets(path, *, complete=True, indices=(0, 1, 2)):
    shard_dir = path / "shards"
    shard_dir.mkdir(parents=True)
    values = np.arange(len(indices) * 4, dtype=np.float32).reshape(len(indices), 2, 2)
    np.savez(shard_dir / "rows.npz", dataset_indices=np.asarray(indices), normalized_null_actions=values)
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "complete": complete,
                "extraction_size": 3,
                "shards": [{"complete": complete, "path": "shards/rows.npz"}],
            }
        )
    )
    return values


def test_replaces_actions_with_aligned_offline_target(tmp_path):
    values = _write_targets(tmp_path)
    dataset = offline_targets.OfflineActionTargetDataset(_Dataset(), tmp_path, target_key="normalized_null_actions")

    np.testing.assert_array_equal(dataset[1]["state"], np.asarray([1]))
    np.testing.assert_array_equal(dataset[1]["actions"], values[1])


def test_rejects_incomplete_extraction(tmp_path):
    _write_targets(tmp_path, complete=False)

    with pytest.raises(ValueError, match="incomplete"):
        offline_targets.OfflineActionTargetDataset(_Dataset(), tmp_path, target_key="normalized_null_actions")


def test_rejects_noncontiguous_indices(tmp_path):
    _write_targets(tmp_path, indices=(0, 2, 1))

    with pytest.raises(AssertionError):
        offline_targets.OfflineActionTargetDataset(_Dataset(), tmp_path, target_key="normalized_null_actions")
