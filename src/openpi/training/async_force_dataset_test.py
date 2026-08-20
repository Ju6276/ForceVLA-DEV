import pickle

import numpy as np

from openpi import transforms
from openpi.training.async_force_dataset import NativeForceSidecarDataset


class _FakeLowRateDataset:
    def __init__(self, timestamp: float):
        self._timestamp = timestamp

    def __len__(self):
        return 1

    def __getitem__(self, index):
        del index
        return {"episode_index": np.asarray(0), "timestamp": np.asarray(self._timestamp)}


def test_native_force_sidecar_dataset_is_picklable_for_spawn_workers(tmp_path):
    np.savez(
        tmp_path / "episode_000000.npz",
        force=np.zeros((2, 6), dtype=np.float32),
        timestamps=np.asarray([0.0, 0.01]),
    )
    dataset = NativeForceSidecarDataset(_FakeLowRateDataset(0.01), data_dir=tmp_path)
    restored = pickle.loads(pickle.dumps(dataset))

    np.testing.assert_array_equal(restored[0]["observation.force"], np.zeros((2, 6), dtype=np.float32))


def test_200hz_sidecar_joins_by_episode_and_extracts_causal_window(tmp_path):
    target_time = 1.0
    timestamps = np.arange(0.8, 1.051, 0.005, dtype=np.float64)
    force = np.repeat(np.arange(timestamps.size, dtype=np.float32)[:, None], 6, axis=1)
    np.savez(tmp_path / "episode_000000.npz", force=force, timestamps=timestamps)

    dataset = NativeForceSidecarDataset(_FakeLowRateDataset(target_time), data_dir=tmp_path)
    item = dataset[0]
    result = transforms.TimestampAlignedForceHistory(
        force_key="observation.force",
        force_timestamps_key="observation.force_timestamps",
        observation_timestamp_key="timestamp",
        window_ms=100,
        max_samples=20,
    )(item)

    selected_timestamps = timestamps[(timestamps > 0.9) & (timestamps <= target_time)]
    assert selected_timestamps.size == 20
    np.testing.assert_array_equal(result["force_history"][:, 0], force[21:41, 0])
    np.testing.assert_array_equal(result["force_history_mask"], np.ones(20, dtype=np.bool_))
    assert not np.any(result["force_history"][:, 0] == force[41, 0])
