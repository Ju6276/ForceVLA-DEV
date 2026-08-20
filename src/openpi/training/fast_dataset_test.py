import json

import jax.numpy as jnp
import numpy as np
import pytest

from openpi.shared import normalize
from openpi.training import fast_dataset


def _write_stage3(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    np.savez(
        shard_dir / "rows_000000_000002.npz",
        dataset_indices=np.arange(3),
        episode_indices=np.array([0, 0, 0]),
        timestamps=np.array([0.0, 0.05, 0.1]),
        normalized_state=np.arange(3 * 32, dtype=np.float32).reshape(3, 32),
        normalized_force_history=np.ones((3, 10, 6), dtype=np.float32),
        force_history_mask=np.ones((3, 10), dtype=np.bool_),
        normalized_full_actions=np.arange(3 * 50 * 32, dtype=np.float32).reshape(3, 50, 32),
        normalized_pose_residual=np.full((3, 50, 6), 2.0, dtype=np.float32),
        normalized_expert_actions=np.full((3, 50, 32), 5.0, dtype=np.float32),
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "extraction_size": 3,
                "shards": [{"complete": True, "path": "shards/rows_000000_000002.npz"}],
            }
        )
    )


def test_load_stage3_fast_arrays_keeps_only_current_targets(tmp_path):
    _write_stage3(tmp_path)
    arrays = fast_dataset.load_stage3_fast_arrays(tmp_path)
    assert arrays.state.shape == (3, 7)
    assert arrays.force_history.shape == (3, 10, 6)
    np.testing.assert_array_equal(arrays.full_pose, np.arange(3 * 50 * 32).reshape(3, 50, 32)[:, 0, :6])
    np.testing.assert_array_equal(arrays.residual_pose, np.full((3, 6), 2.0))
    np.testing.assert_array_equal(arrays.expert_pose, np.full((3, 6), 5.0))
    np.testing.assert_array_equal(arrays.null_pose, arrays.full_pose - arrays.residual_pose)


def test_denormalize_inverts_z_score():
    stats = normalize.NormStats(mean=np.array([1.0, -2.0]), std=np.array([0.5, 4.0]))
    physical = np.array([[3.0, 6.0], [0.0, -2.0]], dtype=np.float32)
    normalized = (physical - stats.mean) / (stats.std + 1e-6)
    np.testing.assert_allclose(fast_dataset.denormalize(normalized, stats), physical, atol=1e-4)


def test_latest_physical_wrench_picks_freshest_valid_slot():
    stats = normalize.NormStats(mean=np.zeros(6), std=np.ones(6))
    history = np.zeros((2, 4, 6), dtype=np.float32)
    history[0, 2] = 7.0
    history[0, 3] = 9.0
    history[1, 1] = 3.0
    mask = np.array([[False, True, True, False], [False, True, False, False]])
    wrench = fast_dataset.latest_physical_wrench(history, mask, stats)
    np.testing.assert_allclose(wrench[0], np.full(6, 7.0), atol=1e-4)
    np.testing.assert_allclose(wrench[1], np.full(6, 3.0), atol=1e-4)


def test_latest_physical_wrench_returns_zero_for_empty_window():
    stats = normalize.NormStats(mean=np.full(6, 2.0), std=np.ones(6))
    history = np.ones((1, 3, 6), dtype=np.float32)
    wrench = fast_dataset.latest_physical_wrench(history, np.zeros((1, 3), dtype=bool), stats)
    np.testing.assert_array_equal(wrench, np.zeros((1, 6)))


def test_select_slow_update_rows_is_causal_and_resets_per_episode():
    episodes = np.array([0, 0, 0, 0, 1, 1, 1])
    times = np.array([0.0, 0.04, 0.1, 0.19, 0.0, 0.09, 0.11])
    keys, mapping = fast_dataset.select_slow_update_rows(episodes, times, update_period_s=0.1)
    np.testing.assert_array_equal(keys, [0, 2, 4, 6])
    np.testing.assert_array_equal(mapping, [0, 0, 1, 1, 2, 2, 3])


def _full_rate_cache(episodes, times, chunks):
    key_rows, mapping = fast_dataset.select_slow_update_rows(episodes, times, update_period_s=None)
    np.testing.assert_array_equal(key_rows, np.arange(len(times)))
    reference, time_features = fast_dataset.build_reference_rollout(
        chunks, mapping, times, times, action_period_s=1 / 30, context_age_scale_s=0.5
    )
    return fast_dataset.SlowCache(
        key_dataset_indices=np.arange(len(times)),
        key_episode_indices=episodes,
        key_timestamps=times,
        context_tokens=np.zeros((len(times), 2, 4), dtype=np.float16),
        context_mask=np.ones((len(times), 2), dtype=bool),
        action_chunks=chunks,
        row_key_positions=mapping,
        reference_actions=reference,
        time_features=time_features,
        context_age_scale_s=0.5,
        action_period_s=1 / 30,
    )


def test_resample_slow_cache_matches_direct_extraction():
    episodes = np.repeat([0, 1], [90, 47]).astype(np.int64)
    times = np.concatenate([np.arange(90), np.arange(47)]) / 30.0
    chunks = np.random.default_rng(0).standard_normal((len(times), 8, 7)).astype(np.float32)
    cache = _full_rate_cache(episodes, times, chunks)

    for rate_hz in (1.0, 5.0, 30.0):
        resampled = fast_dataset.resample_slow_cache(cache, episodes, times, slow_rate_hz=rate_hz)
        key_rows, mapping = fast_dataset.select_slow_update_rows(
            episodes, times, update_period_s=1.0 / rate_hz
        )
        reference, time_features = fast_dataset.build_reference_rollout(
            chunks[key_rows], mapping, times, times[key_rows], action_period_s=1 / 30, context_age_scale_s=0.5
        )
        np.testing.assert_array_equal(resampled.key_dataset_indices, key_rows)
        np.testing.assert_array_equal(resampled.action_chunks, chunks[key_rows])
        np.testing.assert_array_equal(resampled.row_key_positions, mapping)
        np.testing.assert_array_equal(resampled.reference_actions, reference)
        np.testing.assert_array_equal(resampled.time_features, time_features)


def test_resample_slow_cache_rejects_upsampling():
    episodes = np.zeros(60, dtype=np.int64)
    times = np.arange(60) / 30.0
    chunks = np.zeros((60, 8, 7), dtype=np.float32)
    sparse = fast_dataset.resample_slow_cache(
        _full_rate_cache(episodes, times, chunks), episodes, times, slow_rate_hz=2.0
    )
    with pytest.raises(ValueError, match="absent from the cache"):
        fast_dataset.resample_slow_cache(sparse, episodes, times, slow_rate_hz=10.0)


def test_pool_context_tokens_respects_padding():
    hidden = jnp.arange(1 * 4 * 2, dtype=jnp.float32).reshape(1, 4, 2)
    mask = jnp.array([[True, True, False, False]])
    pooled, pooled_mask = fast_dataset.pool_context_tokens(hidden, mask, num_tokens=2)
    np.testing.assert_allclose(pooled[0, 0], np.array([1.0, 2.0]))
    np.testing.assert_array_equal(pooled[0, 1], np.zeros(2))
    np.testing.assert_array_equal(pooled_mask, [[True, False]])


def test_reference_rollout_interpolates_pose_and_holds_gripper():
    chunks = np.array(
        [
            [
                [0, 0, 0, 0, 0, 0, -1],
                [1, 2, 3, 4, 5, 6, 1],
                [2, 4, 6, 8, 10, 12, 1],
            ]
        ],
        dtype=np.float32,
    )
    reference, time_features = fast_dataset.build_reference_rollout(
        chunks,
        np.array([0, 0, 0]),
        np.array([0.0, 0.05, 0.1]),
        np.array([0.0]),
        action_period_s=0.1,
        context_age_scale_s=0.2,
    )
    np.testing.assert_allclose(reference[1, :6], [0.5, 1, 1.5, 2, 2.5, 3])
    assert reference[1, 6] == -1
    np.testing.assert_allclose(reference[2], chunks[0, 1])
    np.testing.assert_allclose(time_features[:, 0], [0.0, 0.25, 0.5])
    np.testing.assert_allclose(time_features[:, 1], [0.0, 0.25, 0.5])
