import dataclasses
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
        normalized_pose_residual=np.full((3, 50, 9), 2.0, dtype=np.float32),
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


def test_load_stage3_fast_arrays_keeps_the_leading_chunk(tmp_path):
    _write_stage3(tmp_path)
    arrays = fast_dataset.load_stage3_fast_arrays(tmp_path, chunk_steps=4)
    assert arrays.state.shape == (3, 10)
    assert arrays.force_history.shape == (3, 10, 6)
    assert arrays.chunk_steps == 4
    np.testing.assert_array_equal(arrays.full_pose, np.arange(3 * 50 * 32).reshape(3, 50, 32)[:, :4, :9])
    np.testing.assert_array_equal(arrays.residual_pose, np.full((3, 4, 9), 2.0))
    np.testing.assert_array_equal(arrays.expert_pose, np.full((3, 4, 9), 5.0))
    np.testing.assert_array_equal(arrays.null_pose, arrays.full_pose - arrays.residual_pose)


def test_load_stage3_fast_arrays_rejects_a_chunk_longer_than_the_horizon(tmp_path):
    _write_stage3(tmp_path)
    with pytest.raises(ValueError, match="shorter than chunk_steps"):
        fast_dataset.load_stage3_fast_arrays(tmp_path, chunk_steps=51)


def test_cached_context_indices_are_explicit_and_current_targets_remain_backward_compatible(tmp_path):
    _write_stage3(tmp_path)
    assert fast_dataset.load_cached_context_dataset_indices(tmp_path) is None

    shard_path = tmp_path / "shards" / "rows_000000_000002.npz"
    with np.load(shard_path, allow_pickle=False) as shard:
        arrays = {name: np.asarray(shard[name]) for name in shard.files}
    arrays["context_dataset_indices"] = np.array([0, 0, 1], dtype=np.int64)
    np.savez(shard_path, **arrays)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["target_context_mode"] = "cached_slow_packet"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    np.testing.assert_array_equal(
        fast_dataset.load_cached_context_dataset_indices(tmp_path),
        np.array([0, 0, 1]),
    )


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
    keys, mapping = fast_dataset.select_slow_update_rows(episodes, times, period_range_s=(0.1, 0.1))
    np.testing.assert_array_equal(keys, [0, 2, 4, 6])
    np.testing.assert_array_equal(mapping, [0, 0, 1, 1, 2, 2, 3])


def test_randomized_slow_period_stays_inside_the_requested_rate_band():
    episodes = np.zeros(3000, dtype=np.int64)
    times = np.arange(3000) / 200.0
    period_range = fast_dataset.period_range_from_rates((5.0, 15.0))
    keys, _ = fast_dataset.select_slow_update_rows(episodes, times, period_range_s=period_range, rng=0)
    intervals = np.diff(times[keys])
    # Keys can only land on the 200 Hz row lattice, so an interval overshoots its
    # drawn period by at most one row.
    assert intervals.min() >= period_range[0] - 1e-9
    assert intervals.max() <= period_range[1] + 1 / 200.0 + 1e-9
    assert np.std(intervals) > 0.01, "a randomized period must not collapse onto one rate"


def _full_rate_cache(episodes, times, chunks, *, chunk_steps=1):
    key_rows, mapping = fast_dataset.select_slow_update_rows(episodes, times, period_range_s=None)
    np.testing.assert_array_equal(key_rows, np.arange(len(times)))
    reference, time_features = fast_dataset.build_reference_rollout(
        chunks,
        mapping,
        times,
        times,
        action_period_s=1 / 30,
        context_age_scale_s=0.5,
        chunk_steps=chunk_steps,
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
        key_grid_timestamps=times,
    )


def test_resample_redraws_jitter_from_the_lattice_not_from_the_source_cache():
    """Jittering an already jittered cache widens the band past the one requested."""
    episodes = np.zeros(90, dtype=np.int64)
    times = np.arange(90) / 30.0
    chunks = np.zeros((90, 8, 7), dtype=np.float32)
    cache = _full_rate_cache(episodes, times, chunks)
    # A source cache that was itself extracted with jitter, as the default does.
    jittered = dataclasses.replace(
        cache, key_timestamps=fast_dataset.jitter_key_timestamps(times, jitter_s=0.01, rng=0)
    )

    resampled = fast_dataset.resample_slow_cache(
        jittered, episodes, times, slow_rate_range_hz=30.0, jitter_s=0.01, rng=1
    )

    offsets = resampled.key_timestamps - times[resampled.key_dataset_indices]
    assert np.max(np.abs(offsets)) <= 0.01 + 1e-12, "Jitter was applied on top of the source cache's jitter"


def test_resample_refuses_a_cache_without_the_lattice_timestamps():
    episodes = np.zeros(90, dtype=np.int64)
    times = np.arange(90) / 30.0
    cache = dataclasses.replace(
        _full_rate_cache(episodes, times, np.zeros((90, 8, 7), dtype=np.float32)), key_grid_timestamps=None
    )

    with pytest.raises(ValueError, match="key_grid_timestamps"):
        fast_dataset.resample_slow_cache(cache, episodes, times, slow_rate_range_hz=10.0)


def test_resample_slow_cache_matches_direct_extraction():
    episodes = np.repeat([0, 1], [90, 47]).astype(np.int64)
    times = np.concatenate([np.arange(90), np.arange(47)]) / 30.0
    chunks = np.random.default_rng(0).standard_normal((len(times), 8, 7)).astype(np.float32)
    cache = _full_rate_cache(episodes, times, chunks)

    for rate_hz in (1.0, 5.0, 30.0):
        resampled = fast_dataset.resample_slow_cache(cache, episodes, times, slow_rate_range_hz=rate_hz, jitter_s=0.0)
        key_rows, mapping = fast_dataset.select_slow_update_rows(
            episodes, times, period_range_s=(1.0 / rate_hz, 1.0 / rate_hz)
        )
        reference, time_features = fast_dataset.build_reference_rollout(
            chunks[key_rows],
            mapping,
            times,
            times[key_rows],
            action_period_s=1 / 30,
            context_age_scale_s=0.5,
            chunk_steps=1,
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
        _full_rate_cache(episodes, times, chunks), episodes, times, slow_rate_range_hz=2.0
    )
    with pytest.raises(ValueError, match="absent from the cache"):
        fast_dataset.resample_slow_cache(sparse, episodes, times, slow_rate_range_hz=10.0)


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
        chunk_steps=1,
    )
    np.testing.assert_allclose(reference[1, 0, :6], [0.5, 1, 1.5, 2, 2.5, 3])
    assert reference[1, 0, 6] == -1
    np.testing.assert_allclose(reference[2, 0], chunks[0, 1])
    np.testing.assert_allclose(time_features[:, 0], [0.0, 0.25, 0.5])
    np.testing.assert_allclose(time_features[:, 1], [0.0, 0.5, 0.0])


def test_reference_rollout_walks_forward_one_action_period_per_step():
    chunks = np.arange(6, dtype=np.float32).reshape(1, 6, 1) * np.ones((1, 1, 2), dtype=np.float32)
    reference, _ = fast_dataset.build_reference_rollout(
        chunks,
        np.array([0]),
        np.array([0.2]),
        np.array([0.0]),
        action_period_s=0.1,
        context_age_scale_s=1.0,
        chunk_steps=3,
    )
    # The row sits two action periods after the key, so the emitted chunk is
    # waypoints 2, 3 and 4 of the Slow chunk.
    np.testing.assert_allclose(reference[0, :, 0], [2.0, 3.0, 4.0], atol=1e-6)


def test_time_features_are_not_collinear():
    # phase and age were the same elapsed time over two constants, so they carried
    # one dimension of information. Age and alpha must not.
    chunks = np.zeros((1, 60, 4), dtype=np.float32)
    row_times = np.linspace(0.0, 0.5, 97)
    _, time_features = fast_dataset.build_reference_rollout(
        chunks,
        np.zeros(len(row_times), dtype=np.int64),
        row_times,
        np.array([0.0]),
        action_period_s=1 / 30,
        context_age_scale_s=0.6,
        chunk_steps=1,
    )
    age, alpha = time_features[:, 0], time_features[:, 1]
    assert abs(float(np.corrcoef(age, alpha)[0, 1])) < 0.2
    assert age.max() < 1.0, "the age token must not saturate inside the operating band"
    # Age is monotone over the sweep while alpha runs through its full sawtooth.
    assert np.all(np.diff(age) > 0)
    assert alpha.max() > 0.9 and np.mean((alpha > 0.01) & (alpha < 0.99)) > 0.8


def test_assign_ready_packets_hides_keys_until_latency_elapses():
    episodes = np.zeros(6, dtype=np.int64)
    times = np.arange(6) / 30.0
    key_rows = np.array([0, 3], dtype=np.int64)
    mapping = fast_dataset.assign_ready_packets(episodes, times, key_rows, times[key_rows], ready_delay_s=0.1)
    # 100 ms is exactly three 30 Hz frames, so rows 0-2 have no ready packet.
    np.testing.assert_array_equal(mapping, [-1, -1, -1, 0, 0, 0])


def test_assign_ready_packets_drops_a_packet_overtaken_in_flight():
    episodes = np.zeros(12, dtype=np.int64)
    times = np.arange(12) / 30.0
    key_rows = np.array([0, 3], dtype=np.int64)
    # The first packet takes 300 ms and the second 40 ms, so the second is installed
    # first and the first is stale on arrival. Serving would discard it.
    mapping = fast_dataset.assign_ready_packets(
        episodes, times, key_rows, times[key_rows], ready_delay_s=np.array([0.3, 0.04])
    )
    # Packet 0 is never selected. Packet 1 is observed at t=0.1 and lands at t=0.14,
    # so it first serves row 5 at t=1/6.
    assert set(mapping.tolist()) == {-1, 1}
    np.testing.assert_array_equal(mapping[:7], [-1, -1, -1, -1, -1, 1, 1])


def test_sampled_ready_delays_span_the_requested_band():
    delays = fast_dataset.sample_ready_delays(2000, delay_range_s=(0.05, 0.30), rng=0)
    assert 0.05 <= delays.min() and delays.max() <= 0.30
    assert delays.min() < 0.06 and delays.max() > 0.29
    np.testing.assert_array_equal(fast_dataset.sample_ready_delays(4, delay_range_s=0.1), np.full(4, 0.1))


def test_jittered_key_times_train_interpolation_alphas():
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
    row_times = np.array([0.0, 1.0 / 30.0, 2.0 / 30.0])
    jittered_times = np.array([0.01])
    reference, jittered_features = fast_dataset.build_reference_rollout(
        chunks,
        np.zeros(3, dtype=np.int64),
        row_times,
        jittered_times,
        action_period_s=1 / 30,
        context_age_scale_s=0.3,
        chunk_steps=1,
    )
    # On the 30 Hz lattice the unjittered ages land on chunk indices, so the
    # interpolated pose equals a stored waypoint. A 10 ms offset does not.
    assert not np.allclose(reference[1, 0, :6], chunks[0, 1, :6])
    age = row_times[1] - jittered_times[0]
    alpha = float(age / (1 / 30) - np.floor(age / (1 / 30)))
    assert 0.0 < alpha < 1.0
    np.testing.assert_allclose(jittered_features[1, 0], np.clip(age / 0.3, 0.0, 1.0), atol=1e-6)
    np.testing.assert_allclose(jittered_features[1, 1], alpha, atol=1e-6)
