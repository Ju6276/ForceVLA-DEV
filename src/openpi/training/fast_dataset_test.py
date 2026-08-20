import json

import jax.numpy as jnp
import numpy as np

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


def test_select_slow_update_rows_is_causal_and_resets_per_episode():
    episodes = np.array([0, 0, 0, 0, 1, 1, 1])
    times = np.array([0.0, 0.04, 0.1, 0.19, 0.0, 0.09, 0.11])
    keys, mapping = fast_dataset.select_slow_update_rows(episodes, times, update_period_s=0.1)
    np.testing.assert_array_equal(keys, [0, 2, 4, 6])
    np.testing.assert_array_equal(mapping, [0, 0, 1, 1, 2, 2, 3])


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
