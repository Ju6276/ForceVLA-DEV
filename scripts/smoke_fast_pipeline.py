"""End-to-end smoke test of the Fast residual pipeline on synthetic arrays.

The unit tests cover the pieces; this exercises the wiring the scripts depend on:
Stage-3 shards -> full-rate Slow cache -> redrawn timing -> batch -> loss -> step,
plus the offline/serving agreement on the reference chunk and the time token.
"""

from __future__ import annotations

import json
import pathlib
import tempfile

from flax import nnx
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models import force_encoder
from openpi.models import slow_fast
from openpi.serving import slow_fast_runtime
from openpi.training import fast_dataset
from openpi.training import slow_fast_distillation

ROWS, HORIZON, ACTION_DIM, CHUNK_STEPS = 240, 50, 10, 5
ACTION_RATE_HZ = 30.0


def _write_stage3(root: pathlib.Path) -> None:
    rng = np.random.default_rng(0)
    shards = root / "shards"
    shards.mkdir(parents=True)
    episodes = np.repeat([0, 1], [140, 100]).astype(np.int64)
    timestamps = np.concatenate([np.arange(140), np.arange(100)]) / ACTION_RATE_HZ
    full = rng.standard_normal((ROWS, HORIZON, 32)).astype(np.float32)
    residual = 0.1 * rng.standard_normal((ROWS, HORIZON, 9)).astype(np.float32)
    np.savez(
        shards / "rows.npz",
        dataset_indices=np.arange(ROWS),
        episode_indices=episodes,
        timestamps=timestamps,
        normalized_state=rng.standard_normal((ROWS, 32)).astype(np.float32),
        normalized_force_history=rng.standard_normal((ROWS, 10, 6)).astype(np.float32),
        force_history_mask=np.ones((ROWS, 10), dtype=np.bool_),
        normalized_full_actions=full,
        normalized_pose_residual=residual,
        normalized_expert_actions=full,
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {"complete": True, "extraction_size": ROWS, "shards": [{"complete": True, "path": "shards/rows.npz"}]}
        )
    )


def _full_rate_cache(arrays) -> fast_dataset.SlowCache:
    rng = np.random.default_rng(1)
    key_rows, mapping = fast_dataset.select_slow_update_rows(
        arrays.episode_indices, arrays.timestamps, period_range_s=None
    )
    chunks = rng.standard_normal((len(key_rows), HORIZON, ACTION_DIM)).astype(np.float32)
    reference, time_features = fast_dataset.build_reference_rollout(
        chunks,
        mapping,
        arrays.timestamps,
        arrays.timestamps[key_rows],
        action_period_s=1.0 / ACTION_RATE_HZ,
        context_age_scale_s=slow_fast.DEFAULT_CONTEXT_AGE_SCALE_S,
        chunk_steps=CHUNK_STEPS,
    )
    return fast_dataset.SlowCache(
        key_dataset_indices=key_rows,
        key_episode_indices=arrays.episode_indices[key_rows],
        key_timestamps=arrays.timestamps[key_rows],
        context_tokens=np.zeros((len(key_rows), 16, 64), dtype=np.float16),
        context_mask=np.ones((len(key_rows), 16), dtype=bool),
        action_chunks=chunks,
        row_key_positions=mapping,
        reference_actions=reference,
        time_features=time_features,
        context_age_scale_s=slow_fast.DEFAULT_CONTEXT_AGE_SCALE_S,
        action_period_s=1.0 / ACTION_RATE_HZ,
        key_ready_delays=np.zeros(len(key_rows)),
        row_ready=np.ones(ROWS, dtype=bool),
        key_grid_timestamps=arrays.timestamps[key_rows],
    )


def _check_serving_matches_offline(cache: fast_dataset.SlowCache, arrays) -> None:
    """Serving must reproduce the offline rollout and token, or Fast is deployed blind."""
    row = int(np.flatnonzero(cache.row_ready)[-1])
    key = int(cache.row_key_positions[row])
    packet = slow_fast_runtime.SlowPacket(
        observation_timestamp=float(cache.key_timestamps[key]),
        ready_timestamp=float(cache.key_timestamps[key] + cache.key_ready_delays[key]),
        reference_start_timestamp=float(cache.key_timestamps[key]),
        intent_tokens=np.zeros((16, 64), dtype=np.float32),
        intent_mask=np.ones(16, dtype=bool),
        reference_actions=cache.action_chunks[key],
        action_period_s=cache.action_period_s,
        version=0,
        context_age_scale_s=cache.context_age_scale_s,
    )
    timestamp = float(arrays.timestamps[row])
    np.testing.assert_allclose(
        slow_fast_runtime.sample_reference(packet, timestamp, steps=CHUNK_STEPS, max_staleness_s=10.0),
        cache.reference_actions[row],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        slow_fast_runtime.reference_time_features(packet, timestamp),
        cache.time_features[row],
        atol=1e-5,
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        _write_stage3(root)
        arrays = fast_dataset.load_stage3_fast_arrays(root, chunk_steps=CHUNK_STEPS)
        assert arrays.residual_pose.shape == (ROWS, CHUNK_STEPS, 9)

        source = _full_rate_cache(arrays)
        _check_serving_matches_offline(source, arrays)

        realizations = [
            fast_dataset.resample_slow_cache(
                source,
                arrays.episode_indices,
                arrays.timestamps,
                slow_rate_range_hz=slow_fast.DEFAULT_SLOW_RATE_RANGE_HZ,
                ready_delay_range_s=slow_fast.DEFAULT_SLOW_LATENCY_RANGE_S,
                rng=seed,
            )
            for seed in (0, 1)
        ]
        for cache in realizations:
            assert cache.reference_actions.shape == (ROWS, CHUNK_STEPS, ACTION_DIM)
            ready = cache.row_ready
            assert 0.5 < ready.mean() <= 1.0, f"too few ready rows: {ready.mean():.3f}"
            age = cache.time_features[ready, 0]
            assert age.max() < 1.0, "the age token saturated; widen context_age_scale_s"
            alpha = cache.time_features[ready, 1]
            interior = float(np.mean((alpha > 1e-3) & (alpha < 1 - 1e-3)))
            assert interior > 0.5, f"alpha collapsed onto the action lattice ({interior:.3f}); jitter is off"
            _check_serving_matches_offline(cache, arrays)
        assert not np.array_equal(realizations[0].key_timestamps, realizations[1].key_timestamps), (
            "redrawing the timing must actually change the realization"
        )

        config = slow_fast.FastResidualConfig(
            chunk_steps=CHUNK_STEPS,
            intent_dim=16,
            width=32,
            mlp_dim=64,
            num_heads=2,
            num_kv_heads=1,
            head_dim=16,
            force_encoder=force_encoder.ForceEncoderConfig(
                type="tcn",
                hidden_dims=(32, 32),
                dilations=(1, 2),
                dropout_rate=0.0,
                sampling_rate_hz=100,
                window_ms=100,
            ),
        )
        model = slow_fast.FastStudentWithIntentProjector(config, slow_context_dim=64, rngs=nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, optax.adamw(1e-3), wrt=nnx.Param)
        loss_config = slow_fast_distillation.FastDistillationLossConfig(reconstruction_weight=1.0)

        cache = realizations[0]
        pool = np.flatnonzero(cache.row_ready)
        indices = pool[:16]
        reference_chunk = jnp.asarray(cache.reference_actions[indices])
        targets = slow_fast_distillation.FastChunkTargets(
            full_action=jnp.asarray(arrays.full_pose[indices]),
            nominal_action=reference_chunk,
            residual_pose=jnp.asarray(arrays.residual_pose[indices]),
        )

        def loss_fn(module):
            predicted, _, _ = module(
                jnp.asarray(cache.context_tokens[cache.row_key_positions[indices]], dtype=jnp.float32),
                jnp.asarray(cache.context_mask[cache.row_key_positions[indices]]),
                jnp.asarray(arrays.force_history[indices]),
                jnp.asarray(arrays.force_history_mask[indices]),
                jnp.asarray(arrays.state[indices]),
                reference_chunk[:, 0],
                jnp.asarray(cache.time_features[indices]),
                train=True,
            )
            assert predicted.shape == (len(indices), CHUNK_STEPS, 9)
            return slow_fast_distillation.fast_residual_loss(predicted, reference_chunk, targets, loss_config)[0]

        before = float(loss_fn(model))
        for _ in range(30):
            loss, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(grads)
        after = float(loss_fn(model))
        assert np.isfinite(before) and after < before, f"loss did not decrease: {before} -> {after}"

        print(f"ok  rows={ROWS}  chunk_steps={CHUNK_STEPS}  loss {before:.5f} -> {after:.5f}")
        for index, cache in enumerate(realizations):
            ready = cache.row_ready
            print(
                f"  realization {index}: packets={len(cache.key_timestamps)} "
                f"ready={ready.mean():.3f} max_age={cache.time_features[ready, 0].max():.3f} "
                f"alpha_interior={np.mean((cache.time_features[ready, 1] > 1e-3) & (cache.time_features[ready, 1] < 1 - 1e-3)):.3f}"
            )


if __name__ == "__main__":
    main()
