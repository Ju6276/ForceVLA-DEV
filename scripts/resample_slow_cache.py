"""Derive a lower-rate Slow cache from a full-rate one without rerunning the model.

Key rows are selected greedily from the first row of each episode, so the key
rows of any Slow rate are a subset of the full-rate rows. A single full-rate
extraction therefore covers an entire Slow-rate sweep.
"""

import argparse
import json
import pathlib

import numpy as np

from openpi.policies import rotation_6d as rot
from openpi.training import fast_dataset


def _band(values: list[float], name: str) -> tuple[float, float]:
    """Read a `VALUE` or `MIN MAX` argument as an ordered band."""
    if len(values) == 1:
        return (float(values[0]), float(values[0]))
    if len(values) == 2 and values[0] <= values[1]:
        return (float(values[0]), float(values[1]))
    raise ValueError(f"{name} takes one value or an ordered MIN MAX pair, got {values}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3-dir", type=pathlib.Path, required=True)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--slow-rate-hz",
        type=float,
        nargs="+",
        required=True,
        help="Target Slow rate, as a single value or a MIN MAX band each interval is drawn from.",
    )
    parser.add_argument(
        "--slow-latency-ms",
        type=float,
        nargs="+",
        default=None,
        help="Optional new latency band. Omit to reuse the latencies already drawn for these packets.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rate_range_hz = _band(args.slow_rate_hz, "--slow-rate-hz")
    if min(rate_range_hz) <= 0:
        raise ValueError("Target Slow rate must be positive")
    latency_range_s = None
    if args.slow_latency_ms is not None:
        latency_range_s = tuple(value / 1000.0 for value in _band(args.slow_latency_ms, "--slow-latency-ms"))
        if min(latency_range_s) < 0:
            raise ValueError("Slow latency must be non-negative")

    cache = fast_dataset.load_slow_cache(args.input)
    stage3 = fast_dataset.load_stage3_fast_arrays(args.stage3_dir, chunk_steps=cache.chunk_steps)
    if len(cache.row_key_positions) != len(stage3.dataset_indices):
        raise ValueError(
            f"Cache covers {len(cache.row_key_positions)} rows but Stage-3 has {len(stage3.dataset_indices)}"
        )
    resampled = fast_dataset.resample_slow_cache(
        cache,
        stage3.episode_indices,
        stage3.timestamps,
        slow_rate_range_hz=rate_range_hz,
        ready_delay_range_s=latency_range_s,
        rng=args.seed,
    )

    arrays = {
        "key_dataset_indices": resampled.key_dataset_indices,
        "key_episode_indices": resampled.key_episode_indices,
        "key_timestamps": resampled.key_timestamps,
        "context_tokens": resampled.context_tokens,
        "context_mask": resampled.context_mask,
        "action_chunks": resampled.action_chunks,
        "row_key_positions": resampled.row_key_positions,
        "reference_actions": resampled.reference_actions,
        "time_features": resampled.time_features,
        "context_age_scale_s": np.float64(resampled.context_age_scale_s),
        "action_period_s": np.float64(resampled.action_period_s),
    }
    if resampled.key_ready_delays is not None:
        arrays["key_ready_delays"] = resampled.key_ready_delays
    if resampled.row_ready is not None:
        arrays["row_ready"] = resampled.row_ready
    if resampled.key_grid_timestamps is not None:
        arrays["key_grid_timestamps"] = resampled.key_grid_timestamps
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(args.output)

    ready = (
        np.asarray(resampled.row_ready, dtype=bool)
        if resampled.row_ready is not None
        else np.ones(len(stage3.dataset_indices), dtype=bool)
    )
    summary = {
        "format_version": 1,
        "source_cache": str(args.input.resolve()),
        "stage3_dir": str(args.stage3_dir.resolve()),
        "rows": len(stage3.dataset_indices),
        "source_slow_packets": len(cache.key_timestamps),
        "slow_packets": len(resampled.key_timestamps),
        "slow_rate_range_hz": list(rate_range_hz),
        "slow_latency_range_ms": (
            [value * 1000.0 for value in latency_range_s] if latency_range_s is not None else None
        ),
        "action_rate_hz": 1.0 / resampled.action_period_s,
        "chunk_steps": resampled.chunk_steps,
        "context_age_scale_ms": resampled.context_age_scale_s * 1000.0,
        "pooled_context_shape": list(resampled.context_tokens.shape),
        "action_chunk_shape": list(resampled.action_chunks.shape),
        "reference_rollout_shape": list(resampled.reference_actions.shape),
        "ready_row_fraction": float(np.mean(resampled.row_ready)) if resampled.row_ready is not None else 1.0,
        "saturated_age_fraction": float(np.mean(resampled.time_features[ready, 0] >= 1.0)) if ready.any() else None,
        "reference_vs_teacher_null_on_ready_rows_mse": float(
            np.mean(
                np.square(
                    resampled.reference_actions[ready][..., : rot.POSE_DIMS]
                    - (stage3.full_pose - stage3.residual_pose)[ready]
                )
            )
        )
        if ready.any()
        else None,
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
