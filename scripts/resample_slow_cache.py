"""Derive a lower-rate Slow cache from a full-rate one without rerunning the model.

Key rows are selected greedily from the first row of each episode, so the key
rows of any Slow rate are a subset of the full-rate rows. A single full-rate
extraction therefore covers an entire Slow-rate sweep.
"""

import argparse
import json
import pathlib

import numpy as np

from openpi.training import fast_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3-dir", type=pathlib.Path, required=True)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--slow-rate-hz", type=float, required=True)
    args = parser.parse_args()

    if args.slow_rate_hz <= 0:
        raise ValueError("Target Slow rate must be positive")

    stage3 = fast_dataset.load_stage3_fast_arrays(args.stage3_dir)
    cache = fast_dataset.load_slow_cache(args.input, expected_rows=len(stage3.dataset_indices))
    resampled = fast_dataset.resample_slow_cache(
        cache,
        stage3.episode_indices,
        stage3.timestamps,
        slow_rate_hz=args.slow_rate_hz,
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(args.output)

    summary = {
        "format_version": 1,
        "source_cache": str(args.input.resolve()),
        "stage3_dir": str(args.stage3_dir.resolve()),
        "rows": len(stage3.dataset_indices),
        "source_slow_packets": len(cache.key_timestamps),
        "slow_packets": len(resampled.key_timestamps),
        "slow_rate_hz": args.slow_rate_hz,
        "action_rate_hz": 1.0 / resampled.action_period_s,
        "context_age_scale_ms": resampled.context_age_scale_s * 1000.0,
        "pooled_context_shape": list(resampled.context_tokens.shape),
        "action_chunk_shape": list(resampled.action_chunks.shape),
        "reference_vs_teacher_null_first_pose_mse": float(
            np.mean(
                np.square(resampled.reference_actions[:, :6] - (stage3.full_pose - stage3.residual_pose))
            )
        ),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
