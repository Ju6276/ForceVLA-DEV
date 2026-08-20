"""Build inspectable, causal 100 Hz force histories for button-press data.

For an observation at time t and the default 100 ms window, the output grid is
    [t - 90 ms, t - 80 ms, ..., t].
Each grid point uses zero-order hold from the newest raw force sample at or
before that point.  Samples from the future are never selected.  Stale or
missing values are zero-filled and marked invalid in ``force_history_mask``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FORCE_COLUMNS = ["force_x", "force_y", "force_z", "moment_x", "moment_y", "moment_z"]


def causal_force_history(
    observation_times: np.ndarray,
    force_times: np.ndarray,
    force_values: np.ndarray,
    *,
    output_rate_hz: float,
    window_ms: float,
    max_sample_age_ms: float,
) -> dict[str, np.ndarray]:
    """Causally resample an asynchronous force stream onto fixed history grids."""
    num_slots_float = output_rate_hz * window_ms / 1000.0
    num_slots = round(num_slots_float)
    if num_slots < 1 or not np.isclose(num_slots, num_slots_float):
        raise ValueError("output_rate_hz * window_ms / 1000 must be a positive integer")
    if force_values.shape != (force_times.size, 6):
        raise ValueError(f"Expected force_values shape ({force_times.size}, 6), got {force_values.shape}")
    if np.any(np.diff(observation_times) < 0) or np.any(np.diff(force_times) < 0):
        raise ValueError("Timestamps must be monotonically non-decreasing")

    period_s = 1.0 / output_rate_hz
    window_s = window_ms / 1000.0
    offsets = np.arange(num_slots - 1, -1, -1, dtype=np.float64) * period_s
    grid_times = observation_times[:, None].astype(np.float64) - offsets[None, :]

    # searchsorted(..., side="right") - 1 implements causal zero-order hold.
    source_indices = np.searchsorted(force_times, grid_times, side="right") - 1
    safe_indices = np.maximum(source_indices, 0)
    source_times = force_times[safe_indices]
    sample_age_s = grid_times - source_times
    lower_bounds = observation_times[:, None] - window_s
    valid = (
        (source_indices >= 0)
        & (source_times > lower_bounds)
        & (source_times <= grid_times + 1e-12)
        & (sample_age_s >= -1e-12)
        & (sample_age_s <= max_sample_age_ms / 1000.0)
    )

    history = force_values[safe_indices].astype(np.float32)
    history[~valid] = 0.0
    source_times = source_times.astype(np.float64)
    source_times[~valid] = np.nan
    sample_age_s = sample_age_s.astype(np.float32)
    sample_age_s[~valid] = np.nan
    return {
        "force_history": history,
        "force_history_mask": valid,
        "grid_timestamps": grid_times,
        "source_timestamps": source_times,
        "sample_age_s": sample_age_s,
    }


def effective_rate_hz(times: np.ndarray) -> float:
    if times.size < 2 or times[-1] <= times[0]:
        return 0.0
    # Overall coverage rate intentionally penalizes long gaps. The reciprocal
    # median interval can overstate bursty streams as being faster than they are.
    return float((times.size - 1) / (times[-1] - times[0]))


def quality_bin(rate_hz: float) -> str:
    if rate_hz >= 200:
        return "at_least_200hz"
    if rate_hz >= 100:
        return "100_to_200hz"
    return "below_100hz"


def assign_episode_split(rows: list[dict], seed: int, train_fraction: float) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    result: dict[str, list[int]] = {"train": [], "val": [], "excluded_below_100hz": []}
    for bin_name in ("at_least_200hz", "100_to_200hz"):
        episodes = np.array([r["episode_index"] for r in rows if r["quality_bin"] == bin_name], dtype=np.int64)
        rng.shuffle(episodes)
        num_train = round(len(episodes) * train_fraction)
        num_train = min(max(num_train, 1), max(len(episodes) - 1, 1)) if len(episodes) > 1 else len(episodes)
        result["train"].extend(episodes[:num_train].tolist())
        result["val"].extend(episodes[num_train:].tolist())
    result["excluded_below_100hz"] = [r["episode_index"] for r in rows if r["quality_bin"] == "below_100hz"]
    for values in result.values():
        values.sort()
    return result


def save_inspection_plot(output_path: Path, examples: list[dict]) -> None:
    fig, axes = plt.subplots(len(examples), 1, figsize=(12, 3.3 * len(examples)), squeeze=False)
    for ax, example in zip(axes[:, 0], examples, strict=True):
        ax.plot(example["raw_t"], example["raw_fz"], color="0.65", linewidth=0.8, label="raw Fz")
        valid = example["mask"]
        ax.scatter(
            example["grid_t"][valid],
            example["history_fz"][valid],
            s=8,
            color="tab:blue",
            label="causal 100 Hz slots",
        )
        ax.scatter(
            example["grid_t"][~valid],
            np.zeros(np.count_nonzero(~valid)),
            s=10,
            marker="x",
            color="tab:red",
            label="invalid/missing slot",
        )
        ax.set_title(
            f"episode {example['episode_index']:06d}: measured {example['rate_hz']:.1f} Hz, "
            f"valid slots {example['valid_fraction']:.1%}"
        )
        ax.set_xlabel("episode time (s)")
        ax.set_ylabel("Fz (N)")
        ax.legend(loc="upper right", ncols=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--standard-data-dir",
        type=Path,
        default=Path("data/panda_button_press_analysis/standard/data/chunk-000"),
    )
    parser.add_argument(
        "--force-data-dir",
        type=Path,
        default=Path("data/panda_button_press_analysis/high_rate_force/chunk-000"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/panda_button_press_100hz_causal"))
    parser.add_argument("--output-rate-hz", type=float, default=100.0)
    parser.add_argument("--window-ms", type=float, default=100.0)
    parser.add_argument("--max-sample-age-ms", type=float, default=12.0)
    parser.add_argument("--min-valid-slots", type=int, default=8)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode_dir = args.output_dir / "episodes"
    episode_dir.mkdir(exist_ok=True)
    sidecar_dir = args.output_dir / "raw_sidecars"
    sidecar_dir.mkdir(exist_ok=True)

    standard_paths = sorted(args.standard_data_dir.glob("episode_*.parquet"))
    if not standard_paths:
        raise FileNotFoundError(f"No standard episodes found in {args.standard_data_dir}")

    rows: list[dict] = []
    examples: list[dict] = []
    for standard_path in standard_paths:
        force_path = args.force_data_dir / standard_path.name
        if not force_path.exists():
            raise FileNotFoundError(f"Missing force episode: {force_path}")
        standard = pd.read_parquet(standard_path)
        force = pd.read_parquet(force_path).sort_values("episode_time_s", kind="stable")
        episode_index = int(standard["episode_index"].iloc[0])
        observation_times = standard["timestamp"].to_numpy(dtype=np.float64)
        force_times = force["episode_time_s"].to_numpy(dtype=np.float64)
        force_values = force[FORCE_COLUMNS].to_numpy(dtype=np.float64)
        np.savez_compressed(
            sidecar_dir / f"episode_{episode_index:06d}.npz",
            force=force_values.astype(np.float32),
            timestamps=force_times,
        )
        result = causal_force_history(
            observation_times,
            force_times,
            force_values,
            output_rate_hz=args.output_rate_hz,
            window_ms=args.window_ms,
            max_sample_age_ms=args.max_sample_age_ms,
        )
        mask = result["force_history_mask"]
        valid_count = mask.sum(axis=1).astype(np.uint8)
        complete_history = observation_times >= args.window_ms / 1000.0 - 1e-9
        usable = complete_history & mask[:, -1] & (valid_count >= args.min_valid_slots)
        rate_hz = effective_rate_hz(force_times)
        positive_dt = np.diff(force_times)
        positive_dt = positive_dt[positive_dt > 0]

        # Assertions are intentionally retained in the builder as a causality audit.
        assert result["force_history"].shape[1:] == (int(args.output_rate_hz * args.window_ms / 1000), 6)
        assert np.all(result["grid_timestamps"] <= observation_times[:, None] + 1e-12)
        assert np.all(result["source_timestamps"][mask] <= result["grid_timestamps"][mask] + 1e-12)
        assert np.nanmax(result["sample_age_s"]) <= args.max_sample_age_ms / 1000.0 + 1e-7

        state = np.stack(standard["observation.state"].to_numpy()).astype(np.float32)
        np.savez_compressed(
            episode_dir / f"episode_{episode_index:06d}.npz",
            episode_index=np.int64(episode_index),
            frame_index=standard["frame_index"].to_numpy(dtype=np.int64),
            observation_timestamps=observation_times,
            grid_timestamps=result["grid_timestamps"],
            source_timestamps=result["source_timestamps"],
            sample_age_s=result["sample_age_s"],
            force_history=result["force_history"],
            force_history_mask=mask,
            valid_count=valid_count,
            usable_window=usable,
            standard_30hz_force=state[:, 7:13],
        )
        row = {
            "episode_index": episode_index,
            "observation_frames": len(observation_times),
            "raw_force_samples": len(force_times),
            "effective_rate_hz": rate_hz,
            "quality_bin": quality_bin(rate_hz),
            "median_raw_dt_ms": float(np.median(positive_dt) * 1000) if positive_dt.size else float("nan"),
            "p95_raw_dt_ms": float(np.quantile(positive_dt, 0.95) * 1000) if positive_dt.size else float("nan"),
            "max_raw_gap_ms": float(np.max(positive_dt) * 1000) if positive_dt.size else float("nan"),
            "valid_slot_fraction": float(mask.mean()),
            "usable_window_count": int(usable.sum()),
            "usable_window_fraction": float(usable.mean()),
        }
        rows.append(row)

        if episode_index in {0, 41}:
            examples.append(
                {
                    "episode_index": episode_index,
                    "raw_t": force_times,
                    "raw_fz": force_values[:, 2],
                    "grid_t": result["grid_timestamps"].reshape(-1),
                    "history_fz": result["force_history"][:, :, 2].reshape(-1),
                    "mask": mask.reshape(-1),
                    "rate_hz": rate_hz,
                    "valid_fraction": float(mask.mean()),
                }
            )

    split = assign_episode_split(rows, args.split_seed, args.train_fraction)
    with (args.output_dir / "manifest.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "split.json").open("w") as file:
        json.dump(split, file, indent=2)

    all_rates = np.array([r["effective_rate_hz"] for r in rows])
    summary = {
        "definition": "10 causal slots at [t-90ms, ..., t], covering ten 10ms bins in (t-100ms, t]",
        "alignment": "causal zero-order hold; newest raw sample <= each grid timestamp",
        "output_rate_hz": args.output_rate_hz,
        "window_ms": args.window_ms,
        "num_slots": int(args.output_rate_hz * args.window_ms / 1000),
        "max_sample_age_ms": args.max_sample_age_ms,
        "min_valid_slots": args.min_valid_slots,
        "episodes": len(rows),
        "measured_rate_hz": {
            "min": float(all_rates.min()),
            "median": float(np.median(all_rates)),
            "max": float(all_rates.max()),
        },
        "episode_quality_counts": {
            name: sum(r["quality_bin"] == name for r in rows)
            for name in ("at_least_200hz", "100_to_200hz", "below_100hz")
        },
        "aggregate_valid_slot_fraction": float(
            np.average([r["valid_slot_fraction"] for r in rows], weights=[r["observation_frames"] for r in rows])
        ),
        "aggregate_usable_window_fraction": float(
            np.average([r["usable_window_fraction"] for r in rows], weights=[r["observation_frames"] for r in rows])
        ),
        "split_counts": {key: len(value) for key, value in split.items()},
    }
    with (args.output_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
    if examples:
        save_inspection_plot(args.output_dir / "causal_100hz_examples.png", examples)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
