"""Score one trained Fast student across a grid of Slow timing operating points.

The Fast student is trained on a randomized band of Slow rates and latencies, so the
claim it supports is not "it works at the timing we measured once" but "it degrades
gracefully as the Slow path gets slower". That claim needs the performance reported as
a function of measured timing, which is what this produces.

Nothing is retrained and no cache is re-extracted: a full-rate cache holds every key
row any lower rate could ask for, so each operating point is derived offline by
`resample_slow_cache`. Points outside the trained band are the interesting ones, since
they are where a student that merely memorized one latency would fall apart.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

import evaluate_fast_residual as evaluator

import openpi.models.slow_fast as slow_fast
import openpi.shared.normalize as normalize_lib
import openpi.training.fast_dataset as fast_dataset


def _band(values: list[float], name: str) -> tuple[float, float]:
    if len(values) == 1:
        return (float(values[0]), float(values[0]))
    if len(values) == 2 and values[0] <= values[1]:
        return (float(values[0]), float(values[1]))
    raise ValueError(f"{name} takes one value or MIN MAX with MIN <= MAX")


def _percent(value: float | None) -> str:
    return "     -" if value is None else f"{value * 100:+6.2f}%"


def _trained_band(fast_run: pathlib.Path | None) -> dict | None:
    if fast_run is None:
        return None
    path = fast_run if fast_run.suffix == ".json" else fast_run / "metadata.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text()).get("trained_timing")


def _in_band(trained: dict | None, rate_hz: float, latency_ms: float) -> bool | None:
    if not trained:
        return None
    rate = trained.get("slow_rate_range_hz")
    latency = trained.get("slow_latency_range_ms")
    if not rate or not latency:
        return None
    return bool(rate[0] <= rate_hz <= rate[1] and latency[0] <= latency_ms <= latency[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=pathlib.Path, required=True)
    parser.add_argument(
        "--slow-cache",
        type=pathlib.Path,
        required=True,
        help=(
            "Cache extracted at full rate (--slow-rate-hz 0). A lower-rate cache cannot be "
            "resampled upward, so its own realization would be the only point available."
        ),
    )
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument(
        "--fast-run",
        type=pathlib.Path,
        default=None,
        help=(
            "Training run directory holding metadata.json. Used only to label each point as "
            "inside or outside the band the student was trained on, which is what separates "
            "interpolation from extrapolation in the resulting curve."
        ),
    )
    parser.add_argument(
        "--norm-stats-dir",
        type=pathlib.Path,
        default=pathlib.Path("assets/forcevla_button_temporal_100hz/panda_button_press_temporal_100hz_train56"),
    )
    parser.add_argument(
        "--slow-rate-hz",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 15.0, 20.0, 30.0],
        help="Slow update rates to score, each held fixed for its point.",
    )
    parser.add_argument(
        "--slow-latency-ms",
        type=float,
        nargs="+",
        default=[50.0, 150.0, 300.0, 450.0, 600.0],
        help="Slow serving latencies to score, each held fixed for its point.",
    )
    parser.add_argument("--contact-threshold-n", type=float, default=5.0)
    parser.add_argument("--baseline-rows", type=int, default=15)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if min(args.slow_rate_hz) <= 0 or min(args.slow_latency_ms) < 0:
        raise ValueError("Slow rates must be positive and latencies non-negative")
    if args.contact_threshold_n <= 0:
        raise ValueError("contact-threshold-n must be positive")

    source = fast_dataset.load_slow_cache(args.slow_cache)
    arrays = fast_dataset.load_stage3_fast_arrays(args.targets, chunk_steps=source.chunk_steps)
    if len(source.row_key_positions) != len(arrays.dataset_indices):
        raise ValueError(
            f"Slow cache covers {len(source.row_key_positions)} rows but Stage-3 has {len(arrays.dataset_indices)}"
        )
    norm_stats = normalize_lib.load(args.norm_stats_dir)
    wrench_by_row = fast_dataset.latest_physical_wrench(
        arrays.force_history, arrays.force_history_mask, norm_stats["force_history"]
    )
    baseline_by_row = fast_dataset.episode_baseline_wrench(
        wrench_by_row, arrays.episode_indices, num_rows=args.baseline_rows
    )
    model, predicts_staleness = evaluator._load_model(args.checkpoint.resolve(), chunk_steps=source.chunk_steps)
    trained = _trained_band(args.fast_run)

    print(
        f"context age scale {source.context_age_scale_s * 1000:.0f} ms; the age token is clipped at 1.0, "
        "so points whose age exceeds it are reported as saturated",
        flush=True,
    )
    points = []
    for rate_hz in args.slow_rate_hz:
        for latency_ms in args.slow_latency_ms:
            cache = fast_dataset.resample_slow_cache(
                source,
                arrays.episode_indices,
                arrays.timestamps,
                slow_rate_range_hz=float(rate_hz),
                ready_delay_range_s=float(latency_ms) / 1000.0,
                rng=args.seed,
            )
            ready = np.flatnonzero(cache.row_ready)
            if len(ready) == 0:
                print(f"  rate={rate_hz:>5.1f}Hz latency={latency_ms:>6.1f}ms  no ready rows, skipped", flush=True)
                continue
            metrics = evaluator.evaluate(
                model,
                arrays,
                cache,
                norm_stats,
                wrench_by_row,
                baseline_by_row,
                predicts_staleness=predicts_staleness,
                contact_threshold_n=args.contact_threshold_n,
                baseline_rows=args.baseline_rows,
                batch_size=args.batch_size,
                seed=args.seed,
            )
            overall = metrics["strata"]["all"]
            action = overall["action_vs_teacher_full"]
            drift = overall.get("staleness_vs_reference_drift")
            # Matching the Teacher is not the same as being right, and a force-conditioned
            # correction is supposed to earn its keep while touching something. Both are
            # already computed per stratum, so dropping them would leave the curve unable
            # to answer whether the advantage survives contact.
            expert = overall["action_vs_expert"]
            contact = metrics["strata"].get("contact")
            ages = cache.time_features[ready, 0]
            point = {
                "slow_rate_hz": float(rate_hz),
                "slow_latency_ms": float(latency_ms),
                "in_trained_band": _in_band(trained, float(rate_hz), float(latency_ms)),
                "rows_scored": int(len(ready)),
                "ready_row_fraction": float(np.mean(cache.row_ready)),
                "slow_packets": int(len(cache.key_timestamps)),
                "mean_normalized_context_age": float(np.mean(ages)),
                "saturated_age_fraction": float(np.mean(ages >= 1.0)),
                "force_residual_gain": overall["residual_vs_teacher_deviation"]["full"]["gain_vs_zero_residual"],
                "staleness_gain": None if drift is None else drift["gain_vs_zero_staleness"],
                "staleness_force_blindness_max_deviation": metrics["staleness_force_blindness_max_deviation"],
                "slow_only_translation_rmse_m": action["slow_reference_only"]["translation_rmse_m"],
                "slow_only_rotation_rmse_rad": action["slow_reference_only"]["rotation_geodesic_rmse_rad"],
                "composed_translation_rmse_m": action["slow_plus_full"]["translation_rmse_m"],
                "composed_rotation_rmse_rad": action["slow_plus_full"]["rotation_geodesic_rmse_rad"],
                "translation_gain_vs_slow_only": action["fast_gain_vs_slow_only"]["translation_rmse_m"],
                "rotation_gain_vs_slow_only": action["fast_gain_vs_slow_only"]["rotation_geodesic_rmse_rad"],
                "expert_slow_only_translation_rmse_m": expert["slow_reference_only"]["translation_rmse_m"],
                "expert_composed_translation_rmse_m": expert["slow_plus_full"]["translation_rmse_m"],
                "expert_translation_gain_vs_slow_only": expert["fast_gain_vs_slow_only"]["translation_rmse_m"],
                "expert_rotation_gain_vs_slow_only": expert["fast_gain_vs_slow_only"]["rotation_geodesic_rmse_rad"],
                "contact_rows": metrics["contact"]["contact_rows"],
                "contact_force_residual_gain": (
                    None if contact is None else contact["residual_vs_teacher_deviation"]["full"]["gain_vs_zero_residual"]
                ),
                "contact_translation_gain_vs_slow_only": (
                    None if contact is None else contact["action_vs_teacher_full"]["fast_gain_vs_slow_only"]["translation_rmse_m"]
                ),
                "contact_expert_translation_gain_vs_slow_only": (
                    None if contact is None else contact["action_vs_expert"]["fast_gain_vs_slow_only"]["translation_rmse_m"]
                ),
            }
            points.append(point)
            band = {True: "in ", False: "OUT", None: "  ?"}[point["in_trained_band"]]
            print(
                f"  rate={rate_hz:>5.1f}Hz latency={latency_ms:>6.1f}ms {band}"
                f"  age={point['mean_normalized_context_age']:.3f}"
                f" sat={point['saturated_age_fraction']:.2f}"
                f"  force={point['force_residual_gain']:.4f}"
                f"  stale={point['staleness_gain'] if point['staleness_gain'] is None else round(point['staleness_gain'], 4)}"
                f"  trans={point['translation_gain_vs_slow_only'] * 100:+6.2f}%"
                f"  rot={point['rotation_gain_vs_slow_only'] * 100:+6.2f}%"
                f"  expert={point['expert_translation_gain_vs_slow_only'] * 100:+6.2f}%"
                f"  contact={_percent(point['contact_translation_gain_vs_slow_only'])}",
                flush=True,
            )

    if not points:
        raise ValueError("No operating point produced a ready row; widen the grid")
    result = {
        "checkpoint": str(args.checkpoint),
        "slow_cache": str(args.slow_cache),
        "norm_stats_dir": str(args.norm_stats_dir),
        "predicts_staleness": predicts_staleness,
        "context_age_scale_s": source.context_age_scale_s,
        "update_jitter_s": slow_fast.DEFAULT_UPDATE_JITTER_S,
        "trained_timing": trained,
        "points": points,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
