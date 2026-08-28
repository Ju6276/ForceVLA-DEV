"""Offline Button reference-delay comparison for Temporal Teacher and ForceDelta.

This isolates the *additional* delay between an already-completed reference query
and publication.  It does not add the measured VLA inference latency a second time.
Both methods use the same query schedule and latest-ready-packet selection.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

import evaluate_fast_residual as evaluator

from openpi.policies import rotation_6d as rot
from openpi.shared import normalize as normalize_lib
from openpi.training import fast_dataset


def _load_full_teacher_chunks(target_dir: pathlib.Path, *, horizon: int) -> np.ndarray:
    manifest = json.loads((target_dir / "manifest.json").read_text())
    chunks = []
    for shard in manifest["shards"]:
        with np.load(target_dir / shard["path"], allow_pickle=False) as values:
            chunks.append(np.asarray(values["normalized_full_actions"][:, :horizon, : rot.POSE_DIMS]))
    result = np.concatenate(chunks).astype(np.float32)
    if len(result) != int(manifest["extraction_size"]):
        raise ValueError("Full-Teacher chunks do not match the Stage-3 manifest")
    return result


def _absolute_pose(delta, base_state, norm_stats):
    return evaluator.to_absolute_pose(delta, base_state, norm_stats["actions"], norm_stats["state"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=pathlib.Path, required=True)
    parser.add_argument("--slow-cache", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--fast-run", type=pathlib.Path, required=True)
    parser.add_argument("--norm-stats-dir", type=pathlib.Path, required=True)
    parser.add_argument("--reference-rate-hz", type=float, default=10.0)
    parser.add_argument("--delay-ms", type=float, nargs="+", default=[0.0, 100.0, 200.0])
    parser.add_argument("--contact-threshold-n", type=float, default=5.0)
    parser.add_argument("--baseline-rows", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    source = fast_dataset.load_slow_cache(args.slow_cache)
    arrays = fast_dataset.load_stage3_fast_arrays(args.targets, chunk_steps=source.chunk_steps)
    full_chunks = _load_full_teacher_chunks(args.targets, horizon=source.action_chunks.shape[1])
    norm_stats = normalize_lib.load(args.norm_stats_dir)
    wrench = fast_dataset.latest_physical_wrench(
        arrays.force_history, arrays.force_history_mask, norm_stats["force_history"]
    )
    baseline = fast_dataset.episode_baseline_wrench(wrench, arrays.episode_indices, num_rows=args.baseline_rows)
    model, predicts_staleness = evaluator._load_model(args.checkpoint.resolve(), chunk_steps=source.chunk_steps)
    metadata = evaluator._run_metadata(args.checkpoint.resolve())

    points = []
    for delay_ms in args.delay_ms:
        cache = fast_dataset.resample_slow_cache(
            source,
            arrays.episode_indices,
            arrays.timestamps,
            slow_rate_range_hz=args.reference_rate_hz,
            jitter_s=0.0,
            ready_delay_range_s=delay_ms / 1000.0,
            rng=args.seed,
        )
        force_delta = evaluator.evaluate(
            model,
            arrays,
            cache,
            norm_stats,
            wrench,
            baseline,
            predicts_staleness=predicts_staleness,
            analytic_rebase=bool(metadata.get("analytic_rebase", False)),
            contact_threshold_n=args.contact_threshold_n,
            baseline_rows=args.baseline_rows,
            batch_size=args.batch_size,
            seed=args.seed,
        )

        ready = np.flatnonzero(cache.row_ready)
        temporal_chunks = full_chunks[cache.key_dataset_indices]
        temporal_reference, _ = fast_dataset.build_reference_rollout(
            temporal_chunks,
            cache.row_key_positions,
            arrays.timestamps,
            cache.key_timestamps,
            action_period_s=cache.action_period_s,
            context_age_scale_s=cache.context_age_scale_s,
            chunk_steps=cache.chunk_steps,
        )
        key_states = arrays.state[cache.key_dataset_indices[cache.row_key_positions[ready]]]
        temporal_absolute = _absolute_pose(temporal_reference[ready, 0], key_states, norm_stats)
        current_full_absolute = _absolute_pose(arrays.full_pose[ready, 0], arrays.state[ready], norm_stats)
        expert_absolute = _absolute_pose(arrays.expert_pose[ready, 0], arrays.state[ready], norm_stats)

        corrected_force = np.linalg.norm((wrench[ready] - baseline[ready])[:, :3], axis=-1)
        contact = corrected_force >= args.contact_threshold_n
        temporal = {
            "vs_current_full": evaluator.physical_pose_error_summary(temporal_absolute, current_full_absolute),
            "vs_expert": evaluator.physical_pose_error_summary(temporal_absolute, expert_absolute),
        }
        if np.any(contact):
            temporal["contact_vs_current_full"] = evaluator.physical_pose_error_summary(
                temporal_absolute[contact], current_full_absolute[contact]
            )

        force_all = force_delta["strata"]["all"]
        force_contact = force_delta["strata"].get("contact")
        point = {
            "additional_publication_delay_ms": float(delay_ms),
            "reference_rate_hz": float(args.reference_rate_hz),
            "rows_scored": int(len(ready)),
            "ready_row_fraction": float(np.mean(cache.row_ready)),
            "temporal_teacher": temporal,
            "force_delta": {
                "vs_current_full": force_all["action_vs_teacher_full"]["slow_plus_full"],
                "vs_expert": force_all["action_vs_expert"]["slow_plus_full"],
                "contact_vs_current_full": (
                    None if force_contact is None else force_contact["action_vs_teacher_full"]["slow_plus_full"]
                ),
            },
        }
        points.append(point)
        print(
            f"delay={delay_ms:>5.0f} ms rows={len(ready):>4d} | "
            f"Temporal/full: {temporal['vs_current_full']['translation_rmse_m']:.5f} m, "
            f"{temporal['vs_current_full']['rotation_geodesic_rmse_rad']:.5f} rad | "
            f"ForceDelta/full: "
            f"{point['force_delta']['vs_current_full']['translation_rmse_m']:.5f} m, "
            f"{point['force_delta']['vs_current_full']['rotation_geodesic_rmse_rad']:.5f} rad",
            flush=True,
        )

    result = {
        "task": "button_press",
        "interpretation": "offline action reconstruction, not closed-loop task success",
        "delay_definition": "additional delay from completed query to packet publication",
        "checkpoint": str(args.checkpoint),
        "fast_run": str(args.fast_run),
        "points": points,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
