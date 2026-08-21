"""Offline conversion of ForceVLA Euler poses to continuous 6D rotations.

Original parquet files are never overwritten. Use --check-only to scan a dataset
for continuity without writing shards.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

from openpi.policies import rotation_6d as rot


def wrap_to_pi(angles: np.ndarray) -> np.ndarray:
    return (np.asarray(angles) + np.pi) % (2 * np.pi) - np.pi


def _stack_column(frame: pd.DataFrame, key: str) -> np.ndarray:
    return np.stack(frame[key].to_numpy())


def summarize_episode(state: np.ndarray, action: np.ndarray) -> dict:
    rpy_state = state[:, rot.RAW_RPY_SLICE]
    r_state = rot.rpy_to_matrix(rpy_state)
    sixd_state = rot.matrix_to_6d(r_state)
    reconstructed = rot.sixd_to_matrix(sixd_state)
    restored_rpy = rot.matrix_to_rpy(reconstructed)
    restored_R = rot.rpy_to_matrix(restored_rpy)

    naive_delta = action[:, :6] - state[:, :6]
    wrapped_delta = naive_delta.copy()
    wrapped_delta[:, 3:6] = wrap_to_pi(naive_delta[:, 3:6])
    sixd_delta = rot.convert_action(action)[:, : rot.POSE_DIMS] - rot.convert_state(state)[:, : rot.POSE_DIMS]

    if len(state) > 1:
        euler_step = np.diff(rpy_state, axis=0)
        geodesic_step = rot.geodesic_angle(r_state[:-1], r_state[1:])
        sixd_step = np.linalg.norm(np.diff(sixd_state, axis=0), axis=-1)
        xyz_step = np.linalg.norm(np.diff(state[:, :3], axis=0), axis=-1)
        wrap = np.max(np.abs(euler_step), axis=-1) > np.pi
        wrap_geodesic_deg = np.degrees(geodesic_step[wrap]).tolist() if wrap.any() else []
        wrap_sixd = sixd_step[wrap].tolist() if wrap.any() else []
    else:
        euler_step = np.zeros((0, 3))
        geodesic_step = np.zeros((0,))
        sixd_step = np.zeros((0,))
        xyz_step = np.zeros((0,))
        wrap = np.zeros((0,), dtype=bool)
        wrap_geodesic_deg = []
        wrap_sixd = []

    return {
        "rows": int(len(state)),
        "roundtrip_geodesic_deg_max": float(np.degrees(rot.geodesic_angle(r_state, reconstructed).max())),
        "rpy_restore_geodesic_deg_max": float(np.degrees(rot.geodesic_angle(r_state, restored_R).max())),
        "naive_rpy_delta_std": naive_delta[:, 3:6].std(axis=0).tolist(),
        "wrapped_rpy_delta_std": wrapped_delta[:, 3:6].std(axis=0).tolist(),
        "sixd_delta_std": sixd_delta[:, 3 : rot.POSE_DIMS].std(axis=0).tolist(),
        "branch_flips": int(wrap.sum()),
        "wrap_geodesic_deg": wrap_geodesic_deg,
        "wrap_sixd_l2": wrap_sixd,
        "geodesic_step_rad_p99": float(np.quantile(geodesic_step, 0.99)) if len(geodesic_step) else 0.0,
        "sixd_step_p99": float(np.quantile(sixd_step, 0.99)) if len(sixd_step) else 0.0,
        "xyz_step_m_p99": float(np.quantile(xyz_step, 0.99)) if len(xyz_step) else 0.0,
        "xyz_step_m_max": float(xyz_step.max()) if len(xyz_step) else 0.0,
        "same_pose_both_branches_l2": float(
            np.linalg.norm(rot.rpy_to_6d(np.array([[np.pi, 0, 0]])) - rot.rpy_to_6d(np.array([[-np.pi, 0, 0]])))
        ),
    }


def _read_pose_frame(path: pathlib.Path) -> pd.DataFrame:
    import pyarrow.parquet as pq

    available = set(pq.ParquetFile(path).schema_arrow.names)
    needed = [name for name in ("observation.state", "action", "timestamp", "episode_index") if name in available]
    return pd.read_parquet(path, columns=needed)


def convert_parquet(path: pathlib.Path, output: pathlib.Path) -> dict:
    frame = _read_pose_frame(path)
    state = _stack_column(frame, "observation.state")
    action = _stack_column(frame, "action")
    converted = frame.copy()
    converted["observation.state"] = list(rot.convert_state(state))
    converted["action"] = list(rot.convert_action(action))
    converted["observation.state_rpy"] = list(state.astype(np.float32))
    converted["action_rpy"] = list(action.astype(np.float32))
    output.parent.mkdir(parents=True, exist_ok=True)
    converted.to_parquet(output, index=False)
    stats = summarize_episode(state, action)
    stats["source"] = str(path)
    stats["output"] = str(output)
    stats["state_dim"] = int(rot.convert_state(state).shape[-1])
    stats["action_dim"] = int(rot.convert_action(action).shape[-1])
    return stats


def _aggregate(reports: list[dict], input_root: pathlib.Path, output_root: pathlib.Path | None) -> dict:
    naive = np.array([r["naive_rpy_delta_std"] for r in reports])
    wrapped = np.array([r["wrapped_rpy_delta_std"] for r in reports])
    wrap_geo = [deg for r in reports for deg in r.get("wrap_geodesic_deg", [])]
    wrap_sixd = [val for r in reports for val in r.get("wrap_sixd_l2", [])]
    label_flips = [deg for deg in wrap_geo if deg < 1.0]
    real_turns = [deg for deg in wrap_geo if deg >= 1.0]
    return {
        "input": str(input_root),
        "output": None if output_root is None else str(output_root),
        "episodes": len(reports),
        "rows": int(sum(r["rows"] for r in reports)),
        "state_dim": reports[0].get("state_dim", rot.ROBOT_DIMS + 6),
        "action_dim": reports[0].get("action_dim", rot.ROBOT_DIMS),
        "branch_flips": int(sum(r["branch_flips"] for r in reports)),
        "label_flips_geodesic_lt_1deg": len(label_flips),
        "real_turns_geodesic_ge_1deg": len(real_turns),
        "wrap_geodesic_deg_max": max(wrap_geo) if wrap_geo else 0.0,
        "wrap_sixd_l2_max_on_label_flips": max((s for s, g in zip(wrap_sixd, wrap_geo, strict=True) if g < 1.0), default=0.0),
        "roundtrip_geodesic_deg_max": max(r["roundtrip_geodesic_deg_max"] for r in reports),
        "rpy_restore_geodesic_deg_max": max(r["rpy_restore_geodesic_deg_max"] for r in reports),
        "same_pose_both_branches_l2": reports[0]["same_pose_both_branches_l2"],
        "naive_rpy_delta_std_mean": naive.mean(axis=0).tolist(),
        "wrapped_rpy_delta_std_mean": wrapped.mean(axis=0).tolist(),
        "inflation_vs_wrapped": (naive.mean(axis=0) / np.maximum(wrapped.mean(axis=0), 1e-12)).tolist(),
        "geodesic_step_rad_p99_mean": float(np.mean([r["geodesic_step_rad_p99"] for r in reports])),
        "sixd_step_p99_mean": float(np.mean([r["sixd_step_p99"] for r in reports])),
        "xyz_step_m_max": max(r["xyz_step_m_max"] for r in reports),
        "continuous": (
            max(r["roundtrip_geodesic_deg_max"] for r in reports) < 1e-3
            and max(r["rpy_restore_geodesic_deg_max"] for r in reports) < 1e-3
            and (max((s for s, g in zip(wrap_sixd, wrap_geo, strict=True) if g < 1.0), default=0.0) < 0.05)
        ),
    }


def convert_dataset(input_root: pathlib.Path, output_root: pathlib.Path) -> dict:
    files = sorted(input_root.glob("data/*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet shards under {input_root}/data")
    reports = []
    for path in files:
        rel = path.relative_to(input_root)
        reports.append(convert_parquet(path, output_root / rel))
        if len(reports) % 10 == 0 or len(reports) == len(files):
            print(f"{input_root.name}: {len(reports)}/{len(files)} shards", flush=True)
    summary = _aggregate(reports, input_root, output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "conversion_report.json").write_text(json.dumps({"summary": summary, "episodes": reports}, indent=2))
    return summary


def check_dataset(input_root: pathlib.Path) -> dict:
    files = sorted(input_root.glob("data/*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet shards under {input_root}/data")
    reports = []
    for path in files:
        frame = _read_pose_frame(path)
        reports.append(summarize_episode(_stack_column(frame, "observation.state"), _stack_column(frame, "action")))
        if len(reports) % 10 == 0 or len(reports) == len(files):
            print(f"check {input_root.name}: {len(reports)}/{len(files)} shards", flush=True)
    return _aggregate(reports, input_root, None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, default=None)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        summary = check_dataset(args.input_root)
    else:
        if args.output_root is None:
            raise ValueError("--output-root is required unless --check-only")
        summary = convert_dataset(args.input_root, args.output_root)
    print(json.dumps(summary, indent=2))
    if not summary["continuous"]:
        raise SystemExit("6D conversion failed the continuity checks")


if __name__ == "__main__":
    main()
