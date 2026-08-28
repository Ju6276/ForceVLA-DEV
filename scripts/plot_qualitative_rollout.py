"""Create a paper-ready rollout figure with images, force, and Fast residual traces."""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import subprocess

import matplotlib.pyplot as plt
import numpy as np

import evaluate_fast_residual as evaluator

from openpi.shared import normalize as normalize_lib
from openpi.training import fast_dataset


def _persistent_onset(signal: np.ndarray, threshold: float, *, samples: int = 3) -> int | None:
    above = np.asarray(signal) >= threshold
    hits = np.convolve(above.astype(np.int16), np.ones(samples, dtype=np.int16), mode="valid")
    candidates = np.flatnonzero(hits == samples)
    return None if len(candidates) == 0 else int(candidates[0])


def _persistent_completion(
    signal: np.ndarray,
    threshold: float,
    *,
    start: int,
    samples: int = 6,
) -> int | None:
    below = np.asarray(signal) < threshold
    hits = np.convolve(below.astype(np.int16), np.ones(samples, dtype=np.int16), mode="valid")
    candidates = np.flatnonzero(hits == samples)
    candidates = candidates[candidates >= start]
    return None if len(candidates) == 0 else int(candidates[0])


def _video_frame(path: pathlib.Path, timestamp: float) -> np.ndarray:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(timestamp, 0.0):.6f}",
        "-i",
        str(path),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-",
    ]
    encoded = subprocess.check_output(command)
    return plt.imread(io.BytesIO(encoded), format="png")


def _adjust_image_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    adjusted = np.asarray(image, dtype=np.float32).copy()
    adjusted[..., :3] = np.power(np.clip(adjusted[..., :3], 0.0, 1.0), gamma)
    return adjusted


def _choose_episode(
    episode_ids: np.ndarray,
    timestamps: np.ndarray,
    force_magnitude: np.ndarray,
    residual_magnitude: np.ndarray,
    ready: np.ndarray,
    threshold: float,
) -> tuple[int, list[dict]]:
    summaries = []
    for episode in np.unique(episode_ids):
        rows = np.flatnonzero((episode_ids == episode) & ready)
        if len(rows) < 30:
            continue
        onset_local = _persistent_onset(force_magnitude[rows], threshold)
        if onset_local is None or onset_local < 8 or onset_local >= len(rows) - 8:
            continue
        onset = rows[onset_local]
        before = rows[timestamps[rows] < timestamps[onset] - 0.25]
        contact = rows[force_magnitude[rows] >= threshold]
        if len(before) < 8 or len(contact) < 8:
            continue
        free_median = float(np.median(residual_magnitude[before]))
        contact_median = float(np.median(residual_magnitude[contact]))
        summaries.append(
            {
                "episode": int(episode),
                "onset_row": int(onset),
                "free_residual_median_mm": free_median,
                "contact_residual_median_mm": contact_median,
                "contact_to_free_ratio": contact_median / max(free_median, 1e-9),
            }
        )
    if not summaries:
        raise ValueError("No episode has enough free-space and persistent-contact rows")

    # Pick the rollout closest to the median positive response, rather than the
    # strongest response, to avoid presenting an extreme cherry-picked example.
    positive = [item for item in summaries if item["contact_to_free_ratio"] > 1.0]
    candidates = positive or summaries
    median_ratio = float(np.median([item["contact_to_free_ratio"] for item in candidates]))
    selected = min(candidates, key=lambda item: abs(item["contact_to_free_ratio"] - median_ratio))
    return int(selected["episode"]), summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=pathlib.Path, required=True)
    parser.add_argument("--slow-cache", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--norm-stats-dir", type=pathlib.Path, required=True)
    parser.add_argument("--dataset-root", type=pathlib.Path, required=True)
    parser.add_argument("--camera", default="observation.image")
    parser.add_argument("--second-camera", default=None)
    parser.add_argument("--image-gamma", type=float, default=1.0)
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--contact-threshold-n", type=float, default=5.0)
    parser.add_argument(
        "--contact-signal",
        choices=("magnitude", "fx", "fy", "fz", "-fx", "-fy", "-fz"),
        default="magnitude",
        help="Force signal used only for contact-phase detection and shading.",
    )
    parser.add_argument("--completion-threshold-n", type=float, default=2.0)
    parser.add_argument("--free-lead-s", type=float, default=1.5)
    parser.add_argument("--release-frame-lead-s", type=float, default=0.0)
    parser.add_argument("--baseline-rows", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-prefix", type=pathlib.Path, required=True)
    args = parser.parse_args()

    cache = fast_dataset.load_slow_cache(args.slow_cache)
    arrays = fast_dataset.load_stage3_fast_arrays(args.targets, chunk_steps=cache.chunk_steps)
    norm_stats = normalize_lib.load(args.norm_stats_dir)
    model, _ = evaluator._load_model(args.checkpoint.resolve(), chunk_steps=cache.chunk_steps)
    predictions, staleness_predictions = evaluator._predict_all(
        model, arrays, cache, batch_size=args.batch_size, seed=args.seed
    )

    wrench = fast_dataset.latest_physical_wrench(
        arrays.force_history, arrays.force_history_mask, norm_stats["force_history"]
    )
    baseline = fast_dataset.episode_baseline_wrench(wrench, arrays.episode_indices, num_rows=args.baseline_rows)
    corrected_wrench = wrench - baseline
    force_magnitude = np.linalg.norm(corrected_wrench[:, :3], axis=-1)
    if args.contact_signal == "magnitude":
        contact_signal = force_magnitude
    else:
        signed_axis = args.contact_signal
        sign = -1.0 if signed_axis.startswith("-") else 1.0
        axis_name = signed_axis.removeprefix("-")
        axis_index = {"fx": 0, "fy": 1, "fz": 2}[axis_name]
        contact_signal = sign * corrected_wrench[:, axis_index]
    action_scale = np.asarray(norm_stats["actions"].std, dtype=np.float64)[:3] + 1e-6
    residual_xyz = predictions["full"][:, 0, :3] * action_scale
    residual_magnitude = 1000.0 * np.linalg.norm(residual_xyz, axis=-1)
    if staleness_predictions is None:
        staleness_xyz = np.zeros_like(residual_xyz)
        predicts_staleness = False
    else:
        staleness_xyz = staleness_predictions["full"][:, 0, :3] * action_scale
        predicts_staleness = True
    staleness_magnitude = 1000.0 * np.linalg.norm(staleness_xyz, axis=-1)
    total_fast_magnitude = 1000.0 * np.linalg.norm(residual_xyz + staleness_xyz, axis=-1)
    ready = np.ones(len(arrays.timestamps), dtype=bool) if cache.row_ready is None else cache.row_ready

    selected, summaries = _choose_episode(
        arrays.episode_indices,
        arrays.timestamps,
        contact_signal,
        total_fast_magnitude,
        ready,
        args.contact_threshold_n,
    )
    if args.episode is not None:
        selected = args.episode
    rows = np.flatnonzero((arrays.episode_indices == selected) & ready)
    onset_local = _persistent_onset(contact_signal[rows], args.contact_threshold_n)
    if onset_local is None:
        raise ValueError(f"Episode {selected} has no persistent contact onset")
    onset_row = rows[onset_local]
    onset_time = float(arrays.timestamps[onset_row])
    start_time, end_time = float(arrays.timestamps[rows[0]]), float(arrays.timestamps[rows[-1]])
    contact_rows = rows[contact_signal[rows] >= args.contact_threshold_n]
    peak_row = int(contact_rows[np.argmax(contact_signal[contact_rows])])
    peak_local = int(np.flatnonzero(rows == peak_row)[0])
    completion_local = _persistent_completion(
        contact_signal[rows],
        args.completion_threshold_n,
        start=peak_local,
    )
    completion_detected = completion_local is not None
    completion_row = int(rows[-1] if completion_local is None else rows[completion_local])
    release_frame_time = max(
        float(arrays.timestamps[peak_row]),
        float(arrays.timestamps[completion_row]) - args.release_frame_lead_s,
    )
    candidate_times = np.asarray(
        [onset_time - args.free_lead_s, onset_time, arrays.timestamps[peak_row], release_frame_time]
    )
    frame_times = np.clip(candidate_times, start_time, end_time)

    cameras = [args.camera] + ([] if args.second_camera is None else [args.second_camera])
    camera_frames = []
    for camera in cameras:
        video = args.dataset_root / "videos" / "chunk-000" / camera / f"episode_{selected:06d}.mp4"
        if not video.is_file():
            raise FileNotFoundError(video)
        camera_frames.append(
            [
                _adjust_image_gamma(_video_frame(video, float(timestamp)), args.image_gamma)
                for timestamp in frame_times
            ]
        )

    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(frame_times)))
    stages = (
        "Free space",
        "Contact onset",
        "Peak contact",
        "Contact release" if completion_detected else "Episode end",
    )
    figure = plt.figure(figsize=(11.0, 7.2 if len(cameras) == 2 else 5.7), constrained_layout=True)
    image_heights = [1.3] * len(cameras)
    grid = figure.add_gridspec(len(cameras) + 2, len(frame_times), height_ratios=image_heights + [1.0, 1.0])
    for camera_index, (camera, frames) in enumerate(zip(cameras, camera_frames, strict=True)):
        row_label = "Front" if camera_index == 0 else "Wrist"
        for index, (frame, timestamp, color, stage) in enumerate(
            zip(frames, frame_times, colors, stages, strict=True)
        ):
            axis = figure.add_subplot(grid[camera_index, index])
            axis.imshow(frame)
            if camera_index == 0:
                axis.set_title(f"{stage}\n{timestamp:.2f} s", color=color, fontsize=8.5, fontweight="bold")
            axis.axis("off")
            if index == 0:
                axis.text(
                    -0.04,
                    0.5,
                    row_label,
                    transform=axis.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=8,
                )
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2.2)
                spine.set_edgecolor(color)

    episode_time = arrays.timestamps[rows]
    force_axis = figure.add_subplot(grid[len(cameras), :])
    residual_axis = figure.add_subplot(grid[len(cameras) + 1, :], sharex=force_axis)
    for component, label, color in zip(
        corrected_wrench[rows, :3].T,
        (r"$F_x$", r"$F_y$", r"$F_z$"),
        ("#2878B5", "#59A14F", "#D1495B"),
        strict=True,
    ):
        force_axis.plot(episode_time, component, color=color, linewidth=1.2, label=label)
    force_axis.axhline(0.0, color="#777777", linewidth=0.7)
    force_axis.set_ylabel("Force (N)")
    residual_axis.plot(
        episode_time,
        total_fast_magnitude[rows],
        color="#E78FB3",
        linewidth=1.55,
        label=r"$\|\Delta p^{fast}\|_2$",
    )
    residual_axis.set_ylabel("Position correction (mm)")
    residual_axis.set_xlabel("Time (s)")
    force_axis.yaxis.set_label_coords(-0.045, 0.5)
    residual_axis.yaxis.set_label_coords(-0.045, 0.5)
    plot_end_time = min(end_time, float(np.ceil(arrays.timestamps[completion_row])))
    for axis in (force_axis, residual_axis):
        axis.axvline(onset_time, color="black", linestyle="--", linewidth=1.1)
        axis.grid(alpha=0.22, linewidth=0.6)
        for timestamp, color in zip(frame_times, colors, strict=True):
            axis.axvline(timestamp, color=color, linewidth=0.8, alpha=0.75)
        axis.set_xlim(start_time, plot_end_time)
    force_axis.legend(loc="lower left", frameon=False, fontsize=8, ncol=3)
    residual_axis.legend(loc="upper left", frameon=False, fontsize=8)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(args.output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)

    selected_summary = next((item for item in summaries if item["episode"] == selected), None)
    precontact = rows[arrays.timestamps[rows] < onset_time - 0.25]
    precontact_values = residual_magnitude[precontact]
    free_median = float(np.median(precontact_values))
    free_sigma = float(1.4826 * np.median(np.abs(precontact_values - free_median)))
    activation_threshold_mm = free_median + max(3.0 * free_sigma, 0.5)
    post_onset_local = int(np.searchsorted(arrays.timestamps[rows], onset_time, side="left"))
    activation_local = _persistent_onset(
        residual_magnitude[rows][post_onset_local:],
        activation_threshold_mm,
        samples=3,
    )
    activation_row = None if activation_local is None else int(rows[post_onset_local + activation_local])
    reaction_latency_ms = (
        None
        if activation_row is None
        else float((arrays.timestamps[activation_row] - onset_time) * 1000.0)
    )
    metadata = {
        "episode": selected,
        "selection": "closest to median contact/free residual ratio among positive held-out episodes",
        "contact_threshold_n": args.contact_threshold_n,
        "contact_signal": args.contact_signal,
        "completion_threshold_n": args.completion_threshold_n,
        "contact_onset_s": onset_time,
        "completion_detected": completion_detected,
        "completion_or_episode_end_s": float(arrays.timestamps[completion_row]),
        "release_frame_s": release_frame_time,
        "plot_end_s": plot_end_time,
        "frame_times_s": frame_times.tolist(),
        "image_gamma": args.image_gamma,
        "checkpoint": str(args.checkpoint.resolve()),
        "residual_source": "Vector sum of raw Fast Student force-head and staleness-head predictions, first chunk step, evaluated at dataset rows",
        "residual_sampling_rate": "dataset state/action timestamps (approximately 30 Hz), not realized runtime packet timestamps",
        "plotted_fast_terms": ["vector_sum"] if predicts_staleness else ["force_head"],
        "offline_reaction_latency": {
            "definition": f"Delay from the first persistent {args.contact_threshold_n:g} N crossing of {args.contact_signal} to the first of three consecutive force-head residual samples above the robust pre-contact threshold.",
            "precontact_median_mm": free_median,
            "precontact_robust_sigma_mm": free_sigma,
            "activation_threshold_mm": activation_threshold_mm,
            "activation_time_s": None if activation_row is None else float(arrays.timestamps[activation_row]),
            "latency_ms": reaction_latency_ms,
            "resolution_limit_ms": float(1000.0 / 30.0),
            "caveat": "This is threshold-to-output latency on 30 Hz offline rows, not physical-contact-to-command latency or realized asynchronous publication latency.",
        },
        "selected_summary": selected_summary,
        "all_episode_summaries": summaries,
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata["selected_summary"], indent=2))
    print(f"wrote {args.output_prefix.with_suffix('.pdf')}")
    print(f"wrote {args.output_prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
