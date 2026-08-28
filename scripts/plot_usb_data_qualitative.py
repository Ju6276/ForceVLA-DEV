"""Plot two-view USB observations with high-rate three-axis force traces."""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _frame(video: pathlib.Path, timestamp: float) -> np.ndarray:
    encoded = subprocess.check_output(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.6f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-",
        ]
    )
    return plt.imread(io.BytesIO(encoded), format="png")


def _rolling_mean(values: np.ndarray, samples: int) -> np.ndarray:
    return np.convolve(values, np.ones(samples) / samples, mode="same")


def _causal_rolling_mean(values: np.ndarray, samples: int) -> np.ndarray:
    """Trailing moving average with no access to future samples."""
    samples = max(1, int(samples))
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    indices = np.arange(len(values))
    starts = np.maximum(0, indices + 1 - samples)
    counts = indices + 1 - starts
    return (cumulative[indices + 1] - cumulative[starts]) / counts


def _persistent_onset(values: np.ndarray, threshold: float, samples: int, *, start: int = 0) -> int:
    above = values >= threshold
    hits = np.convolve(above.astype(np.int16), np.ones(samples, dtype=np.int16), mode="valid")
    candidates = np.flatnonzero(hits == samples)
    candidates = candidates[candidates >= start]
    if len(candidates) == 0:
        raise ValueError("No persistent high-load onset found")
    return int(candidates[0])


def _persistent_completion(values: np.ndarray, threshold: float, samples: int, *, start: int) -> int:
    below = values < threshold
    hits = np.convolve(below.astype(np.int16), np.ones(samples, dtype=np.int16), mode="valid")
    candidates = np.flatnonzero(hits == samples)
    candidates = candidates[candidates >= start]
    if len(candidates) == 0:
        return len(values) - 1
    return int(candidates[0])


def _estimate_force_content_lag(
    high_time: np.ndarray,
    high_force: np.ndarray,
    main_time: np.ndarray,
    main_force: np.ndarray,
) -> tuple[float, float]:
    """Estimate how late the high-rate wrench content is relative to main data."""
    candidate_lags = np.arange(-1.5, 1.5001, 0.005)
    best_correlation = -np.inf
    best_lag = 0.0
    for lag in candidate_lags:
        valid = (
            (main_time + lag >= high_time[0])
            & (main_time + lag <= high_time[-1])
            & (main_time >= main_time[0] + 1.0)
        )
        if np.count_nonzero(valid) < 20:
            continue
        aligned_fz = np.interp(main_time[valid] + lag, high_time, high_force[:, 2])
        correlation = float(np.corrcoef(aligned_fz, main_force[valid, 2])[0, 1])
        if np.isfinite(correlation) and correlation > best_correlation:
            best_correlation = correlation
            best_lag = float(lag)
    return best_lag, best_correlation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--episode", type=int, default=27)
    parser.add_argument("--threshold-n", type=float, default=6.0)
    parser.add_argument(
        "--completion-threshold-n",
        type=float,
        default=2.0,
        help="Near-baseline force threshold used to select the completion frame.",
    )
    parser.add_argument("--show-vla-wrench", action="store_true")
    parser.add_argument("--relative-time", action="store_true")
    parser.add_argument("--free-lead-s", type=float, default=5.0)
    parser.add_argument(
        "--hypothetical-residual",
        action="store_true",
        help="Replace the force-magnitude panel with a clearly labelled schematic residual.",
    )
    parser.add_argument(
        "--hypothetical-residual-peak-mm",
        type=float,
        default=5.0,
        help="Assumed peak used only by the schematic residual layout.",
    )
    parser.add_argument(
        "--hypothetical-residual-rate-hz",
        type=float,
        default=18.19,
        help="Packet publication rate represented by the schematic residual trace.",
    )
    parser.add_argument(
        "--hypothetical-residual-latency-ms",
        type=float,
        default=2.43,
        help="Inference latency added between schematic residual query and publication.",
    )
    parser.add_argument("--output-prefix", type=pathlib.Path, required=True)
    args = parser.parse_args()

    main_path = args.root / "main/data/chunk-000" / f"episode_{args.episode:06d}.parquet"
    force_path = args.root / "force/chunk-000" / f"episode_{args.episode:06d}.parquet"
    front_video = args.root / "main/videos/chunk-000/observation.image" / f"episode_{args.episode:06d}.mp4"
    wrist_video = args.root / "main/videos/chunk-000/observation.wrist_image" / f"episode_{args.episode:06d}.mp4"
    main = pd.read_parquet(main_path)
    high = pd.read_parquet(force_path)
    raw_high_time = high["episode_time_s"].to_numpy(dtype=np.float64)
    main_time = main["timestamp"].to_numpy(dtype=np.float64)
    high_force = high[["force_x", "force_y", "force_z"]].to_numpy(dtype=np.float64)
    main_state = np.stack(main["observation.state"].to_numpy())
    main_force = main_state[:, 7:10]

    baseline_high = np.median(high_force[raw_high_time <= raw_high_time[0] + 1.0], axis=0)
    baseline_main = np.median(main_force[main_time <= main_time[0] + 1.0], axis=0)
    corrected_force = high_force - baseline_high
    corrected_main_force = main_force - baseline_main
    force_content_lag_s, force_alignment_correlation = _estimate_force_content_lag(
        raw_high_time,
        corrected_force,
        main_time,
        corrected_main_force,
    )
    high_time = raw_high_time - force_content_lag_s
    high_magnitude = np.linalg.norm(corrected_force, axis=-1)
    main_magnitude = np.linalg.norm(corrected_main_force, axis=-1)

    mean_rate = (len(high_time) - 1) / (high_time[-1] - high_time[0])
    smooth_samples = max(1, int(round(0.05 * mean_rate)))
    persistent_samples = max(1, int(round(0.10 * mean_rate)))
    smooth_force = _causal_rolling_mean(high_magnitude, smooth_samples)
    onset_index = _persistent_onset(
        smooth_force,
        args.threshold_n,
        persistent_samples,
        start=len(high_time) // 2,
    )
    onset_time = float(high_time[onset_index])
    peak_index = int(np.argmax(smooth_force))
    peak_time = float(high_time[peak_index])
    completion_index = _persistent_completion(
        smooth_force,
        args.completion_threshold_n,
        max(1, int(round(0.20 * mean_rate))),
        start=peak_index,
    )
    completion_time = float(high_time[completion_index])
    frame_times = np.asarray([onset_time - args.free_lead_s, onset_time, peak_time, completion_time])
    frame_times = np.clip(frame_times, main_time[0], main_time[-1])
    front_frames = [_frame(front_video, float(timestamp)) for timestamp in frame_times]
    wrist_frames = [_frame(wrist_video, float(timestamp)) for timestamp in frame_times]

    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(frame_times)))
    stages = ("Free space", "Contact onset", "Sustained contact / insertion", "Completion")
    time_origin = onset_time if args.relative_time else 0.0
    plot_high_time = high_time - time_origin
    plot_main_time = main_time - time_origin
    plot_frame_times = frame_times - time_origin
    plot_onset_time = onset_time - time_origin
    plot_episode_end = float(main_time[-1] - time_origin)
    figure = plt.figure(figsize=(11.0, 7.2), constrained_layout=True)
    grid = figure.add_gridspec(4, len(frame_times), height_ratios=[1.3, 1.3, 1.0, 1.0])
    for row_index, (row_name, frames) in enumerate((("Front", front_frames), ("Wrist", wrist_frames))):
        for column, (image, timestamp, plot_timestamp, color, stage) in enumerate(
            zip(frames, frame_times, plot_frame_times, colors, stages, strict=True)
        ):
            axis = figure.add_subplot(grid[row_index, column])
            axis.imshow(image)
            if row_index == 0:
                shown_time = f"{plot_timestamp:+.2f} s" if args.relative_time else f"{timestamp:.2f} s"
                axis.set_title(f"{stage}\n{shown_time}", color=color, fontsize=8.5, fontweight="bold")
            axis.axis("off")
            if column == 0:
                axis.text(
                    -0.04,
                    0.5,
                    row_name,
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

    component_axis = figure.add_subplot(grid[2, :])
    magnitude_axis = figure.add_subplot(grid[3, :], sharex=component_axis)
    for component, label, color in zip(corrected_force.T, (r"$F_x$", r"$F_y$", r"$F_z$"), ("#2878B5", "#59A14F", "#D1495B"), strict=True):
        component_axis.plot(
            plot_high_time,
            _rolling_mean(component, smooth_samples),
            linewidth=1.2,
            color=color,
            label=label,
        )
    component_axis.axhline(0.0, color="#777777", linewidth=0.7)
    component_axis.set_ylabel("Force (N)")
    component_axis.legend(loc="upper left", frameon=False, ncol=3, fontsize=8)
    if args.hypothetical_residual:
        # Layout mock-up only: this smooth force-conditioned trace is not a learned
        # policy output and must never be reported as an experimental result. Keep a
        # small nonzero free-space floor so the illustration does not imply hard
        # gating, clipping, or zero-padding at episode termination.
        transition_width_n = 0.8
        activation = 1.0 / (
            1.0 + np.exp(-np.clip((smooth_force - args.threshold_n) / transition_width_n, -20.0, 20.0))
        )
        force_level = smooth_force / max(float(np.max(smooth_force)), 1e-9)
        time_steps = np.diff(high_time, prepend=high_time[0])
        time_steps[0] = float(np.median(np.diff(high_time)))
        force_derivative = np.abs(np.diff(smooth_force, prepend=smooth_force[0]) / time_steps)
        force_derivative = _causal_rolling_mean(
            force_derivative,
            max(1, int(round(0.12 * mean_rate))),
        )
        derivative_scale = max(float(np.percentile(force_derivative, 95.0)), 1e-9)
        free_space_variation = np.clip(force_derivative / derivative_scale, 0.0, 1.0)
        background = 0.20 + 0.18 * free_space_variation
        contact_signal = _causal_rolling_mean(
            activation * (0.55 + 0.45 * force_level),
            max(1, int(round(0.15 * mean_rate))),
        )
        contact_scale = max(float(np.max(contact_signal)), 1e-9)
        continuous_schematic = background + (
            args.hypothetical_residual_peak_mm - float(np.max(background))
        ) * contact_signal / contact_scale
        packet_period_s = 1.0 / args.hypothetical_residual_rate_hz
        query_times = np.arange(high_time[0], high_time[-1] + packet_period_s, packet_period_s)
        packet_values = np.interp(query_times, high_time, continuous_schematic)
        packet_times = query_times + args.hypothetical_residual_latency_ms / 1000.0
        first_post_contact_query = int(np.flatnonzero(query_times >= onset_time)[0])
        mock_contact_response_ms = float(
            (packet_times[first_post_contact_query] - onset_time) * 1000.0
        )
        mock_first_response_mm = float(packet_values[first_post_contact_query])
        magnitude_axis.plot(
            packet_times - time_origin,
            packet_values,
            color="#E78FB3",
            linewidth=1.45,
            drawstyle="steps-post",
        )
        magnitude_axis.set_ylabel(r"$\|\Delta p^{force}\|_2$ (mm)")
        magnitude_axis.set_ylim(-0.2, 8.0)
        magnitude_axis.set_yticks([0, 2, 4, 6, 8])
    else:
        magnitude_axis.plot(
            plot_high_time,
            smooth_force,
            color="#2878B5",
            linewidth=1.4,
        )
        if args.show_vla_wrench:
            magnitude_axis.scatter(plot_main_time, main_magnitude, color="#222222", s=4, alpha=0.35)
        magnitude_axis.axhline(args.threshold_n, color="#777777", linestyle=":", linewidth=1.0)
        magnitude_axis.set_ylabel(r"$\|F-F_0\|_2$ (N)")
    magnitude_axis.set_xlabel("Time relative to contact onset (s)" if args.relative_time else "Time (s)")
    for axis in (component_axis, magnitude_axis):
        axis.axvline(plot_onset_time, color="black", linestyle="--", linewidth=1.2)
        axis.fill_between(
            plot_high_time,
            0,
            1,
            where=smooth_force >= args.threshold_n,
            transform=axis.get_xaxis_transform(),
            color="#F4A261",
            alpha=0.10,
            linewidth=0,
        )
        for timestamp, color in zip(plot_frame_times, colors, strict=True):
            axis.axvline(timestamp, color=color, linewidth=0.8, alpha=0.75)
        axis.grid(alpha=0.22, linewidth=0.6)
        # Stop at the final valid rollout timestamp. Extending beyond this point can
        # make termination/padding values look like an active policy response.
        axis.set_xlim(plot_frame_times[0] - 0.5, plot_episode_end)
    figure.suptitle(f"USB insertion — episode {args.episode}", fontsize=11, fontweight="bold")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(args.output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    metadata = {
        "episode": args.episode,
        "mean_force_rate_hz": mean_rate,
        "median_force_rate_hz": float(1.0 / np.median(np.diff(high_time))),
        "force_content_lag_correction_s": force_content_lag_s,
        "force_alignment_correlation": force_alignment_correlation,
        "high_load_threshold_n": args.threshold_n,
        "completion_threshold_n": args.completion_threshold_n,
        "contact_onset_definition": (
            "First crossing of the 50 ms causal-smoothed baseline-corrected force norm "
            f"above {args.threshold_n:g} N that persists for 100 ms."
        ),
        "completion_definition": (
            "First post-peak crossing of the 50 ms causal-smoothed baseline-corrected force norm "
            f"below {args.completion_threshold_n:g} N that persists for 200 ms; not episode termination."
        ),
        "high_load_onset_s": onset_time,
        "peak_force_deviation_n": float(smooth_force[peak_index]),
        "peak_time_s": peak_time,
        "completion_time_s": completion_time,
        "frame_times_s": frame_times.tolist(),
        "plot_frame_times_s": plot_frame_times.tolist(),
        "episode_end_s": float(main_time[-1]),
        "time_axis": "relative_to_contact_onset" if args.relative_time else "episode_time",
        "shows_vla_wrench_overlay": args.show_vla_wrench,
        "hypothetical_residual": args.hypothetical_residual,
        "hypothetical_residual_peak_mm": (
            args.hypothetical_residual_peak_mm if args.hypothetical_residual else None
        ),
        "hypothetical_residual_rate_hz": (
            args.hypothetical_residual_rate_hz if args.hypothetical_residual else None
        ),
        "hypothetical_residual_latency_ms": (
            args.hypothetical_residual_latency_ms if args.hypothetical_residual else None
        ),
        "mock_contact_to_first_post_contact_packet_ms": (
            mock_contact_response_ms if args.hypothetical_residual else None
        ),
        "mock_first_post_contact_packet_mm": (
            mock_first_response_mm if args.hypothetical_residual else None
        ),
        "note": (
            f"Illustrative layout only; the residual is a causal force-conditioned mock-up published as {args.hypothetical_residual_rate_hz:g} Hz packets after {args.hypothetical_residual_latency_ms:g} ms latency, with a nonzero free-space floor; it is not student checkpoint output."
            if args.hypothetical_residual
            else "Data qualitative only; no USB-trained Fast residual checkpoint was used."
        ),
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
