"""Drop episodes whose force stream is slower than the rate the model assumes.

The Button Press recordings range from roughly 57 Hz to 314 Hz of wrench. The
force front-end resamples onto a fixed causal grid and invalidates slots older
than `max_sample_age_ms`, so a slow episode does not crash training; it quietly
contributes windows that are mostly padding. Removing those episodes is a data
decision, and this script makes it reproducible.

Episode rate is measured over the whole span, `(n - 1) / (t_last - t_first)`,
rather than from the median gap. The median only sees the typical sample and
stays high on a stream that drops packets, which is exactly the stream this
filter is meant to catch.

A LeRobot dataset numbers episodes contiguously and carries a dataset-wide frame
index, so episodes cannot simply be deleted. This writes a new dataset with both
renumbered and leaves the input untouched.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def episode_force_rates(sidecar_dir: pathlib.Path) -> dict[int, float]:
    rates = {}
    for path in sorted(sidecar_dir.glob("episode_*.npz")):
        index = int(path.stem.split("_")[-1])
        timestamps = np.asarray(np.load(path)["timestamps"], dtype=np.float64)
        if timestamps.size < 2:
            rates[index] = 0.0
            continue
        span = timestamps[-1] - timestamps[0]
        rates[index] = (timestamps.size - 1) / span if span > 0 else 0.0
    return rates


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: pathlib.Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=pathlib.Path, required=True, help="LeRobot dataset root")
    parser.add_argument("--sidecars", type=pathlib.Path, required=True, help="Force sidecar directory")
    parser.add_argument("--output", type=pathlib.Path, required=True, help="New dataset root")
    parser.add_argument("--sidecar-output", type=pathlib.Path, required=True, help="New sidecar directory")
    parser.add_argument("--min-force-rate-hz", type=float, default=100.0)
    parser.add_argument("--link", action="store_true", help="Hardlink parquet/video instead of copying")
    parser.add_argument("--dry-run", action="store_true", help="Report the selection and stop")
    args = parser.parse_args()

    rates = episode_force_rates(args.sidecars)
    if not rates:
        raise ValueError(f"No sidecars under {args.sidecars}")
    info = json.loads((args.dataset / "meta" / "info.json").read_text())
    episodes = _read_jsonl(args.dataset / "meta" / "episodes.jsonl")
    stats = _read_jsonl(args.dataset / "meta" / "episodes_stats.jsonl")
    if len(episodes) != info["total_episodes"]:
        raise ValueError(f"info.json claims {info['total_episodes']} episodes but episodes.jsonl has {len(episodes)}")
    missing = {record["episode_index"] for record in episodes} - rates.keys()
    if missing:
        raise ValueError(f"No force sidecar for episodes {sorted(missing)}")

    keep = [record["episode_index"] for record in episodes if rates[record["episode_index"]] >= args.min_force_rate_hz]
    dropped = [index for index in sorted(rates) if index not in set(keep)]
    kept_frames = sum(record["length"] for record in episodes if record["episode_index"] in set(keep))
    print(f"kept {len(keep)} / {len(episodes)} episodes, {kept_frames} / {info['total_frames']} frames")
    print(f"dropped ({len(dropped)}): {dropped}")
    if args.dry_run:
        return

    stats_by_index = {record["episode_index"]: record for record in stats}
    episodes_by_index = {record["episode_index"]: record for record in episodes}
    video_keys = [name for name, feature in info["features"].items() if feature.get("dtype") == "video"]

    for directory in (args.output, args.sidecar_output):
        if directory.exists():
            raise FileExistsError(f"{directory} already exists; remove it or pick another path")
    (args.output / "meta").mkdir(parents=True)
    args.sidecar_output.mkdir(parents=True)

    transfer = _hardlink if args.link else shutil.copy2
    new_episodes, new_stats, frame_offset = [], [], 0
    for new_index, old_index in enumerate(keep):
        chunk = new_index // info["chunks_size"]
        source = args.dataset / info["data_path"].format(episode_chunk=old_index // info["chunks_size"], episode_index=old_index)
        target = args.output / info["data_path"].format(episode_chunk=chunk, episode_index=new_index)
        target.parent.mkdir(parents=True, exist_ok=True)

        table = pq.read_table(source)
        rows = table.num_rows
        table = table.set_column(
            table.schema.get_field_index("episode_index"),
            "episode_index",
            pa.array(np.full(rows, new_index, dtype=np.int64)),
        )
        # `index` is a dataset-wide frame counter, so it has to be rebuilt rather
        # than carried over from the original numbering.
        table = table.set_column(
            table.schema.get_field_index("index"),
            "index",
            pa.array(np.arange(frame_offset, frame_offset + rows, dtype=np.int64)),
        )
        pq.write_table(table, target)
        frame_offset += rows

        for key in video_keys:
            video_source = args.dataset / info["video_path"].format(
                episode_chunk=old_index // info["chunks_size"], video_key=key, episode_index=old_index
            )
            video_target = args.output / info["video_path"].format(
                episode_chunk=chunk, video_key=key, episode_index=new_index
            )
            video_target.parent.mkdir(parents=True, exist_ok=True)
            transfer(video_source, video_target)

        transfer(args.sidecars / f"episode_{old_index:06d}.npz", args.sidecar_output / f"episode_{new_index:06d}.npz")

        episode = dict(episodes_by_index[old_index])
        episode["episode_index"] = new_index
        if episode["length"] != rows:
            raise ValueError(f"Episode {old_index}: metadata says {episode['length']} frames, parquet has {rows}")
        new_episodes.append(episode)
        stat = dict(stats_by_index[old_index])
        stat["episode_index"] = new_index
        new_stats.append(stat)

    info = {
        **info,
        "total_episodes": len(new_episodes),
        "total_frames": frame_offset,
        "total_videos": len(new_episodes) * len(video_keys),
        "total_chunks": (len(new_episodes) - 1) // info["chunks_size"] + 1,
        "splits": {"train": f"0:{len(new_episodes)}"},
    }
    (args.output / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    _write_jsonl(args.output / "meta" / "episodes.jsonl", new_episodes)
    _write_jsonl(args.output / "meta" / "episodes_stats.jsonl", new_stats)
    for name in ("tasks.jsonl", "crisp_meta.json"):
        source = args.dataset / "meta" / name
        if source.is_file():
            shutil.copy2(source, args.output / "meta" / name)

    summary = {
        "min_force_rate_hz": args.min_force_rate_hz,
        "kept_episodes": len(new_episodes),
        "dropped_episodes": dropped,
        "total_frames": frame_offset,
        "kept_force_rate_hz": {
            "min": min(rates[index] for index in keep),
            "max": max(rates[index] for index in keep),
        },
    }
    (args.output / "meta" / "force_rate_filter.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def _hardlink(source, target) -> None:
    pathlib.Path(target).hardlink_to(source)


if __name__ == "__main__":
    main()
