"""Download only the episode videos selected for button-press train/validation."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import gdown
import requests

FOLDER_ID = "1toz0vAk15v26asD3Jug0z67StmemIysV"
FOLDER_URL = f"https://drive.google.com/drive/folders/{FOLDER_ID}"


def _walk(node, path: tuple[str, ...] = ()):
    current = (*path, node.name)
    if node.type.endswith("folder"):
        for child in node.children:
            yield from _walk(child, current)
    else:
        yield current, node


def _download_file(file_id: str, output: Path) -> None:
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    temporary = output.with_suffix(output.suffix + ".part")
    with requests.get(url, stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        with temporary.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("data/panda_button_press_100hz_causal/split.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/panda_button_press_selected/panda_button_press"),
    )
    parser.add_argument("--all-episodes", action="store_true")
    args = parser.parse_args()

    split = json.loads(args.split.read_text())
    selected = set(range(100)) if args.all_episodes else set(split["train"]) | set(split["val"])
    module = importlib.import_module("gdown.download_folder")
    major_version = int(gdown.__version__.split(".", maxsplit=1)[0])
    folder_reference = FOLDER_ID if major_version >= 5 else FOLDER_URL
    root = module._download_and_parse_google_drive_link(  # noqa: SLF001
        requests.Session(), folder_reference, quiet=True
    )
    files = []
    for path, node in _walk(root):
        if "panda_button_press" not in path or "videos" not in path or not node.name.endswith(".mp4"):
            continue
        episode_index = int(node.name.removeprefix("episode_").removesuffix(".mp4"))
        if episode_index in selected:
            camera = path[-2]
            files.append((episode_index, camera, node))

    expected = 2 * len(selected)
    if len(files) != expected:
        raise RuntimeError(f"Expected {expected} selected videos, found {len(files)}")
    for position, (episode_index, camera, node) in enumerate(sorted(files), start=1):
        output = args.output_root / "videos" / "chunk-000" / camera / node.name
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.is_file() and output.stat().st_size > 0:
            print(f"[{position}/{expected}] exists: {output}")
            continue
        print(f"[{position}/{expected}] downloading episode {episode_index}, {camera}")
        _download_file(node.id, output)


if __name__ == "__main__":
    main()
