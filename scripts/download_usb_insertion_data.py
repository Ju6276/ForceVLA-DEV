"""Download USB insertion metadata/parquets and selected episode videos from Drive."""

from __future__ import annotations

import argparse
import importlib
import pathlib

import gdown
import requests

MAIN_FOLDER_ID = "1xb-PiK3OF3Rw5qYnNsGPhYfnOXbAPsU8"
FORCE_FOLDER_ID = "1FvMRHeofFFniQ1zvStXsQXuUfrOsB3dM"


def _tree(folder_id: str):
    module = importlib.import_module("gdown.download_folder")
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    result = module._download_and_parse_google_drive_link(  # noqa: SLF001
        requests.Session(), url, quiet=True, remaining_ok=True
    )
    return result[1] if isinstance(result, tuple) else result


def _walk(node, path=()):
    current = (*path, node.name)
    if node.type.endswith("folder"):
        for child in node.children:
            yield from _walk(child, current)
    else:
        yield current, node


def _download(node, output: pathlib.Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and output.stat().st_size > 0:
        return
    temporary = output.with_suffix(output.suffix + ".part")
    try:
        gdown.download(id=node.id, output=str(temporary), quiet=True)
    except gdown.exceptions.FileURLRetrievalError:
        url = f"https://drive.usercontent.google.com/download?id={node.id}&export=download&confirm=t"
        with requests.get(url, stream=True, timeout=(20, 180)) as response:
            response.raise_for_status()
            with temporary.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--video-episodes", type=int, nargs="*", default=[])
    args = parser.parse_args()
    selected_videos = set(args.video_episodes)

    main_root = _tree(MAIN_FOLDER_ID)
    force_root = _tree(FORCE_FOLDER_ID)
    downloads = []
    for path, node in _walk(main_root):
        relative = pathlib.Path(*path[1:])
        if relative.parts[0] in {"data", "meta"}:
            downloads.append((node, args.output / "main" / relative))
        elif relative.parts[0] == "videos":
            episode = int(relative.name.removeprefix("episode_").removesuffix(".mp4"))
            if episode in selected_videos:
                downloads.append((node, args.output / "main" / relative))
    for path, node in _walk(force_root):
        downloads.append((node, args.output / "force" / pathlib.Path(*path[1:])))

    for index, (node, output) in enumerate(downloads, start=1):
        print(f"[{index}/{len(downloads)}] {output}", flush=True)
        _download(node, output)


if __name__ == "__main__":
    main()
