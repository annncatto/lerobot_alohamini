#!/usr/bin/env python3
"""Create a hard-linked AlohaMini dataset view with visual observations only."""

import argparse
import json
import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq

KEEP_CAMERAS = ("forward", "wrist_right")
DROP_FEATURES = ("observation.state", "observation.images.chest")


def link_tree(source: Path, target: Path) -> None:
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        destination = target / relative
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, destination)


def rewrite_parquet_without_prefixes(source: Path, target: Path, prefixes: tuple[str, ...]) -> None:
    table = pq.read_table(source)
    keep_columns = [name for name in table.column_names if not name.startswith(prefixes)]
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.select(keep_columns), target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    target = args.target.expanduser().resolve()
    if not (source / "meta/info.json").is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset: {source}")
    if target.exists():
        raise FileExistsError(f"Target already exists; refusing to modify it: {target}")

    target.mkdir(parents=True)
    info = json.loads((source / "meta/info.json").read_text())
    stats = json.loads((source / "meta/stats.json").read_text())
    for feature in DROP_FEATURES:
        if feature not in info["features"]:
            raise KeyError(f"Source dataset is missing expected feature {feature!r}")
        del info["features"][feature]
        stats.pop(feature, None)

    (target / "meta").mkdir()
    (target / "meta/info.json").write_text(json.dumps(info, ensure_ascii=False, indent=4) + "\n")
    (target / "meta/stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=4) + "\n")
    shutil.copy2(source / "meta/tasks.parquet", target / "meta/tasks.parquet")

    for episode_file in (source / "meta/episodes").rglob("*.parquet"):
        rewrite_parquet_without_prefixes(
            episode_file,
            target / episode_file.relative_to(source),
            (
                "videos/observation.images.chest/",
                "stats/observation.images.chest/",
                "stats/observation.state/",
            ),
        )

    # Numeric data is small; rewrite it so observation.state is absent from the
    # dataloader batch instead of merely being ignored by the policy.
    for data_file in (source / "data").rglob("*.parquet"):
        rewrite_parquet_without_prefixes(
            data_file,
            target / data_file.relative_to(source),
            ("observation.state",),
        )

    for camera in KEEP_CAMERAS:
        video_dir = source / "videos" / f"observation.images.{camera}"
        if not video_dir.is_dir():
            raise FileNotFoundError(video_dir)
        link_tree(video_dir, target / "videos" / video_dir.name)

    print(f"Created visual-only dataset view: {target}")
    print(f"Kept cameras: {', '.join(KEEP_CAMERAS)}")
    print(f"Removed features: {', '.join(DROP_FEATURES)}")


if __name__ == "__main__":
    main()
