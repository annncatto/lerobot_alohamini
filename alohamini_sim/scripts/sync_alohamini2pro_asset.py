#!/usr/bin/env python3
"""Check or synchronize the AlohaMini2Pro runtime asset into alohamini_sim.

RoboTwin is the canonical robot-description source. This script copies only
missing or changed runtime files and never deletes simulator-local files.
Documentation, maintenance references, and tests stay in RoboTwin instead of
being duplicated into every consumer repository.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


DEFAULT_SOURCE = Path("/home/anncatto/RoboTwin/assets/embodiments/alohamini2pro")
DEFAULT_DESTINATION = (
    Path(__file__).resolve().parents[1]
    / "data_engine/agents/aloha_mini/assets/alohamini2pro"
)
IGNORED_PARTS = {"__pycache__", ".pytest_cache"}
IGNORED_SUFFIXES = {".pyc", ".md"}
IGNORED_TOP_LEVEL_DIRS = {"references", "tests"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files(root: Path) -> dict[Path, Path]:
    result: dict[Path, Path] = {}
    if not root.exists():
        return result
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in IGNORED_PARTS for part in relative.parts):
            continue
        if relative.parts[0] in IGNORED_TOP_LEVEL_DIRS:
            continue
        if path.suffix in IGNORED_SUFFIXES:
            continue
        result[relative] = path
    return result


def compare(source: Path, destination: Path) -> tuple[list[Path], list[Path]]:
    source_files = files(source)
    destination_files = files(destination)
    missing: list[Path] = []
    different: list[Path] = []
    for relative, source_path in sorted(source_files.items()):
        destination_path = destination_files.get(relative)
        if destination_path is None:
            missing.append(relative)
        elif source_path.stat().st_size != destination_path.stat().st_size or sha256(
            source_path
        ) != sha256(destination_path):
            different.append(relative)
    return missing, different


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--sync", action="store_true")
    args = parser.parse_args()

    if not args.source.is_dir():
        raise SystemExit(f"canonical asset does not exist: {args.source}")
    missing, different = compare(args.source, args.destination)
    if args.sync:
        for relative in missing + different:
            destination = args.destination / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.source / relative, destination)
            print(f"[COPY] {relative}")
        missing, different = compare(args.source, args.destination)

    print(
        f"[CHECK] source_files={len(files(args.source))} "
        f"missing={len(missing)} different={len(different)}"
    )
    for label, paths in (("MISSING", missing), ("DIFFERENT", different)):
        for path in paths:
            print(f"[{label}] {path}")
    if missing or different:
        raise SystemExit(1)
    print("[OK] alohamini_sim asset matches RoboTwin upstream")


if __name__ == "__main__":
    main()
