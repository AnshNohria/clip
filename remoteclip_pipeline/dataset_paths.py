#!/usr/bin/env python3
"""RSICD dataset path helpers for the RemoteCLIP pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

# RSICD is split into these subfolders under the dataset root.
RSICD_SPLIT_DIRS = ("train", "test", "valid", "val", "validation")

IMAGE_EXTENSIONS = (
    "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff",
    "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIF", "*.TIFF",
)


def get_repo_root() -> Path:
    """Absolute path to the clip repo root (e.g. /home/jovyan/clip)."""
    return Path(__file__).resolve().parent.parent


def resolve_rsicd_source_dir(repo_root: Path, explicit: Optional[str] = None) -> Path:
    """
    Resolve the RSICD source directory.

    Default (Linux server): /home/jovyan/clip/Datasets/rsicd
    """
    if explicit:
        return Path(explicit).expanduser().resolve()

    candidates = [
        repo_root / "Datasets" / "rsicd",
        repo_root / "RS-TransCLIP" / "datasets" / "rsicd_images",
        repo_root / "datasets" / "rsicd_images",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return candidates[0].resolve()


def collect_rsicd_images(source_path: Path) -> List[Path]:
    """
    Collect image paths from an RSICD layout.

    Expects images under split subfolders:
        <source>/train/
        <source>/test/
        <source>/valid/

    Falls back to a recursive search of the whole tree if no split folders
    are found (flat or custom layouts).
    """
    source_path = source_path.resolve()
    if not source_path.exists():
        return []

    split_dirs = [source_path / name for name in RSICD_SPLIT_DIRS if (source_path / name).is_dir()]
    search_roots = split_dirs if split_dirs else [source_path]

    found: dict[str, Path] = {}
    for root in search_roots:
        for pattern in IMAGE_EXTENSIONS:
            for path in root.rglob(pattern):
                if path.is_file():
                    found[str(path.resolve())] = path.resolve()

    images = sorted(found.values(), key=lambda p: str(p))
    return images


def describe_rsicd_layout(source_path: Path) -> str:
    """Human-readable summary of what was found under the RSICD root."""
    source_path = source_path.resolve()
    lines = [f"RSICD root: {source_path}"]
    any_split = False
    for name in RSICD_SPLIT_DIRS:
        split = source_path / name
        if split.is_dir():
            any_split = True
            n = len(collect_rsicd_images(split))
            lines.append(f"  {name}/: {n} images")
    if not any_split:
        lines.append("  (no train/test/valid subfolders — scanning tree recursively)")
    return "\n".join(lines)
