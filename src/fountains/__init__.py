"""fountains: will the drinking-water fountains along your ride be running?

Built with PriorLabs-TabPFN.
"""

from __future__ import annotations

import os
from pathlib import Path

__version__ = "0.1.0"

DISCLAIMER = (
    "A score is never a reason to leave with less water. Carry what you need for the "
    "whole ride as if every fountain were dry. This tool says nothing about whether "
    "the water is safe to drink."
)

# Corsica bounding box (with a small margin). The model is trained on Corsican
# streams and Corsican rain gauges only, so the tool refuses tracks elsewhere.
CORSICA_BBOX = (41.30, 8.50, 43.05, 9.60)  # south, west, north, east

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("FOUNTAINS_DATA_DIR", REPO_ROOT / "data"))


def cache_dir(*parts: str) -> Path:
    """Per-user cache for downloads (never inside the repository)."""
    base = Path(os.environ.get("FOUNTAINS_CACHE_DIR", Path.home() / ".cache" / "fountains"))
    path = base.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def in_corsica(lat: float, lon: float) -> bool:
    s, w, n, e = CORSICA_BBOX
    return s <= lat <= n and w <= lon <= e
