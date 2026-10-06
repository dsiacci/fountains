"""Compare the scores with what a rider remembers seeing at fountains.

Memories are not measurements: a rider remembers the fountain they stopped
at, not every one they passed, and remembers dates loosely. This check is a
sanity test of the stream-to-fountain assumption on a handful of cases. The
model is never adjusted to agree with it.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np

from . import DATA_DIR, meteo
from .calibrate import wilson
from .elevation import elevations
from .features import RainContext, rain_features
from .model import band, fit_predict, load_bands, read_table
from .osm import load_fountains

REMEMBERED = {"flowing": 1, "weak": 1, "dry": 0}


def check_memories(path: Path, n_estimators: int = 4, data_dir: Path = DATA_DIR) -> str:
    with open(path, encoding="utf-8") as f:
        mem = [r for r in csv.DictReader(f) if r.get("state") in REMEMBERED]
    if not mem:
        return "No usable memory (state must be flowing, weak or dry)."
    fountains = {f["osm_id"]: f for f in load_fountains(data_dir / "fountains-corsica.geojson")}
    mem = [m for m in mem if m["osm_id"] in fountains]
    days = [dt.date.fromisoformat(m["date"]) for m in mem]
    g = meteo.read_gauges(meteo.download_gauge_files(("latest",)), start=min(days) - dt.timedelta(days=400))
    ctx = RainContext.load(g, data_dir / "gauge-normals-1991-2020.json")
    zs = elevations([(fountains[m["osm_id"]]["lat"], fountains[m["osm_id"]]["lon"]) for m in mem])
    rows = []
    for m, d, z in zip(mem, days, zs):
        f = fountains[m["osm_id"]]
        r = rain_features(ctx, f["lat"], f["lon"], d, forecast=None)
        r["elevation_m"] = z
        rows.append(r)
    cal = json.loads((data_dir / "calibration.json").read_text(encoding="utf-8"))
    p = fit_predict(read_table(data_dir / "training.csv"), rows, tuple(cal["features"]), n_estimators=n_estimators)
    bands = load_bands(data_dir / "calibration.json")
    y = np.array([REMEMBERED[m["state"]] for m in mem])
    lines = ["| Date | Fountain | Remembered | Band | 90-day rain |", "|---|---|---|---|---|"]
    per_band: dict[str, list[int]] = {}
    for m, d, r, pi, yi in zip(mem, days, rows, p, y):
        f = fountains[m["osm_id"]]
        b = band(float(pi), bands)
        per_band.setdefault(b, []).append(int(yi))
        name = f.get("name") or f["kind"]
        lines.append(f"| {d.isoformat()} | {name} ({m['osm_id']}) | {m['state']} | {b} | {r['rain_90']} mm |")
    lines.append("")
    for b in ("likely", "uncertain", "unlikely"):
        v = per_band.get(b, [])
        if v:
            lo, hi = wilson(sum(v), len(v))
            lines.append(f"- {b}: {sum(v)} of {len(v)} remembered running (95 % interval {lo:.0%} to {hi:.0%})")
        else:
            lines.append(f"- {b}: no case")
    lines.append(f"- Brier score on these {len(y)} memories: {float(np.mean((p - y) ** 2)):.3f} (memories, not measurements; n is small)")
    return "\n".join(lines)
