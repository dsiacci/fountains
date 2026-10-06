"""The model's inputs for a point and a day.

The same function builds the training rows (ONDE stream observations, past
dates, measured rain only) and the rows scored for a ride (fountains, the ride
day, measured rain up to the last gauge report, then the Météo-France forecast).
"""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import meteo

# Everything the model may use. A feature set picks among them (see model.py).
ALL_FEATURES = ("doy", "elevation_m", "rain_30", "rain_90", "rain_180", "ratio_30", "ratio_90", "ratio_180")


@dataclass
class RainContext:
    gauges: meteo.Gauges
    normals: dict
    gauge_meta: dict[str, tuple[float, float]] = field(default_factory=dict)

    @classmethod
    def load(cls, gauges: meteo.Gauges, normals_path: Path) -> "RainContext":
        normals, meta = load_normals(normals_path)
        return cls(gauges=gauges, normals=normals, gauge_meta=meta)


def save_normals(normals: dict, g: meteo.Gauges, path: Path) -> None:
    meta = {sid: [float(g.lat[j]), float(g.lon[j]), g.names[j], float(g.alt[j])] for j, sid in enumerate(g.ids) if sid in normals}
    payload = {
        "source": "Météo-France, données climatologiques de base quotidiennes (Licence Ouverte 2.0), normals over 1991-2020 computed by fountains",
        "windows": list(meteo.WINDOWS),
        "gauges": meta,
        "normals": {sid: {str(w): v for w, v in normals[sid].items()} for sid in normals},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def load_normals(path: Path) -> tuple[dict, dict[str, tuple[float, float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    normals = {sid: {int(w): v for w, v in per.items()} for sid, per in payload["normals"].items()}
    meta = {sid: (m[0], m[1]) for sid, m in payload["gauges"].items()}
    return normals, meta


def rain_features(
    ctx: RainContext,
    lat: float,
    lon: float,
    day: dt.date,
    forecast: dict[dt.date, float | None] | None = None,
) -> dict:
    """Rain before `day` at a point: sums over 30/90/180 days and ratios to normal.

    Days after the last gauge report are filled from `forecast` when given;
    forecast days the model does not cover count as dry (the safe side: a
    missing forecast never makes a fountain look wetter) and are reported in
    `assumed_dry_days`.
    """
    series, used = meteo.point_series(ctx.gauges, lat, lon)
    start = ctx.gauges.start
    need_until = day - dt.timedelta(days=1)
    last_measured = ctx.gauges.last_day
    if need_until > last_measured:
        extra_days = (need_until - last_measured).days
        series = np.concatenate([series[: ctx.gauges.day_index(last_measured) + 1], np.zeros(extra_days, dtype=np.float32)])
    forecast_mm = 0.0
    assumed_dry = []
    for i in range(1, (need_until - last_measured).days + 1):
        d = last_measured + dt.timedelta(days=i)
        v = (forecast or {}).get(d)
        if v is None:
            assumed_dry.append(d.isoformat())
            v = 0.0
        series[(d - start).days] = v
        forecast_mm += v
    end = (day - start).days
    out = {
        "doy": meteo.doy_slot(day) + 1,
        "gauges": used,
        "last_measured": last_measured.isoformat(),
        "forecast_mm": round(forecast_mm, 1),
        "assumed_dry_days": assumed_dry,
    }
    for w in meteo.WINDOWS:
        s = meteo.window_sum(series, end, w)
        n = meteo.point_normal(ctx.normals, ctx.gauge_meta, lat, lon, day, w)
        out[f"rain_{w}"] = None if math.isnan(s) else round(s, 1)
        out[f"normal_{w}"] = None if math.isnan(n) else round(n, 1)
        out[f"ratio_{w}"] = None if (math.isnan(s) or math.isnan(n) or n <= 0) else round(s / n, 3)
    return out


def vector(row: dict, features: tuple[str, ...]) -> list[float]:
    """Model input in a fixed order; missing values become NaN (TabPFN handles them)."""
    return [math.nan if row.get(f) is None else float(row[f]) for f in features]
