"""Rain: Météo-France daily gauges (measured) and the Météo-France forecast.

Sources
- Météo-France, « Données climatologiques de base - quotidiennes », department 20
  (Corsica), Licence Ouverte 2.0 / Etalab. RR is the rain measured from 06:00 UTC
  on day J to 06:00 UTC on day J+1, attributed to day J, in mm.
  https://www.data.gouv.fr/fr/datasets/donnees-climatologiques-de-base-quotidiennes/
- Forecast: the Météo-France models (AROME, ARPEGE) served by Open-Meteo,
  https://open-meteo.com/en/docs/meteofrance-api, data under CC BY 4.0.

The rain at a point is interpolated from the nearest gauges that reported that
day (inverse distance weighting). Altitude is left to the model as its own
feature: interpolation does not try to correct for it.
"""

from __future__ import annotations

import csv
import datetime as dt
import gzip
import json
import math
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import cache_dir
from .geo import haversine_m

MF_BASE = "https://meteofrance.s3.sbg.io.cloud.ovh.net/data/synchro_ftp/BASE/QUOT/"
MF_FILES = {
    "previous": "Q_20_previous-1950-2024_RR-T-Vent.csv.gz",
    "latest": "Q_20_latest-2025-2026_RR-T-Vent.csv.gz",
}
OPEN_METEO_MF = "https://api.open-meteo.com/v1/meteofrance"
USER_AGENT = "fountains/0.1 (+https://github.com/dsiacci/fountains)"

WINDOWS = (30, 90, 180)  # days of rain before the day of interest
IDW_K = 3  # gauges used for each day
IDW_CANDIDATES = 8  # nearest gauges considered (the first K that reported are used)
IDW_MAX_KM = 40.0
NORMAL_YEARS = (1991, 2020)


def _download(url: str, path: Path, max_age_s: float | None) -> Path:
    if path.exists() and (max_age_s is None or time.time() - path.stat().st_mtime < max_age_s):
        return path
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=300) as r:
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(r.read())
        tmp.replace(path)
    return path


def download_gauge_files(which: tuple[str, ...] = ("latest",)) -> list[Path]:
    """Fetch the Météo-France daily files (the 'latest' one is refreshed every 6 h)."""
    out = []
    for key in which:
        name = MF_FILES[key]
        max_age = 6 * 3600 if key == "latest" else None
        out.append(_download(MF_BASE + name, cache_dir("meteofrance") / name, max_age))
    return out


@dataclass
class Gauges:
    """Daily rain for every gauge: rr[day_index, gauge_index] in mm (NaN = no report)."""

    ids: list[str]
    names: list[str]
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    start: dt.date
    rr: np.ndarray

    def day_index(self, d: dt.date) -> int:
        return (d - self.start).days

    @property
    def last_day(self) -> dt.date:
        """Last day on which at least one gauge reported."""
        rows = np.where(~np.isnan(self.rr).all(axis=1))[0]
        return self.start + dt.timedelta(days=int(rows[-1]))


def read_gauges(paths: list[Path], start: dt.date = dt.date(1990, 1, 1)) -> Gauges:
    """Parse Météo-France daily files into a day x gauge rain matrix."""
    meta: dict[str, tuple[str, float, float, float]] = {}
    values: list[tuple[str, int, float]] = []
    last = start
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter=";")
            head = next(reader)
            ix = {k: i for i, k in enumerate(head)}
            i_id, i_day, i_rr = ix["NUM_POSTE"], ix["AAAAMMJJ"], ix["RR"]
            for row in reader:
                rr = row[i_rr]
                if not rr:
                    continue
                day = row[i_day]
                d = dt.date(int(day[:4]), int(day[4:6]), int(day[6:8]))
                if d < start:
                    continue
                sid = row[i_id]
                if sid not in meta:
                    meta[sid] = (row[ix["NOM_USUEL"]], float(row[ix["LAT"]]), float(row[ix["LON"]]), float(row[ix["ALTI"]] or "nan"))
                values.append((sid, (d - start).days, float(rr)))
                last = max(last, d)
    ids = sorted(meta)
    col = {sid: j for j, sid in enumerate(ids)}
    rr = np.full(((last - start).days + 1, len(ids)), np.nan, dtype=np.float32)
    for sid, i, v in values:
        rr[i, col[sid]] = v
    return Gauges(
        ids=ids,
        names=[meta[s][0] for s in ids],
        lat=np.array([meta[s][1] for s in ids]),
        lon=np.array([meta[s][2] for s in ids]),
        alt=np.array([meta[s][3] for s in ids]),
        start=start,
        rr=rr,
    )


def point_series(g: Gauges, lat: float, lon: float) -> tuple[np.ndarray, list[tuple[str, float]]]:
    """Daily rain interpolated at a point, and the gauges it leans on.

    For each day, the IDW_K nearest gauges (among the IDW_CANDIDATES nearest,
    within IDW_MAX_KM) that reported that day are weighted by 1/d².
    """
    d_km = np.array([haversine_m(lat, lon, a, b) / 1000 for a, b in zip(g.lat, g.lon)])
    order = [j for j in np.argsort(d_km)[:IDW_CANDIDATES] if d_km[j] <= IDW_MAX_KM]
    if not order:
        return np.full(g.rr.shape[0], np.nan, dtype=np.float32), []
    vals = g.rr[:, order]  # days x candidates, nearest first
    w = 1.0 / np.maximum(d_km[order], 0.5) ** 2
    ok = ~np.isnan(vals)
    rank = np.cumsum(ok, axis=1)
    use = ok & (rank <= IDW_K)
    wsum = (use * w).sum(axis=1)
    num = np.where(use, np.nan_to_num(vals) * w, 0.0).sum(axis=1)
    series = np.where(wsum > 0, num / np.where(wsum > 0, wsum, 1), np.nan).astype(np.float32)
    used = [(g.names[j], round(float(d_km[j]), 1)) for j in order[:IDW_K]]
    return series, used


def window_sum(series: np.ndarray, end_index: int, days: int, max_missing: float = 0.1) -> float:
    """Sum of the `days` values before `end_index` (exclusive); NaN if too many are missing."""
    lo = end_index - days
    if lo < 0 or end_index > len(series):
        return math.nan
    w = series[lo:end_index]
    missing = np.isnan(w).sum()
    if missing > max_missing * days:
        return math.nan
    # Scale up for the few missing days so a gap does not read as a dry spell.
    return float(np.nansum(w) * days / (days - missing))


def window_sums(series: np.ndarray, days: int, max_missing: float = 0.1) -> np.ndarray:
    """Vectorised `window_sum` for every end index: out[i] = window_sum(series, i, days)."""
    vals = np.nan_to_num(series.astype(np.float64))
    miss = np.isnan(series).astype(np.int64)
    cs = np.concatenate([[0.0], np.cumsum(vals)])
    cn = np.concatenate([[0], np.cumsum(miss)])
    out = np.full(len(series) + 1, np.nan)
    if len(series) < days:
        return out
    total = cs[days:] - cs[:-days]
    missing = cn[days:] - cn[:-days]
    ok = missing <= max_missing * days
    scaled = np.where(ok, total * days / np.maximum(days - missing, 1), np.nan)
    out[days:] = scaled
    return out


def gauge_normals(g: Gauges, years: tuple[int, int] = NORMAL_YEARS, min_years: int = 20) -> dict[str, dict[int, list[float]]]:
    """Per gauge, the mean rain of each window ending on each day of the year.

    Returns {gauge_id: {window: [366 values]}}, only for gauges with at least
    `min_years` complete windows (at most 5 % missing days) for every day of the
    year in the normal period. Day-of-year slots follow a leap-year calendar
    (Feb 29 included) so every date has one; Feb 29 uses Feb 28 in common years.
    """
    ends: list[list[int]] = [[] for _ in range(366)]
    for doy in range(366):
        d = dt.date(2000, 1, 1) + dt.timedelta(days=doy)  # 2000 is a leap year
        for y in range(years[0], years[1] + 1):
            try:
                end = dt.date(y, d.month, d.day)
            except ValueError:
                end = dt.date(y, 2, 28)
            ends[doy].append(g.day_index(end))
    out: dict[str, dict[int, list[float]]] = {}
    for j, sid in enumerate(g.ids):
        per_window = {}
        for win in WINDOWS:
            s = window_sums(g.rr[:, j], win, max_missing=0.05)
            means = []
            for idx in ends:
                v = s[[i for i in idx if 0 <= i < len(s)]]
                v = v[~np.isnan(v)]
                if len(v) < min_years:
                    break
                means.append(round(float(v.mean()), 1))
            if len(means) < 366:
                break
            per_window[win] = means
        if len(per_window) == len(WINDOWS):
            out[sid] = per_window
    return out


def doy_slot(d: dt.date) -> int:
    """Index of a date in a leap-year calendar (0..365)."""
    return (dt.date(2000, d.month, d.day) - dt.date(2000, 1, 1)).days


def point_normal(normals: dict, meta: dict[str, tuple[float, float]], lat: float, lon: float, d: dt.date, window: int) -> float:
    """Normal rain for a window at a point: IDW of the nearest gauges that have normals."""
    near = sorted((haversine_m(lat, lon, *meta[sid]) / 1000, sid) for sid in normals if sid in meta)[:IDW_K]
    near = [(km, sid) for km, sid in near if km <= IDW_MAX_KM]
    if not near:
        return math.nan
    slot = doy_slot(d)
    w = [1.0 / max(km, 0.5) ** 2 for km, _ in near]
    v = [normals[sid][window][slot] for _, sid in near]
    return sum(a * b for a, b in zip(w, v)) / sum(w)


def forecast_mm(lat: float, lon: float, past_days: int = 3, forecast_days: int = 4) -> dict[dt.date, float | None]:
    """Daily rain from the Météo-France models at a point (Open-Meteo, CC BY 4.0).

    Days the model does not cover come back as None.
    """
    url = (
        f"{OPEN_METEO_MF}?latitude={lat:.5f}&longitude={lon:.5f}&daily=precipitation_sum"
        f"&past_days={past_days}&forecast_days={forecast_days}&timezone=Europe%2FParis"
    )
    key = f"mf-{lat:.4f}-{lon:.4f}-{past_days}-{forecast_days}.json"
    path = cache_dir("open-meteo") / key
    if not path.exists() or time.time() - path.stat().st_mtime > 3 * 3600:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as r:
            path.write_bytes(r.read())
    data = json.loads(path.read_text())
    days = data["daily"]["time"]
    vals = data["daily"]["precipitation_sum"]
    return {dt.date.fromisoformat(t): (None if v is None else float(v)) for t, v in zip(days, vals)}
