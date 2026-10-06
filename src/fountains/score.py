"""Score the fountains along a track for a given day."""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from . import DATA_DIR, in_corsica, meteo, onde
from .elevation import elevations
from .features import RainContext, rain_features
from .geo import Polyline, haversine_m
from .gpx import Track
from .model import FEATURE_SETS, band, fit_predict, load_bands, read_table
from .network import build_graph, detour_m, distances_from_track
from .osm import load_fountains, ways_near_points

NEARBY_STREAM_KM = 15.0


@dataclass
class ScoredFountain:
    osm_id: str
    name: str
    kind: str
    lat: float
    lon: float
    km: float  # distance along the track
    off_track_m: float  # straight line
    detour_m: float | None  # along roads and paths, one way
    reachable: bool
    access: str
    elevation_m: float | None
    p: float
    band: str
    rain: dict
    last_confirmed: str
    stream: dict | None
    notes: list[str] = field(default_factory=list)

    def label(self) -> str:
        return self.name or self.kind


def nearest_stream(obs: list[dict], lat: float, lon: float, day: dt.date) -> dict | None:
    """The closest ONDE station and its last observation on or before `day`."""
    by_station: dict[str, dict] = {}
    for o in obs:
        if o["date"] > day.isoformat():
            continue
        cur = by_station.get(o["station"])
        if cur is None or o["date"] > cur["date"]:
            by_station[o["station"]] = o
    best = None
    for o in by_station.values():
        km = haversine_m(lat, lon, o["lat"], o["lon"]) / 1000
        if km <= NEARBY_STREAM_KM and (best is None or km < best[0]):
            best = (km, o)
    if best is None:
        return None
    km, o = best
    return {"station": o["station_name"], "stream": o["stream"], "km": round(km, 1), "date": o["date"], "flowing": bool(o["flowing"]), "code": o["code"]}


def score_track(
    track: Track,
    day: dt.date,
    max_detour_m: float = 250.0,
    use_forecast: bool = True,
    n_estimators: int = 4,
    data_dir: Path = DATA_DIR,
    log=print,
) -> tuple[list[ScoredFountain], dict]:
    if not all(in_corsica(lat, lon) for lat, lon in track.points[:: max(1, len(track.points) // 50)]):
        raise SystemExit("This model is trained on Corsican streams and rain gauges only; the track leaves Corsica.")
    line = Polyline(track.points)

    # 1. Fountains near the track (straight line), then along the network.
    near = []
    excluded_private = []
    for f in load_fountains(data_dir / "fountains-corsica.geojson"):
        d, s = line.nearest(f["lat"], f["lon"], within_m=max_detour_m)
        if math.isinf(d):
            continue
        if f.get("access") in ("private", "no"):
            excluded_private.append(f)
            continue
        near.append((f, d, s))
    log(f"{len(near)} drinking-water points within {max_detour_m:.0f} m of the track (straight line).")

    detours: dict[str, float | None] = {}
    if near:
        log("Fetching the roads and paths around those points from OpenStreetMap...")
        ways = ways_near_points([(f["lat"], f["lon"]) for f, _, _ in near], max_detour_m + 150)
        g = build_graph(ways, line.proj)
        dist = distances_from_track(g, line)
        for f, _, _ in near:
            detours[f["osm_id"]] = detour_m(g, dist, f["lat"], f["lon"])

    # 2. Rain: Météo-France gauges up to the last report, then the forecast.
    which = ("latest",) if day - dt.timedelta(days=200) >= dt.date(2025, 1, 1) else ("previous", "latest")
    gauges = meteo.read_gauges(meteo.download_gauge_files(which), start=day - dt.timedelta(days=400))
    ctx = RainContext.load(gauges, data_dir / "gauge-normals-1991-2020.json")
    need_forecast = use_forecast and day - dt.timedelta(days=1) > gauges.last_day
    zs = elevations([(f["lat"], f["lon"]) for f, _, _ in near]) if near else []
    rows = []
    for (f, d, s), z in zip(near, zs):
        fc = meteo.forecast_mm(f["lat"], f["lon"]) if need_forecast else None
        r = rain_features(ctx, f["lat"], f["lon"], day, forecast=fc)
        r["elevation_m"] = z
        rows.append(r)

    # 3. TabPFN, given every ONDE observation as context.
    cal = json.loads((data_dir / "calibration.json").read_text(encoding="utf-8"))
    features = tuple(cal["features"])
    assert features in FEATURE_SETS.values(), "calibration.json names an unknown feature set"
    probs = []
    if rows:
        log(f"Scoring {len(rows)} points with TabPFN on CPU (a minute or two)...")
        probs = fit_predict(read_table(data_dir / "training.csv"), rows, features, n_estimators=n_estimators)
    bands = load_bands(data_dir / "calibration.json")
    obs = onde.read_observations_csv(data_dir / "onde-observations.csv")

    out = []
    for (f, d, s), r, p in zip(near, rows, probs):
        det = detours.get(f["osm_id"])
        notes = []
        if f.get("access") == "customers":
            notes.append("customers only (access=customers)")
        if f.get("seasonal"):
            notes.append(f"OSM says seasonal={f['seasonal']}")
        if f.get("drinking_water") == "conditional":
            notes.append("OSM says drinking_water=conditional")
        if r["assumed_dry_days"]:
            notes.append(f"no forecast yet for {len(r['assumed_dry_days'])} day(s) before the ride, counted as dry")
        out.append(
            ScoredFountain(
                osm_id=f["osm_id"], name=f.get("name", ""), kind=f["kind"], lat=f["lat"], lon=f["lon"],
                km=round(s / 1000, 1), off_track_m=round(d), detour_m=None if det is None else round(det),
                reachable=det is not None and det <= max_detour_m, access=f.get("access", ""),
                elevation_m=r["elevation_m"], p=float(p), band=band(float(p), bands), rain=r,
                last_confirmed=f.get("last_confirmed", ""), stream=nearest_stream(obs, f["lat"], f["lon"], day), notes=notes,
            )
        )
    out.sort(key=lambda x: x.km)
    meta = {
        "track": track.name,
        "day": day.isoformat(),
        "max_detour_m": max_detour_m,
        "length_km": round(line.length_m / 1000, 1),
        "gauges_last_day": gauges.last_day.isoformat(),
        "forecast_used": need_forecast,
        "excluded_private": len(excluded_private),
        "calibration": {k: cal[k] for k in ("brier", "brier_month_rate", "n_observations", "n_stations", "per_band")},
    }
    return out, meta
