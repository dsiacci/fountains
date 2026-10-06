"""Score the fountains along a track for a given day."""

from __future__ import annotations

import bisect
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
from .ign import load_ign, merge
from .model import BAND_LIKELY, FEATURE_SETS, band, fit_predict, load_bands, read_table
from .network import build_graph, detour_m, distances_from_track
from .osm import load_fountains, ways_near_points
from .stops import is_open, passing_minutes
from .views import aerial_url, plan_url, street_view

NEARBY_STREAM_KM = 15.0


@dataclass
class ScoredFountain:
    ref: str  # "node/123" (OpenStreetMap) or "IGN PAIHYDRO..." (BD TOPO)
    source: str  # "OSM", "IGN" or "OSM + IGN"
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
    eta: str = ""  # passing time, when a start time is given
    views: dict = field(default_factory=dict)  # street (Panoramax picture or None), plan, aerial (image URLs)

    def label(self) -> str:
        return self.name or self.kind


@dataclass
class Stop:
    ref: str
    name: str
    kind: str
    lat: float
    lon: float
    km: float
    detour_m: float | None
    opening_hours: str
    eta: str
    open_then: bool | None


@dataclass
class Gap:
    from_km: float
    to_km: float
    stops: list[Stop]

    @property
    def length_km(self) -> float:
        return round(self.to_km - self.from_km, 1)


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


def water_points(data_dir: Path) -> list[dict]:
    """OpenStreetMap drinking-water points, plus IGN fountains OSM does not have."""
    osm = [dict(f, ref=f["osm_id"]) for f in load_fountains(data_dir / "fountains-corsica.geojson")]
    ign_path = data_dir / "fountains-ign-corsica.geojson"
    if not ign_path.exists():
        return [dict(p, source="OSM", ign_id="") for p in osm]
    pts = merge(osm, load_ign(ign_path))
    for p in pts:
        p.setdefault("ref", "")
        if not p["ref"]:
            p["ref"] = f"IGN {p['ign_id']}"
    return pts


def find_gaps(length_km: float, water_km: list[float], min_gap_km: float) -> list[tuple[float, float]]:
    """Stretches of at least `min_gap_km` between the start, the likely fountains and the end."""
    marks = [0.0] + sorted(water_km) + [length_km]
    return [(a, b) for a, b in zip(marks, marks[1:]) if b - a >= min_gap_km]


def _clock(start: dt.datetime | None, minutes: float) -> tuple[str, dt.datetime | None]:
    if start is None:
        return "", None
    t = start + dt.timedelta(minutes=minutes)
    return t.strftime("%H:%M"), t


def score_track(
    track: Track,
    day: dt.date,
    max_detour_m: float = 250.0,
    use_forecast: bool = True,
    n_estimators: int = 4,
    data_dir: Path = DATA_DIR,
    start: dt.time | None = None,
    flat_kmh: float = 20.0,
    climb_mh: float = 500.0,
    gap_km: float = 10.0,
    photos: bool = True,
    points: list[dict] | None = None,
    log=print,
) -> tuple[list[ScoredFountain], list[Gap], dict]:
    """Score the water points along `track` for `day`.

    `points` replaces the map's water points with a list chosen by the rider
    (for example the ones validated on photos), each a dict with at least
    ref, source, name, kind, lat and lon.
    """
    if not all(in_corsica(lat, lon) for lat, lon in track.points[:: max(1, len(track.points) // 50)]):
        raise SystemExit("This model is trained on Corsican streams and rain gauges only; the track leaves Corsica.")
    line = Polyline(track.points)
    minutes = passing_minutes(line.cum, track.ele, flat_kmh, climb_mh)
    start_dt = dt.datetime.combine(day, start) if start else None

    def minutes_at(s_m: float) -> float:
        i = min(bisect.bisect_left(line.cum, s_m), len(minutes) - 1)
        return minutes[i]

    # 1. Water points near the track (straight line), then along the network.
    near, excluded_private = [], []
    for f in points if points is not None else water_points(data_dir):
        d, s = line.nearest(f["lat"], f["lon"], within_m=max_detour_m)
        if math.isinf(d):
            continue
        if f.get("access") in ("private", "no"):
            excluded_private.append(f)
            continue
        near.append((f, d, s))
    log(f"{len(near)} water points within {max_detour_m:.0f} m of the track (straight line).")

    stops_near = []
    stops_path = data_dir / "stops-corsica.geojson"
    if stops_path.exists():
        for f in load_fountains(stops_path):  # same GeoJSON shape: flat dicts with lat/lon
            d, s = line.nearest(f["lat"], f["lon"], within_m=max_detour_m)
            if not math.isinf(d):
                stops_near.append((f, d, s))

    detours: dict[str, float | None] = {}
    positions = [(f["lat"], f["lon"]) for f, _, _ in near + stops_near]
    if positions:
        log("Fetching the roads and paths around those points from OpenStreetMap...")
        g = build_graph(ways_near_points(positions, max_detour_m + 150), line.proj)
        dist = distances_from_track(g, line)
        for f, _, _ in near + stops_near:
            detours[f["ref"] if "ref" in f else f["osm_id"]] = detour_m(g, dist, f["lat"], f["lon"])

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

    def look(f: dict) -> dict:
        street = None
        if photos:
            try:
                street = street_view(f["lat"], f["lon"])
            except Exception as e:  # noqa: BLE001 - a missing photo never stops the score
                log(f"Panoramax unavailable for {f['ref']}: {e}")
        return {"street": street, "plan": plan_url(f["lat"], f["lon"]), "aerial": aerial_url(f["lat"], f["lon"])}

    if photos and near:
        log("Looking for street photos of those points on Panoramax...")
    scored = []
    for (f, d, s), r, p in zip(near, rows, probs):
        det = detours.get(f["ref"])
        notes = []
        if f.get("access") == "customers":
            notes.append("customers only (access=customers)")
        if f.get("seasonal"):
            notes.append(f"OSM says seasonal={f['seasonal']}")
        if f.get("drinking_water") == "conditional":
            notes.append("OSM says drinking_water=conditional")
        if f["source"] == "IGN":
            notes.append("from IGN only: not in OpenStreetMap, no drinking-water information")
        if r["assumed_dry_days"]:
            notes.append(f"no forecast yet for {len(r['assumed_dry_days'])} day(s) before the ride, counted as dry")
        scored.append(
            ScoredFountain(
                ref=f["ref"], source=f["source"], name=f.get("name", ""), kind=f["kind"], lat=f["lat"], lon=f["lon"],
                km=round(s / 1000, 1), off_track_m=round(d), detour_m=None if det is None else round(det),
                reachable=det is not None and det <= max_detour_m, access=f.get("access", ""),
                elevation_m=r["elevation_m"], p=float(p), band=band(float(p), bands), rain=r,
                last_confirmed=f.get("last_confirmed", ""), stream=nearest_stream(obs, f["lat"], f["lon"], day),
                notes=notes, eta=_clock(start_dt, minutes_at(s))[0], views=look(f),
            )
        )
    scored.sort(key=lambda x: x.km)

    # 4. Long stretches without a likely fountain, and the shops along them.
    likely_km = [f.km for f in scored if f.reachable and f.band == BAND_LIKELY]
    gaps = []
    for a, b in find_gaps(round(line.length_m / 1000, 1), likely_km, gap_km):
        stops = []
        for f, d, s in stops_near:
            km = s / 1000
            det = detours.get(f["osm_id"])
            if not (a <= km <= b) or det is None or det > max_detour_m:
                continue
            eta, when = _clock(start_dt, minutes_at(s))
            stops.append(Stop(ref=f["osm_id"], name=f.get("name", ""), kind=f["kind"], lat=f["lat"], lon=f["lon"],
                              km=round(km, 1), detour_m=round(det), opening_hours=f.get("opening_hours", ""),
                              eta=eta, open_then=is_open(f.get("opening_hours"), when) if when else None))
        gaps.append(Gap(from_km=a, to_km=b, stops=sorted(stops, key=lambda x: x.km)))

    meta = {
        "track": track.name,
        "day": day.isoformat(),
        "start": start.strftime("%H:%M") if start else "",
        "pace": {"flat_kmh": flat_kmh, "climb_mh": climb_mh, "uses_elevation": bool(track.ele)},
        "max_detour_m": max_detour_m,
        "gap_km": gap_km,
        "length_km": round(line.length_m / 1000, 1),
        "gauges_last_day": gauges.last_day.isoformat(),
        "forecast_used": need_forecast,
        "excluded_private": len(excluded_private),
        "calibration": {k: cal[k] for k in ("brier", "brier_month_rate", "n_observations", "n_stations", "per_band")},
    }
    return scored, gaps, meta
