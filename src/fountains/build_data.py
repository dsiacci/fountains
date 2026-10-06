"""Rebuild the files in data/ from the open sources (maintainer command).

    fountains build-data --all

1. OpenStreetMap drinking-water points in Corsica      -> fountains-corsica.geojson
2. ONDE stream observations in Corsica (Hub'Eau)      -> onde-observations.csv
3. Météo-France gauge normals over 1991-2020           -> gauge-normals-1991-2020.json
4. The training table (features + label per observation) -> training.csv
"""

from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

from . import DATA_DIR, meteo, onde
from .elevation import elevations
from .features import RainContext, save_normals
from .osm import FOUNTAIN_QUERY, fountains_from_overpass, overpass


def refresh_osm(data_dir: Path = DATA_DIR, log=print) -> None:
    fc = fountains_from_overpass(overpass(FOUNTAIN_QUERY))
    (data_dir / "fountains-corsica.geojson").write_text(json.dumps(fc, ensure_ascii=False, indent=0), encoding="utf-8")
    log(f"OSM: {len(fc['features'])} drinking-water points (data as of {fc['osm_base']})")


def refresh_onde(data_dir: Path = DATA_DIR, log=print) -> None:
    n = onde.write_observations_csv(onde.fetch_observations(), data_dir / "onde-observations.csv")
    log(f"ONDE: {n} usable observations")


def build_normals(data_dir: Path = DATA_DIR, log=print) -> None:
    g = meteo.read_gauges(meteo.download_gauge_files(("previous", "latest")), start=dt.date(1990, 1, 1))
    normals = meteo.gauge_normals(g)
    save_normals(normals, g, data_dir / "gauge-normals-1991-2020.json")
    log(f"Météo-France: {len(g.ids)} gauges since 1990, {len(normals)} with 1991-2020 normals")


def build_training(data_dir: Path = DATA_DIR, log=print) -> None:
    obs = onde.read_observations_csv(data_dir / "onde-observations.csv")
    g = meteo.read_gauges(meteo.download_gauge_files(("previous", "latest")), start=dt.date(2010, 1, 1))
    ctx = RainContext.load(g, data_dir / "gauge-normals-1991-2020.json")
    stations = {}
    for o in obs:
        stations.setdefault(o["station"], (o["lat"], o["lon"]))
    zs = dict(zip(stations, elevations(list(stations.values()))))
    series = {s: meteo.point_series(g, *ll)[0] for s, ll in stations.items()}
    sums = {s: {w: meteo.window_sums(series[s], w) for w in meteo.WINDOWS} for s in stations}
    out = []
    for o in obs:
        day = dt.date.fromisoformat(o["date"])
        i = g.day_index(day)
        row = {k: o[k] for k in ("station", "station_name", "stream", "date", "code", "flowing")}
        row["doy"] = meteo.doy_slot(day) + 1
        row["elevation_m"] = zs[o["station"]]
        for w in meteo.WINDOWS:
            s = float(sums[o["station"]][w][i]) if 0 <= i < len(series[o["station"]]) + 1 else float("nan")
            n = meteo.point_normal(ctx.normals, ctx.gauge_meta, o["lat"], o["lon"], day, w)
            row[f"rain_{w}"] = "" if s != s else round(s, 1)
            row[f"normal_{w}"] = "" if n != n else round(n, 1)
            row[f"ratio_{w}"] = "" if (s != s or n != n or n <= 0) else round(s / n, 3)
        out.append(row)
    with open(data_dir / "training.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    missing = sum(1 for r in out if r["rain_180"] == "")
    log(f"Training table: {len(out)} rows from {len(stations)} stations ({missing} without a complete 180-day rain window)")
