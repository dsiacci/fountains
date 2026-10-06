"""Fountains from IGN's BD TOPO, the second source next to OpenStreetMap.

Source: IGN, BD TOPO®, class « détail hydrographique », nature « Fontaine »,
through the Géoplateforme WFS service (https://data.geopf.fr/wfs). Open data
under the Licence Ouverte 2.0 / Etalab since 1 January 2021.

In Corsica, BD TOPO knows about 1,500 fountains, OpenStreetMap about 600
drinking-water points, and only a few hundred are in both. A point from both
sources within MERGE_M metres is shown once.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

from .geo import haversine_m

WFS_URL = "https://data.geopf.fr/wfs/ows"
USER_AGENT = "fountains/0.1 (+https://github.com/dsiacci/fountains)"
CORSICA_BBOX_WFS = "41.30,8.50,43.05,9.60"
MERGE_M = 30.0


def fetch_fountains() -> dict:
    """All BD TOPO « Fontaine » features in Corsica, as GeoJSON."""
    cql = f"nature='Fontaine' AND BBOX(geometrie,{CORSICA_BBOX_WFS},'urn:ogc:def:crs:EPSG::4326')"
    q = urllib.parse.urlencode(
        {
            "SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature",
            "TYPENAMES": "BDTOPO_V3:detail_hydrographique", "OUTPUTFORMAT": "application/json",
            "COUNT": "10000", "CQL_FILTER": cql,
        }
    )
    req = urllib.request.Request(f"{WFS_URL}?{q}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.load(r)
    if len(data["features"]) >= 10000:
        raise RuntimeError("IGN returned a full page; add paging")
    feats = []
    for f in data["features"]:
        p = f["properties"]
        lon, lat = f["geometry"]["coordinates"][:2]
        feats.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(lon, 7), round(lat, 7)]},
                "properties": {
                    "ign_id": p["cleabs"],
                    "name": p.get("toponyme") or "",
                    "state": p.get("etat_de_l_objet") or "",
                    "ign_confirmed": (p.get("date_de_confirmation") or "")[:10],
                },
            }
        )
    feats.sort(key=lambda f: f["properties"]["ign_id"])
    return {
        "type": "FeatureCollection",
        "license": "Licence Ouverte 2.0 (Etalab). Source: IGN, BD TOPO®",
        "features": feats,
    }


def load_ign(path: str | Path) -> list[dict]:
    fc = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for f in fc["features"]:
        lon, lat = f["geometry"]["coordinates"]
        p = f["properties"]
        if p.get("state") and p["state"] != "En service":
            continue
        out.append(
            {
                "osm_id": "",
                "ign_id": p["ign_id"],
                "source": "IGN",
                "name": p.get("name", ""),
                "kind": "fountain",
                "lat": lat,
                "lon": lon,
                "last_confirmed": p.get("ign_confirmed", ""),
            }
        )
    return out


def merge(osm_points: list[dict], ign_points: list[dict], within_m: float = MERGE_M) -> list[dict]:
    """OpenStreetMap points, plus the IGN fountains that are not already mapped there.

    A matched pair keeps the OpenStreetMap point (it carries access and
    drinking-water tags), takes the IGN name when OSM has none, and the most
    recent check date of the two.
    """
    out = [dict(p, source="OSM", ign_id="") for p in osm_points]
    for g in ign_points:
        best, best_d = None, within_m
        for p in out:
            if abs(p["lat"] - g["lat"]) > 0.001 or abs(p["lon"] - g["lon"]) > 0.0015:
                continue
            d = haversine_m(p["lat"], p["lon"], g["lat"], g["lon"])
            if d <= best_d:
                best, best_d = p, d
        if best is None:
            out.append(dict(g))
        elif not best["ign_id"]:
            best["source"] = "OSM + IGN"
            best["ign_id"] = g["ign_id"]
            if not best.get("name"):
                best["name"] = g["name"]
            best["last_confirmed"] = max(best.get("last_confirmed", ""), g["last_confirmed"])
    return out
