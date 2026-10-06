"""OpenStreetMap data through the Overpass API (read only).

The tool never writes to OpenStreetMap. Data © OpenStreetMap contributors,
available under the Open Database License (ODbL 1.0).
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from . import cache_dir

OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
)
# What a query around a ride declares it may use. Public servers turn away
# (HTTP 504) queries that ask for more than they can spare at the moment, even
# small ones: [timeout:180] with the default 512 MiB was refused where this
# passed in a second.
SMALL_QUERY = "[out:json][timeout:90][maxsize:268435456];"
USER_AGENT = "fountains/0.1 (+https://github.com/dsiacci/fountains)"

# Corsica: the two departments, by their ISO 3166-2 codes in OSM.
CORSICA_AREA = '(area["ISO3166-2"="FR-2A"];area["ISO3166-2"="FR-2B"];)->.c;'

FOUNTAIN_QUERY = (
    "[out:json][timeout:180];\n"
    + CORSICA_AREA
    + """
(
 nwr["amenity"="drinking_water"](area.c);
 nwr["amenity"="water_point"](area.c);
 nwr["natural"="spring"]["drinking_water"](area.c);
 nwr["man_made"="water_tap"]["drinking_water"](area.c);
 nwr["man_made"="water_well"]["drinking_water"](area.c);
 nwr["amenity"="fountain"]["drinking_water"](area.c);
);
out center tags meta;
"""
)

KEEP_TAGS = (
    "name", "amenity", "natural", "man_made", "drinking_water", "access", "seasonal",
    "check_date", "check_date:drinking_water", "survey:date", "intermittent", "ele",
)


def overpass(query: str, *, retries: int = 3, timeout: int = 150) -> dict:
    """Run an Overpass query; on a busy or failing server, try the next public instance."""
    body = urllib.parse.urlencode({"data": query}).encode()
    last = None
    for attempt in range(retries):
        for url in OVERPASS_URLS:
            req = urllib.request.Request(url, data=body, headers={"User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    return json.load(r)
            except Exception as e:  # noqa: BLE001 - network errors of every kind get the same retry
                last = e
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"Overpass query failed after {retries} rounds on {len(OVERPASS_URLS)} servers: {last}")


def fountain_kind(tags: dict) -> str:
    if tags.get("natural") == "spring":
        return "spring"
    if tags.get("man_made") == "water_tap":
        return "tap"
    if tags.get("man_made") == "water_well":
        return "well"
    if tags.get("amenity") == "water_point":
        return "water point"
    if tags.get("amenity") == "fountain":
        return "fountain"
    return "drinking water"


def is_drinking_water_point(tags: dict) -> bool:
    """The selection rule: what OpenStreetMap marks as a place to fill a bottle.

    `amenity=drinking_water` and `amenity=water_point` count unless tagged
    `drinking_water=no`; springs, taps, wells and decorative fountains count
    only when tagged `drinking_water=yes` or `conditional`. Whether the water is
    safe to drink is not something this tool knows or judges.
    """
    dw = tags.get("drinking_water")
    if dw == "no":
        return False
    if tags.get("amenity") in ("drinking_water", "water_point"):
        return True
    if dw in ("yes", "conditional"):
        return (
            tags.get("natural") == "spring"
            or tags.get("man_made") in ("water_tap", "water_well")
            or tags.get("amenity") == "fountain"
        )
    return False


def last_confirmation(tags: dict) -> str:
    """Most recent date a mapper recorded checking this point, or ''."""
    dates = [tags.get(k, "") for k in ("check_date", "check_date:drinking_water", "survey:date")]
    return max((d for d in dates if d), default="")


def fountains_from_overpass(data: dict) -> dict:
    """Turn an Overpass answer into a GeoJSON FeatureCollection of fountains."""
    feats = []
    for e in data["elements"]:
        tags = e.get("tags", {})
        if not is_drinking_water_point(tags):
            continue
        lat = e.get("lat", e.get("center", {}).get("lat"))
        lon = e.get("lon", e.get("center", {}).get("lon"))
        if lat is None or lon is None:
            continue
        props = {k: tags[k] for k in KEEP_TAGS if k in tags}
        props.update(
            osm_id=f"{e['type']}/{e['id']}",
            kind=fountain_kind(tags),
            last_confirmed=last_confirmation(tags),
            last_edited=(e.get("timestamp") or "")[:10],
        )
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(lon, 7), round(lat, 7)]}, "properties": props})
    feats.sort(key=lambda f: f["properties"]["osm_id"])
    return {
        "type": "FeatureCollection",
        "osm_base": data.get("osm3s", {}).get("timestamp_osm_base", ""),
        "license": "ODbL 1.0, © OpenStreetMap contributors (https://www.openstreetmap.org/copyright)",
        "features": feats,
    }


def load_fountains(path: str | Path) -> list[dict]:
    """Fountains as flat dicts with lat/lon."""
    fc = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for f in fc["features"]:
        lon, lat = f["geometry"]["coordinates"]
        out.append(dict(f["properties"], lat=lat, lon=lon))
    return out


def ways_near_points(points: list[tuple[float, float]], radius_m: float) -> dict:
    """Every highway within `radius_m` of any of the points, with its nodes (cached on disk).

    The points are the fountains found near the track, which are public
    OpenStreetMap objects: the track itself is never sent anywhere.
    """
    parts = "\n".join(f' way["highway"](around:{int(radius_m)},{lat:.6f},{lon:.6f});' for lat, lon in points)
    query = SMALL_QUERY + "\n(\n" + parts + "\n);\n(._;>;);\nout body qt;\n"
    key = hashlib.sha256(query.encode()).hexdigest()[:20]
    path = cache_dir("overpass") / f"ways-{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    data = overpass(query)
    path.write_text(json.dumps(data))
    return data
