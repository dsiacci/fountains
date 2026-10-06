"""Altitude of a point from the IGN altimetry service (Géoplateforme).

Source: IGN, RGE ALTI® / BD ALTI®, through https://data.geopf.fr/altimetrie,
Licence Ouverte 2.0 / Etalab.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

from . import cache_dir

IGN_URL = "https://data.geopf.fr/altimetrie/1.0/calcul/alti/rest/elevation.json"
USER_AGENT = "fountains/0.1 (+https://github.com/dsiacci/fountains)"
BATCH = 50


def _key(lat: float, lon: float) -> str:
    return f"{lat:.5f},{lon:.5f}"


def elevations(points: list[tuple[float, float]]) -> list[float | None]:
    """Altitude in metres for each (lat, lon), cached on disk."""
    path = cache_dir("ign") / "elevations.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    todo = [p for p in dict.fromkeys(points) if _key(*p) not in cache]
    for i in range(0, len(todo), BATCH):
        chunk = todo[i : i + BATCH]
        q = urllib.parse.urlencode(
            {
                "lon": "|".join(f"{lon:.6f}" for _, lon in chunk),
                "lat": "|".join(f"{lat:.6f}" for lat, _ in chunk),
                "resource": "ign_rge_alti_wld",
                "zonly": "true",
            }
        )
        req = urllib.request.Request(f"{IGN_URL}?{q}", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as r:
            zs = json.load(r)["elevations"]
        for p, z in zip(chunk, zs):
            # The service answers -99999 where it has no data (at sea).
            cache[_key(*p)] = None if z is None or z <= -1000 else round(float(z), 1)
    path.write_text(json.dumps(cache))
    return [cache[_key(*p)] for p in points]
