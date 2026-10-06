"""See where a water point is before riding there: street photo, plan, aerial.

- Street level: Panoramax, the open street-imagery commons (https://panoramax.fr).
  In Corsica most pictures near roads are IGN's 360° captures of 2025, under the
  Licence Ouverte 2.0; other contributors publish under their own licence,
  which is kept with each picture. The tool asks the federated API for pictures
  that look at the point, keeps a 360° one when there is one (it can be turned
  toward the point), else the closest.
- Plan and aerial: Plan IGN and BD ORTHO® through the Géoplateforme WMS, open
  under the Licence Ouverte 2.0 since 2021.

Only the positions of the water points are sent, never the track.
"""

from __future__ import annotations

import json
import math
import urllib.request

from . import cache_dir
from .geo import haversine_m

PANORAMAX_SEARCH = "https://api.panoramax.xyz/api/search"
PANORAMAX_VIEWER = "https://api.panoramax.xyz/"
IGN_WMS = "https://data.geopf.fr/wms-r/wms"
USER_AGENT = "fountains/0.1 (+https://github.com/dsiacci/fountains)"
EARTH_MERCATOR_R = 6378137.0


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial compass bearing from point 1 to point 2 (0 = north, 90 = east)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def relative_deg(target: float, azimuth: float) -> float:
    """Angle from the picture's centre to the target, in (-180, 180]."""
    r = ((target - azimuth + 540) % 360) - 180
    return 180.0 if r == -180 else float(r) + 0.0  # + 0.0 turns -0.0 into 0.0


def pick(features: list[dict], lat: float, lon: float) -> dict | None:
    """The picture to show: 360° first, then the closest."""
    best, best_key = None, None
    for f in features:
        plon, plat = f["geometry"]["coordinates"][:2]
        p = f["properties"]
        fov = (p.get("pers:interior_orientation") or {}).get("field_of_view")
        dist = haversine_m(plat, plon, lat, lon)
        key = (fov == 360, -dist)
        if best_key is None or key > best_key:
            best_key = key
            b = bearing_deg(plat, plon, lat, lon)
            az = p.get("view:azimuth")
            best = {
                "id": f["id"],
                "date": (p.get("datetime") or "")[:10],
                "distance_m": round(dist),
                "fov": fov,
                "bearing": round(b),
                "azimuth": az,
                "relative": None if az is None else round(relative_deg(b, az)),
                "license": p.get("license") or "",
                "instance": next((l["href"] for l in f.get("links", []) if l.get("rel") == "via"), ""),
                "authors": [x.get("name") for x in p.get("providers", []) if x.get("name")],
                "sd": f.get("assets", {}).get("sd", {}).get("href", ""),
                "thumb": f.get("assets", {}).get("thumb", {}).get("href", ""),
                "viewer": f"{PANORAMAX_VIEWER}?focus=pic&map=19/{lat:.6f}/{lon:.6f}&pic={f['id']}&xyz={round(b)}/0/30",
            }
    return best


def street_view(lat: float, lon: float, max_m: int = 60) -> dict | None:
    """The best Panoramax picture looking at the point, or None (cached on disk)."""
    path = cache_dir("panoramax") / f"{lat:.5f}_{lon:.5f}_{max_m}.json"
    if path.exists():
        data = json.loads(path.read_text())
    else:
        url = f"{PANORAMAX_SEARCH}?place_position={lon:.6f},{lat:.6f}&place_distance=0-{max_m}&place_fov_tolerance=60&limit=20"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.load(r)
        path.write_text(json.dumps(data))
    return pick(data.get("features", []), lat, lon)


def _wms(layer: str, fmt: str, lat: float, lon: float, width_m: float, px: tuple[int, int]) -> str:
    """A WMS GetMap URL for a view `width_m` wide, centred on the point (Web Mercator)."""
    x = math.radians(lon) * EARTH_MERCATOR_R
    y = math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * EARTH_MERCATOR_R
    half = width_m / 2 / math.cos(math.radians(lat))  # Mercator metres are stretched by 1/cos(lat)
    hh = half * px[1] / px[0]
    return (
        f"{IGN_WMS}?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS={layer}&STYLES=&CRS=EPSG:3857"
        f"&BBOX={x - half:.1f},{y - hh:.1f},{x + half:.1f},{y + hh:.1f}&WIDTH={px[0]}&HEIGHT={px[1]}&FORMAT={fmt}"
    )


def plan_url(lat: float, lon: float, width_m: float = 400, px: tuple[int, int] = (600, 340)) -> str:
    return _wms("GEOGRAPHICALGRIDSYSTEMS.PLANIGNV2", "image/png", lat, lon, width_m, px)


def aerial_url(lat: float, lon: float, width_m: float = 160, px: tuple[int, int] = (600, 340)) -> str:
    return _wms("ORTHOIMAGERY.ORTHOPHOTOS", "image/jpeg", lat, lon, width_m, px)
