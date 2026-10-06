"""Where else to fill a bottle on long stretches without a likely fountain.

No model here: cafés, bars, bakeries, small shops and fuel stations from
OpenStreetMap, the time you should pass them (from your start time and a
simple pace), and whether their `opening_hours` say they are open then.
Many places have no opening hours in OpenStreetMap; they are shown as unknown.
"""

from __future__ import annotations

import datetime as dt
import re

from .osm import CORSICA_AREA

STOP_QUERY = (
    "[out:json][timeout:180];\n"
    + CORSICA_AREA
    + """
(
 nwr["amenity"~"^(cafe|bar|pub|restaurant|fast_food|fuel|ice_cream)$"](area.c);
 nwr["shop"~"^(bakery|pastry|supermarket|convenience|general|greengrocer|deli|farm)$"](area.c);
);
out center tags;
"""
)

KIND_LABEL = {
    "cafe": "café", "bar": "bar", "pub": "bar", "restaurant": "restaurant", "fast_food": "snack bar",
    "fuel": "fuel station", "ice_cream": "ice cream", "bakery": "bakery", "pastry": "bakery",
    "supermarket": "supermarket", "convenience": "small shop", "general": "small shop",
    "greengrocer": "greengrocer", "deli": "delicatessen", "farm": "farm shop",
}


def stops_from_overpass(data: dict) -> dict:
    feats = []
    for e in data["elements"]:
        tags = e.get("tags", {})
        kind = tags.get("amenity") if tags.get("amenity") in KIND_LABEL else tags.get("shop")
        if kind not in KIND_LABEL or tags.get("disused") == "yes":
            continue
        lat = e.get("lat", e.get("center", {}).get("lat"))
        lon = e.get("lon", e.get("center", {}).get("lon"))
        if lat is None or lon is None:
            continue
        props = {"osm_id": f"{e['type']}/{e['id']}", "kind": KIND_LABEL[kind], "name": tags.get("name", "")}
        if tags.get("opening_hours"):
            props["opening_hours"] = tags["opening_hours"]
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(lon, 7), round(lat, 7)]}, "properties": props})
    feats.sort(key=lambda f: f["properties"]["osm_id"])
    return {
        "type": "FeatureCollection",
        "osm_base": data.get("osm3s", {}).get("timestamp_osm_base", ""),
        "license": "ODbL 1.0, © OpenStreetMap contributors (https://www.openstreetmap.org/copyright)",
        "features": feats,
    }


# --- opening_hours: a deliberately small subset of the OSM syntax -----------

DAYS = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_TIME = r"(\d{1,2}):(\d{2})"
_RANGE = re.compile(rf"^{_TIME}-{_TIME}$")


class Unknown(Exception):
    """The value uses syntax this parser does not handle: say 'unknown', never guess."""


def _expand(spec: str, names: list[str]) -> set[int]:
    out: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            if a not in names or b not in names:
                raise Unknown(part)
            i, j = names.index(a), names.index(b)
            out.update(range(i, j + 1) if i <= j else list(range(i, len(names))) + list(range(0, j + 1)))
        elif part in names:
            out.add(names.index(part))
        else:
            raise Unknown(part)
    return out


def _minutes(h: str, m: str) -> int:
    return int(h) * 60 + int(m)


def open_intervals(value: str, day: dt.date) -> list[tuple[int, int]]:
    """Opening intervals (minutes since midnight) on `day`. Raises Unknown."""
    value = value.strip()
    if value == "24/7":
        return [(0, 24 * 60)]
    if '"' in value or "||" in value:
        raise Unknown(value)
    intervals: list[tuple[int, int]] | None = None
    for rule in [r.strip() for r in value.split(";") if r.strip()]:
        tokens = rule.split()
        months = set(range(12))
        days = set(range(7))
        if tokens and re.fullmatch(r"[A-Z][a-z]{2}(-[A-Z][a-z]{2})?(,[A-Z][a-z]{2}(-[A-Z][a-z]{2})?)*:?", tokens[0]) and tokens[0][:3] in MONTHS:
            months = _expand(tokens.pop(0).rstrip(":"), MONTHS)
        if tokens and re.fullmatch(r"(Mo|Tu|We|Th|Fr|Sa|Su|PH|SH)([-,](Mo|Tu|We|Th|Fr|Sa|Su|PH|SH))*", tokens[0]):
            spec = tokens.pop(0)
            if "PH" in spec or "SH" in spec:
                parts = [p for p in spec.split(",") if p not in ("PH", "SH")]
                if not parts:
                    continue  # a holidays-only rule: holidays are not modelled, skip it
                spec = ",".join(parts)
            days = _expand(spec, DAYS)
        if day.month - 1 not in months or day.weekday() not in days:
            continue
        rest = " ".join(tokens)
        if rest in ("off", "closed"):
            intervals = []
            continue
        if rest == "" or rest == "open":
            raise Unknown(rule)
        found = []
        for rng in rest.split(","):
            m = _RANGE.match(rng.strip())
            if not m:
                raise Unknown(rng)
            a, b = _minutes(m[1], m[2]), _minutes(m[3], m[4])
            if b <= a:
                b += 24 * 60  # past midnight
            found.append((a, b))
        intervals = found  # a later rule for the same day overrides an earlier one
    if intervals is None:
        return []  # no rule covers that day: closed
    return intervals


def is_open(value: str | None, when: dt.datetime) -> bool | None:
    """True / False when the hours say so, None when they are missing or unclear."""
    if not value:
        return None
    try:
        minute = when.hour * 60 + when.minute
        today = open_intervals(value, when.date())
        yesterday = open_intervals(value, when.date() - dt.timedelta(days=1))
    except Unknown:
        return None
    if any(a <= minute < b for a, b in today):
        return True
    return any(a <= minute + 24 * 60 < b for a, b in yesterday)


# --- when will I be there? ---------------------------------------------------

def passing_minutes(cum_m: list[float], ele: list[float | None] | None, flat_kmh: float, climb_mh: float) -> list[float]:
    """Minutes from the start to each track point: distance at a flat speed, plus climbing time.

    Climbing counts only rises of at least 2 m between kept points, so GPS
    noise does not add phantom metres.
    """
    out = [0.0]
    climbed = 0.0
    ref = ele[0] if ele else None
    for i in range(1, len(cum_m)):
        if ele and ele[i] is not None and ref is not None:
            if ele[i] - ref >= 2.0:
                climbed += ele[i] - ref
                ref = ele[i]
            elif ele[i] < ref:
                ref = ele[i]
        out.append(cum_m[i] / 1000 / flat_kmh * 60 + climbed / climb_mh * 60)
    return out
