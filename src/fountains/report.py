"""What the rider reads before leaving: a table, a GeoJSON, waypoints, a map."""

from __future__ import annotations

import html
import json
from pathlib import Path

from . import DISCLAIMER
from .gpx import write_waypoints
from .score import ScoredFountain

STREAM_STATE = {"1": "flowing", "1a": "flowing", "1f": "flowing (weak)", "2": "no visible flow", "3": "dry"}
TRANSFER_NOTE = (
    "The model learned from small streams, not from fountains: it assumes a fountain fed by a "
    "spring dries like a headwater stream. A fountain on the town mains does not depend on rain, "
    "and OpenStreetMap rarely says which is which."
)


def _fmt_date(iso: str) -> str:
    return f"{iso[8:10]}/{iso[5:7]}/{iso[:4]}" if iso else ""


def reason(f: ScoredFountain) -> str:
    r = f.rain
    parts = []
    if r.get("rain_90") is not None:
        txt = f"rain in the last 90 days {r['rain_90']:.0f} mm"
        if r.get("ratio_90") is not None:
            txt += f" ({r['ratio_90']:.0%} of normal)"
        parts.append(txt)
    if r.get("rain_30") is not None:
        parts.append(f"last 30 days {r['rain_30']:.0f} mm")
    if r.get("forecast_mm"):
        parts.append(f"{r['forecast_mm']:.0f} mm forecast before the ride")
    if f.elevation_m is not None:
        parts.append(f"{f.elevation_m:.0f} m up")
    if f.stream:
        s = f.stream
        parts.append(f"nearest observed stream ({s['stream'] or s['station']}, {s['km']:.0f} km): {STREAM_STATE.get(s['code'], '?')} on {_fmt_date(s['date'])}")
    parts.append(f"last checked in OSM {_fmt_date(f.last_confirmed)}" if f.last_confirmed else "no check date in OSM")
    return "; ".join(parts)


def text_table(scored: list[ScoredFountain], meta: dict) -> str:
    lines = [
        f"{meta['track']} ({meta['length_km']} km), ride on {_fmt_date(meta['day'])}",
        f"!! {DISCLAIMER}",
        "",
    ]
    near = [f for f in scored if f.reachable]
    far = [f for f in scored if not f.reachable]
    if not scored:
        lines.append(f"No drinking-water point in OpenStreetMap within {meta['max_detour_m']:.0f} m of this track.")
    if near:
        lines.append(f"Within {meta['max_detour_m']:.0f} m of the track along roads and paths, in riding order:")
        for f in near:
            lines.append(f"  km {f.km:5.1f}  {f.band.upper():<9}  {f.label()} ({f.kind}), {f.detour_m} m off the route  [osm {f.osm_id}]")
            lines.append(f"            {reason(f)}")
            for n in f.notes:
                lines.append(f"            note: {n}")
    if far:
        lines.append("")
        lines.append(f"Close as the crow flies, but more than {meta['max_detour_m']:.0f} m away by road or path:")
        for f in far:
            how = f"{f.detour_m} m by road or path" if f.detour_m is not None else "no mapped path nearby"
            lines.append(f"  km {f.km:5.1f}  {f.band.upper():<9}  {f.label()} ({f.kind}), {f.off_track_m} m straight, {how}  [osm {f.osm_id}]")
            lines.append(f"            {reason(f)}")
    if meta.get("excluded_private"):
        lines.append(f"\n{meta['excluded_private']} private point(s) (access=private or no) left out.")
    cal = meta["calibration"]
    pb = cal["per_band"]

    def share(b: str) -> str:
        x = pb.get(b) or {}
        return f"{x['rate']:.0%} of {x['n']}" if x.get("n") else "no case"

    lines += [
        "",
        f"Rain: Météo-France gauges up to {_fmt_date(meta['gauges_last_day'])}"
        + (", then the Météo-France forecast." if meta["forecast_used"] else "."),
        f"How to read the bands: when {cal['n_stations']} Corsican streams were each hidden from the model in turn, "
        f"'likely' streams were flowing in {share('likely')} cases, 'uncertain' in {share('uncertain')}, 'unlikely' in {share('unlikely')}.",
        TRANSFER_NOTE,
        f"!! {DISCLAIMER}",
    ]
    return "\n".join(lines)


def to_geojson(scored: list[ScoredFountain], meta: dict) -> dict:
    feats = []
    for f in scored:
        feats.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [f.lon, f.lat]},
                "properties": {
                    "osm_id": f.osm_id, "name": f.name, "kind": f.kind, "band": f.band,
                    "km": f.km, "detour_m": f.detour_m, "within_detour_limit": f.reachable,
                    "elevation_m": f.elevation_m, "reason": reason(f), "notes": f.notes,
                    "last_checked_in_osm": f.last_confirmed,
                    "model_probability_streams": round(f.p, 3),
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "disclaimer": DISCLAIMER,
        "note": TRANSFER_NOTE,
        "meta": meta,
        "attribution": "Fountains © OpenStreetMap contributors (ODbL). Rain: Météo-France (Licence Ouverte 2.0); forecast via Open-Meteo (CC BY 4.0). Streams: OFB, ONDE via Hub'Eau (Licence Ouverte 2.0). Built with PriorLabs-TabPFN.",
        "features": feats,
    }


def waypoint_name(f: ScoredFountain, max_detour_m: float) -> str:
    name = f"{f.label()} - {f.band}"
    if not f.reachable:
        name += f" - {f.detour_m} m off" if f.detour_m is not None else " - no path"
    return name[:40]


def write_outputs(scored: list[ScoredFountain], meta: dict, out_dir: Path, stem: str, with_html: bool = False) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    p = out_dir / f"{stem}.txt"
    p.write_text(text_table(scored, meta) + "\n", encoding="utf-8")
    paths.append(p)
    p = out_dir / f"{stem}.geojson"
    p.write_text(json.dumps(to_geojson(scored, meta), ensure_ascii=False, indent=1), encoding="utf-8")
    paths.append(p)
    p = out_dir / f"{stem}-waypoints.gpx"
    write_waypoints(
        [{"lat": f.lat, "lon": f.lon, "name": waypoint_name(f, meta["max_detour_m"]), "desc": reason(f)} for f in scored],
        p,
        metadata_desc=DISCLAIMER,
    )
    paths.append(p)
    if with_html:
        p = out_dir / f"{stem}.html"
        p.write_text(html_map(scored, meta), encoding="utf-8")
        paths.append(p)
    return paths


def html_map(scored: list[ScoredFountain], meta: dict) -> str:
    data = json.dumps(to_geojson(scored, meta), ensure_ascii=False)
    title = html.escape(f"Fountains on {meta['track']}, {_fmt_date(meta['day'])}")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
:root {{ --bg:#f4f6f5; --fg:#1b2320; --likely:#2a7a57; --uncertain:#a8761c; --unlikely:#a24c2a; }}
body {{ margin:0; font:15px/1.4 system-ui,sans-serif; background:var(--bg); color:var(--fg); }}
header {{ padding:10px 16px; }} h1 {{ font-size:18px; margin:0 0 4px; }}
.warn {{ font-weight:600; }} #map {{ height:70vh; }} footer {{ padding:10px 16px; font-size:13px; }}
</style></head><body>
<header><h1>{title}</h1><p class="warn">{html.escape(DISCLAIMER)}</p></header>
<div id="map"></div>
<footer><p>{html.escape(TRANSFER_NOTE)}</p><p>Map and fountains © OpenStreetMap contributors. Rain: Météo-France. Streams: OFB (ONDE). Built with PriorLabs-TabPFN.</p></footer>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const data = {data};
const color = {{likely: getComputedStyle(document.documentElement).getPropertyValue('--likely'), uncertain: getComputedStyle(document.documentElement).getPropertyValue('--uncertain'), unlikely: getComputedStyle(document.documentElement).getPropertyValue('--unlikely')}};
const map = L.map('map');
L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'}}).addTo(map);
const layer = L.geoJSON(data, {{
  pointToLayer: (f, ll) => L.circleMarker(ll, {{radius: 8, color: color[f.properties.band], fillOpacity: 0.85}}),
  onEachFeature: (f, l) => {{ const p = f.properties; const e = document.createElement('div');
    const h = document.createElement('strong'); h.textContent = (p.name || p.kind) + ' (' + p.band + ')'; e.append(h);
    const r = document.createElement('p'); r.textContent = 'km ' + p.km + ', ' + (p.detour_m ?? '?') + ' m off the route. ' + p.reason; e.append(r);
    l.bindPopup(e); }}
}}).addTo(map);
if (data.features.length) map.fitBounds(layer.getBounds().pad(0.2)); else map.setView([42.0, 9.0], 8);
</script></body></html>
"""
