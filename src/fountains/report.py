"""What the rider reads before leaving: a table, a GeoJSON, waypoints, a map."""

from __future__ import annotations

import html
import json
from pathlib import Path

from . import DISCLAIMER
from .gpx import write_waypoints
from .score import Gap, ScoredFountain

STREAM_STATE = {"1": "flowing", "1a": "flowing", "1f": "flowing (weak)", "2": "no visible flow", "3": "dry"}
TRANSFER_NOTE = (
    "The model learned from small streams, not from fountains: it assumes a fountain fed by a "
    "spring dries like a headwater stream. A fountain on the town mains does not depend on rain, "
    "and the maps rarely say which is which."
)
ATTRIBUTION = (
    "Water points © OpenStreetMap contributors (ODbL) and IGN BD TOPO® (Licence Ouverte 2.0). "
    "Rain: Météo-France (Licence Ouverte 2.0); forecast via Open-Meteo (CC BY 4.0). "
    "Streams: OFB, ONDE via Hub'Eau (Licence Ouverte 2.0). Built with PriorLabs-TabPFN."
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
    parts.append(f"last checked {_fmt_date(f.last_confirmed)}" if f.last_confirmed else "no check date on the maps")
    return "; ".join(parts)


def _stop_line(s) -> str:
    when = ""
    if s.eta:
        state = {True: "open", False: "closed", None: "hours unknown"}[s.open_then]
        when = f" around {s.eta}: {state}"
    hours = f" ({s.opening_hours})" if s.opening_hours else ""
    return f"  km {s.km:5.1f}  {s.kind}: {s.name or 'no name'}, {s.detour_m} m off the route{when}{hours}  [osm {s.ref}]"


def text_table(scored: list[ScoredFountain], meta: dict, gaps: list[Gap] | None = None) -> str:
    lines = [
        f"{meta['track']} ({meta['length_km']} km), ride on {_fmt_date(meta['day'])}" + (f", start {meta['start']}" if meta.get("start") else ""),
        f"!! {DISCLAIMER}",
        "",
    ]
    near = [f for f in scored if f.reachable]
    far = [f for f in scored if not f.reachable]
    if not scored:
        lines.append(f"No water point on the maps within {meta['max_detour_m']:.0f} m of this track.")
    if near:
        lines.append(f"Within {meta['max_detour_m']:.0f} m of the track along roads and paths, in riding order:")
        for f in near:
            at = f" ~{f.eta}" if f.eta else ""
            lines.append(f"  km {f.km:5.1f}{at}  {f.band.upper():<9}  {f.label()} ({f.kind}), {f.detour_m} m off the route  [{f.source}: {f.ref}]")
            lines.append(f"            {reason(f)}")
            for n in f.notes:
                lines.append(f"            note: {n}")
    if far:
        lines.append("")
        lines.append(f"Close as the crow flies, but more than {meta['max_detour_m']:.0f} m away by road or path:")
        for f in far:
            how = f"{f.detour_m} m by road or path" if f.detour_m is not None else "no mapped path nearby"
            lines.append(f"  km {f.km:5.1f}  {f.band.upper():<9}  {f.label()} ({f.kind}), {f.off_track_m} m straight, {how}  [{f.source}: {f.ref}]")
            lines.append(f"            {reason(f)}")
    if meta.get("excluded_private"):
        lines.append(f"\n{meta['excluded_private']} private point(s) (access=private or no) left out.")
    if gaps:
        lines.append("")
        lines.append(f"Stretches of {meta['gap_km']:.0f} km or more without a likely fountain, and where else to fill a bottle:")
        for g in gaps:
            lines.append(f"  km {g.from_km:.1f} to {g.to_km:.1f} ({g.length_km} km)")
            if g.stops:
                lines.extend(_stop_line(s) for s in g.stops)
            else:
                lines.append("    no café, shop or fuel station on the maps within reach")
        if not meta.get("start"):
            lines.append("  (give --start HH:MM to check opening hours at the time you pass)")
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


def to_geojson(scored: list[ScoredFountain], meta: dict, gaps: list[Gap] | None = None) -> dict:
    feats = []
    for f in scored:
        feats.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [f.lon, f.lat]},
                "properties": {
                    "ref": f.ref, "source": f.source, "name": f.name, "kind": f.kind, "band": f.band,
                    "km": f.km, "eta": f.eta, "detour_m": f.detour_m, "within_detour_limit": f.reachable,
                    "elevation_m": f.elevation_m, "reason": reason(f), "notes": f.notes,
                    "last_checked": f.last_confirmed,
                    "model_probability_streams": round(f.p, 3),
                },
            }
        )
    for g in gaps or []:
        for s in g.stops:
            feats.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [s.lon, s.lat]},
                    "properties": {"ref": s.ref, "source": "OSM", "name": s.name, "kind": s.kind, "stop": True, "km": s.km,
                                   "eta": s.eta, "detour_m": s.detour_m, "opening_hours": s.opening_hours,
                                   "open_then": s.open_then, "gap_km": [g.from_km, g.to_km]},
                }
            )
    return {"type": "FeatureCollection", "disclaimer": DISCLAIMER, "note": TRANSFER_NOTE, "meta": meta, "attribution": ATTRIBUTION, "features": feats}


def waypoint_name(f: ScoredFountain) -> str:
    name = f"{f.label()} - {f.band}"
    if not f.reachable:
        name += f" - {f.detour_m} m off" if f.detour_m is not None else " - no path"
    return name[:40]


def write_outputs(scored: list[ScoredFountain], meta: dict, out_dir: Path, stem: str, with_html: bool = False, gaps: list[Gap] | None = None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    p = out_dir / f"{stem}.txt"
    p.write_text(text_table(scored, meta, gaps) + "\n", encoding="utf-8")
    paths.append(p)
    p = out_dir / f"{stem}.geojson"
    p.write_text(json.dumps(to_geojson(scored, meta, gaps), ensure_ascii=False, indent=1), encoding="utf-8")
    paths.append(p)
    p = out_dir / f"{stem}-waypoints.gpx"
    wpts = [{"lat": f.lat, "lon": f.lon, "name": waypoint_name(f), "desc": reason(f)} for f in scored]
    for g in gaps or []:
        for s in g.stops:
            state = {True: "open", False: "closed", None: "hours?"}[s.open_then] if s.eta else "hours?"
            wpts.append({"lat": s.lat, "lon": s.lon, "name": f"{s.kind} - {state}"[:40], "desc": f"{s.name} {s.opening_hours}".strip()})
    write_waypoints(wpts, p, metadata_desc=DISCLAIMER)
    paths.append(p)
    if with_html:
        p = out_dir / f"{stem}.html"
        p.write_text(html_map(scored, meta, gaps), encoding="utf-8")
        paths.append(p)
    return paths


def html_map(scored: list[ScoredFountain], meta: dict, gaps: list[Gap] | None = None) -> str:
    data = json.dumps(to_geojson(scored, meta, gaps), ensure_ascii=False)
    title = html.escape(f"Fountains on {meta['track']}, {_fmt_date(meta['day'])}")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
:root {{ --bg:#f4f6f5; --fg:#1b2320; --likely:#2a7a57; --uncertain:#a8761c; --unlikely:#a24c2a; --stop:#4b5b8c; }}
body {{ margin:0; font:15px/1.4 system-ui,sans-serif; background:var(--bg); color:var(--fg); }}
header {{ padding:10px 16px; }} h1 {{ font-size:18px; margin:0 0 4px; }}
.warn {{ font-weight:600; }} #map {{ height:70vh; }} footer {{ padding:10px 16px; font-size:13px; }}
</style></head><body>
<header><h1>{title}</h1><p class="warn">{html.escape(DISCLAIMER)}</p></header>
<div id="map"></div>
<footer><p>{html.escape(TRANSFER_NOTE)}</p><p>{html.escape(ATTRIBUTION)}</p></footer>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const data = {data};
const css = getComputedStyle(document.documentElement);
const colour = (p) => css.getPropertyValue(p.stop ? '--stop' : '--' + p.band);
const map = L.map('map');
L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'}}).addTo(map);
const layer = L.geoJSON(data, {{
  pointToLayer: (f, ll) => L.circleMarker(ll, {{radius: f.properties.stop ? 6 : 8, color: colour(f.properties), fillOpacity: 0.85}}),
  onEachFeature: (f, l) => {{ const p = f.properties; const e = document.createElement('div');
    const h = document.createElement('strong'); h.textContent = (p.name || p.kind) + (p.stop ? '' : ' (' + p.band + ')'); e.append(h);
    const r = document.createElement('p'); r.textContent = 'km ' + p.km + (p.eta ? ' around ' + p.eta : '') + ', ' + (p.detour_m ?? '?') + ' m off the route. ' + (p.reason || p.opening_hours || ''); e.append(r);
    l.bindPopup(e); }}
}}).addTo(map);
if (data.features.length) map.fitBounds(layer.getBounds().pad(0.2)); else map.setView([42.0, 9.0], 8);
</script></body></html>
"""
