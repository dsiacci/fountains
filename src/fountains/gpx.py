"""Read a GPX track exported by the rider, write waypoints for a bike computer.

Only files are read. The tool never talks to Strava or any other account: the
rider exports a GPX from whatever app they use and passes the file.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

from .geo import Polyline


@dataclass
class Track:
    name: str
    points: list[tuple[float, float]]
    ele: list[float | None] | None = None  # metres, when the file has them


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def read_gpx(path: str | Path) -> Track:
    """Return the track points (trkpt, or rtept when there is no track)."""
    root = ET.parse(path).getroot()
    trk, rte, name = [], [], None
    for el in root.iter():
        tag = _local(el.tag)
        if tag in ("trkpt", "rtept"):
            ele = next((c.text for c in el if _local(c.tag) == "ele" and c.text), None)
            (trk if tag == "trkpt" else rte).append((float(el.get("lat")), float(el.get("lon")), None if ele is None else float(ele)))
        elif tag == "name" and name is None and el.text:
            name = el.text.strip()
    points = trk or rte
    if len(points) < 2:
        raise ValueError(f"{path}: no track or route with at least two points")
    eles = [p[2] for p in points]
    return Track(
        name=name or Path(path).stem,
        points=[(p[0], p[1]) for p in points],
        ele=eles if any(e is not None for e in eles) else None,
    )


def trim_track(track: Track, start_m: float, end_m: float) -> Track:
    """Drop the first `start_m` and last `end_m` metres (to keep a home address private)."""
    line = Polyline(track.points)
    idx = [i for i, s in enumerate(line.cum) if start_m <= s <= line.length_m - end_m]
    if len(idx) < 2:
        raise ValueError("nothing left after trimming")
    return Track(
        name=track.name,
        points=[track.points[i] for i in idx],
        ele=[track.ele[i] for i in idx] if track.ele else None,
    )


def _trk(track: Track) -> str:
    eles = track.ele or [None] * len(track.points)
    pts = "\n".join(
        f'      <trkpt lat="{lat:.6f}" lon="{lon:.6f}">' + (f"<ele>{e:.1f}</ele>" if e is not None else "") + "</trkpt>"
        for (lat, lon), e in zip(track.points, eles)
    )
    return f"  <trk>\n    <name>{escape(track.name)}</name>\n    <trkseg>\n{pts}\n    </trkseg>\n  </trk>\n"


def _wpt(w: dict) -> str:
    kind = f"    <type>{escape(w['type'])}</type>\n" if w.get("type") else ""
    return (
        f'  <wpt lat="{w["lat"]:.6f}" lon="{w["lon"]:.6f}">\n'
        f"    <name>{escape(w['name'])}</name>\n"
        f"    <desc>{escape(w.get('desc', ''))}</desc>\n"
        f"    <sym>{escape(w.get('sym', 'Drinking Water'))}</sym>\n"
        + kind
        + "  </wpt>\n"
    )


def _gpx(body: str, metadata_desc: str = "") -> str:
    meta = f"  <metadata><desc>{escape(metadata_desc)}</desc></metadata>\n" if metadata_desc else ""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<gpx version="1.1" creator="fountains" xmlns="http://www.topografix.com/GPX/1/1">\n'
        + meta
        + body
        + "</gpx>\n"
    )


def write_track(track: Track, path: str | Path) -> None:
    """Write a bare track: no time, no author, no device metadata (elevation kept)."""
    Path(path).write_text(_gpx(_trk(track)), encoding="utf-8")


def write_waypoints(waypoints: list[dict], path: str | Path, metadata_desc: str = "") -> None:
    """Write GPX waypoints (`lat`, `lon`, `name`, `desc`, optional `sym` and `type`) for a bike computer.

    Names stay short because most head units truncate them; the full reason goes
    in `desc`.
    """
    Path(path).write_text(_gpx("".join(_wpt(w) for w in waypoints), metadata_desc), encoding="utf-8")


def course_gpx(track: Track, waypoints: list[dict], metadata_desc: str = "") -> str:
    """One file for a bike computer: the waypoints, then the track they belong to."""
    return _gpx("".join(_wpt(w) for w in waypoints) + _trk(track), metadata_desc)
