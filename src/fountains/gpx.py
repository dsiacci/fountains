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


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def read_gpx(path: str | Path) -> Track:
    """Return the track points (trkpt, or rtept when there is no track)."""
    root = ET.parse(path).getroot()
    trk, rte, name = [], [], None
    for el in root.iter():
        tag = _local(el.tag)
        if tag == "trkpt":
            trk.append((float(el.get("lat")), float(el.get("lon"))))
        elif tag == "rtept":
            rte.append((float(el.get("lat")), float(el.get("lon"))))
        elif tag == "name" and name is None and el.text:
            name = el.text.strip()
    points = trk or rte
    if len(points) < 2:
        raise ValueError(f"{path}: no track or route with at least two points")
    return Track(name=name or Path(path).stem, points=points)


def trim_track(track: Track, start_m: float, end_m: float) -> Track:
    """Drop the first `start_m` and last `end_m` metres (to keep a home address private)."""
    line = Polyline(track.points)
    keep = [p for p, s in zip(track.points, line.cum) if start_m <= s <= line.length_m - end_m]
    if len(keep) < 2:
        raise ValueError("nothing left after trimming")
    return Track(name=track.name, points=keep)


def write_track(track: Track, path: str | Path) -> None:
    """Write a bare track: no time, no author, no device metadata."""
    pts = "\n".join(f'      <trkpt lat="{lat:.6f}" lon="{lon:.6f}"/>' for lat, lon in track.points)
    Path(path).write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<gpx version="1.1" creator="fountains" xmlns="http://www.topografix.com/GPX/1/1">\n'
        f"  <trk>\n    <name>{escape(track.name)}</name>\n    <trkseg>\n{pts}\n    </trkseg>\n  </trk>\n</gpx>\n",
        encoding="utf-8",
    )


def write_waypoints(waypoints: list[dict], path: str | Path, metadata_desc: str = "") -> None:
    """Write GPX waypoints (`lat`, `lon`, `name`, `desc`) for a bike computer.

    Names stay short because most head units truncate them; the full reason goes
    in `desc`.
    """
    items = []
    for w in waypoints:
        items.append(
            f'  <wpt lat="{w["lat"]:.6f}" lon="{w["lon"]:.6f}">\n'
            f"    <name>{escape(w['name'])}</name>\n"
            f"    <desc>{escape(w.get('desc', ''))}</desc>\n"
            "    <sym>Drinking Water</sym>\n"
            "  </wpt>"
        )
    meta = f"  <metadata><desc>{escape(metadata_desc)}</desc></metadata>\n" if metadata_desc else ""
    Path(path).write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<gpx version="1.1" creator="fountains" xmlns="http://www.topografix.com/GPX/1/1">\n'
        + meta
        + "\n".join(items)
        + "\n</gpx>\n",
        encoding="utf-8",
    )
