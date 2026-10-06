"""Small planar geometry helpers.

Distances in this project never exceed a few hundred kilometres, so points are
projected onto a local equirectangular plane around a reference latitude. The
error is well under 1 % at Corsican latitudes, far below the precision of the
data (OpenStreetMap positions, GPS tracks).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

EARTH_RADIUS_M = 6_371_008.8


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


@dataclass(frozen=True)
class Projection:
    """Local equirectangular projection (metres) around `lat0`."""

    lat0: float

    def xy(self, lat: float, lon: float) -> tuple[float, float]:
        k = math.cos(math.radians(self.lat0))
        return (math.radians(lon) * EARTH_RADIUS_M * k, math.radians(lat) * EARTH_RADIUS_M)

    def latlon(self, x: float, y: float) -> tuple[float, float]:
        k = math.cos(math.radians(self.lat0))
        return (math.degrees(y / EARTH_RADIUS_M), math.degrees(x / (EARTH_RADIUS_M * k)))


def project_on_segment(p, a, b) -> tuple[float, float, tuple[float, float]]:
    """Project point p on segment ab (planar coordinates).

    Returns (distance from p to the segment, fraction u along ab, projected point).
    """
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    u = 0.0 if l2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / l2))
    q = (ax + u * dx, ay + u * dy)
    return math.hypot(p[0] - q[0], p[1] - q[1]), u, q


class SegmentGrid:
    """Buckets segments into square cells so nearby segments are found fast."""

    def __init__(self, segments: list[tuple[tuple[float, float], tuple[float, float]]], cell_m: float = 250.0):
        self.cell = cell_m
        self.segments = segments
        self.cells: dict[tuple[int, int], list[int]] = {}
        for i, (a, b) in enumerate(segments):
            x0, x1 = sorted((a[0], b[0]))
            y0, y1 = sorted((a[1], b[1]))
            for cx in range(int(math.floor(x0 / cell_m)), int(math.floor(x1 / cell_m)) + 1):
                for cy in range(int(math.floor(y0 / cell_m)), int(math.floor(y1 / cell_m)) + 1):
                    self.cells.setdefault((cx, cy), []).append(i)

    def near(self, p: tuple[float, float], radius_m: float) -> set[int]:
        """Indices of the segments that may lie within `radius_m` of p."""
        r = int(math.ceil(radius_m / self.cell))
        cx, cy = int(math.floor(p[0] / self.cell)), int(math.floor(p[1] / self.cell))
        out: set[int] = set()
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                out.update(self.cells.get((cx + dx, cy + dy), ()))
        return out


class Polyline:
    """A track as a planar polyline with cumulative distance along it."""

    def __init__(self, points: list[tuple[float, float]], proj: Projection | None = None):
        if len(points) < 2:
            raise ValueError("a track needs at least two points")
        self.points = points
        self.proj = proj or Projection(sum(p[0] for p in points) / len(points))
        self.xy = [self.proj.xy(lat, lon) for lat, lon in points]
        self.cum = [0.0]
        for i in range(1, len(self.xy)):
            self.cum.append(self.cum[-1] + math.dist(self.xy[i - 1], self.xy[i]))
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        self.bbox = (min(lats), min(lons), max(lats), max(lons))
        self._grid: SegmentGrid | None = None

    @property
    def length_m(self) -> float:
        return self.cum[-1]

    def nearest(self, lat: float, lon: float, within_m: float | None = None) -> tuple[float, float]:
        """(distance to the track in metres, distance along the track in metres).

        With `within_m`, only segments that can be that close are examined, and
        the distance is `inf` when none is.
        """
        return self.nearest_xy(self.proj.xy(lat, lon), within_m)

    def nearest_xy(self, p: tuple[float, float], within_m: float | None = None) -> tuple[float, float]:
        """`nearest` for a point already in this polyline's projection."""
        if within_m is None:
            candidates = range(1, len(self.xy))
        else:
            if self._grid is None:
                self._grid = SegmentGrid([(self.xy[i - 1], self.xy[i]) for i in range(1, len(self.xy))])
            candidates = sorted(i + 1 for i in self._grid.near(p, within_m))
        best_d, best_s = math.inf, 0.0
        for i in candidates:
            d, u, _ = project_on_segment(p, self.xy[i - 1], self.xy[i])
            if d < best_d:
                best_d = d
                best_s = self.cum[i - 1] + u * (self.cum[i] - self.cum[i - 1])
        if within_m is not None and best_d > within_m:
            return math.inf, 0.0
        return best_d, best_s

    def resample(self, step_m: float) -> list[tuple[float, float]]:
        """Points every `step_m` metres along the track (first and last included)."""
        out = [self.points[0]]
        target = step_m
        for i in range(1, len(self.xy)):
            while self.cum[i] >= target:
                seg = self.cum[i] - self.cum[i - 1]
                u = 0.0 if seg == 0 else (target - self.cum[i - 1]) / seg
                x = self.xy[i - 1][0] + u * (self.xy[i][0] - self.xy[i - 1][0])
                y = self.xy[i - 1][1] + u * (self.xy[i][1] - self.xy[i - 1][1])
                out.append(self.proj.latlon(x, y))
                target += step_m
        if out[-1] != self.points[-1]:
            out.append(self.points[-1])
        return out
