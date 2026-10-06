"""How far off the track is each fountain, along roads and paths?

The track is matched to the OpenStreetMap ways it follows; each fountain is
attached to the nearest way; the detour is the shortest distance along the
network from the track to the fountain (one way). A fountain 40 m from the
road as the crow flies can be 400 m away if the only access is a loop through
the village.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

from .geo import Polyline, Projection, SegmentGrid, project_on_segment

# Ways a rider can use to reach a fountain, on the bike or pushing it for a
# few metres. Ways that are not built or not public are left out.
EXCLUDED_HIGHWAYS = {"proposed", "construction", "abandoned", "disused", "razed", "platform", "bus_stop", "elevator", "corridor", "via_ferrata", "raceway"}
PRIVATE_ACCESS = {"private", "no"}

# A graph node closer than this to the track is "on the track".
ON_TRACK_M = 25.0
# A fountain further than this from any way is not attached to the network.
MAX_SNAP_M = 80.0


@dataclass
class RoadGraph:
    proj: Projection
    xy: dict[int, tuple[float, float]] = field(default_factory=dict)
    adj: dict[int, list[tuple[int, float]]] = field(default_factory=dict)
    edges: list[tuple[int, int]] = field(default_factory=list)
    grid: SegmentGrid | None = None

    def add_edge(self, a: int, b: int) -> None:
        d = math.dist(self.xy[a], self.xy[b])
        self.adj.setdefault(a, []).append((b, d))
        self.adj.setdefault(b, []).append((a, d))
        self.edges.append((a, b))


def build_graph(overpass_data: dict, proj: Projection) -> RoadGraph:
    """Graph of usable ways from an Overpass answer (`way` + their `node`s)."""
    g = RoadGraph(proj=proj)
    nodes = {e["id"]: (e["lat"], e["lon"]) for e in overpass_data["elements"] if e["type"] == "node"}
    for e in overpass_data["elements"]:
        if e["type"] != "way":
            continue
        tags = e.get("tags", {})
        if tags.get("highway") in EXCLUDED_HIGHWAYS or tags.get("access") in PRIVATE_ACCESS:
            continue
        ids = [n for n in e.get("nodes", []) if n in nodes]
        for n in ids:
            if n not in g.xy:
                g.xy[n] = proj.xy(*nodes[n])
        for a, b in zip(ids, ids[1:]):
            if a != b:
                g.add_edge(a, b)
    return g


def distances_from_track(g: RoadGraph, track: Polyline) -> dict[int, float]:
    """Shortest distance along the network from the track to every node.

    Nodes on the track start at their (small) distance to it; everything else
    is reached through the network (multi-source Dijkstra).
    """
    dist: dict[int, float] = {}
    heap: list[tuple[float, int]] = []
    for n, p in g.xy.items():
        d, _ = track.nearest(*g.proj.latlon(*p), within_m=ON_TRACK_M)
        if d <= ON_TRACK_M:
            dist[n] = d
            heap.append((d, n))
    heapq.heapify(heap)
    while heap:
        d, n = heapq.heappop(heap)
        if d > dist.get(n, math.inf):
            continue
        for m, w in g.adj.get(n, ()):
            nd = d + w
            if nd < dist.get(m, math.inf):
                dist[m] = nd
                heapq.heappush(heap, (nd, m))
    return dist


def detour_m(g: RoadGraph, dist: dict[int, float], lat: float, lon: float) -> float | None:
    """One-way distance from the track to the point along the network.

    None when the point is more than MAX_SNAP_M from every way, or when its way
    is not connected to the track.
    """
    if g.grid is None:
        g.grid = SegmentGrid([(g.xy[a], g.xy[b]) for a, b in g.edges])
    p = g.proj.xy(lat, lon)
    best: float | None = None
    for i in g.grid.near(p, MAX_SNAP_M):
        a, b = g.edges[i]
        snap, u, _ = project_on_segment(p, g.xy[a], g.xy[b])
        if snap > MAX_SNAP_M:
            continue
        seg = math.dist(g.xy[a], g.xy[b])
        via = min(dist.get(a, math.inf) + u * seg, dist.get(b, math.inf) + (1 - u) * seg)
        if math.isinf(via):
            continue
        total = via + snap
        if best is None or total < best:
            best = total
    return best
