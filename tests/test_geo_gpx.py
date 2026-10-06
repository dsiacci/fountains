import math
import xml.etree.ElementTree as ET

import pytest

from fountains.geo import Polyline, SegmentGrid, haversine_m, project_on_segment
from fountains.gpx import Track, read_gpx, trim_track, write_track, write_waypoints


def test_haversine_one_degree_of_latitude():
    assert haversine_m(42.0, 9.0, 43.0, 9.0) == pytest.approx(111_195, rel=1e-3)


def test_project_on_segment_clamps_to_ends():
    d, u, q = project_on_segment((5.0, 5.0), (0.0, 0.0), (10.0, 0.0))
    assert (d, u, q) == (5.0, 0.5, (5.0, 0.0))
    d, u, _ = project_on_segment((-3.0, 4.0), (0.0, 0.0), (10.0, 0.0))
    assert (d, u) == (5.0, 0.0)


def straight_track(n=50, lat=42.0, lon0=9.0, step=0.001):
    return [(lat, lon0 + i * step) for i in range(n)]


def test_polyline_nearest_and_along_distance():
    line = Polyline(straight_track())
    # A point 100 m north of the 10th vertex.
    d, s = line.nearest(42.0 + 100 / 111_195, 9.0 + 10 * 0.001)
    assert d == pytest.approx(100, abs=1)
    assert s == pytest.approx(line.cum[10], abs=1)


def test_polyline_nearest_with_grid_matches_brute_force():
    line = Polyline(straight_track())
    for dlat in (0.0005, 0.002, 0.01):
        p = (42.0 + dlat, 9.017)
        brute = line.nearest(*p)
        fast = line.nearest(*p, within_m=500)
        if brute[0] <= 500:
            assert fast == pytest.approx(brute)
        else:
            assert math.isinf(fast[0])


def test_segment_grid_finds_neighbours():
    segs = [((0.0, 0.0), (100.0, 0.0)), ((1000.0, 1000.0), (1100.0, 1000.0))]
    g = SegmentGrid(segs, cell_m=50)
    assert g.near((50.0, 20.0), 30) == {0}
    assert 1 not in g.near((50.0, 20.0), 30)


def test_resample_spacing():
    line = Polyline(straight_track(n=20))
    pts = line.resample(100.0)
    assert pts[0] == line.points[0] and pts[-1] == line.points[-1]
    assert len(pts) == math.floor(line.length_m / 100.0) + 2


GPX_TRK = """<?xml version="1.0"?>
<gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1">
 <metadata><name>Meta name</name><author><name>Someone</name></author></metadata>
 <trk><name>Boucle</name><trkseg>
  <trkpt lat="42.0" lon="9.0"><ele>10</ele><time>2026-09-13T07:23:00Z</time></trkpt>
  <trkpt lat="42.0" lon="9.01"/>
  <trkpt lat="42.0" lon="9.02"/>
 </trkseg></trk>
</gpx>"""

GPX_RTE = """<?xml version="1.0"?>
<gpx version="1.0" creator="test" xmlns="http://www.topografix.com/GPX/1/0">
 <rte><name>Route</name><rtept lat="42.0" lon="9.0"/><rtept lat="42.1" lon="9.0"/></rte>
</gpx>"""


def test_read_gpx_track_and_route(tmp_path):
    p = tmp_path / "a.gpx"
    p.write_text(GPX_TRK)
    t = read_gpx(p)
    assert len(t.points) == 3 and t.points[1] == (42.0, 9.01)
    p.write_text(GPX_RTE)
    assert read_gpx(p).points == [(42.0, 9.0), (42.1, 9.0)]


def test_trim_removes_start_and_end_and_write_drops_metadata(tmp_path):
    t = Track("Boucle", straight_track(n=100, step=0.0005))  # about 4 km
    trimmed = trim_track(t, 1000, 1000)
    line = Polyline(t.points)
    assert haversine_m(*t.points[0], *trimmed.points[0]) >= 1000 - 50
    assert haversine_m(*t.points[-1], *trimmed.points[-1]) >= 1000 - 50
    assert len(trimmed.points) < len(t.points)
    assert line.length_m > 3000
    out = tmp_path / "shared.gpx"
    write_track(trimmed, out)
    text = out.read_text()
    assert "<time>" not in text and "author" not in text
    assert read_gpx(out).points[0] == pytest.approx(trimmed.points[0], abs=1e-6)


def test_trim_refuses_to_leave_nothing():
    with pytest.raises(ValueError):
        trim_track(Track("short", straight_track(n=5)), 1000, 1000)


def test_waypoints_are_valid_gpx(tmp_path):
    out = tmp_path / "w.gpx"
    write_waypoints([{"lat": 42.0, "lon": 9.0, "name": "Funtana <vechja> - likely", "desc": "rain & sun"}], out, metadata_desc="Carry water.")
    root = ET.parse(out).getroot()
    ns = {"g": "http://www.topografix.com/GPX/1/1"}
    assert root.find("g:wpt/g:name", ns).text == "Funtana <vechja> - likely"
    assert root.find("g:metadata/g:desc", ns).text == "Carry water."
