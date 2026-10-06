import pytest

from fountains.geo import Polyline, Projection
from fountains.network import build_graph, detour_m, distances_from_track
from fountains.osm import fountain_kind, fountains_from_overpass, is_drinking_water_point, last_confirmation


@pytest.mark.parametrize(
    "tags,expected",
    [
        ({"amenity": "drinking_water"}, True),
        ({"amenity": "drinking_water", "drinking_water": "no"}, False),
        ({"amenity": "water_point"}, True),
        ({"natural": "spring"}, False),
        ({"natural": "spring", "drinking_water": "yes"}, True),
        ({"natural": "spring", "drinking_water": "conditional"}, True),
        ({"man_made": "water_tap"}, False),
        ({"man_made": "water_tap", "drinking_water": "yes"}, True),
        ({"amenity": "fountain"}, False),
        ({"amenity": "fountain", "drinking_water": "yes"}, True),
        ({"man_made": "water_well", "drinking_water": "yes"}, True),
    ],
)
def test_selection_rule(tags, expected):
    assert is_drinking_water_point(tags) is expected


def test_kind_and_confirmation():
    assert fountain_kind({"natural": "spring", "amenity": "drinking_water"}) == "spring"
    assert fountain_kind({"amenity": "drinking_water", "man_made": "water_tap"}) == "tap"
    assert last_confirmation({"check_date": "2022-05-24", "survey:date": "2024-08-13"}) == "2024-08-13"
    assert last_confirmation({}) == ""


def test_fountains_from_overpass_keeps_only_drinking_water_points():
    data = {
        "osm3s": {"timestamp_osm_base": "2026-10-06T07:33:42Z"},
        "elements": [
            {"type": "node", "id": 1, "lat": 42.0, "lon": 9.0, "timestamp": "2024-01-02T00:00:00Z", "tags": {"amenity": "drinking_water", "name": "Funtana"}},
            {"type": "node", "id": 2, "lat": 42.0, "lon": 9.1, "tags": {"natural": "spring"}},
            {"type": "way", "id": 3, "center": {"lat": 42.1, "lon": 9.0}, "tags": {"amenity": "fountain", "drinking_water": "yes"}},
        ],
    }
    fc = fountains_from_overpass(data)
    ids = [f["properties"]["osm_id"] for f in fc["features"]]
    assert ids == ["node/1", "way/3"]
    assert fc["features"][0]["properties"]["last_edited"] == "2024-01-02"
    assert "ODbL" in fc["license"]


def meters_to_deg(dx, dy, lat0=42.0):
    proj = Projection(lat0)
    x0, y0 = proj.xy(lat0, 9.0)
    return proj.latlon(x0 + dx, y0 + dy)


def node(i, dx, dy):
    lat, lon = meters_to_deg(dx, dy)
    return {"type": "node", "id": i, "lat": lat, "lon": lon}


def way(i, nodes, **tags):
    return {"type": "way", "id": i, "nodes": nodes, "tags": {"highway": "residential", **tags}}


def test_detour_follows_the_network():
    # Main road along x from 0 to 1000 m (the track). A side street leaves at
    # x=600, goes 150 m north, then turns back west 300 m. The fountain sits at
    # the end of the side street: 150 m from the track as the crow flies, but
    # 150 + 300 = 450 m away by road.
    elements = [node(1, 0, 0), node(2, 600, 0), node(3, 1000, 0), node(4, 600, 150), node(5, 300, 150)]
    elements += [way(10, [1, 2, 3], highway="primary"), way(11, [2, 4, 5])]
    track = Polyline([meters_to_deg(0, 0), meters_to_deg(1000, 0)], proj=Projection(42.0))
    g = build_graph({"elements": elements}, track.proj)
    dist = distances_from_track(g, track)
    lat, lon = meters_to_deg(300, 155)
    d = detour_m(g, dist, lat, lon)
    assert d == pytest.approx(455, abs=2)
    assert track.nearest(lat, lon)[0] == pytest.approx(155, abs=1)


def test_private_way_is_not_a_way_in():
    elements = [node(1, 0, 0), node(2, 1000, 0), node(3, 500, 0), node(4, 500, 100)]
    elements += [way(10, [1, 3, 2], highway="primary"), way(11, [3, 4], access="private")]
    track = Polyline([meters_to_deg(0, 0), meters_to_deg(1000, 0)], proj=Projection(42.0))
    g = build_graph({"elements": elements}, track.proj)
    dist = distances_from_track(g, track)
    # 100 m north of the road, only reachable through the private drive: the
    # nearest public way is the main road itself, 100 m away (beyond the snap
    # distance), so there is no mapped way in.
    assert detour_m(g, dist, *meters_to_deg(500, 100)) is None
    # Next to the main road, the detour is just the few metres off the road.
    assert detour_m(g, dist, *meters_to_deg(500, 10)) == pytest.approx(10, abs=1)
