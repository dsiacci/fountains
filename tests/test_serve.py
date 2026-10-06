"""The local page: what it lists, what the rider decides, what the GPX holds, who may call it."""

import http.client
import json
import threading
import xml.etree.ElementTree as ET
from http.server import ThreadingHTTPServer

import pytest

from fountains import DISCLAIMER, score, serve
from fountains.gpx import Track, write_track

GPX_NS = "{http://www.topografix.com/GPX/1/1}"


def track_bytes(tmp_path, points=None) -> bytes:
    pts = points or [(41.90, 8.80 + i * 0.0005) for i in range(41)]  # about 1.7 km eastward, south of Ajaccio's hills
    path = tmp_path / "ride.gpx"
    write_track(Track("Test loop", pts, [100.0 + 2 * i for i in range(len(pts))]), path)
    return path.read_bytes()


def fake_points(data_dir):
    return [
        {"ref": "node/1", "osm_id": "node/1", "source": "OSM", "name": "Funtana", "kind": "drinking water", "lat": 41.9003, "lon": 8.805, "last_confirmed": "2025-07-01"},
        {"ref": "IGN X", "osm_id": "", "source": "IGN", "name": "", "kind": "fountain", "lat": 41.9009, "lon": 8.810, "last_confirmed": ""},
        {"ref": "node/2", "osm_id": "node/2", "source": "OSM", "name": "Far", "kind": "drinking water", "lat": 41.91, "lon": 8.81},
        {"ref": "node/3", "osm_id": "node/3", "source": "OSM", "name": "Private", "kind": "tap", "lat": 41.9001, "lon": 8.812, "access": "private"},
    ]


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(score, "water_points", fake_points)
    monkeypatch.setattr(serve.App, "start", lambda self, kind, **kw: None)  # no background job in these tests
    return serve.App(tmp_path / "work", data_dir=tmp_path, vision=False)


def test_climb_counts_rises_and_ignores_noise():
    assert serve.climb_m([100, 101, 99, 100, 120, 118, 140]) == 43  # 99 -> 120 and 118 -> 140
    assert serve.climb_m(None) is None
    assert serve.climb_m([None, None]) is None


def test_simplify_keeps_both_ends():
    pts = [(42.0 + i * 1e-4, 9.0) for i in range(5001)]
    out = serve.simplify(pts, keep=1000)
    assert len(out) <= 1002
    assert out[0] == [42.0, 9.0] and out[-1] == [round(pts[-1][0], 5), 9.0]


def test_upload_lists_mapped_points_near_the_track_as_prefilled_fountains(app, tmp_path):
    app.upload(track_bytes(tmp_path), "ride.gpx")
    pts = app.ride.state["points"]
    assert [p["ref"] for p in pts] == ["node/1", "IGN X"]  # far and private points left out, in riding order
    assert all(p["verdict"] == "fountain" and p["prefilled"] for p in pts)
    assert pts[0]["km"] < pts[1]["km"]
    assert pts[0]["plan"].startswith("https://data.geopf.fr/") and pts[0]["photo_state"] == "waiting"
    assert app.ride.state["length_km"] == pytest.approx(1.7, abs=0.1)
    assert app.ride.state["climb_m"] == 78  # 2 m rises, counted by steps of 6 m: the last 2 m stay below the 5 m step


def test_same_file_keeps_decisions_and_another_ride_replaces_it(app, tmp_path):
    gpx = track_bytes(tmp_path)
    app.upload(gpx, "ride.gpx")
    app.decide("M01", "not")
    app.upload(gpx, "ride.gpx")
    assert app.ride.point("M01")["verdict"] == "not"
    first = app.ride
    app.upload(track_bytes(tmp_path, [(41.95, 8.85 + i * 0.0005) for i in range(30)]), "other.gpx")
    assert app.ride is not first and first.cancelled
    reopened = serve.App(tmp_path / "work", data_dir=tmp_path, vision=False)
    assert reopened.ride.state["id"] == app.ride.state["id"]


def test_tracks_outside_corsica_and_gpx_with_entities_are_refused(app, tmp_path):
    with pytest.raises(ValueError, match="Corsica"):
        app.upload(track_bytes(tmp_path, [(45.0, 6.0), (45.01, 6.0)]), "alps.gpx")
    with pytest.raises(ValueError, match="DTD"):
        app.upload(b'<?xml version="1.0"?><!DOCTYPE gpx [<!ENTITY a "aaaa">]><gpx/>', "bomb.gpx")


def test_decisions_are_checked(app, tmp_path):
    app.upload(track_bytes(tmp_path), "ride.gpx")
    app.decide("M02", "unsure")
    p = app.ride.point("M02")
    assert p["verdict"] == "unsure" and not p["prefilled"]
    with pytest.raises(ValueError):
        app.decide("M02", "maybe")
    with pytest.raises(ValueError):
        app.decide("M02", photo=3)  # no photo yet
    with pytest.raises(KeyError):
        app.decide("Z99", "not")


def test_a_fountain_the_rider_knows_can_be_added_without_any_photo(app, tmp_path):
    app.upload(track_bytes(tmp_path), "ride.gpx")
    pid = app.add(41.9002, 8.815, "Funtana di u Pastore")
    p = app.ride.point(pid)
    assert pid == "U01" and p["verdict"] == "fountain" and p["origin"] == "you" and p["photo_state"] == "none"
    with pytest.raises(ValueError, match="from the track"):
        app.add(41.95, 8.815, "too far")


def test_waypoint_goes_where_the_photo_was_taken_only_when_the_map_may_be_off():
    photo = {"lat": 41.0, "lon": 9.0}
    osm = {"origin": "map", "source": "OSM", "lat": 42.0, "lon": 9.1, "verdict": "fountain", "photos": [photo], "road_m": 40}
    ign_far = {"origin": "map", "source": "IGN", "lat": 42.0, "lon": 9.1, "verdict": "fountain", "photos": [photo], "road_m": 60}
    ign_near = dict(ign_far, road_m=5)
    clue = {"origin": "clue", "source": "clue", "lat": 42.0, "lon": 9.1, "road_lat": 42.2, "road_lon": 9.2, "verdict": "fountain", "photos": [photo]}
    assert serve.where(osm) == (42.0, 9.1)
    assert serve.where(ign_far) == (41.0, 9.0)
    assert serve.where(ign_near) == (42.0, 9.1)
    assert serve.where(clue) == (41.0, 9.0)
    assert serve.where(dict(clue, verdict="unsure")) == (42.2, 9.2)
    assert serve.where(dict(clue, photos=[])) == (42.2, 9.2)


def test_gpx_has_the_track_and_only_the_points_kept_or_to_check(app, tmp_path):
    app.upload(track_bytes(tmp_path), "ride.gpx")
    st = app.ride.state
    st["points"].append(dict(st["points"][0], id="C01", origin="clue", source="clue", name="", kind="stream crossing",
                             verdict=None, prefilled=False, road_lat=41.9, road_lon=8.806, km=0.6))
    app.decide("M02", "unsure")
    app.add(41.9002, 8.815, "Pastore")
    st["score"] = {"meta": {"day": "2026-10-10"}, "gaps": [], "fountains": [
        {"ref": "M01", "band": "likely", "reason": "rain in the last 90 days 120 mm", "notes": ["OSM says seasonal=summer"], "reachable": True,
         "waypoint": "likely: Funtana"}]}
    fname, body = app.gpx()
    assert fname == "Test-loop-fountains-2026-10-10.gpx"
    root = ET.fromstring(body)
    names = [w.find(GPX_NS + "name").text for w in root.iter(GPX_NS + "wpt")]
    assert names == ["likely: Funtana", "check: fountain", "fountain: Pastore"]
    assert len(list(root.iter(GPX_NS + "trkpt"))) == 41
    assert root.find(GPX_NS + "metadata/" + GPX_NS + "desc").text == DISCLAIMER
    first = next(root.iter(GPX_NS + "wpt"))
    assert "seasonal=summer" in first.find(GPX_NS + "desc").text
    assert first.find(GPX_NS + "sym").text == "Drinking Water"


@pytest.fixture
def server(app):
    handler = type("H", (serve.Handler,), {"app": app})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def call(port, method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request(method, path, body=body, headers=headers or {})
    r = c.getresponse()
    return r.status, r.getheader("Content-Type") or "", r.read()


def test_page_and_state_are_served_to_this_computer(server):
    status, ctype, body = call(server, "GET", "/")
    assert status == 200 and ctype.startswith("text/html") and b"fountains" in body
    status, _, body = call(server, "GET", "/api/state")
    data = json.loads(body)
    assert data["ride"] is None and data["disclaimer"] == DISCLAIMER and "PriorLabs-TabPFN" in data["attribution"]


def test_other_hosts_and_non_json_posts_are_refused(server, tmp_path):
    assert call(server, "GET", "/", headers={"Host": "evil.example:8765"})[0] == 403
    assert call(server, "POST", "/api/decide", body="id=M01&verdict=not", headers={"Content-Type": "application/x-www-form-urlencoded"})[0] == 415


def test_upload_decide_and_download_over_http(server, tmp_path):
    gpx = track_bytes(tmp_path).decode()
    status, _, body = call(server, "POST", "/api/ride", json.dumps({"gpx": gpx, "filename": "ride.gpx"}), {"Content-Type": "application/json"})
    assert status == 200 and len(json.loads(body)["ride"]["points"]) == 2
    version = json.loads(body)["ride"]["version"]
    assert json.loads(call(server, "GET", f"/api/state?v={version}")[2]) == {"same": True}
    status, _, body = call(server, "POST", "/api/decide", json.dumps({"id": "M01", "verdict": "not"}), {"Content-Type": "application/json"})
    assert status == 200 and json.loads(body)["ride"]["points"][0]["verdict"] == "not"
    status, _, body = call(server, "POST", "/api/decide", json.dumps({"id": "M01", "verdict": "perhaps"}), {"Content-Type": "application/json"})
    assert status == 400 and "verdict" in json.loads(body)["error"]
    status, ctype, body = call(server, "GET", "/api/gpx")
    assert status == 200 and ctype == "application/gpx+xml" and b"<trk>" in body
    for bad in ("/crops/../state.json", "/crops/..%2Fstate.json", "/crops/x.jpg"):
        assert call(server, "GET", bad)[0] == 404
