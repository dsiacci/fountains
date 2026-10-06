import datetime as dt

import pytest

from fountains.ign import merge
from fountains.report import text_table
from fountains.score import Gap, Stop, find_gaps
from fountains.stops import is_open, open_intervals, passing_minutes, stops_from_overpass

SAT = dt.date(2026, 10, 10)
SUN = dt.date(2026, 10, 11)


def at(day, hhmm):
    return dt.datetime.combine(day, dt.time.fromisoformat(hhmm))


@pytest.mark.parametrize(
    "hours,when,expected",
    [
        ("Mo-Sa 07:00-12:30,15:30-19:30", at(SAT, "10:00"), True),
        ("Mo-Sa 07:00-12:30,15:30-19:30", at(SAT, "13:00"), False),
        ("Mo-Sa 07:00-12:30,15:30-19:30", at(SUN, "10:00"), False),
        ("Mo-Sa 07:00-12:30, 15:30-19:30; Su 07:00-12:00", at(SUN, "10:00"), True),
        ("Mo-Fr 08:00-18:00; Sa 08:00-12:00; Su,PH off", at(SUN, "10:00"), False),
        ("24/7", at(SUN, "03:00"), True),
        ("08:00-20:00", at(SUN, "19:59"), True),
        ("Jul-Aug Mo-Su 08:00-21:00; Sep-Jun Mo-Sa 08:00-19:00", at(SUN, "10:00"), False),
        ("Jun-Sep: Mo-Su 08:00-20:00", at(SAT, "10:00"), False),
        ("Tu-Su 18:00-02:00", at(SUN, "01:30"), True),  # Saturday night spills into Sunday
        ("sunrise-sunset", at(SUN, "10:00"), None),
        ('Mo-Sa 08:00-12:00 "sur rendez-vous"', at(SAT, "10:00"), None),
        ("", at(SAT, "10:00"), None),
        (None, at(SAT, "10:00"), None),
    ],
)
def test_opening_hours(hours, when, expected):
    assert is_open(hours, when) is expected


def test_later_rule_overrides_earlier_for_the_same_day():
    assert open_intervals("Mo-Su 08:00-20:00; Su 09:00-12:00", SUN) == [(540, 720)]


def test_passing_minutes_adds_climbing_time_and_ignores_gps_noise():
    cum = [0.0, 10_000.0, 20_000.0, 30_000.0]
    flat = passing_minutes(cum, None, 20.0, 500.0)
    assert flat[-1] == pytest.approx(90.0)
    hilly = passing_minutes(cum, [100.0, 600.0, 601.0, 100.0], 20.0, 500.0)
    assert hilly[1] == pytest.approx(30.0 + 60.0)  # 500 m climbed at 500 m/h
    assert hilly[2] == pytest.approx(60.0 + 60.0)  # +1 m is noise
    assert hilly[3] == pytest.approx(90.0 + 60.0)  # descending costs nothing extra


def test_gaps_between_likely_fountains():
    assert find_gaps(70.6, [18.0, 44.8], 15.0) == [(0.0, 18.0), (18.0, 44.8), (44.8, 70.6)]
    assert find_gaps(30.0, [10.0, 20.0], 15.0) == []


def test_merge_keeps_osm_point_and_adds_unmapped_ign_fountains():
    osm = [{"ref": "node/1", "osm_id": "node/1", "name": "", "kind": "drinking water", "lat": 42.0, "lon": 9.0, "last_confirmed": "2022-05-24"}]
    ign = [
        {"ign_id": "A", "source": "IGN", "name": "Funtana", "kind": "fountain", "lat": 42.0001, "lon": 9.0001, "last_confirmed": "2025-06-27"},
        {"ign_id": "B", "source": "IGN", "name": "Funtana di Leccia", "kind": "fountain", "lat": 41.82036, "lon": 8.87251, "last_confirmed": ""},
    ]
    out = merge(osm, ign)
    assert len(out) == 2
    a = [p for p in out if p.get("ref") == "node/1"][0]
    assert a["source"] == "OSM + IGN" and a["name"] == "Funtana" and a["last_confirmed"] == "2025-06-27"
    b = [p for p in out if p["ign_id"] == "B"][0]
    assert b["source"] == "IGN" and b["name"] == "Funtana di Leccia"


def test_stops_from_overpass_labels_and_keeps_hours():
    data = {"elements": [
        {"type": "node", "id": 7, "lat": 41.9, "lon": 8.8, "tags": {"shop": "bakery", "name": "U Fornu", "opening_hours": "Mo-Sa 06:30-13:00"}},
        {"type": "node", "id": 8, "lat": 41.9, "lon": 8.8, "tags": {"amenity": "fuel"}},
        {"type": "node", "id": 9, "lat": 41.9, "lon": 8.8, "tags": {"amenity": "bench"}},
    ]}
    fc = stops_from_overpass(data)
    kinds = [f["properties"]["kind"] for f in fc["features"]]
    assert kinds == ["bakery", "fuel station"]
    assert fc["features"][0]["properties"]["opening_hours"] == "Mo-Sa 06:30-13:00"


def test_gaps_are_printed_with_opening_state():
    meta = {
        "track": "Boucle", "day": "2026-10-11", "start": "08:30", "max_detour_m": 250.0, "gap_km": 15.0, "length_km": 70.6,
        "gauges_last_day": "2026-10-09", "forecast_used": True, "excluded_private": 0,
        "calibration": {"n_stations": 33, "per_band": {}},
    }
    gaps = [Gap(18.0, 39.2, [Stop("node/7", "U Fornu", "bakery", 41.9, 8.8, 27.3, 60, "Mo-Sa 06:30-13:00", "10:05", False)]), Gap(0.0, 18.0, [])]
    text = text_table([], meta, gaps)
    assert "km 18.0 to 39.2 (21.2 km)" in text
    assert "   km  27.3  bakery: U Fornu, 60 m off the route around 10:05: closed (Mo-Sa 06:30-13:00)" in text
    assert "no café, shop or fuel station on the maps within reach" in text


def stop(km, kind="café", open_then=None, detour=50, name=""):
    return Stop(f"node/{int(km * 10)}", name or kind, kind, 41.9, 8.8, km, detour, "", "10:00", open_then)


def test_a_few_stops_are_suggested_never_a_closed_one_nor_one_just_after_the_start():
    from fountains.score import suggest_stops

    stops = [stop(0.7, "fuel station", True), stop(4.0, "restaurant", True), stop(5.0, "bakery", None), stop(8.0, "café", False),
             stop(14.0, "bar", None, detour=200), stop(16.0, "bar", True, detour=240), stop(16.5, "small shop", None, detour=10),
             stop(29.0, "restaurant", None), stop(43.0, "café", False)]
    got = [(s.km, s.kind) for s in suggest_stops(stops, 0.0, 44.8, 10.0)]
    # windows of 8.36 km from km 3: [3, 11.4) [11.4, 19.7) [19.7, 28.1) [28.1, 36.4) [36.4, 44.8]
    assert got == [(4.0, "restaurant"), (16.0, "bar"), (29.0, "restaurant")]
    assert suggest_stops([stop(12.0, "café", False)], 0.0, 20.0, 10.0) == []


def test_only_suggested_stops_go_in_the_gpx(tmp_path):
    from fountains.report import write_outputs

    meta = {
        "track": "Boucle", "day": "2026-10-11", "start": "08:30", "max_detour_m": 250.0, "gap_km": 10.0, "length_km": 30.0,
        "gauges_last_day": "2026-10-09", "forecast_used": True, "excluded_private": 0,
        "calibration": {"n_stations": 33, "per_band": {}},
    }
    a, b = stop(12.0, "bar", True), stop(13.0, "bakery", None)
    a.suggested = True
    paths = write_outputs([], meta, tmp_path, "t", gaps=[Gap(0.0, 30.0, [a, b])])
    gpx = next(p for p in paths if p.suffix == ".gpx").read_text()
    assert "bar - open" in gpx and "bakery" not in gpx
    assert "-> km  12.0  bar" in next(p for p in paths if p.suffix == ".txt").read_text()
