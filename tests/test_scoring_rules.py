"""The rules the tool must never break, whatever the model says."""

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from fountains import DISCLAIMER, onde
from fountains.calibrate import band_thresholds, isotonic, summarise, wilson
from fountains.features import ALL_FEATURES
from fountains.gpx import Track
from fountains.model import FEATURE_SETS, band
from fountains.report import text_table, to_geojson, write_outputs
from fountains.score import ScoredFountain, nearest_stream, score_track

BANDS = {"likely_from": 0.9, "unlikely_to": 0.4}


def test_onde_labels():
    assert [onde.label(c) for c in ("1", "1a", "1f", "2", "3", None, "x")] == [1, 1, 1, 0, 0, None, None]


def test_feature_sets_only_use_known_features():
    for fs in FEATURE_SETS.values():
        assert set(fs) <= set(ALL_FEATURES)


def test_band_edges():
    assert band(0.95, BANDS) == "likely"
    assert band(0.9, BANDS) == "likely"
    assert band(0.6, BANDS) == "uncertain"
    assert band(0.4, BANDS) == "unlikely"
    assert band(float("nan"), BANDS) == "uncertain"


def test_isotonic_is_monotone_and_thresholds_follow_targets():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(int)  # perfectly calibrated
    ps, fit = isotonic(p, y)
    assert np.all(np.diff(fit) >= -1e-12)
    t = band_thresholds(p, y)
    assert t["likely_from"] == pytest.approx(0.9, abs=0.05)
    assert t["unlikely_to"] == pytest.approx(0.5, abs=0.05)


def test_no_likely_band_when_never_reliable_enough():
    p = np.linspace(0, 1, 200)
    y = np.zeros(200, dtype=int)
    y[::2] = 1  # half flow whatever the score
    assert band_thresholds(p, y)["likely_from"] > 1


def test_wilson_interval():
    lo, hi = wilson(9, 10)
    assert 0.55 < lo < 0.6 and 0.98 < hi <= 1.0


def test_summary_counts_every_prediction():
    preds = [{"station": s, "date": "2025-08-01", "y": y, "p": p, "p_logistic": 0.5, "p_base": 0.8, "p_month": 0.7}
             for s, y, p in [("A", 1, 0.95), ("A", 1, 0.9), ("B", 0, 0.2), ("B", 1, 0.6), ("C", 0, 0.1)]]
    s = summarise(preds, ("doy",), 4)
    assert s["n_observations"] == 5 and s["n_stations"] == 3
    assert sum(b["n"] for b in s["reliability"]) == 5
    assert sum(v["n"] for v in s["per_band"].values()) == 5


def fountain(osm_id, km, band_, reachable=True, detour=40):
    return ScoredFountain(
        osm_id=osm_id, name="", kind="drinking water", lat=42.0, lon=9.0 + km / 100, km=km, off_track_m=30,
        detour_m=detour, reachable=reachable, access="", elevation_m=500.0, p=0.5, band=band_,
        rain={"rain_30": 10.0, "rain_90": 40.0, "ratio_90": 0.5, "forecast_mm": 0.0, "assumed_dry_days": []},
        last_confirmed="", stream=None,
    )


META = {
    "track": "Boucle", "day": "2026-10-11", "max_detour_m": 250.0, "length_km": 60.0,
    "gauges_last_day": "2026-10-09", "forecast_used": True, "excluded_private": 1,
    "calibration": {"brier": 0.1, "brier_month_rate": 0.14, "n_observations": 3831, "n_stations": 33,
                    "per_band": {"likely": {"n": 10, "rate": 0.92}, "uncertain": {"n": 5, "rate": 0.7}, "unlikely": {"n": 3, "rate": 0.3}}},
}


def test_every_fountain_is_shown_everywhere_and_the_warning_too(tmp_path):
    scored = [fountain("node/1", 3.0, "likely"), fountain("node/2", 12.5, "unlikely"),
              fountain("node/3", 20.0, "uncertain", reachable=False, detour=420), fountain("node/4", 30.0, "uncertain", reachable=False, detour=None)]
    text = text_table(scored, META)
    for f in scored:
        assert f.osm_id in text
    assert text.count(DISCLAIMER) == 2
    assert "no mapped path" in text and "420 m by road or path" in text
    gj = to_geojson(scored, META)
    assert len(gj["features"]) == len(scored) and gj["disclaimer"] == DISCLAIMER
    paths = write_outputs(scored, META, tmp_path, "ride", with_html=True)
    gpx = ET.parse([p for p in paths if p.suffix == ".gpx"][0]).getroot()
    ns = {"g": "http://www.topografix.com/GPX/1/1"}
    names = [w.find("g:name", ns).text for w in gpx.findall("g:wpt", ns)]
    assert len(names) == len(scored)
    assert all(any(b in n for b in ("likely", "uncertain", "unlikely")) for n in names)
    assert gpx.find("g:metadata/g:desc", ns).text == DISCLAIMER
    html = [p for p in paths if p.suffix == ".html"][0].read_text()
    assert DISCLAIMER in html and "Built with PriorLabs-TabPFN" in html


def test_order_is_the_riding_order_not_a_ranking():
    scored = [fountain("node/1", 3.0, "unlikely"), fountain("node/2", 12.5, "likely")]
    text = text_table(scored, META)
    assert text.index("node/1") < text.index("node/2")


def test_tracks_outside_corsica_are_refused():
    with pytest.raises(SystemExit):
        score_track(Track("Paris", [(48.85, 2.35), (48.86, 2.36)]), __import__("datetime").date(2026, 10, 11))


def test_nearest_stream_ignores_later_observations():
    obs = [
        {"station": "Y1", "station_name": "Pont", "stream": "Prunelli", "lat": 42.0, "lon": 9.0, "date": "2026-09-28", "code": "3", "flowing": 0},
        {"station": "Y1", "station_name": "Pont", "stream": "Prunelli", "lat": 42.0, "lon": 9.0, "date": "2026-10-20", "code": "1a", "flowing": 1},
    ]
    s = nearest_stream(obs, 42.01, 9.0, __import__("datetime").date(2026, 10, 11))
    assert s["date"] == "2026-09-28" and s["flowing"] is False and s["km"] == pytest.approx(1.1, abs=0.1)
    assert nearest_stream(obs, 43.0, 9.0, __import__("datetime").date(2026, 10, 11)) is None


def test_geojson_is_serialisable():
    json.dumps(to_geojson([fountain("node/1", 3.0, "likely")], META))
