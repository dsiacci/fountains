import datetime as dt
import gzip
import math

import numpy as np
import pytest

from conftest import make_gauges
from fountains import meteo
from fountains.features import RainContext, rain_features, save_normals, vector


def test_window_sum_and_vectorised_version_agree():
    rng = np.random.default_rng(1)
    s = rng.gamma(0.5, 4.0, 400).astype(np.float32)
    s[[10, 50, 51, 300]] = np.nan
    all30 = meteo.window_sums(s, 30)
    for i in (0, 29, 30, 31, 60, 301, 400):
        a, b = meteo.window_sum(s, i, 30), all30[i]
        assert (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b, rel=1e-5)


def test_window_sum_scales_up_a_few_missing_days_and_refuses_many():
    s = np.ones(100, dtype=np.float32)
    s[95:98] = np.nan  # 3 missing of the last 30
    assert meteo.window_sum(s, 100, 30) == pytest.approx(30.0)
    s[80:95] = np.nan  # 18 missing of the last 30
    assert math.isnan(meteo.window_sum(s, 100, 30))


def test_point_series_uses_nearest_reporting_gauges():
    g = make_gauges(n_days=10)
    series, used = meteo.point_series(g, 42.00, 9.00)  # on the CENTRE gauge
    assert series[0] == pytest.approx(2.0, abs=0.01)
    assert used[0][0] == "CENTRE"
    g.rr[5, 1] = np.nan  # CENTRE silent on day 5: the other two take over
    series, _ = meteo.point_series(g, 42.00, 9.00)
    assert series[5] == pytest.approx(2.5, abs=0.01)  # equidistant NORD (1) and SUD (4)


def test_point_series_far_from_every_gauge_is_nan():
    g = make_gauges(n_days=10)
    series, used = meteo.point_series(g, 43.5, 9.0)
    assert used == [] and np.isnan(series).all()


def test_gauge_normals_constant_rain():
    g = make_gauges(n_days=(dt.date(2021, 1, 1) - dt.date(1990, 1, 1)).days, start=dt.date(1990, 1, 1))
    normals = meteo.gauge_normals(g)
    assert set(normals) == set(g.ids)
    assert normals["20000002"][90][meteo.doy_slot(dt.date(2000, 7, 1))] == pytest.approx(180.0)


def test_read_gauges_parses_meteo_france_format(tmp_path):
    head = "NUM_POSTE;NOM_USUEL;LAT;LON;ALTI;AAAAMMJJ;RR;QRR;TN\n"
    rows = "20004002;AJACCIO;41.918000;8.792667;5;20250101;0.0;1;4.6\n20004002;AJACCIO;41.918000;8.792667;5;20250103;9.9;1;4.8\n20004003;PARATA;41.9;8.6;124;20250102;;;\n"
    p = tmp_path / "Q_20.csv.gz"
    with gzip.open(p, "wt") as f:
        f.write(head + rows)
    g = meteo.read_gauges([p], start=dt.date(2025, 1, 1))
    assert g.ids == ["20004002"]  # the gauge with no rain value is ignored
    assert g.rr.shape == (3, 1)
    assert np.isnan(g.rr[1, 0]) and g.rr[2, 0] == pytest.approx(9.9)
    assert g.last_day == dt.date(2025, 1, 3)


def context(tmp_path, n_days=400):
    g = make_gauges(n_days=n_days, start=dt.date(2025, 9, 1))
    normals = {sid: {w: [w * 1.0] * 366 for w in meteo.WINDOWS} for sid in g.ids}  # normal = 1 mm/day
    save_normals(normals, g, tmp_path / "normals.json")
    return g, RainContext.load(g, tmp_path / "normals.json")


def test_rain_features_measured_only(tmp_path):
    g, ctx = context(tmp_path)
    day = dt.date(2026, 6, 1)
    r = rain_features(ctx, 42.0, 9.0, day)
    assert r["rain_30"] == pytest.approx(60.0, abs=0.5)  # 2 mm/day at CENTRE
    assert r["ratio_30"] == pytest.approx(2.0, abs=0.02)
    assert r["forecast_mm"] == 0 and r["assumed_dry_days"] == []


def test_rain_features_fills_the_gap_with_forecast_and_counts_missing_days_as_dry(tmp_path):
    g, ctx = context(tmp_path, n_days=30)  # gauges stop on 2025-09-30
    day = dt.date(2025, 10, 5)  # needs Oct 1..4
    fc = {dt.date(2025, 10, 1): 10.0, dt.date(2025, 10, 2): 5.0, dt.date(2025, 10, 3): None}
    r = rain_features(ctx, 42.0, 9.0, day, forecast=fc)
    assert r["forecast_mm"] == pytest.approx(15.0)
    assert r["assumed_dry_days"] == ["2025-10-03", "2025-10-04"]
    assert r["rain_30"] == pytest.approx(26 * 2.0 + 15.0, abs=0.5)


def test_vector_maps_missing_to_nan():
    v = vector({"doy": 200, "rain_30": None}, ("doy", "rain_30", "absent"))
    assert v[0] == 200 and math.isnan(v[1]) and math.isnan(v[2])
