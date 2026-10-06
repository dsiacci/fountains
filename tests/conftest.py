import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fountains import meteo  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Tests never touch the user's cache or the network caches."""
    monkeypatch.setenv("FOUNTAINS_CACHE_DIR", str(tmp_path / "cache"))


def make_gauges(n_days: int = 800, start: dt.date = dt.date(2025, 1, 1), daily_mm=(1.0, 2.0, 4.0)) -> meteo.Gauges:
    """Three synthetic gauges in Corsica, each with constant daily rain."""
    rr = np.zeros((n_days, len(daily_mm)), dtype=np.float32)
    for j, v in enumerate(daily_mm):
        rr[:, j] = v
    return meteo.Gauges(
        ids=["20000001", "20000002", "20000003"][: len(daily_mm)],
        names=["NORD", "CENTRE", "SUD"][: len(daily_mm)],
        lat=np.array([42.05, 42.00, 41.95][: len(daily_mm)]),
        lon=np.array([9.00, 9.00, 9.00][: len(daily_mm)]),
        alt=np.array([100.0, 500.0, 900.0][: len(daily_mm)]),
        start=start,
        rr=rr,
    )
