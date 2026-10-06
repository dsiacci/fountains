"""TabPFN v2 as the scorer, and the three confidence bands.

TabPFN is a tabular foundation model (Prior Labs): it is pre-trained on
synthetic tables and predicts on a new table in one forward pass, given the
training rows as context. Nothing is fine-tuned here. Weights: TabPFN v2
classifier, Prior Labs License 1.1 (Apache 2.0 with an attribution clause).
Built with PriorLabs-TabPFN.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np

from .features import vector

FEATURE_SETS = {
    # Rain before the day, season, altitude.
    "base": ("doy", "elevation_m", "rain_30", "rain_90", "rain_180"),
    # The same, plus how that rain compares with the 1991-2020 normal there.
    "anomaly": ("doy", "elevation_m", "rain_30", "rain_90", "rain_180", "ratio_90", "ratio_180"),
}

BAND_LIKELY = "likely"
BAND_UNCERTAIN = "uncertain"
BAND_UNLIKELY = "unlikely"


def read_table(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            if k in ("station", "station_name", "stream", "date", "code"):
                continue
            r[k] = None if v in ("", "nan", "None") else float(v)
    return rows


def make_classifier(n_estimators: int = 4, random_state: int = 0):
    """A TabPFN v2 classifier on CPU (the v2 weights, not the newer non-commercial ones)."""
    os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    return TabPFNClassifier.create_default_for_version(
        ModelVersion.V2,
        n_estimators=n_estimators,
        device="cpu",
        random_state=random_state,
    )


def fit_predict(train: list[dict], test: list[dict], features: tuple[str, ...], n_estimators: int = 4) -> np.ndarray:
    """Probability that flowing water is visible, for each test row."""
    X = np.array([vector(r, features) for r in train], dtype=np.float32)
    y = np.array([int(r["flowing"]) for r in train])
    Xt = np.array([vector(r, features) for r in test], dtype=np.float32)
    clf = make_classifier(n_estimators=n_estimators)
    clf.fit(X, y)
    proba = clf.predict_proba(Xt)
    return proba[:, list(clf.classes_).index(1)]


def load_bands(path: Path) -> dict:
    """Band thresholds measured by `fountains calibrate` (see calibration.json)."""
    cal = json.loads(Path(path).read_text(encoding="utf-8"))
    return cal["bands"]


def band(p: float, bands: dict) -> str:
    if math.isnan(p):
        return BAND_UNCERTAIN
    if p >= bands["likely_from"]:
        return BAND_LIKELY
    if p <= bands["unlikely_to"]:
        return BAND_UNLIKELY
    return BAND_UNCERTAIN
