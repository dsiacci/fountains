"""How much can the score be trusted? Leave-one-station-out validation.

Each ONDE station is left out in turn: the model is given every observation of
the other stations and predicts the left-out station's observations, which it
has never seen. Pooled over all stations, these predictions give
- the Brier score (mean squared error of the probability, 0 is perfect),
- the same score for two baselines (the base rate, and the rate for the month),
- the reliability curve: among the cases scored around p, how many flowed,
- the three bands, whose thresholds are read off that curve (isotonic fit):
  "likely" where at least 90 % of the left-out cases flowed, "unlikely" where at
  most 50 % did, "uncertain" in between. These targets were fixed before the
  first run.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .model import fit_predict

LIKELY_TARGET = 0.90
UNLIKELY_TARGET = 0.50


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson interval for a proportion k/n."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def leave_one_station_out(rows: list[dict], features: tuple[str, ...], n_estimators: int, cache_path: Path | None = None, log=print) -> list[dict]:
    """Out-of-station predictions for every row (resumable through `cache_path`)."""
    done: dict[str, list[dict]] = {}
    if cache_path and cache_path.exists():
        for line in cache_path.read_text().splitlines():
            rec = json.loads(line)
            done.setdefault(rec["station"], []).append(rec)
    stations = sorted({r["station"] for r in rows})
    out = []
    for i, s in enumerate(stations, 1):
        if s in done:
            out.extend(done[s])
            continue
        train = [r for r in rows if r["station"] != s]
        test = [r for r in rows if r["station"] == s]
        p = fit_predict(train, test, features, n_estimators=n_estimators)
        p_logit = logistic_predict(train, test, features)
        base = float(np.mean([r["flowing"] for r in train]))
        month_rate = defaultdict(list)
        for r in train:
            month_rate[r["date"][5:7]].append(r["flowing"])
        recs = [
            {
                "station": s,
                "date": r["date"],
                "y": int(r["flowing"]),
                "p": float(pi),
                "p_logistic": float(pl),
                "p_base": base,
                "p_month": float(np.mean(month_rate.get(r["date"][5:7], [base]))),
            }
            for r, pi, pl in zip(test, p, p_logit)
        ]
        if cache_path:
            with open(cache_path, "a") as f:
                for rec in recs:
                    f.write(json.dumps(rec) + "\n")
        out.extend(recs)
        log(f"[{i}/{len(stations)}] station {s}: {len(test)} observations")
    return out


def logistic_predict(train: list[dict], test: list[dict], features: tuple[str, ...]) -> np.ndarray:
    """A plain baseline on the same inputs: logistic regression, standardised, median-imputed."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from .features import vector

    X = np.array([vector(r, features) for r in train], dtype=float)
    y = np.array([int(r["flowing"]) for r in train])
    Xt = np.array([vector(r, features) for r in test], dtype=float)
    clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(X, y)
    return clf.predict_proba(Xt)[:, 1]


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def isotonic(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators fit of y on p: returns (sorted p, fitted rate)."""
    order = np.argsort(p, kind="mergesort")
    ps, ys = p[order], y[order].astype(float)
    blocks = [[v, 1.0] for v in ys]  # [mean, weight]
    merged: list[list[float]] = []
    for b in blocks:
        merged.append(b)
        while len(merged) > 1 and merged[-2][0] > merged[-1][0]:
            m2, w2 = merged.pop()
            m1, w1 = merged.pop()
            merged.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2])
    fitted = np.concatenate([np.full(int(w), m) for m, w in merged])
    return ps, fitted


def band_thresholds(p: np.ndarray, y: np.ndarray) -> dict:
    ps, fit = isotonic(p, y)
    hi = ps[fit >= LIKELY_TARGET]
    lo = ps[fit <= UNLIKELY_TARGET]
    return {
        "likely_from": float(hi.min()) if len(hi) else 1.01,
        "unlikely_to": float(lo.max()) if len(lo) else -0.01,
        "likely_target": LIKELY_TARGET,
        "unlikely_target": UNLIKELY_TARGET,
    }


def summarise(preds: list[dict], features: tuple[str, ...], n_estimators: int) -> dict:
    y = np.array([r["y"] for r in preds])
    p = np.array([r["p"] for r in preds])
    bands = band_thresholds(p, y)
    bins = []
    edges = np.linspace(0, 1, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        n = int(m.sum())
        if n:
            k = int(y[m].sum())
            bins.append({"from": round(float(lo), 2), "to": round(float(hi), 2), "n": n, "mean_p": round(float(p[m].mean()), 3), "flowing": k, "rate": round(k / n, 3), "ci95": [round(v, 3) for v in wilson(k, n)]})
    per_band = {}
    for name, m in (
        ("likely", p >= bands["likely_from"]),
        ("unlikely", p <= bands["unlikely_to"]),
        ("uncertain", (p < bands["likely_from"]) & (p > bands["unlikely_to"])),
    ):
        n, k = int(m.sum()), int(y[m].sum())
        per_band[name] = {"n": n, "flowing": k, "rate": round(k / n, 3) if n else None, "ci95": [round(v, 3) for v in wilson(k, n)] if n else None}
    b_model = brier(y, p)
    b_logit = brier(y, np.array([r["p_logistic"] for r in preds]))
    b_base = brier(y, np.array([r["p_base"] for r in preds]))
    b_month = brier(y, np.array([r["p_month"] for r in preds]))
    return {
        "method": "leave-one-station-out over the ONDE observations in Corsica",
        "features": list(features),
        "n_estimators": n_estimators,
        "n_observations": len(preds),
        "n_stations": len({r["station"] for r in preds}),
        "flowing_rate": round(float(y.mean()), 3),
        "brier": round(b_model, 4),
        "brier_logistic_regression": round(b_logit, 4),
        "brier_base_rate": round(b_base, 4),
        "brier_month_rate": round(b_month, 4),
        "brier_skill_vs_month": round(1 - b_model / b_month, 3),
        "reliability": bins,
        "bands": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in bands.items()},
        "per_band": per_band,
    }


def reliability_svg(summary: dict, path: Path) -> None:
    """A small, dependency-free reliability diagram (reads in light and dark)."""
    W, H, M = 420, 420, 56
    size = W - 2 * M

    def sx(v: float) -> float:
        return M + v * size

    def sy(v: float) -> float:
        return H - M - v * size

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="sans-serif" font-size="12">',
        "<style>.t{fill:#1b2320}.g{stroke:#c8d1cd}.a{stroke:#1b2320}.d{fill:#1d6b86}.l{stroke:#1d6b86}"
        "@media (prefers-color-scheme: dark){.t{fill:#e4ebe8}.g{stroke:#3a4643}.a{stroke:#e4ebe8}.d{fill:#63b7d4}.l{stroke:#63b7d4}}</style>",
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="none"/>',
    ]
    for v in (0, 0.25, 0.5, 0.75, 1):
        parts.append(f'<line class="g" x1="{sx(v)}" y1="{sy(0)}" x2="{sx(v)}" y2="{sy(1)}" stroke-width="1"/>')
        parts.append(f'<line class="g" x1="{sx(0)}" y1="{sy(v)}" x2="{sx(1)}" y2="{sy(v)}" stroke-width="1"/>')
        parts.append(f'<text class="t" x="{sx(v)}" y="{sy(0) + 18}" text-anchor="middle">{v:g}</text>')
        parts.append(f'<text class="t" x="{sx(0) - 8}" y="{sy(v) + 4}" text-anchor="end">{v:g}</text>')
    parts.append(f'<line class="a" x1="{sx(0)}" y1="{sy(0)}" x2="{sx(1)}" y2="{sy(1)}" stroke-width="1" stroke-dasharray="4 4"/>')
    pts = [(b["mean_p"], b["rate"], b["n"]) for b in summary["reliability"]]
    if pts:
        poly = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b, _ in pts)
        parts.append(f'<polyline class="l" points="{poly}" fill="none" stroke-width="2"/>')
        nmax = max(n for _, _, n in pts)
        for a, b, n in pts:
            r = 3 + 7 * math.sqrt(n / nmax)
            parts.append(f'<circle class="d" cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="{r:.1f}"><title>{n} observations, {b:.0%} flowing</title></circle>')
    parts.append(f'<text class="t" x="{W / 2}" y="{H - 14}" text-anchor="middle">predicted probability of visible flow</text>')
    parts.append(f'<text class="t" transform="translate(16 {H / 2}) rotate(-90)" text-anchor="middle">observed share flowing</text>')
    parts.append(f'<text class="t" x="{M}" y="22">Leave-one-station-out, {summary["n_observations"]} observations, {summary["n_stations"]} streams</text>')
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts) + "\n", encoding="utf-8")
