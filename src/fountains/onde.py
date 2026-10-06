"""Training labels: summer observations of small Corsican streams (réseau ONDE).

Source: Office français de la biodiversité (OFB), Observatoire national des
étiages (ONDE), through the Hub'Eau API « Écoulement des cours d'eau »,
https://hubeau.eaufrance.fr/page/api-ecoulement. Licence Ouverte 2.0 / Etalab.

Each observation says what an agent saw at a fixed point of a stream:
- 1  « Écoulement visible » (used when the network did not split it, see below)
- 1a « Écoulement visible acceptable »
- 1f « Écoulement visible faible »
- 2  « Écoulement non visible » (water in pools, no current)
- 3  « Assec » (dry bed)

The target is "flowing water visible" (1, 1a, 1f) against "no visible flow"
(2, 3). Two reasons: a fountain fills a bottle only if water flows, and in
Corsica the network recorded only « écoulement visible », without the
acceptable / weak split, for most observations from 2016 to 2019 and in 2022;
the visible / not visible split is the only one that means the same thing in
every year.
"""

from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path

HUBEAU_OBS = "https://hubeau.eaufrance.fr/api/v1/ecoulement/observations"
USER_AGENT = "fountains/0.1 (+https://github.com/dsiacci/fountains)"

FLOWING = {"1", "1a", "1f"}
NOT_FLOWING = {"2", "3"}


def fetch_observations(departments: tuple[str, ...] = ("2A", "2B")) -> list[dict]:
    rows = []
    for dep in departments:
        url = f"{HUBEAU_OBS}?code_departement={dep}&size=20000&format=json"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.load(r)
        if data.get("next"):
            raise RuntimeError("Hub'Eau returned more than one page; add paging")
        rows.extend(data["data"])
    return rows


def label(code: str | None) -> int | None:
    if code in FLOWING:
        return 1
    if code in NOT_FLOWING:
        return 0
    return None


def write_observations_csv(rows: list[dict], path: Path) -> int:
    """Keep the usable observations (a date and a known class), sorted."""
    keep = []
    for o in rows:
        y = label(o.get("code_ecoulement"))
        if y is None or not o.get("date_observation"):
            continue
        keep.append(
            {
                "station": o["code_station"],
                "station_name": o["libelle_station"],
                "stream": o.get("libelle_cours_eau") or "",
                "lat": round(o["latitude"], 6),
                "lon": round(o["longitude"], 6),
                "date": o["date_observation"],
                "code": o["code_ecoulement"],
                "flowing": y,
            }
        )
    keep.sort(key=lambda r: (r["station"], r["date"]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(keep[0]))
        w.writeheader()
        w.writerows(keep)
    return len(keep)


def read_observations_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["lat"], r["lon"], r["flowing"] = float(r["lat"]), float(r["lon"]), int(r["flowing"])
    return rows
