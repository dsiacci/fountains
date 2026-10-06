"""Command line: `fountains score my-ride.gpx --date 2026-10-11`."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from . import DATA_DIR, DISCLAIMER, __version__


def cmd_score(a: argparse.Namespace) -> None:
    from .gpx import read_gpx
    from .report import text_table, write_outputs
    from .score import score_track

    track = read_gpx(a.track)
    day = dt.date.fromisoformat(a.date) if a.date else dt.date.today()
    start = dt.time.fromisoformat(a.start) if a.start else None
    log = (lambda *_: None) if a.quiet else (lambda m: print(m, file=sys.stderr))
    scored, gaps, meta = score_track(
        track, day, max_detour_m=a.max_detour, use_forecast=not a.no_forecast, n_estimators=a.n_estimators,
        start=start, flat_kmh=a.flat_kmh, climb_mh=a.climb_mh, gap_km=a.gap_km, log=log,
    )
    print(text_table(scored, meta, gaps))
    stem = Path(a.track).stem + f"-{day.isoformat()}"
    for p in write_outputs(scored, meta, Path(a.out), stem, with_html=a.html, gaps=gaps):
        log(f"wrote {p}")


def cmd_calibrate(a: argparse.Namespace) -> None:
    from .calibrate import leave_one_station_out, reliability_svg, summarise
    from .model import FEATURE_SETS, read_table

    rows = read_table(DATA_DIR / "training.csv")
    features = FEATURE_SETS[a.features]
    cache = Path(a.cache) if a.cache else None
    preds = leave_one_station_out(rows, features, a.n_estimators, cache_path=cache, log=lambda m: print(m, file=sys.stderr))
    summary = summarise(preds, features, a.n_estimators)
    summary["feature_set"] = a.features
    out = Path(a.out)
    out.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    if a.svg:
        reliability_svg(summary, Path(a.svg))
    print(json.dumps({k: summary[k] for k in ("feature_set", "brier", "brier_logistic_regression", "brier_month_rate", "brier_base_rate", "bands", "per_band")}, indent=1))


def cmd_build_data(a: argparse.Namespace) -> None:
    from . import build_data as b

    steps = {"osm": b.refresh_osm, "ign": b.refresh_ign, "stops": b.refresh_stops, "onde": b.refresh_onde, "normals": b.build_normals, "training": b.build_training}
    chosen = list(steps) if a.all else a.steps
    if not chosen:
        raise SystemExit("name steps (osm, ign, stops, onde, normals, training) or pass --all")
    for s in chosen:
        steps[s]()


def cmd_check_memories(a: argparse.Namespace) -> None:
    from .memories import check_memories

    print(check_memories(Path(a.memories), n_estimators=a.n_estimators))


def cmd_trim(a: argparse.Namespace) -> None:
    from .gpx import read_gpx, trim_track, write_track

    t = trim_track(read_gpx(a.track), a.start, a.end)
    write_track(t, a.out)
    print(f"wrote {a.out}: {len(t.points)} points, first {a.start:.0f} m and last {a.end:.0f} m removed, no metadata kept")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="fountains", description=f"Will the fountains along your ride be running? {DISCLAIMER}")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="score the drinking-water points along a GPX track")
    s.add_argument("track", help="GPX file exported from your app or bike computer")
    s.add_argument("--date", help="day of the ride, YYYY-MM-DD (default: today)")
    s.add_argument("--max-detour", type=float, default=250.0, help="one-way detour along roads and paths, in metres (default 250)")
    s.add_argument("--no-forecast", action="store_true", help="use measured rain only (days without data count as dry)")
    s.add_argument("--n-estimators", type=int, default=4, help="TabPFN ensemble size (default 4, as in the calibration)")
    s.add_argument("--out", default="out", help="output directory (default ./out)")
    s.add_argument("--html", action="store_true", help="also write a map page")
    s.add_argument("--start", help="start time, HH:MM: gives passing times and checks opening hours of cafés and shops")
    s.add_argument("--flat-kmh", type=float, default=20.0, help="pace on the flat for passing times (default 20 km/h)")
    s.add_argument("--climb-mh", type=float, default=500.0, help="metres climbed per hour for passing times (default 500)")
    s.add_argument("--gap-km", type=float, default=15.0, help="list cafés and shops on stretches this long without a likely fountain (default 15 km)")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_score)

    c = sub.add_parser("calibrate", help="leave-one-station-out validation and band thresholds")
    c.add_argument("--features", default="base", choices=["base", "anomaly"])
    c.add_argument("--n-estimators", type=int, default=4)
    c.add_argument("--cache", help="JSONL file to resume an interrupted run")
    c.add_argument("--out", default=str(DATA_DIR / "calibration.json"))
    c.add_argument("--svg", default=None, help="write the reliability diagram here")
    c.set_defaults(func=cmd_calibrate)

    b = sub.add_parser("build-data", help="rebuild data/ from the open sources")
    b.add_argument("steps", nargs="*", choices=["osm", "ign", "stops", "onde", "normals", "training"])
    b.add_argument("--all", action="store_true")
    b.set_defaults(func=cmd_build_data)

    m = sub.add_parser("check-memories", help="compare the scores with remembered fountain states")
    m.add_argument("memories", help="CSV with ref (node/123 or IGN PAIHYDRO...), date, state (flowing, weak or dry)")
    m.add_argument("--n-estimators", type=int, default=4)
    m.set_defaults(func=cmd_check_memories)

    t = sub.add_parser("trim", help="cut the start and end of a track before sharing it")
    t.add_argument("track")
    t.add_argument("out")
    t.add_argument("--start", type=float, default=1500.0, help="metres removed at the start (default 1500)")
    t.add_argument("--end", type=float, default=1500.0, help="metres removed at the end (default 1500)")
    t.set_defaults(func=cmd_trim)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
