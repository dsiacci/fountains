"""A local web page for one ride: find the water points, check them on photos, score them, take a GPX.

`fountains serve` starts a small web server on this computer and nothing else:
the track is read here and stays here. What leaves the machine is the same as
for the command line: the bounding box of the ride (roads, streams and IGN
water features in it), the positions of the water points (street photos,
elevation, rain forecast), and the map tiles your browser loads.

The flow:
1. Upload a GPX exported from your app or bike computer.
2. The water points the maps know (OpenStreetMap, IGN BD TOPO) near the track
   are listed at once. In the background, clues for unmapped fountains are
   gathered along the track (IGN springs, wash houses, stream crossings), and
   every point gets the Panoramax street photos around it, ranked by OWLv2.
3. You decide for each point: fountain, not a fountain, or not sure. You can
   decide without a photo, and add a fountain you know by clicking the map.
4. TabPFN gives a band (likely, uncertain, unlikely) to the fountains you kept,
   for the day of your ride; cafés and shops are listed on long stretches
   without a likely fountain.
5. Download one GPX with the track and the waypoints for your bike computer.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
import shutil
import threading
import traceback
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import DATA_DIR, DISCLAIMER, in_corsica
from .geo import Polyline
from .gpx import Track, course_gpx, read_gpx
from .views import aerial_url, plan_url

STATIC = Path(__file__).with_name("static")
MAX_BODY = 25 << 20
VERDICTS = ("fountain", "not", "unsure")
CROP_NAME = re.compile(r"^[A-Za-z0-9_-]+\.jpg$")
SEEN = 0.15  # an OWLv2 box this sure is worth a look (the Marato fountain scored 0.20, empty roadsides under 0.10)
STOP_SYMBOL = {"fuel station": "Gas Station", "supermarket": "Shopping Center", "small shop": "Convenience Store", "bakery": "Restaurant"}


def climb_m(ele: list[float | None] | None, step: float = 5.0) -> float | None:
    """Metres climbed, counting rises of at least `step` metres (GPS noise ignored)."""
    vals = [e for e in ele or [] if e is not None]
    if len(vals) < 2:
        return None
    total, ref = 0.0, vals[0]
    for e in vals[1:]:
        if e - ref >= step:
            total += e - ref
            ref = e
        elif e < ref:
            ref = e
    return round(total)


def simplify(points: list[tuple[float, float]], keep: int = 1500) -> list[list[float]]:
    k = max(1, math.ceil(len(points) / keep))
    out = [[round(a, 5), round(b, 5)] for a, b in points[::k]]
    if points and out[-1] != [round(points[-1][0], 5), round(points[-1][1], 5)]:
        out.append([round(points[-1][0], 5), round(points[-1][1], 5)])
    return out


def point_label(p: dict) -> str:
    if p.get("name"):
        return p["name"]
    if p["origin"] == "clue":
        return "fountain" if p.get("verdict") == "fountain" else p["kind"]
    return p["kind"]


def where(p: dict) -> tuple[float, float]:
    """Where the waypoint goes.

    OpenStreetMap positions are kept. For a place found from clues, or an
    IGN-only point mapped away from the road, a fountain confirmed on a street
    photo is put where that photo was taken: on the road, a few metres from
    it (IGN positions can be tens of metres off, and a clue may lie in the
    woods).
    """
    away = p["origin"] == "clue" or (p["source"] == "IGN" and (p.get("road_m") or 0) > 15)
    if p.get("verdict") == "fountain" and away and p.get("photos"):
        ph = p["photos"][min(p.get("chosen", 0), len(p["photos"]) - 1)]
        return ph["lat"], ph["lon"]
    if p["origin"] == "clue" and p.get("road_lat") is not None:
        return p["road_lat"], p["road_lon"]
    return p["lat"], p["lon"]


class Ride:
    """One uploaded track and everything decided about it, saved in its own folder."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.lock = threading.RLock()
        self.cancelled = False
        self.track: Track = read_gpx(folder / "track.gpx")
        self.line = Polyline(self.track.points)
        self.state = json.loads((folder / "state.json").read_text(encoding="utf-8"))

    @property
    def crops(self) -> Path:
        return self.folder / "crops"

    def save(self) -> None:
        with self.lock:
            self.state["version"] += 1
            tmp = self.folder / "state.json.tmp"
            tmp.write_text(json.dumps(self.state, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.folder / "state.json")

    def point(self, pid: str) -> dict:
        p = next((p for p in self.state["points"] if p["id"] == pid), None)
        if p is None:
            raise KeyError(pid)
        return p

    def log(self, job: str, msg: str) -> None:
        with self.lock:
            j = self.state["jobs"].setdefault(job, {})
            j["message"] = msg
            j.setdefault("log", []).append(f"{dt.datetime.now():%H:%M:%S} {msg}")
            j["log"] = j["log"][-40:]
            self.save()


def new_ride(gpx: bytes, filename: str, rides_dir: Path, data_dir: Path, max_detour_m: float) -> Ride:
    """Store an uploaded GPX and list the water points the maps know near it."""
    text = gpx.decode("utf-8", errors="replace")
    if "<!DOCTYPE" in text or "<!ENTITY" in text:
        raise ValueError("this GPX declares a DTD or entities, which a GPX never needs")
    rid = hashlib.sha256(gpx).hexdigest()[:12]
    folder = rides_dir / rid
    if (folder / "state.json").exists():
        return Ride(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "track.gpx").write_bytes(gpx)
    try:
        try:
            track = read_gpx(folder / "track.gpx")
        except Exception as e:
            raise ValueError(f"could not read a track in this file: {e}") from e
        sample = track.points[:: max(1, len(track.points) // 50)] + [track.points[-1]]
        if not all(in_corsica(lat, lon) for lat, lon in sample):
            raise ValueError("this tool only knows Corsica (its rain gauges and streams): the track leaves the island")
    except ValueError:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    line = Polyline(track.points)
    name = track.name if track.name and track.name != "track" else Path(filename).stem
    state = {
        "id": rid, "version": 0, "name": name, "file": Path(filename).name,
        "length_km": round(line.length_m / 1000, 1), "climb_m": climb_m(track.ele),
        "line": simplify(track.points), "points": [], "jobs": {}, "score": None,
        "settings": {"max_detour_m": max_detour_m},
    }
    state["points"] = known_points(line, data_dir, max_detour_m)
    (folder / "state.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    return Ride(folder)


def known_points(line: Polyline, data_dir: Path, max_detour_m: float) -> list[dict]:
    """OpenStreetMap and IGN water points within max_detour_m of the track (straight line).

    They start as fountains: a mapper or IGN recorded them. The rider looks at
    the photo and changes it when the map is wrong.
    """
    from .score import water_points

    out = []
    for f in water_points(data_dir):
        d, s = line.nearest(f["lat"], f["lon"], within_m=max_detour_m)
        if math.isinf(d) or f.get("access") in ("private", "no"):
            continue
        tags = {k: f[k] for k in ("access", "seasonal", "drinking_water", "last_confirmed") if f.get(k)}
        out.append({
            "origin": "map", "ref": f["ref"], "source": f["source"], "name": f.get("name", ""), "kind": f["kind"],
            "lat": f["lat"], "lon": f["lon"], "km": round(s / 1000, 2), "off_track_m": round(d), "tags": tags,
            "verdict": "fountain", "prefilled": True,
        })
    out.sort(key=lambda p: p["km"])
    for i, p in enumerate(out, 1):
        p["id"] = f"M{i:02d}"
        _views(p)
    return out


def _views(p: dict) -> None:
    p.setdefault("photos", [])
    p.setdefault("photo_state", "waiting")
    p.setdefault("chosen", 0)
    p["plan"] = plan_url(p["lat"], p["lon"], width_m=300, px=(480, 270))
    p["aerial"] = aerial_url(p["lat"], p["lon"], width_m=120, px=(480, 270))


class App:
    """The server's state: the current ride, the background jobs and their options."""

    def __init__(self, work_dir: Path, data_dir: Path = DATA_DIR, vision: bool = True, max_pictures: int = 4,
                 corridor_m: float = 100.0, max_detour_m: float = 250.0, allowed_hosts: tuple[str, ...] = ()):
        self.work_dir = work_dir
        self.rides_dir = work_dir / "rides"
        self.rides_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = data_dir
        self.vision = vision
        self.max_pictures = max_pictures
        self.corridor_m = corridor_m
        self.max_detour_m = max_detour_m
        self.allowed_hosts = {"localhost", "127.0.0.1", "::1", *allowed_hosts}
        self.ride: Ride | None = None
        self.threads: dict[str, threading.Thread] = {}
        cur = work_dir / "current"
        if cur.exists() and (self.rides_dir / cur.read_text().strip() / "state.json").exists():
            self.ride = Ride(self.rides_dir / cur.read_text().strip())
            if self._unfinished(self.ride):
                self.start("look")

    def _unfinished(self, ride: Ride) -> bool:
        """Whether the look job left work: points without photos yet, or photos to rank now that the model is on."""
        pts = ride.state["points"]
        return (not ride.state["jobs"].get("look", {}).get("finished")
                or any(p["photo_state"] in ("waiting", "looking") for p in pts)
                or (self.vision and any(p.get("photos") and not p.get("ranked") for p in pts)))

    # --- jobs -----------------------------------------------------------------

    def busy(self, kind: str, ride: Ride | None = None) -> bool:
        ride = ride or self.ride
        t = self.threads.get((ride.state["id"], kind)) if ride else None
        return t is not None and t.is_alive()

    def start(self, kind: str, **kw) -> None:
        ride = self._need_ride()
        if self.busy(kind, ride):
            raise RuntimeError(f"already running: {kind}")
        target = {"look": self._look, "score": self._score}[kind]

        def run():
            with ride.lock:
                ride.state["jobs"][kind] = {"running": True, "started": dt.datetime.now().isoformat(timespec="seconds"), "log": []}
                ride.save()
            try:
                target(ride, **kw)
                with ride.lock:
                    ride.state["jobs"][kind].update(running=False, finished=dt.datetime.now().isoformat(timespec="seconds"), error="")
            except BaseException as e:  # noqa: BLE001 - shown on the page, the server keeps running
                traceback.print_exc()
                with ride.lock:
                    ride.state["jobs"][kind].update(running=False, error=str(e) or e.__class__.__name__)
            ride.save()

        t = threading.Thread(target=run, name=f"fountains-{kind}", daemon=True)
        self.threads[(ride.state["id"], kind)] = t
        t.start()

    def _look(self, ride: Ride) -> None:
        """Clues along the track, then street photos for every point (best crops first)."""
        from . import discover as d

        log = lambda m: ride.log("look", m)  # noqa: E731
        corridor = None
        log("Fetching the paved roads and streams around the ride from OpenStreetMap (only its bounding box is sent)...")
        try:
            corridor = d.Corridor(ride.line, pad_m=max(400.0, self.max_detour_m + 150))
        except Exception as e:  # noqa: BLE001
            log(f"OpenStreetMap roads unavailable ({e}): no clues this time; photos are still searched.")
        if corridor is not None and not ride.state.get("clues_done"):
            log("Looking for clues of unmapped fountains: IGN springs, wash houses and water points, stream crossings...")
            with ride.lock:
                known = [p for p in ride.state["points"] if p["origin"] == "map"]
            clues = d.clues_along(corridor, known, self.corridor_m)
            new = []
            for c in clues:
                if c.known:
                    continue
                dist, s = ride.line.nearest(c.road_lat, c.road_lon)
                p = {
                    "origin": "clue", "ref": c.id, "source": "clue", "name": c.name, "kind": ", ".join(c.kinds),
                    "lat": c.lat, "lon": c.lon, "km": round(s / 1000, 2), "off_track_m": round(dist),
                    "road": c.road, "road_lat": c.road_lat, "road_lon": c.road_lon, "road_m": c.road_m,
                    "tags": {}, "verdict": None, "prefilled": False, "id": c.id,
                }
                _views(p)
                new.append(p)
            with ride.lock:
                ride.state["points"] = [p for p in ride.state["points"] if p["origin"] != "clue"] + new
                ride.state["clues_done"] = True
            log(f"{len(new)} places to check from clues ({len(clues) - len(new)} more were next to a mapped point).")
        if corridor is not None:
            with ride.lock:
                for p in ride.state["points"]:
                    if p["origin"] == "map" and "road_m" not in p:
                        dist, road, rlat, rlon, _ = corridor.road_at(p["lat"], p["lon"], 150.0)
                        if not math.isinf(dist):
                            p.update(road=road, road_lat=rlat, road_lon=rlon, road_m=round(dist))
        vision = self.vision
        if vision:
            try:
                import transformers  # noqa: F401
            except ImportError:
                vision = False
                log("No vision model installed (pip install transformers pillow): photos are shown unranked.")
        with ride.lock:
            todo = sorted((p for p in ride.state["points"] if p["photo_state"] not in ("done", "none")), key=lambda p: p["km"])
        for n, p in enumerate(todo, 1):
            if ride.cancelled:
                return
            log(f"Street photos {n}/{len(todo)}: km {p['km']:.1f} {point_label(p)}")
            with ride.lock:
                p["photo_state"] = "looking"
                ride.save()
            try:
                photos = self._fetch(ride, p, corridor)
                with ride.lock:
                    p.update(photos=photos, photo_state="done" if photos else "none", ranked=False, chosen=0)
            except Exception as e:  # noqa: BLE001 - one point without photos never stops the others
                with ride.lock:
                    p["photo_state"] = "error"
                    p["photo_error"] = str(e)[:200]
            ride.save()
        if vision:
            with ride.lock:
                todo = sorted((p for p in ride.state["points"] if p.get("photos") and not p.get("ranked")), key=lambda p: p["km"])
            for n, p in enumerate(todo, 1):
                if ride.cancelled:
                    return
                log(f"Vision model (OWLv2) {n}/{len(todo)}: km {p['km']:.1f} {point_label(p)}, {len(p['photos'])} views")
                self._rank(ride, p)
                ride.save()
        log("Done: every point has its photos, or a note saying there is none within 100 m."
            + ("" if vision else " The photos are not ranked: start again with the vision model to rank them."))

    def _fetch(self, ride: Ride, p: dict, corridor) -> list[dict]:
        """Every crop around the point: the ones aimed at its mapped position first, then by picture."""
        from PIL import Image

        from . import discover as d

        center = (p["road_lat"], p["road_lon"]) if p.get("road_lat") is not None else (p["lat"], p["lon"])
        target = (p["lat"], p["lon"]) if p["origin"] == "map" or (p.get("road_m") or 0) > 15 else None
        crops = d.look_around(p["id"], *center, corridor, ride.crops, target=target, max_pictures=self.max_pictures)
        crops.sort(key=lambda c: c.aim != "toward the point")  # stable: pictures stay closest first
        out = []
        for c in crops:
            with Image.open(c.path) as im:
                w, h = im.size
            out.append({"file": Path(c.path).name, "picture": c.picture, "owl": 0.0, "box": [], "w": w, "h": h, "date": c.date,
                        "viewer": c.viewer, "license": c.license, "lat": c.lat, "lon": c.lon, "aim": c.aim, "heading": c.heading})
        return out

    def _rank(self, ride: Ride, p: dict) -> None:
        """OWLv2 on every crop of the point; the best crop of each picture first, then the others."""
        from . import discover as d

        crops = [d.Crop(clue=p["id"], picture=x["picture"], date=x["date"], lat=x["lat"], lon=x["lon"], heading=x["heading"],
                        aim=x["aim"], path=str(ride.crops / x["file"]), viewer=x["viewer"], license=x["license"]) for x in p["photos"]]
        d.owl_score(crops)
        for x, c in zip(p["photos"], crops):
            x["owl"], x["box"] = c.owl, c.box
        ranked = sorted(p["photos"], key=lambda x: (x["owl"], x["aim"] == "toward the point"), reverse=True)
        first, seen = [], set()
        for x in ranked:
            if x["picture"] not in seen:
                first.append(x)
                seen.add(x["picture"])
        with ride.lock:
            p["photos"] = first + [x for x in ranked if x not in first]
            p["ranked"] = True
            p["chosen"] = 0

    def _score(self, ride: Ride, day: dt.date, start: dt.time | None, flat_kmh: float, climb_mh: float, gap_km: float) -> None:
        from .report import reason, waypoint_name
        from .score import score_track

        with ride.lock:
            chosen = [p for p in ride.state["points"] if p["verdict"] == "fountain"]
            pts = []
            for p in chosen:
                lat, lon = where(p)
                src = {"clue": "found from clues", "you": "added by you"}.get(p["origin"], p["source"])
                pts.append(dict(p.get("tags", {}), ref=p["id"], source=src, name=p.get("name", ""), lat=lat, lon=lon,
                                kind="fountain" if p["origin"] != "map" else p["kind"]))
        ride.log("score", f"Scoring {len(pts)} fountains for {day:%d/%m/%Y} (rain gauges, forecast, then TabPFN)...")
        scored, gaps, meta = score_track(ride.track, day, max_detour_m=self.max_detour_m, start=start, flat_kmh=flat_kmh,
                                         climb_mh=climb_mh, gap_km=gap_km, photos=False, points=pts,
                                         data_dir=self.data_dir, log=lambda m: ride.log("score", m))
        by_id = {p["id"]: p for p in chosen}
        fountains = []
        for f in scored:
            p = by_id.get(f.ref)
            if p and p["origin"] == "clue":
                ph = p["photos"][min(p.get("chosen", 0), len(p["photos"]) - 1)] if p.get("photos") else None
                f.notes.append("not on any map: found from clues" + (f", confirmed by you on a street photo of {ph['date']}" if ph else ", confirmed by you"))
            elif p and p["origin"] == "you":
                f.notes.append("added by you")
            row = asdict(f)
            row.update(reason=reason(f), waypoint=waypoint_name(f))
            fountains.append(row)
        with ride.lock:
            ride.state["score"] = {"meta": meta, "fountains": fountains, "gaps": [dict(asdict(g), length_km=g.length_km) for g in gaps]}
        ride.log("score", f"Scored {len(fountains)} fountains; {len(gaps)} stretch(es) of {gap_km:.0f} km or more without a likely one.")

    # --- what the page asks for -------------------------------------------------

    def upload(self, gpx: bytes, filename: str) -> None:
        if self.ride is not None and self.ride.state["id"] == hashlib.sha256(gpx).hexdigest()[:12]:
            return  # the same file again: keep the decisions already made
        ride = new_ride(gpx, filename, self.rides_dir, self.data_dir, self.max_detour_m)
        if self.ride is not None:
            self.ride.cancelled = True
        self.ride = ride
        (self.work_dir / "current").write_text(self.ride.state["id"])
        if not self.busy("look") and self._unfinished(self.ride):
            self.start("look")

    def decide(self, pid: str, verdict: str | None = None, photo: int | None = None) -> None:
        ride = self._need_ride()
        with ride.lock:
            p = ride.point(pid)
            if verdict is not None:
                if verdict not in VERDICTS:
                    raise ValueError(f"verdict must be one of {', '.join(VERDICTS)}")
                p["verdict"] = verdict
                p["prefilled"] = False
            if photo is not None:
                if not 0 <= photo < max(1, len(p.get("photos", []))):
                    raise ValueError("no such photo")
                p["chosen"] = photo
            ride.save()

    def add(self, lat: float, lon: float, name: str) -> str:
        ride = self._need_ride()
        d, s = ride.line.nearest(lat, lon)
        if d > self.max_detour_m:
            raise ValueError(f"this place is {d:.0f} m from the track; points within {self.max_detour_m:.0f} m are scored")
        with ride.lock:
            n = 1 + sum(1 for p in ride.state["points"] if p["origin"] == "you")
            p = {"id": f"U{n:02d}", "origin": "you", "ref": "", "source": "you", "name": name.strip()[:60], "kind": "fountain",
                 "lat": lat, "lon": lon, "km": round(s / 1000, 2), "off_track_m": round(d), "tags": {},
                 "verdict": "fountain", "prefilled": False, "photo_state": "none"}
            _views(p)
            p["photo_state"] = "none"
            ride.state["points"].append(p)
            ride.save()
        return p["id"]

    def score(self, body: dict) -> None:
        ride = self._need_ride()
        day = dt.date.fromisoformat(body["date"])
        start = dt.time.fromisoformat(body["start"]) if body.get("start") else None
        kw = dict(day=day, start=start, flat_kmh=float(body.get("flat_kmh") or 20), climb_mh=float(body.get("climb_mh") or 500),
                  gap_km=float(body.get("gap_km") or 10))
        with ride.lock:
            ride.state["settings"].update(date=day.isoformat(), start=body.get("start") or "", flat_kmh=kw["flat_kmh"],
                                          climb_mh=kw["climb_mh"], gap_km=kw["gap_km"])
            ride.save()
        self.start("score", **kw)

    def _need_ride(self) -> Ride:
        if self.ride is None:
            raise ValueError("upload a GPX first")
        return self.ride

    def gpx(self) -> tuple[str, str]:
        """The track with a waypoint for every fountain kept, every point to check, and the shops on long stretches."""
        ride = self._need_ride()
        with ride.lock:
            st = ride.state
            scored = {f["ref"]: f for f in (st.get("score") or {}).get("fountains", [])}
            wpts = []
            for p in sorted(st["points"], key=lambda p: p["km"]):
                lat, lon = where(p)
                label = point_label(p)
                if p["verdict"] == "fountain":
                    f = scored.get(p["id"])
                    if f:
                        desc = f["reason"] + "".join(f". Note: {n}" for n in f["notes"])
                        wpts.append({"lat": lat, "lon": lon, "name": f"{f['band']}: {label}"[:30], "desc": desc, "type": "Water"})
                    else:
                        wpts.append({"lat": lat, "lon": lon, "name": f"fountain: {label}"[:30], "desc": "not scored for a day yet", "type": "Water"})
                elif p["verdict"] == "unsure":
                    wpts.append({"lat": lat, "lon": lon, "name": f"check: {label}"[:30], "type": "Water",
                                 "desc": "not confirmed: look on the spot, and do not count on it"})
            for g in (st.get("score") or {}).get("gaps", []):
                for s in g["stops"]:
                    state = {True: "open", False: "closed", None: "hours?"}[s["open_then"]] if s["eta"] else "hours?"
                    wpts.append({"lat": s["lat"], "lon": s["lon"], "name": f"{s['kind']} {state}: {s['name']}"[:30],
                                 "desc": " ".join(x for x in (s["name"], s["opening_hours"], f"around {s['eta']}" if s["eta"] else "") if x),
                                 "sym": STOP_SYMBOL.get(s["kind"], "Restaurant"), "type": "Food"})
            day = (st.get("score") or {}).get("meta", {}).get("day", "")
            name = re.sub(r"[^A-Za-z0-9_-]+", "-", st["name"]).strip("-")[:60] or "ride"
        fname = f"{name}-fountains{'-' + day if day else ''}.gpx"
        return fname, course_gpx(ride.track, wpts, metadata_desc=DISCLAIMER)

    def public_state(self) -> dict:
        from .report import ATTRIBUTION, TRANSFER_NOTE

        out = {"ride": None, "disclaimer": DISCLAIMER, "note": TRANSFER_NOTE, "attribution": ATTRIBUTION, "seen": SEEN}
        if self.ride is None:
            return out
        with self.ride.lock:
            st = json.loads(json.dumps(self.ride.state))
        for kind in ("look", "score"):
            if kind in st["jobs"]:
                st["jobs"][kind]["running"] = self.busy(kind)
        return dict(out, ride=st)


class Handler(BaseHTTPRequestHandler):
    server_version = "fountains"
    app: App

    def log_message(self, fmt, *args):  # noqa: D102 - quiet: the page shows what happens
        pass

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").strip().lower()
        host = host[1:].split("]")[0] if host.startswith("[") else host.split(":")[0]
        return host in self.app.allowed_hosts

    def _send(self, code: int, body: bytes, ctype: str, headers: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self):  # noqa: N802 - http.server API
        if not self._host_ok():
            return self._send(403, b"forbidden host", "text/plain")
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8",
                              {"Content-Security-Policy": "frame-ancestors 'none'"})
        if url.path == "/api/state":
            st = self.app.public_state()
            v = parse_qs(url.query).get("v", [None])[0]
            if v is not None and st["ride"] and str(st["ride"]["version"]) == v:
                return self._json({"same": True})
            return self._json(st)
        if url.path.startswith("/crops/"):
            name = url.path[len("/crops/"):]
            ride = self.app.ride
            if ride is None or not CROP_NAME.match(name) or not (ride.crops / name).is_file():
                return self._send(404, b"not found", "text/plain")
            return self._send(200, (ride.crops / name).read_bytes(), "image/jpeg")
        if url.path == "/api/gpx":
            try:
                fname, body = self.app.gpx()
            except ValueError as e:
                return self._json({"error": str(e)}, 400)
            return self._send(200, body.encode(), "application/gpx+xml", {"Content-Disposition": f'attachment; filename="{fname}"'})
        return self._send(404, b"not found", "text/plain")

    def do_POST(self):  # noqa: N802 - http.server API
        if not self._host_ok():
            return self._send(403, b"forbidden host", "text/plain")
        # JSON only: a page from another site cannot send it without the browser asking first (CORS).
        if not (self.headers.get("Content-Type") or "").startswith("application/json"):
            return self._json({"error": "send JSON"}, 415)
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            return self._json({"error": "file too large"}, 413)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
            path = urlparse(self.path).path
            if path == "/api/ride":
                self.app.upload(body["gpx"].encode("utf-8"), body.get("filename") or "ride.gpx")
            elif path == "/api/decide":
                self.app.decide(body["id"], body.get("verdict"), body.get("photo"))
            elif path == "/api/add":
                self.app.add(float(body["lat"]), float(body["lon"]), body.get("name") or "")
            elif path == "/api/score":
                self.app.score(body)
            elif path == "/api/look":
                self.app.start("look")
            else:
                return self._json({"error": "unknown action"}, 404)
        except (KeyError, ValueError, RuntimeError, TypeError) as e:
            return self._json({"error": str(e)}, 400)
        return self._json(self.app.public_state())


def serve(host: str = "127.0.0.1", port: int = 8765, work_dir: Path | None = None, **options) -> None:
    from . import cache_dir

    app = App(work_dir or cache_dir("serve"), **options)
    handler = type("BoundHandler", (Handler,), {"app": app})
    httpd = ThreadingHTTPServer((host, port), handler)
    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0", "::") else host
    print(f"fountains: open http://{shown}:{port} in your browser (Ctrl+C to stop)")
    print(DISCLAIMER)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
