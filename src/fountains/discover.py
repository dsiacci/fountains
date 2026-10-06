"""Find fountains that no map records: a bundle of clues, checked on street photos.

1. Clues, from open data only: IGN BD TOPO fountains, springs, captured
   springs, wash houses and water points near a paved road, and the places
   where a stream crosses a paved road (a spout is often set where the slope
   brings water to the road).
2. Street photos: for each clue, the Panoramax 360° pictures along the road
   within SWEEP_M metres, cropped toward both roadsides and toward the clue.
3. Open vision models, Apache 2.0: OWLv2 looks for a fountain, a spout or a
   trough in every crop and draws a box; the best crops are shown first.
   (SigLIP, a zero-shot classifier, was tried first and is kept for the
   `discover` command, but it barely separated a fountain from a roadside.)

Along a whole ride (`clues_along`), only the roads and streams in a corridor
around the track are kept, and the clues must sit on a road that passes
within `corridor_m` of the track.

The output is a list of photos for a person to look at. Nothing is added to a
map: a rider checks on the spot, and may then map it by hand.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import cache_dir
from .geo import Polyline, Projection, SegmentGrid, haversine_m, project_on_segment
from .osm import SMALL_QUERY, overpass
from .views import bearing_deg

UA = {"User-Agent": "fountains/0.1 (+https://github.com/dsiacci/fountains)"}
WFS = "https://data.geopf.fr/wfs/ows"
PANORAMAX_SEARCH = "https://api.panoramax.xyz/api/search"
PAVED = {"trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "living_street"}
CLUE_RANGE_M = {"Fontaine": 300, "Lavoir": 150, "Point d'eau": 100, "Source captée": 60, "Source": 60}
MERGE_M = 60.0
SWEEP_M = 100.0
MAX_PICTURES = 8  # per clue, spread along the road
SPACING_M = 12.0  # minimum distance between two pictures of one clue

POSITIVE = [
    "a stone water fountain by a road",
    "a public drinking fountain with a water spout",
    "a stone trough with running water",
    "a village fountain with a basin",
]
NEGATIVE = [
    "a road with trees and bushes",
    "a stone retaining wall by a road",
    "a rocky embankment",
    "a house wall",
    "a car on a road",
    "a guard rail",
]
DETECT = ["a water fountain", "a water spout", "a stone water trough", "a water tap"]
BOTTOM_BAND = 0.9  # boxes reaching below 90 % of the crop height sit on the camera car


@dataclass
class Clue:
    id: str
    kinds: list[str]
    name: str
    lat: float  # the clue itself (may be off the road)
    lon: float
    road: str
    road_lat: float  # nearest point on a paved road
    road_lon: float
    road_m: float
    known: str = ""  # an OpenStreetMap drinking-water point already there


@dataclass
class Crop:
    clue: str
    picture: str
    date: str
    lat: float
    lon: float
    heading: float  # compass direction of the crop centre
    aim: str  # "roadside" or "toward the clue"
    path: str
    viewer: str
    license: str
    siglip: float = 0.0
    owl: float = 0.0
    box: list[float] = field(default_factory=list)


def _get_json(url: str, data: bytes | None = None, timeout: int = 120) -> dict:
    req = urllib.request.Request(url, data=data, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def ign_hydro(bbox: tuple[float, float, float, float], page: int = 5000) -> list[dict]:
    """IGN BD TOPO fountains, wash houses, water points and springs in the box."""
    s, w, n, e = bbox
    natures = ",".join("'" + k.replace("'", "''") + "'" for k in CLUE_RANGE_M)
    feats: list[dict] = []
    while True:
        q = {
            "SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature", "TYPENAMES": "BDTOPO_V3:detail_hydrographique",
            "OUTPUTFORMAT": "application/json", "COUNT": str(page), "STARTINDEX": str(len(feats)), "SORTBY": "cleabs",
            "CQL_FILTER": f"nature IN ({natures}) AND BBOX(geometrie,{s},{w},{n},{e},'urn:ogc:def:crs:EPSG::4326')",
        }
        got = _get_json(f"{WFS}?{urllib.parse.urlencode(q)}")["features"]
        feats += got
        if len(got) < page:
            return feats


def osm_ways(bbox: tuple[float, float, float, float]) -> dict:
    """Paved roads and streams in the box, from Overpass (cached on disk: the box is all that is sent)."""
    s, w, n, e = bbox
    roads = "|".join(sorted(PAVED))
    query = SMALL_QUERY + f"""
(way["highway"~"^({roads})$"]({s:.5f},{w:.5f},{n:.5f},{e:.5f}); way["waterway"~"^(stream|ditch|drain|river)$"]({s:.5f},{w:.5f},{n:.5f},{e:.5f}););
out tags geom qt;"""
    path = cache_dir("overpass") / f"box-{hashlib.sha256(query.encode()).hexdigest()[:20]}.json"
    if path.exists():
        return json.loads(path.read_text())
    data = overpass(query)
    path.write_text(json.dumps(data))
    return data


def _segments(data: dict, proj: Projection):
    """Paved roads and streams as planar segments, from ways with inline geometry or with their nodes."""
    nodes = {e["id"]: (e["lat"], e["lon"]) for e in data["elements"] if e["type"] == "node"}
    roads, streams = [], []
    for e in data["elements"]:
        if e["type"] != "way":
            continue
        if "geometry" in e:
            pts = [proj.xy(g["lat"], g["lon"]) for g in e["geometry"] if g]
        else:
            pts = [proj.xy(*nodes[n]) for n in e.get("nodes", []) if n in nodes]
        segs = list(zip(pts, pts[1:]))
        tags = e.get("tags", {})
        if tags.get("highway") in PAVED:
            roads.append((tags.get("ref") or tags.get("name") or tags["highway"], segs))
        elif "waterway" in tags:
            streams.append((tags.get("name", ""), segs))
    return roads, streams


def _intersection(a, b, c, d):
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = a, b, c, d
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1)) if 0 <= t <= 1 and 0 <= u <= 1 else None


def find_clues(bbox: tuple[float, float, float, float], known_points: list[dict]) -> list[Clue]:
    """Clues near paved roads in the box, merged when closer than MERGE_M along the road."""
    proj = Projection((bbox[0] + bbox[2]) / 2)
    roads, streams = _segments(osm_ways(bbox), proj)

    def nearest_road(xy):
        best = (math.inf, "", xy)
        for name, segs in roads:
            for a, b in segs:
                d, _, q = project_on_segment(xy, a, b)
                if d < best[0]:
                    best = (d, name, q)
        return best

    raw = []
    for f in ign_hydro(bbox):
        p = f["properties"]
        lim = CLUE_RANGE_M.get(p.get("nature"))
        if not lim:
            continue
        lon, lat = f["geometry"]["coordinates"][:2]
        d, name, q = nearest_road(proj.xy(lat, lon))
        if d <= lim:
            raw.append(("IGN " + p["nature"].lower(), p.get("toponyme") or "", lat, lon, name, q, d))
    for sname, ssegs in streams:
        for rname, rsegs in roads:
            for a, b in ssegs:
                for c, d in rsegs:
                    x = _intersection(a, b, c, d)
                    if x:
                        lat, lon = proj.latlon(*x)
                        raw.append(("stream crossing", sname, lat, lon, rname, x, 0.0))
    order = ["IGN fontaine", "IGN lavoir", "IGN source captée", "IGN point d'eau", "IGN source", "stream crossing"]
    raw.sort(key=lambda r: order.index(r[0]))
    clues: list[Clue] = []
    for kind, name, lat, lon, rname, q, d in raw:
        rlat, rlon = proj.latlon(*q)
        same = next((c for c in clues if haversine_m(c.road_lat, c.road_lon, rlat, rlon) < MERGE_M), None)
        if same:
            if kind not in same.kinds:
                same.kinds.append(kind)
            same.name = same.name or name
            continue
        clues.append(Clue(id=f"C{len(clues) + 1:02d}", kinds=[kind], name=name, lat=lat, lon=lon, road=rname,
                          road_lat=rlat, road_lon=rlon, road_m=round(d)))
    for c in clues:
        k = next((p for p in known_points if haversine_m(p["lat"], p["lon"], c.road_lat, c.road_lon) < 120), None)
        if k:
            c.known = k.get("ref") or k.get("osm_id", "")
    return clues


def road_bearing(roads, proj: Projection, lat: float, lon: float) -> float | None:
    p = proj.xy(lat, lon)
    best = (math.inf, None)
    for _, segs in roads:
        for a, b in segs:
            d, _, _ = project_on_segment(p, a, b)
            if d < best[0]:
                best = (d, (a, b))
    if best[1] is None or best[0] > 25:
        return None
    (ax, ay), (bx, by) = best[1]
    return (math.degrees(math.atan2(bx - ax, by - ay)) + 360) % 360


def pictures_near(lat: float, lon: float, radius_m: float = SWEEP_M) -> list[dict]:
    """360° Panoramax pictures within radius_m, spread along the road (closest first)."""
    url = f"{PANORAMAX_SEARCH}?place_position={lon:.6f},{lat:.6f}&place_distance=0-{int(radius_m)}&limit=100"
    feats = _get_json(url, timeout=60).get("features", [])
    pics = []
    for f in feats:
        fov = (f["properties"].get("pers:interior_orientation") or {}).get("field_of_view")
        if fov != 360 or f["properties"].get("view:azimuth") is None:
            continue
        plon, plat = f["geometry"]["coordinates"][:2]
        pics.append((haversine_m(plat, plon, lat, lon), plat, plon, f))
    pics.sort(key=lambda x: x[0])
    chosen = []
    for d, plat, plon, f in pics:
        if all(haversine_m(plat, plon, q[1], q[2]) >= SPACING_M for q in chosen):
            chosen.append((d, plat, plon, f))
        if len(chosen) >= MAX_PICTURES:
            break
    return [f for _, _, _, f in chosen]


def crop_equirect(img, heading_rel: float, fov_deg: float = 100.0, up_deg: float = 12.0, down_deg: float = 30.0):
    """A window of an equirectangular panorama, `heading_rel` degrees right of its centre."""
    from PIL import Image

    W, H = img.size
    x_c = (0.5 + heading_rel / 360.0) * W
    half = fov_deg / 360.0 * W / 2
    top, bottom = int((90 - up_deg) / 180 * H), int((90 + down_deg) / 180 * H)
    x0, x1 = int(round(x_c - half)), int(round(x_c + half))
    if x0 >= 0 and x1 <= W:
        return img.crop((x0, top, x1, bottom))
    # wrap around the seam
    x0 %= W
    x1 %= W
    left = img.crop((x0, top, W, bottom))
    right = img.crop((0, top, x1, bottom))
    out = Image.new("RGB", (left.width + right.width, bottom - top))
    out.paste(left, (0, 0))
    out.paste(right, (left.width, 0))
    return out


def _picture(f: dict, keep_hd: bool = True):
    """The HD equirectangular image of a Panoramax picture (cached unless keep_hd is False)."""
    from PIL import Image

    hd = f["assets"].get("hd", f["assets"]["sd"])["href"]
    cache = cache_dir("panoramax-hd") / f"{f['id']}.jpg"
    if cache.exists():
        return Image.open(cache).convert("RGB")
    with urllib.request.urlopen(urllib.request.Request(hd, headers=UA), timeout=120) as r:
        raw = r.read()
    if keep_hd:
        cache.write_bytes(raw)
    with tempfile.SpooledTemporaryFile(max_size=64 << 20) as tmp:
        tmp.write(raw)
        tmp.seek(0)
        return Image.open(tmp).convert("RGB")


def _crops_of_picture(f: dict, prefix: str, headings: list[tuple[str, float]], out_dir: Path, keep_hd: bool = True) -> list[Crop]:
    p = f["properties"]
    plon, plat = f["geometry"]["coordinates"][:2]
    az = float(p["view:azimuth"])
    img = _picture(f, keep_hd)
    out = []
    for aim, heading in headings:
        rel = ((heading - az + 540) % 360) - 180
        path = out_dir / f"{prefix}_{f['id'][:8]}_{int(heading):03d}.jpg"
        crop_equirect(img, rel).save(path, quality=85)
        out.append(Crop(clue=prefix, picture=f["id"], date=(p.get("datetime") or "")[:10], lat=plat, lon=plon,
                        heading=round(heading), aim=aim, path=str(path),
                        viewer=f"https://api.panoramax.xyz/?focus=pic&map=19/{plat:.6f}/{plon:.6f}&pic={f['id']}&xyz={round(heading)}/0/30",
                        license=p.get("license") or ""))
    return out


def make_crops(clues: list[Clue], bbox, out_dir: Path, max_pictures: int = MAX_PICTURES, log=print) -> list[Crop]:
    proj = Projection((bbox[0] + bbox[2]) / 2)
    roads, _ = _segments(osm_ways(bbox), proj)
    out_dir.mkdir(parents=True, exist_ok=True)
    crops: list[Crop] = []
    for c in clues:
        pics = pictures_near(c.road_lat, c.road_lon)[:max_pictures]
        log(f"{c.id} {'/'.join(c.kinds)} {c.name}: {len(pics)} pictures")
        for f in pics:
            plon, plat = f["geometry"]["coordinates"][:2]
            az = float(f["properties"]["view:azimuth"])
            rb = road_bearing(roads, proj, plat, plon)
            headings = [("roadside", (rb + 90) % 360), ("roadside", (rb + 270) % 360)] if rb is not None else [("roadside", az), ("roadside", (az + 180) % 360)]
            if c.road_m > 15:
                headings.append(("toward the clue", bearing_deg(plat, plon, c.lat, c.lon)))
            crops += _crops_of_picture(f, c.id, headings, out_dir)
    return crops


# --- Along a whole ride --------------------------------------------------------

class Corridor:
    """Paved roads and streams within `pad_m` of a track, indexed for fast lookups.

    Only the bounding box of the track (plus `pad_m`) is sent to Overpass and
    IGN, never the track itself.
    """

    def __init__(self, line: Polyline, pad_m: float = 400.0):
        self.line, self.proj, self.pad = line, line.proj, pad_m
        s, w, n, e = line.bbox
        dlat = pad_m / 111_195
        dlon = dlat / math.cos(math.radians((s + n) / 2))
        self.bbox = (s - dlat, w - dlon, n + dlat, e + dlon)
        roads, streams = _segments(osm_ways(self.bbox), self.proj)
        self.roads = [(name, a, b) for name, segs in roads for a, b in segs if self._near(a, b)]
        self.streams = [(name, a, b) for name, segs in streams for a, b in segs if self._near(a, b)]
        self.grid = SegmentGrid([(a, b) for _, a, b in self.roads], cell_m=100.0)

    def _near(self, a, b) -> bool:
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        return not math.isinf(self.line.nearest_xy(mid, within_m=self.pad + math.dist(a, b) / 2)[0])

    def nearest_road(self, xy, max_m: float) -> tuple[float, str, tuple[float, float], float | None]:
        """(distance, road name, closest point, road bearing) of the closest paved road within max_m."""
        best = (math.inf, "", xy, None)
        for i in self.grid.near(xy, max_m):
            name, a, b = self.roads[i]
            d, _, q = project_on_segment(xy, a, b)
            if d < best[0]:
                best = (d, name, q, (math.degrees(math.atan2(b[0] - a[0], b[1] - a[1])) + 360) % 360)
        return best if best[0] <= max_m else (math.inf, "", xy, None)

    def road_at(self, lat: float, lon: float, max_m: float) -> tuple[float, str, float, float, float | None]:
        """(distance, road name, lat, lon, bearing) of the closest paved road point, or distance inf."""
        d, name, q, bearing = self.nearest_road(self.proj.xy(lat, lon), max_m)
        rlat, rlon = self.proj.latlon(*q)
        return d, name, rlat, rlon, bearing


def clues_along(corridor: Corridor, known_points: list[dict], corridor_m: float = 100.0) -> list[Clue]:
    """Clues on paved roads that pass within `corridor_m` of the track, merged and ordered along it."""
    line, proj = corridor.line, corridor.proj
    raw = []
    for f in ign_hydro(corridor.bbox):
        p = f["properties"]
        lim = CLUE_RANGE_M.get(p.get("nature"))
        if not lim:
            continue
        lon, lat = f["geometry"]["coordinates"][:2]
        d, name, rlat, rlon, _ = corridor.road_at(lat, lon, lim)
        if not math.isinf(d):
            raw.append(("IGN " + p["nature"].lower(), p.get("toponyme") or "", lat, lon, name, rlat, rlon, d))
    for sname, a, b in corridor.streams:
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        for i in corridor.grid.near(mid, math.dist(a, b) / 2 + 1):
            rname, c, d = corridor.roads[i]
            x = _intersection(a, b, c, d)
            if x:
                lat, lon = proj.latlon(*x)
                raw.append(("stream crossing", sname, lat, lon, rname, lat, lon, 0.0))
    order = ["IGN fontaine", "IGN lavoir", "IGN source captée", "IGN point d'eau", "IGN source", "stream crossing"]
    raw.sort(key=lambda r: order.index(r[0]))
    clues: list[Clue] = []
    for kind, name, lat, lon, rname, rlat, rlon, d in raw:
        if math.isinf(line.nearest(rlat, rlon, within_m=corridor_m)[0]):
            continue
        same = next((c for c in clues if haversine_m(c.road_lat, c.road_lon, rlat, rlon) < MERGE_M), None)
        if same:
            if kind not in same.kinds:
                same.kinds.append(kind)
            same.name = same.name or name
            continue
        clues.append(Clue(id="", kinds=[kind], name=name, lat=lat, lon=lon, road=rname, road_lat=rlat, road_lon=rlon, road_m=round(d)))
    for c in clues:
        k = next((p for p in known_points if haversine_m(p["lat"], p["lon"], c.road_lat, c.road_lon) < 120), None)
        if k:
            c.known = k.get("ref") or k.get("osm_id", "")
    clues.sort(key=lambda c: line.nearest(c.road_lat, c.road_lon)[1])
    for i, c in enumerate(clues, 1):
        c.id = f"C{i:02d}"
    return clues


def look_around(prefix: str, lat: float, lon: float, corridor: Corridor | None, out_dir: Path, target: tuple[float, float] | None = None,
                max_pictures: int = 4, spacing_m: float = 15.0, keep_hd: bool = False) -> list[Crop]:
    """Crops of the 360° pictures within SWEEP_M of (lat, lon) that cover every direction.

    Four 100° crops per picture, set diagonally to the road, see both roadsides
    ahead and behind, so a fountain a few metres off the road is in frame
    whatever the spacing of the pictures; a fifth looks toward `target` (a
    mapped position away from the road).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    feats = pictures_near(lat, lon)
    chosen = []
    for f in feats:
        plon, plat = f["geometry"]["coordinates"][:2]
        if all(haversine_m(plat, plon, *g["geometry"]["coordinates"][1::-1]) >= spacing_m for g in chosen):
            chosen.append(f)
        if len(chosen) >= max_pictures:
            break
    crops: list[Crop] = []
    for f in chosen:
        plon, plat = f["geometry"]["coordinates"][:2]
        rb = corridor.road_at(plat, plon, 25.0)[4] if corridor is not None else None
        base = rb if rb is not None else float(f["properties"]["view:azimuth"])
        headings = [("roadside", (base + k) % 360) for k in (45, 135, 225, 315)]
        if target and haversine_m(plat, plon, *target) > 15:
            headings.append(("toward the point", bearing_deg(plat, plon, *target)))
        crops += _crops_of_picture(f, prefix, headings, out_dir, keep_hd=keep_hd)
    return crops


def score_crops(crops: list[Crop], owl_top: int | None = None, log=print) -> None:
    """SigLIP scores every crop; OWLv2 looks for a fountain, a spout or a trough.

    On the first test (Marato), SigLIP's zero-shot scores barely separated the
    fountain from the roadside, while OWLv2 boxed it in the best crop: OWLv2
    looks at every crop unless `owl_top` limits it to SigLIP's best.
    """
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    torch.set_num_threads(max(1, torch.get_num_threads()))
    proc = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    model = AutoModel.from_pretrained("google/siglip-base-patch16-224").eval()
    texts = POSITIVE + NEGATIVE
    with torch.no_grad():
        for i in range(0, len(crops), 16):
            batch = crops[i : i + 16]
            images = [Image.open(c.path).convert("RGB") for c in batch]
            inputs = proc(text=texts, images=images, padding="max_length", return_tensors="pt")
            logits = model(**inputs).logits_per_image  # images x texts
            for c, row in zip(batch, logits):
                c.siglip = round(float(row[: len(POSITIVE)].max() - row[len(POSITIVE) :].max()), 3)
    log(f"SigLIP scored {len(crops)} crops")
    oproc, omodel = owl_models()
    best = sorted(crops, key=lambda c: c.siglip, reverse=True)[:owl_top] if owl_top else crops
    owl_score(best, oproc, omodel)
    log(f"OWLv2 looked at the {len(best)} best crops")


_OWL = None


def owl_models():
    """The OWLv2 processor and model, loaded once per process."""
    global _OWL
    if _OWL is None:
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        _OWL = (Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble"),
                Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").eval())
    return _OWL


def owl_score(crops: list[Crop], oproc=None, omodel=None) -> None:
    """Best OWLv2 box per crop, ignoring boxes that touch the bottom edge.

    The bottom of a car-mounted 360° picture shows the car itself (bonnet,
    roof rails), which OWLv2 sometimes takes for a trough; a fountain by the
    road sits above that band.

    One crop per forward pass. Stacking two crops in one image halves the
    time (OWLv2 pads every image to a square, and a crop is 2.4 times wider
    than tall), but it changes the scores: on the Marato crops, the view of
    the fountain fell from 0.20 to 0.13 while empty roadsides rose.
    """
    import torch
    from PIL import Image

    if oproc is None or omodel is None:
        oproc, omodel = owl_models()
    post = getattr(oproc, "post_process_grounded_object_detection", None) or oproc.post_process_object_detection
    with torch.no_grad():
        for c in crops:
            img = Image.open(c.path).convert("RGB")
            out = omodel(**oproc(text=[DETECT], images=img, return_tensors="pt"))
            size = max(img.size)
            res = post(out, threshold=0.0, target_sizes=torch.tensor([[size, size]]))[0]
            c.owl, c.box = 0.0, []
            for k in res["scores"].argsort(descending=True).tolist():
                box = [float(v) for v in res["boxes"][k]]
                if box[3] < BOTTOM_BAND * img.height:
                    c.owl = round(float(res["scores"][k]), 3)
                    c.box = [round(v) for v in box]
                    break


def rescore(out_dir: Path, log=print) -> list[Crop]:
    """Run OWLv2 again on the crops of an earlier search (results.json in out_dir)."""
    data = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    names = {f for f in Crop.__dataclass_fields__}
    crops = [Crop(**{k: v for k, v in c.items() if k in names}) for c in data["crops"]]
    owl_score(crops)
    log(f"OWLv2 scored {len(crops)} crops again")
    data["crops"] = [asdict(c) for c in crops]
    (out_dir / "results.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    return crops


def write_report(clues: list[Clue], crops: list[Crop], out_dir: Path, title: str) -> Path:
    """A local page with the clues and their crops, best first, for a person to look at."""
    import html as h

    by_clue: dict[str, list[Crop]] = {}
    for c in crops:
        by_clue.setdefault(c.clue, []).append(c)
    rows = []
    ranked = sorted(clues, key=lambda c: max((x.siglip for x in by_clue.get(c.id, [])), default=-9), reverse=True)
    for c in ranked:
        cs = sorted(by_clue.get(c.id, []), key=lambda x: (x.owl, x.siglip), reverse=True)[:4]
        figs = "".join(
            f'<figure><img src="{h.escape(Path(x.path).name)}" loading="lazy" alt="street photo"><figcaption>SigLIP {x.siglip:+.2f}'
            f'{f" · OWLv2 {x.owl:.2f}" if x.owl else ""} · {h.escape(x.aim)} · {h.escape(x.date)} · <a href="{h.escape(x.viewer)}">Panoramax</a></figcaption></figure>'
            for x in cs
        )
        known = f" · already mapped: {h.escape(c.known)}" if c.known else ""
        rows.append(f"<section><h2>{c.id} · {h.escape(', '.join(c.kinds))} {h.escape(c.name)}</h2><p>{h.escape(c.road)}, "
                    f"{c.road_m:.0f} m from the road · {c.road_lat:.5f}, {c.road_lon:.5f}{known}</p>{figs or '<p>No 360° street photo within 100 m.</p>'}</section>")
    page = out_dir / "index.html"
    page.write_text(
        f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{h.escape(title)}</title>
<style>body{{font:15px/1.4 system-ui,sans-serif;margin:0 auto;max-width:72rem;padding:16px}}section{{border-top:1px solid #ccc;padding:8px 0}}
figure{{display:inline-block;margin:4px;width:340px;vertical-align:top}}img{{width:340px;border-radius:6px}}figcaption{{font-size:12px;color:#555}}</style>
<h1>{h.escape(title)}</h1><p>Clues from open data, checked on Panoramax street photos by open vision models (SigLIP, OWLv2). A list for a person to look at: nothing here is a confirmed fountain.</p>
{''.join(rows)}""",
        encoding="utf-8",
    )
    (out_dir / "results.json").write_text(json.dumps({"clues": [asdict(c) for c in clues], "crops": [asdict(c) for c in crops]}, indent=1), encoding="utf-8")
    return page
