import math
import urllib.parse

import pytest

from fountains.views import aerial_url, bearing_deg, pick, plan_url, relative_deg


def test_bearing_cardinal_directions():
    assert bearing_deg(42.0, 9.0, 42.1, 9.0) == pytest.approx(0, abs=0.01)
    assert bearing_deg(42.0, 9.0, 42.0, 9.1) == pytest.approx(90, abs=0.1)
    assert bearing_deg(42.0, 9.0, 41.9, 9.0) == pytest.approx(180, abs=0.01)
    assert bearing_deg(42.0, 9.0, 42.0, 8.9) == pytest.approx(270, abs=0.1)


@pytest.mark.parametrize("target,azimuth,expected", [(265, 84, -179), (74, 85, -11), (10, 350, 20), (350, 10, -20), (180, 0, 180)])
def test_relative_angle_wraps(target, azimuth, expected):
    assert relative_deg(target, azimuth) == pytest.approx(expected)


def feature(fid, lon, lat, fov, azimuth=90):
    return {
        "id": fid,
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": {"datetime": "2025-04-01T13:29:42+00:00", "view:azimuth": azimuth, "license": "etalab-2.0",
                       "pers:interior_orientation": {"field_of_view": fov}},
        "assets": {"sd": {"href": f"https://example.org/{fid}/sd.jpg"}, "thumb": {"href": f"https://example.org/{fid}/thumb.jpg"}},
        "links": [{"rel": "via", "href": "https://panoramax.ign.fr"}],
    }


def test_pick_prefers_a_360_picture_then_the_closest():
    lat, lon = 41.82036, 8.87251
    near_flat = feature("flat", lon + 0.0001, lat, 72)
    far_360 = feature("pano", lon + 0.0003, lat, 360, azimuth=84)
    best = pick([near_flat, far_360], lat, lon)
    assert best["id"] == "pano" and best["fov"] == 360
    assert best["bearing"] == 270 and best["relative"] == -174
    assert "pic=pano" in best["viewer"] and "xyz=270/0/30" in best["viewer"]
    assert pick([], lat, lon) is None


def test_wms_views_are_centred_on_the_point():
    lat, lon = 41.82036, 8.87251
    for url, width in ((plan_url(lat, lon), 400), (aerial_url(lat, lon), 160)):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        x0, y0, x1, y1 = map(float, q["BBOX"][0].split(","))
        R = 6378137.0
        cx = math.degrees((x0 + x1) / 2 / R)
        cy = math.degrees(2 * math.atan(math.exp((y0 + y1) / 2 / R)) - math.pi / 2)
        assert cx == pytest.approx(lon, abs=1e-6) and cy == pytest.approx(lat, abs=1e-6)
        assert (x1 - x0) * math.cos(math.radians(lat)) == pytest.approx(width, rel=1e-6)
