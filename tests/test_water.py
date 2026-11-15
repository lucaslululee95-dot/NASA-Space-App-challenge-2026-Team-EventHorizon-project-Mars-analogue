import os

import numpy as np
from streamlit.testing.v1 import AppTest

from mars_analog.bundle import save_bundle
from mars_analog.pipeline import run, run_from_earth
from mars_analog.sources import synthetic_tiles
from mars_analog.tile import Tile
from mars_analog.water import is_water, water_reason

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def _sea(name, lat, lon, ndvi=-0.3, image=True):
    rng = np.random.default_rng(0)
    rgb = rng.random((3, 64, 64)) * 0.1 if image else np.zeros((3, 64, 64))
    return Tile(name=name, planet="earth", lat=lat, lon=lon, tile_km=10,
                dem=np.zeros((64, 64)), rgb=rgb, ndvi=ndvi, chem={"ferric_index": 0.9})


def test_three_signs_of_water():
    land = synthetic_tiles("earth", 3, seed=2)[1]
    assert water_reason(land) is None
    assert water_reason(_sea("a", 0, 0, ndvi=0.05)) == "mostly at sea level"
    assert water_reason(_sea("b", 0, 0, ndvi=0.05, image=False)) == "no satellite image"
    wet = synthetic_tiles("earth", 3, seed=2)[1]
    wet.ndvi = -0.3
    assert water_reason(wet) == "plant cover reads as water"
    assert not is_water(synthetic_tiles("mars", 1, seed=1)[0])


def test_sea_is_never_a_match():
    mars = synthetic_tiles("mars", 8, seed=1)
    earth = synthetic_tiles("earth", 30, seed=2) + [_sea("earth_sea_1", -24.0, 14.0)]
    res = run(mars, earth, target=(17.0, 77.0), top_k=100)
    assert "earth_sea_1" not in set(res["earth_matches"]["earth_tile"])
    try:
        run_from_earth(mars, earth, "earth_sea_1")
        assert False, "a sea place shouldn't be usable as a starting point"
    except ValueError:
        pass


def test_typed_sea_coordinates_get_a_clear_message(tmp_path, monkeypatch):
    earth = synthetic_tiles("earth", 60, seed=2)
    earth[0].lat, earth[0].lon = -24.0, 15.0                # land
    earth.append(_sea("earth_sea_1", -24.0, 14.0))          # sea, 100 km west
    save_bundle(str(tmp_path), synthetic_tiles("mars", 8, seed=1), earth)
    monkeypatch.setenv("MARS_BUNDLE_DIR", str(tmp_path))
    monkeypatch.setenv("MARS_MODEL_PATH", str(tmp_path / "none.json"))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    radio = lambda label: [r for r in at.sidebar.radio if r.label == label][0]
    number = lambda start: [n for n in at.sidebar.number_input if n.label.startswith(start)][0]
    radio("A place on").set_value("Earth").run()
    places = [o for s in at.sidebar.selectbox for o in s.options]
    assert not any("14.00° E" in p for p in places)          # sea isn't offered in the lists
    radio("Choose it").set_value("By coordinates").run()
    number("Latitude").set_value(-24.0).run()
    number("Longitude").set_value(14.05).run()
    assert not at.exception
    assert any("looks like sea" in i.value and "km away" in i.value for i in at.info)
    number("Longitude").set_value(14.95).run()               # near the land place
    assert any(c.value.startswith("Using our nearest place") for c in at.sidebar.caption)


def test_summer_only_near_the_poles():
    from mars_analog.geo import summer_months
    assert summer_months(75.4) == (6, 8)          # Devon Island
    assert summer_months(-77.5) == (12, 2)        # Dry Valleys, wrapping round the new year
    assert summer_months(-24.0) is None           # Atacama: the whole year
    assert summer_months(65.1) == (6, 8)          # Iceland highlands


def test_analog_regions_are_named():
    from mars_analog.regions import region_of
    assert region_of(-77.5, 161.5) == "Antarctic Dry Valleys"
    assert region_of(75.5, -89.5) == "Devon Island, Canadian Arctic"
    assert region_of(35.5, -111.5) == "San Francisco Volcanic Field, Arizona"


def test_catalogue_sites_reach_the_new_regions():
    import make_label_sheet
    (site, *_), km = make_label_sheet.nearest_site(75.5, -89.5)
    assert site.startswith("Haughton") and km < 75
    (site, *_), km = make_label_sheet.nearest_site(-77.25, 161.0)
    assert site.startswith("Wright Valley") and km < 75
