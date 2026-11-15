import json
import pickle

import numpy as np

import refresh_tiles
from mars_analog.bundle import load_bundle, save_bundle
from mars_analog.sources import _load_cache, _save_cache, synthetic_tiles
from mars_analog.tile import Tile
from mars_analog.water import snow_fraction, water_reason


def test_snow_index_tells_snow_from_rock_and_water():
    n = (8, 8)
    snow = snow_fraction(np.full(n, 0.8), np.full(n, 0.7), np.full(n, 0.1))    # bright, dark in SWIR
    rock = snow_fraction(np.full(n, 0.2), np.full(n, 0.25), np.full(n, 0.3))   # SWIR brighter than green
    water = snow_fraction(np.full(n, 0.05), np.full(n, 0.02), np.full(n, 0.01))  # dark in near-infrared
    assert snow == 1.0 and rock == 0.0 and water == 0.0


def test_mostly_snow_is_left_out_with_the_right_reason():
    t = synthetic_tiles("earth", 2, seed=2)[1]
    t.snow_fraction = 0.7
    assert water_reason(t) == "mostly snow or ice"
    t.snow_fraction = 0.3
    assert water_reason(t) is None


def test_old_cached_squares_without_the_field_still_work(tmp_path):
    t = synthetic_tiles("earth", 1, seed=2)[0]
    del t.__dict__["snow_fraction"]                  # as if pickled before the field existed
    path = tmp_path / "cache.pkl"
    with open(path, "wb") as f:
        pickle.dump({"k": t}, f)
    old = _load_cache(str(path))["k"]
    assert old.snow_fraction is None and water_reason(old) is None


def test_bundle_keeps_the_snow_share(tmp_path):
    earth = synthetic_tiles("earth", 4, seed=2)
    earth[0].snow_fraction = 0.25
    save_bundle(str(tmp_path), synthetic_tiles("mars", 2, seed=1), earth)
    _, e = load_bundle(str(tmp_path))
    assert e[0].snow_fraction == 0.25 and e[1].snow_fraction is None


def test_refresh_clears_only_the_named_regions(tmp_path):
    tiles = synthetic_tiles("earth", 3, seed=2)
    tiles[0].lat, tiles[0].lon = -77.5, 161.5        # Antarctic Dry Valleys
    tiles[1].lat, tiles[1].lon = -24.0, 15.0         # Namib
    tiles[2].lat, tiles[2].lon = 65.0, -16.5         # Iceland highlands
    cache = {("v", t.lat, t.lon): t for t in tiles}
    path = tmp_path / "cache.pkl"
    _save_cache(str(path), cache)
    (tmp_path / "config.json").write_text(json.dumps({"earth": {"cache": str(path)}}))
    assert refresh_tiles.main(["--config", str(tmp_path / "config.json")]) == 0
    left = _load_cache(str(path))
    assert [t.lat for t in left.values()] == [-24.0]
    assert refresh_tiles.main(["--config", str(tmp_path / "config.json"), "--region", "Atlantis"]) == 1


def test_typed_glacier_coordinates_get_a_snow_message(tmp_path, monkeypatch):
    import os
    from streamlit.testing.v1 import AppTest
    earth = synthetic_tiles("earth", 40, seed=2)
    earth[0].lat, earth[0].lon = -77.25, 161.0          # rock
    earth[1].lat, earth[1].lon = -77.75, 161.5          # glacier
    earth[1].snow_fraction = 0.9
    save_bundle(str(tmp_path), synthetic_tiles("mars", 8, seed=1), earth)
    monkeypatch.setenv("MARS_BUNDLE_DIR", str(tmp_path))
    monkeypatch.setenv("MARS_MODEL_PATH", str(tmp_path / "none.json"))
    app = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
    at = AppTest.from_file(app, default_timeout=60)
    at.run()
    radio = lambda label: [r for r in at.sidebar.radio if r.label == label][0]
    number = lambda start: [n for n in at.sidebar.number_input if n.label.startswith(start)][0]
    radio("A place on").set_value("Earth").run()
    radio("Choose it").set_value("By coordinates").run()
    number("Latitude").set_value(-77.75).run()
    number("Longitude").set_value(161.5).run()
    assert not at.exception
    assert any("mostly snow or ice" in i.value and "km away" in i.value for i in at.info)
