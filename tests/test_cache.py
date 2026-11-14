import sys
from unittest import mock

import numpy as np

from mars_analog import sources
from mars_analog.tile import Tile


def _tile(lat, lon):
    return Tile(name=f"earth_{lat}_{lon}", planet="earth", lat=lat, lon=lon, tile_km=10,
                dem=np.zeros((64, 64)), rgb=np.zeros((3, 64, 64)), ndvi=0.05,
                chem={"ferric_index": 1.2})


def test_cache_round_trip(tmp_path):
    path = tmp_path / "sub" / "earth.pkl"
    sources._save_cache(str(path), {"k": _tile(-23.0, -69.0)})
    loaded = sources._load_cache(str(path))
    assert loaded["k"].lat == -23.0 and loaded["k"].dem.shape == (64, 64)


def test_cached_tiles_are_used_without_calling_earth_engine(tmp_path, monkeypatch):
    centers = [(-23.0, -69.0), (-23.5, -69.0)]
    cache = {}
    for lat, lon in centers:
        key = (sources.EARTH_CACHE_VERSION, round(lat, 4), round(lon, 4), 10, 64, 2023)
        cache[key] = _tile(lat, lon)
    path = tmp_path / "earth.pkl"
    sources._save_cache(str(path), cache)
    monkeypatch.setitem(sys.modules, "ee", mock.MagicMock())   # any real Earth Engine call would fail here
    tiles = sources.earth_tiles_from_gee(centers, 10, 64, project="x", year=2023, cache_path=str(path))
    assert len(tiles) == 2
    assert {t.name for t in tiles} == {f"earth_{lat}_{lon}" for lat, lon in centers}
