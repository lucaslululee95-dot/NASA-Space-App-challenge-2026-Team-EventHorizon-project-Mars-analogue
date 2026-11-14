import numpy as np

from bundle_data import earth_keys, select_earth
from mars_analog.bundle import load_bundle, save_bundle
from mars_analog.pipeline import run
from mars_analog.sources import synthetic_tiles


def test_bundle_round_trip_keeps_numbers_and_labels(tmp_path):
    mars = synthetic_tiles("mars", 12, seed=1)
    earth = synthetic_tiles("earth", 30, seed=2)
    mars[0].chem = {"Fe": np.nan, "Fe_pct": np.nan}
    save_bundle(str(tmp_path), mars, earth)
    mars2, earth2 = load_bundle(str(tmp_path))
    assert [t.name for t in mars2] == [t.name for t in mars]
    assert np.allclose(earth2[0].dem, earth[0].dem, rtol=1e-6, atol=1e-3)
    assert earth2[0].ndvi == earth[0].ndvi and mars2[0].ndvi is None
    assert earth2[0].chem == earth[0].chem
    assert np.isnan(mars2[0].chem["Fe"])


def test_results_from_bundle_match_the_original(tmp_path):
    mars = synthetic_tiles("mars", 30, seed=1)
    earth = synthetic_tiles("earth", 90, seed=2)
    save_bundle(str(tmp_path), mars, earth)
    mars2, earth2 = load_bundle(str(tmp_path))
    a = run(mars, earth, target=(17.0, 77.0), top_k=10)["earth_matches"]
    b = run(mars2, earth2, target=(17.0, 77.0), top_k=10)["earth_matches"]
    assert np.allclose(a["final_score"], b["final_score"], atol=1e-4)


def test_select_earth_finds_configured_tiles_and_reports_missing():
    keys = earth_keys([[0, 0, 1, 1]], 0.5, 10, 64, 2023)     # 3 x 3 = 9 tiles
    cache = {k: "tile" for k in keys[:7]}
    tiles, missing = select_earth(cache, keys)
    assert len(tiles) == 7 and len(missing) == 2
