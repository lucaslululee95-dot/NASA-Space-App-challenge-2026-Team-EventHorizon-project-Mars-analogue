import numpy as np
import pandas as pd

from mars_analog.chemistry import chemistry_scores_for_earth
from mars_analog.geo import km_between, nearest_tile
from mars_analog.pipeline import run, run_from_earth
from mars_analog.sources import synthetic_tiles

MARS = synthetic_tiles("mars", 30, seed=1)
EARTH = synthetic_tiles("earth", 90, seed=2)
style = lambda name: name.split("_")[1]


def test_earth_to_mars_finds_the_same_kind_of_terrain():
    hits = []
    for e in EARTH[1:40:3]:
        if e.ndvi is not None and e.ndvi > 0.15:
            continue
        best = run_from_earth(MARS, EARTH, e.name, top_k=5)["mars_matches"].iloc[0]
        hits.append(style(best["mars_tile"]) == style(e.name))
    assert np.mean(hits) > 0.8


def test_both_directions_give_the_same_terrain_score_for_a_pair():
    m, e = MARS[4], EARTH[10]
    forward = run(MARS, EARTH, target=(m.lat, m.lon), top_k=200)["earth_matches"]
    backward = run_from_earth(MARS, EARTH, e.name, top_k=200)["mars_matches"]
    f = forward.set_index("earth_tile").loc[e.name, "morph_similarity"]
    b = backward.set_index("mars_tile").loc[m.name, "morph_similarity"]
    assert abs(f - b) < 1e-9


def test_a_green_earth_place_can_still_be_the_starting_point():
    green = next(e for e in EARTH if e.ndvi is not None and e.ndvi > 0.15)
    res = run_from_earth(MARS, EARTH, green.name, top_k=10)
    assert len(res["mars_matches"]) == 10
    assert res["mars_matches"]["final_score"].between(0, 1).all()
    assert res["chem_compared"] == ["iron"]


def test_reverse_iron_score_compares_ranks_on_each_planet():
    mars_df = pd.DataFrame({"chem_Fe": [10.0, 15.0, 20.0], "chem_Fe_pct": [0.2, 0.5, 0.9]})
    earth_df = pd.DataFrame({"chem_ferric_index": [1.0, 2.0, 3.0, 4.0]})   # ranks .25 .5 .75 1
    res = chemistry_scores_for_earth(mars_df, earth_df, 2)                 # Earth place at 0.75
    assert np.allclose(res["score"], [1 - 0.55, 1 - 0.25, 1 - 0.15])


def test_nearest_tile_and_distance():
    t, km = nearest_tile(EARTH, EARTH[5].lat, EARTH[5].lon, "earth")
    assert t is EARTH[5] and km < 1e-6
    assert abs(km_between(0, 0, 1, 0, "earth") - 111.19) < 0.1
    assert abs(km_between(0, 0, 1, 0, "mars") - 59.16) < 0.1
