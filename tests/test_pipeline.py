import numpy as np

from mars_analog.features import feature_columns, feature_table
from mars_analog.geo import box_from_center
from mars_analog.pipeline import run
from mars_analog.sources import synthetic_tiles


def _style(name):
    return name.split("_")[1]


def test_ten_km_tile_is_ten_km_on_earth():
    w, s, e, n = box_from_center(0.0, 0.0, 10.0, "earth")
    assert abs((n - s) * 111.195 - 10.0) < 0.1


def test_features_finite_and_present():
    df = feature_table(synthetic_tiles("earth", 9, seed=3))
    cols = feature_columns(df, df)
    assert {"slope_mean_deg", "roughness_m", "texture_rel", "red_blue_ratio"} <= set(cols)
    assert np.isfinite(df[cols].to_numpy()).all()


def test_greyscale_mars_drops_colour_feature():
    m_df = feature_table(synthetic_tiles("mars", 6, seed=4))
    e_df = feature_table(synthetic_tiles("earth", 12, seed=5))
    assert "red_blue_ratio" not in feature_columns(m_df, e_df)


def test_terrain_style_is_recovered():
    mars = synthetic_tiles("mars", 30, seed=1)
    earth = synthetic_tiles("earth", 90, seed=2)
    ranking = run(mars, earth)["mars_ranking"]
    hits = [_style(m) == _style(e)
            for m, e in zip(ranking["mars_tile"], ranking["closest_earth_tile"])]
    assert np.mean(hits) > 0.8


def test_target_run_and_bounds():
    mars = synthetic_tiles("mars", 30, seed=1)
    earth = synthetic_tiles("earth", 90, seed=2)
    res = run(mars, earth, target=(17.0, 77.0), top_k=10)
    m = res["earth_matches"]
    assert len(m) == 10
    assert "iron" in res["chem_compared"]
    assert m["chem_score"].between(0, 1).all()
    assert m["final_score"].between(0, 1).all()
    assert res["target"].startswith("mars_")
