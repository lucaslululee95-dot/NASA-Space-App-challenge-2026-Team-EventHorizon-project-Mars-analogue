import numpy as np
import pandas as pd

from mars_analog.features import (EXTRA_FEATURES, FEATURE_WEIGHTS, feature_columns, feature_table,
                                  terrain_features)
from mars_analog.matching import prepare
from mars_analog.sources import synthetic_tiles

N = 64
yy, xx = np.mgrid[0:N, 0:N].astype(float)
r = np.hypot(xx - N / 2, yy - N / 2)


def test_extra_measurements_exist_and_are_finite():
    df = feature_table(synthetic_tiles("earth", 9, seed=3))
    assert set(EXTRA_FEATURES) <= set(df.columns)
    assert np.isfinite(df[EXTRA_FEATURES].to_numpy()).all()


def test_slope_directions_one_way_versus_every_way():
    tilted = terrain_features(xx * 20.0, 10)          # a plane rising to the east
    bowl = terrain_features(r ** 2 * 2.0, 10)         # a round hollow
    assert tilted["aspect_spread"] < 0.05
    assert bowl["aspect_spread"] > 0.9


def test_hills_and_hollows_are_counted():
    flat = terrain_features(np.full((N, N), 500.0), 10)
    rng = np.random.default_rng(0)
    from scipy.ndimage import gaussian_filter
    bumpy = gaussian_filter(rng.normal(size=(N, N)), 2) * 300.0
    assert flat["peaks_per_100km2"] == 0
    assert terrain_features(bumpy, 10)["peaks_per_100km2"] > 0


def test_share_of_high_ground():
    cone = terrain_features(-r * 30.0, 10)            # one peak, mostly low ground
    bowl = terrain_features(r * 30.0, 10)             # one hollow, mostly high ground
    assert cone["hypsometric"] < 0.5 < bowl["hypsometric"]


def test_extra_flag_adds_the_landform_measurements():
    m = feature_table(synthetic_tiles("mars", 6, seed=1))
    e = feature_table(synthetic_tiles("earth", 12, seed=2))
    basic, extended = feature_columns(m, e), feature_columns(m, e, extra=True)
    assert set(basic) <= set(FEATURE_WEIGHTS)
    assert extended == basic + EXTRA_FEATURES


def test_scaling_each_column_alone_keeps_the_hand_set_score_unchanged():
    m = feature_table(synthetic_tiles("mars", 6, seed=1))
    e = feature_table(synthetic_tiles("earth", 12, seed=2))
    basic, extended = feature_columns(m, e), feature_columns(m, e, extra=True)
    m1, e1 = prepare(m, e, basic)
    m2, e2 = prepare(m, e, extended)
    pd.testing.assert_frame_equal(m1, m2[basic])
    pd.testing.assert_frame_equal(e1, e2[basic])
