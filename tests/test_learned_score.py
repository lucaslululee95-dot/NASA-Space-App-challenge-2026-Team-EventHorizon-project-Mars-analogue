import numpy as np
import pandas as pd
import pytest

from mars_analog.matching import learned_similarity_matrix
from mars_analog.pipeline import run
from mars_analog.sources import synthetic_tiles


def _model(features, weight=1.0, intercept=2.0):
    return {"features": features, "weights": {f: weight for f in features}, "intercept": intercept,
            "labels_similar": 20, "labels_different": 20}


def test_identical_tiles_score_highest():
    m = pd.DataFrame({"a": [0.0], "b": [0.0]})
    e = pd.DataFrame({"a": [0.0, 1.0, 3.0], "b": [0.0, 1.0, 3.0]})
    S = learned_similarity_matrix(m, e, _model(["a", "b"]))
    assert S.shape == (1, 3)
    assert S[0, 0] > S[0, 1] > S[0, 2]
    assert ((S > 0) & (S < 1)).all()


def test_zero_weights_give_the_same_score_everywhere():
    m = pd.DataFrame({"a": [0.0, 2.0]})
    e = pd.DataFrame({"a": [0.0, 5.0]})
    S = learned_similarity_matrix(m, e, _model(["a"], weight=0.0, intercept=0.0))
    assert np.allclose(S, 0.5)


def test_pipeline_uses_the_learned_model():
    mars, earth = synthetic_tiles("mars", 12, seed=1), synthetic_tiles("earth", 40, seed=2)
    cols = ["slope_mean_deg", "slope_p90_deg", "relief_m", "roughness_m", "curvature_per_m"]
    hand = run(mars, earth, target=(17.0, 77.0), top_k=40)["earth_matches"]
    learned = run(mars, earth, target=(17.0, 77.0), top_k=40,
                  terrain_model=_model(cols))["earth_matches"]
    assert learned["morph_similarity"].between(0, 1).all()
    assert not np.allclose(hand.sort_values("earth_tile")["morph_similarity"].to_numpy(),
                           learned.sort_values("earth_tile")["morph_similarity"].to_numpy())


def test_model_needing_missing_measures_is_explained():
    mars, earth = synthetic_tiles("mars", 6, seed=1), synthetic_tiles("earth", 20, seed=2)
    with pytest.raises(ValueError, match="Retrain"):
        run(mars, earth, target=(17.0, 77.0), terrain_model=_model(["not_a_measure"]))


def test_hand_set_mode_explains_with_the_original_measurements():
    from mars_analog.features import FEATURE_WEIGHTS
    res = run(synthetic_tiles("mars", 8, seed=1), synthetic_tiles("earth", 30, seed=2), target=(17.0, 77.0))
    assert set(res["feature_cols"]) <= set(FEATURE_WEIGHTS)


def test_learned_mode_can_use_landform_measurements():
    model = _model(["slope_mean_deg", "aspect_spread", "peaks_per_100km2"])
    model["weights"]["peaks_per_100km2"] = 0.0
    res = run(synthetic_tiles("mars", 8, seed=1), synthetic_tiles("earth", 30, seed=2),
              target=(17.0, 77.0), terrain_model=model)
    assert res["feature_cols"] == ["slope_mean_deg", "aspect_spread"]     # weight-zero ones left out
    assert res["earth_matches"]["morph_similarity"].between(0, 1).all()
