"""End-to-end comparison.

Step 1: for each Mars tile, how close is its terrain to the nearest Earth tile?
Step 2: choose a Mars target (given, or the most Earth-like Mars tile).
Step 3: rank Earth tiles by terrain similarity to the target, then blend in chemistry.

run_from_earth does the reverse: it ranks the Mars spots against one Earth place.
"""
import numpy as np
import pandas as pd

from .chemistry import chemistry_scores, chemistry_scores_for_earth
from .features import feature_columns, feature_table
from .matching import explain, learned_similarity_matrix, prepare, similarity_matrix


def score_matrix(mars_tiles, earth_tiles, max_ndvi=0.15, terrain_model=None, keep_earth=None):
    """Measure every tile and score every Mars-Earth pair.

    Returns (mars_df, earth_df, m_s, e_s, S, used): the feature tables, the scaled
    measurements, the Mars-by-Earth score matrix, and the measurements the score uses.
    keep_earth names an Earth tile to keep even if the vegetation filter would drop it.
    """
    mars_df = feature_table(mars_tiles).reset_index(drop=True)
    earth_df = feature_table(earth_tiles).reset_index(drop=True)

    if "ndvi" in earth_df.columns:
        keep = (earth_df["ndvi"] <= max_ndvi) | (earth_df["name"] == keep_earth)
        earth_df = earth_df[keep].reset_index(drop=True)
    if earth_df.empty:
        raise ValueError("No Earth tiles left after the vegetation filter. "
                         "Raise max_ndvi or add candidate regions.")

    cols = feature_columns(mars_df, earth_df)
    if not cols:
        raise ValueError("No features are available on both planets.")

    all_cols = feature_columns(mars_df, earth_df, extra=True)
    m_s, e_s = prepare(mars_df, earth_df, all_cols)   # each column is scaled on its own
    if terrain_model is None:
        S = similarity_matrix(m_s[cols], e_s[cols], cols)   # shape (mars tiles, earth tiles)
        used = cols
    else:
        missing = [f for f in terrain_model["features"] if f not in all_cols]
        if missing:
            raise ValueError("The learned model needs measurements this data doesn't have: "
                             + ", ".join(missing) + ". Retrain it with train_model.py.")
        S = learned_similarity_matrix(m_s, e_s, terrain_model)
        used = [f for f in terrain_model["features"] if terrain_model["weights"][f] > 0] \
            or list(terrain_model["features"])
    return mars_df, earth_df, m_s, e_s, S, used


def _blend(morph, chem_score, w_morph, w_chem):
    w_c = np.where(np.isfinite(chem_score), w_chem, 0.0)  # if no chemistry, use terrain alone
    return (w_morph * morph + w_c * np.nan_to_num(chem_score)) / (w_morph + w_c)


def _explain_rows(m_s, e_s, pairs, used):
    agrees, differs = [], []
    for i, j in pairs:
        a, d = explain(m_s.iloc[i], e_s.iloc[j], used)
        agrees.append(", ".join(a))
        differs.append(", ".join(d))
    return agrees, differs


def run(mars_tiles, earth_tiles, *, w_morph=0.7, w_chem=0.3,
        max_ndvi=0.15, target=None, top_k=25, terrain_model=None):
    """Rank Earth places against one Mars spot.

    target: optional (lat, lon) of a Mars tile. If None, the Mars tile whose
    terrain is closest to any Earth tile is used.
    """
    mars_df, earth_df, m_s, e_s, S, used = score_matrix(
        mars_tiles, earth_tiles, max_ndvi, terrain_model)

    # Step 1: which Mars tiles have terrain closest to something on Earth?
    closest = S.argmax(axis=1)
    ranking = pd.DataFrame({
        "mars_tile": mars_df["name"],
        "lat": mars_df["lat"],
        "lon": mars_df["lon"],
        "best_earth_similarity": S.max(axis=1),
        "closest_earth_tile": earth_df["name"].to_numpy()[closest],
    }).sort_values("best_earth_similarity", ascending=False).reset_index(drop=True)

    # Step 2: choose the Mars target
    if target is None:
        t = int(np.argmax(S.max(axis=1)))
    else:
        lat, lon = target
        t = int(((mars_df["lat"] - lat) ** 2 + (mars_df["lon"] - lon) ** 2).idxmin())

    # Step 3: rank Earth tiles against the target
    morph = S[t]
    chem = chemistry_scores(mars_df, t, earth_df)
    matches = pd.DataFrame({
        "earth_tile": earth_df["name"],
        "lat": earth_df["lat"],
        "lon": earth_df["lon"],
        "morph_similarity": morph,
        "chem_score": chem["score"],
        "final_score": _blend(morph, chem["score"], w_morph, w_chem),
    }).sort_values("final_score", ascending=False).head(top_k).copy()
    matches["agrees_on"], matches["differs_on"] = _explain_rows(
        m_s, e_s, [(t, j) for j in matches.index], used)

    return {
        "target": mars_df["name"].iloc[t],
        "mars_ranking": ranking,
        "earth_matches": matches.reset_index(drop=True),
        "feature_cols": used,
        "chem_compared": chem["compared"],
        "chem_skipped": chem["skipped"],
    }


def run_from_earth(mars_tiles, earth_tiles, earth_name, *, w_morph=0.7, w_chem=0.3,
                   max_ndvi=0.15, top_k=25, terrain_model=None):
    """Rank the Mars spots against one Earth place, named by its tile."""
    mars_df, earth_df, m_s, e_s, S, used = score_matrix(
        mars_tiles, earth_tiles, max_ndvi, terrain_model, keep_earth=earth_name)
    hits = np.flatnonzero(earth_df["name"].to_numpy() == earth_name)
    if not len(hits):
        raise ValueError(f"The Earth place {earth_name} isn't in the data.")
    j = int(hits[0])

    morph = S[:, j]
    chem = chemistry_scores_for_earth(mars_df, earth_df, j)
    matches = pd.DataFrame({
        "mars_tile": mars_df["name"],
        "lat": mars_df["lat"],
        "lon": mars_df["lon"],
        "morph_similarity": morph,
        "chem_score": chem["score"],
        "final_score": _blend(morph, chem["score"], w_morph, w_chem),
    }).sort_values("final_score", ascending=False).head(top_k).copy()
    matches["agrees_on"], matches["differs_on"] = _explain_rows(
        m_s, e_s, [(i, j) for i in matches.index], used)

    return {
        "target": earth_name,
        "mars_matches": matches.reset_index(drop=True),
        "feature_cols": used,
        "chem_compared": chem["compared"],
        "chem_skipped": chem["skipped"],
    }
