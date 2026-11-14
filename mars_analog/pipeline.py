"""End-to-end comparison.

Step 1: for each Mars tile, how close is its terrain to the nearest Earth tile?
Step 2: choose a Mars target (given, or the most Earth-like Mars tile).
Step 3: rank Earth tiles by terrain similarity to the target, then blend in chemistry.
"""
import numpy as np
import pandas as pd

from .chemistry import chemistry_scores
from .features import feature_columns, feature_table
from .matching import explain, prepare, similarity_matrix


def run(mars_tiles, earth_tiles, *, w_morph=0.7, w_chem=0.3,
        max_ndvi=0.15, target=None, top_k=25):
    """Run the full comparison.

    target: optional (lat, lon) of a Mars tile. If None, the Mars tile whose
    terrain is closest to any Earth tile is used.
    """
    mars_df = feature_table(mars_tiles).reset_index(drop=True)
    earth_df = feature_table(earth_tiles).reset_index(drop=True)

    if "ndvi" in earth_df.columns:
        earth_df = earth_df[earth_df["ndvi"] <= max_ndvi].reset_index(drop=True)
    if earth_df.empty:
        raise ValueError("No Earth tiles left after the vegetation filter. "
                         "Raise max_ndvi or add candidate regions.")

    cols = feature_columns(mars_df, earth_df)
    if not cols:
        raise ValueError("No features are available on both planets.")

    m_s, e_s = prepare(mars_df, earth_df, cols)
    S = similarity_matrix(m_s, e_s, cols)            # shape

   
    closest = S.argmax(axis=1)
    ranking = pd.DataFrame({
        "mars_tile": mars_df["name"],
        "lat": mars_df["lat"],
        "lon": mars_df["lon"],
        "best_earth_similarity": S.max(axis=1),
        "closest_earth_tile": earth_df["name"].to_numpy()[closest],
    }).sort_values("best_earth_similarity", ascending=False).reset_index(drop=True)

    
    if target is None:
        t = int(np.argmax(S.max(axis=1)))
    else:
        lat, lon = target
        t = int(((mars_df["lat"] - lat) ** 2 + (mars_df["lon"] - lon) ** 2).idxmin())

    
    morph = S[t]
    chem = chemistry_scores(mars_df, t, earth_df)
    chem_score = chem["score"]
    w_c = np.where(np.isfinite(chem_score), w_chem, 0.0)  # if no chemistry, use terrain alone
    final = (w_morph * morph + w_c * np.nan_to_num(chem_score)) / (w_morph + w_c)

    matches = pd.DataFrame({
        "earth_tile": earth_df["name"],
        "lat": earth_df["lat"],
        "lon": earth_df["lon"],
        "morph_similarity": morph,
        "chem_score": chem_score,
        "final_score": final,
    }).sort_values("final_score", ascending=False).head(top_k).copy()

    agrees, differs = [], []
    for j in matches.index:
        a, d = explain(m_s.iloc[t], e_s.iloc[j], cols)
        agrees.append(", ".join(a))
        differs.append(", ".join(d))
    matches["agrees_on"] = agrees
    matches["differs_on"] = differs

    return {
        "target": mars_df["name"].iloc[t],
        "mars_ranking": ranking,
        "earth_matches": matches.reset_index(drop=True),
        "feature_cols": cols,
        "chem_compared": chem["compared"],
        "chem_skipped": chem["skipped"],
    }