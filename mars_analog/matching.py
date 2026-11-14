"""Turn feature tables into a Mars-by-Earth similarity matrix."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .features import FEATURE_WEIGHTS, LOG_FEATURES


def prepare(mars_df, earth_df, cols):
    """Log-transform skewed features, then scale both planets with one shared median and IQR.

    Sharing the scale matters: if each planet were scaled separately, every
    tile would be judged only against its own planet's spread.
    """
    def transform(df):
        out = df[cols].astype(float).copy()
        for c in cols:
            if c in LOG_FEATURES:
                out[c] = np.log1p(out[c].clip(lower=0))
        return out

    m, e = transform(mars_df), transform(earth_df)
    both = pd.concat([m, e], ignore_index=True)
    med = both.median()
    iqr = both.quantile(0.75) - both.quantile(0.25)
    spread = iqr.where(iqr > 1e-9, both.std()).replace(0, 1.0).fillna(1.0)
    return (m - med) / spread, (e - med) / spread


def similarity_matrix(m_scaled, e_scaled, cols):
    """Similarity in (0, 1] between every Mars tile (rows) and Earth tile (columns)."""
    w = np.array([FEATURE_WEIGHTS[c] for c in cols], dtype=float)
    sw = np.sqrt(w)
    # Weighted root-mean-square difference, so the result does not depend on how many features there are.
    dist = cdist(m_scaled.to_numpy() * sw, e_scaled.to_numpy() * sw, "euclidean") / np.sqrt(w.sum())
    return 1.0 / (1.0 + dist)


def explain(m_row, e_row, cols, k=3):
    """Features where the two tiles agree most and disagree most (scaled values)."""
    diffs = sorted((abs(m_row[c] - e_row[c]), c) for c in cols)
    agree = [c for _, c in diffs[:k]]
    disagree = [c for _, c in diffs[-k:][::-1]]
    return agree, disagree
