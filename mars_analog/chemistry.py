"""Chemistry comparison between a Mars target and Earth candidates.

Mars and Earth measurements use different units and instruments, so values
are never compared directly. Each value is converted to a percentile within
its own planet's sample, and the percentiles are compared. A high score means
both places sit at a similar relative level of the same property on their
own planet.

CHEM_AXES lists every element the tool knows about. Only axes with an Earth
proxy and compare=True are scored. The rest are reported as "not compared",
so the output never implies more chemistry was checked than actually was.
"""
import numpy as np
import pandas as pd

CHEM_AXES = {
    "iron": {
        "mars_field": "Fe",
        "earth_field": "ferric_index",
        "mars_pct_field": "Fe_pct",   # rank of the cell among all valid Mars cells (convert_grs.py)
        "compare": True,
        "note": "Mars: gamma-ray iron abundance. Earth: Sentinel-2 red/blue ratio, an iron-oxide proxy.",
    },
    "silicon": {"mars_field": "Si", "earth_field": None, "compare": False,
                "note": "Mars only. No Earth proxy configured."},
    "sulfur": {"mars_field": "S", "earth_field": None, "compare": False,
               "note": "Mars only. No Earth proxy configured."},
    "chlorine": {"mars_field": "Cl", "earth_field": None, "compare": False,
                 "note": "Mars only. Soil salinity could become an Earth proxy."},
    "potassium": {"mars_field": "K", "earth_field": None, "compare": False,
                  "note": "Mars only. No Earth proxy configured."},
}


def chemistry_scores(mars_df, target_idx, earth_df):
    """Score each Earth tile against the Mars target, in [0, 1]. NaN where nothing was comparable."""
    n = len(earth_df)
    total, count = np.zeros(n), np.zeros(n)
    compared, skipped = [], []

    for axis, spec in CHEM_AXES.items():
        mcol = f"chem_{spec['mars_field']}"
        ecol = f"chem_{spec['earth_field']}" if spec["earth_field"] else None
        usable = spec["compare"] and ecol is not None and mcol in mars_df and ecol in earth_df
        if not usable:
            skipped.append(axis)
            continue

        m_vals = mars_df[mcol].astype(float).to_numpy()
        e_vals = earth_df[ecol].astype(float).to_numpy()
        t_val = m_vals[target_idx]
        if not np.isfinite(t_val):
            skipped.append(axis)
            continue

        pct_col = f"chem_{spec['mars_pct_field']}" if spec.get("mars_pct_field") else None
        if pct_col and pct_col in mars_df and np.isfinite(mars_df[pct_col].iloc[target_idx]):
            # Percentile against the whole Mars table, from convert_grs.py
            p_target = float(mars_df[pct_col].iloc[target_idx])
        else:
            # Fallback: percentile among the Mars tiles, with ties at the middle rank
            pool = m_vals[np.isfinite(m_vals)]
            p_target = float((np.sum(pool < t_val) + 0.5 * np.sum(pool == t_val)) / len(pool))
        e_pct = pd.Series(e_vals).rank(pct=True).to_numpy()  # each Earth tile's percentile on Earth
        score = 1.0 - np.abs(e_pct - p_target)

        ok = np.isfinite(score)
        total[ok] += score[ok]
        count[ok] += 1
        compared.append(axis)

    scores = np.full(n, np.nan)
    has = count > 0
    scores[has] = total[has] / count[has]
    return {"score": scores, "compared": compared, "skipped": skipped}
