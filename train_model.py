"""Learn how much each terrain measurement should count, from labelled pairs.

    python train_model.py

Reads every sheet in labels/ (rows labelled 1 or 0) and the data bundle, fits a
small logistic regression, tests it on pairs it didn't see, and writes
models/learned_weights.json.

The model has one weight per measurement, and a weight can't be negative, so a
bigger difference never makes two places more alike. That keeps every learned
number readable.

When several people label the same pair, each answer counts as an example, and
the report says how often people agreed. Testing leaves out a whole pair at a
time, so one person's answer can't help predict another's answer for that pair.
"""
import argparse
import datetime
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from mars_analog.bundle import load_bundle
from mars_analog.features import FEATURE_WEIGHTS, feature_columns, feature_table
from mars_analog.matching import prepare

MIN_PER_CLASS = 10


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit(D, y, l2=1.0):
    """Fit P(similar) = sigmoid(b - D @ w) with w >= 0. Returns (b, w)."""
    k = D.shape[1]

    def loss(params):
        b, w = params[0], params[1:]
        z = b - D @ w
        nll = np.sum(np.logaddexp(0.0, z) - y * z)
        grad_z = _sigmoid(z) - y
        grad = np.concatenate([[grad_z.sum()], -(D.T @ grad_z) + l2 * w])
        return nll + 0.5 * l2 * w @ w, grad

    start = np.concatenate([[0.0], np.full(k, 0.1)])
    res = minimize(loss, start, jac=True, method="L-BFGS-B",
                   bounds=[(None, None)] + [(0.0, None)] * k)
    return float(res.x[0]), res.x[1:]


def predict(D, b, w):
    return _sigmoid(b - D @ w)


def auc(scores, y):
    """Chance a random similar pair outscores a random different one (0.5 = guessing)."""
    scores, y = np.asarray(scores, float), np.asarray(y, int)
    pos, neg = scores[y == 1], scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def leave_one_out(D, y, l2=1.0, groups=None):
    """Predict each pair from a model trained without it (all its rows left out together)."""
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    preds = np.empty(len(y))
    for g in np.unique(groups):
        out = groups == g
        b, w = fit(D[~out], y[~out], l2)
        preds[out] = predict(D[out], b, w)
    return preds


def likely_range(learned, hand, y, groups, n=2000, seed=0):
    """95% ranges for both scores and their difference, resampling whole pairs."""
    rng = np.random.default_rng(seed)
    keys = np.unique(groups)
    rows = {g: np.flatnonzero(groups == g) for g in keys}
    a, b, d = [], [], []
    for _ in range(n):
        idx = np.concatenate([rows[g] for g in rng.choice(keys, size=len(keys))])
        al, ah = auc(learned[idx], y[idx]), auc(hand[idx], y[idx])
        if np.isfinite(al) and np.isfinite(ah):
            a.append(al), b.append(ah), d.append(al - ah)
    pct = lambda v: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
    return pct(a), pct(b), pct(d)


def verdict(diff_range):
    """Plain words for the likely range of (learned - hand-set)."""
    lo, hi = diff_range
    if lo > 0:
        return "The learned weights are better: the whole likely range is above zero."
    if hi < 0:
        return ("The hand-set weights do better on these labels: the whole likely range is "
                "below zero. Keep the hand-set weights, or add more labels and retrain.")
    return "Not clearly different yet: the likely range includes zero. More labels will tell."


def hand_set_scores(D, cols):
    """The app's original score, from the hand-set weights."""
    w = np.array([FEATURE_WEIGHTS[c] for c in cols])
    return 1.0 / (1.0 + np.sqrt((D ** 2) @ w / w.sum()))


def sheet_paths(labels):
    """A single sheet, or every .csv in a folder."""
    if os.path.isdir(labels):
        return sorted(glob.glob(os.path.join(labels, "*.csv")))
    return [labels]


def read_labels(paths):
    """Every row labelled 1 or 0, with who labelled it."""
    frames = []
    for p in paths:
        df = pd.read_csv(p, encoding="utf-8-sig", encoding_errors="replace",
                         dtype=str, keep_default_na=False)
        if not {"mars_tile", "earth_tile", "label"} <= set(df.columns):
            continue
        df["label"] = pd.to_numeric(df["label"].str.strip(), errors="coerce")
        df = df[df["label"].isin([0, 1])].copy()
        who = df["labeller"].str.strip() if "labeller" in df.columns else pd.Series("", index=df.index)
        named = who[who != ""]
        # A row saved without initials belongs to whoever labelled the rest of that sheet.
        fallback = named.mode().iloc[0] if len(named) else os.path.basename(p)
        df["who"] = who.where(who != "", fallback)
        df["source"] = os.path.basename(p)
        frames.append(df[["mars_tile", "earth_tile", "label", "who", "source"]])
    if not frames:
        return pd.DataFrame(columns=["mars_tile", "earth_tile", "label", "who", "source"])
    out = pd.concat(frames, ignore_index=True)
    out["label"] = out["label"].astype(int)
    return out


def agreement(pairs):
    """How often people agreed, over pairs that more than one person labelled."""
    g = pairs.groupby(["mars_tile", "earth_tile"])
    shared = g.filter(lambda x: x["who"].nunique() > 1).groupby(["mars_tile", "earth_tile"])
    n = shared.ngroups
    if n == 0:
        return {"pairs_shared": 0, "agreed": None}
    agreed = sum(1 for _, x in shared if x["label"].nunique() == 1)
    return {"pairs_shared": int(n), "agreed": round(agreed / n, 3)}


def scaled(mars_tiles, earth_tiles, max_ndvi, extra=True):
    """Feature values on the shared Mars-and-Earth scale, indexed by tile name."""
    m_df = feature_table(mars_tiles).reset_index(drop=True)
    e_df = feature_table(earth_tiles).reset_index(drop=True)
    if "ndvi" in e_df.columns:
        e_df = e_df[e_df["ndvi"] <= max_ndvi].reset_index(drop=True)
    cols = feature_columns(m_df, e_df, extra=extra)
    m_s, e_s = prepare(m_df, e_df, cols)
    m_s.index, e_s.index = m_df["name"], e_df["name"]
    return m_s, e_s, cols


def differences(mars_tiles, earth_tiles, pairs, max_ndvi):
    m_s, e_s, cols = scaled(mars_tiles, earth_tiles, max_ndvi)
    ok = pairs["mars_tile"].isin(m_s.index) & pairs["earth_tile"].isin(e_s.index)
    used = pairs[ok].reset_index(drop=True)
    D = np.abs(m_s.loc[used["mars_tile"]].to_numpy() - e_s.loc[used["earth_tile"]].to_numpy())
    groups = (used["mars_tile"] + "|" + used["earth_tile"]).to_numpy()
    return D, used["label"].to_numpy(int), cols, int((~ok).sum()), groups


def main(argv=None):
    ap = argparse.ArgumentParser(description="Learn measurement weights from labelled pairs.")
    ap.add_argument("--labels", default="labels", help="a label sheet, or a folder of them")
    ap.add_argument("--bundle", default="bundle")
    ap.add_argument("--out", default=os.path.join("models", "learned_weights.json"))
    ap.add_argument("--max-ndvi", type=float, default=0.15)
    args = ap.parse_args(argv)

    paths = sheet_paths(args.labels)
    pairs = read_labels(paths)
    mars, earth = load_bundle(args.bundle)
    D, y, cols, missing, groups = differences(mars, earth, pairs, args.max_ndvi)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if missing:
        print(f"{missing} labelled pairs were skipped because their tiles aren't in the bundle.")
    if n_pos < MIN_PER_CLASS or n_neg < MIN_PER_CLASS:
        print(f"Not enough labels yet: {n_pos} marked 1 and {n_neg} marked 0. "
              f"The model needs at least {MIN_PER_CLASS} of each.")
        return 1

    # Two candidates: the original 7 measurements, and those plus the extra landform ones.
    basic = [i for i, c in enumerate(cols) if c in FEATURE_WEIGHTS]
    D_basic, cols_basic = D[:, basic], [cols[i] for i in basic]
    hand = hand_set_scores(D_basic, cols_basic)
    preds_basic = leave_one_out(D_basic, y, groups=groups)
    preds_all = leave_one_out(D, y, groups=groups)
    auc_basic, auc_all = auc(preds_basic, y), auc(preds_all, y)
    _, _, r_all_vs_basic = likely_range(preds_all, preds_basic, y, groups)
    if len(cols) > len(cols_basic) and auc_all > auc_basic:
        learned, chosen_D, chosen_cols = preds_all, D, cols
    else:
        learned, chosen_D, chosen_cols = preds_basic, D_basic, cols_basic
    cols = chosen_cols
    b, w = fit(chosen_D, y)
    learned_auc, hand_auc = auc(learned, y), auc(hand, y)
    r_learned, r_hand, r_diff = likely_range(learned, hand, y, groups)
    agree = agreement(pairs)
    weights = {c: round(float(v), 4) for c, v in zip(cols, w)}
    model = {
        "features": cols, "weights": weights, "intercept": round(b, 4),
        "labels_similar": n_pos, "labels_different": n_neg,
        "pairs": int(len(np.unique(groups))), "labellers": int(pairs["who"].nunique()),
        "label_files": sorted(pairs["source"].unique().tolist()),
        "leave_one_out_auc": round(learned_auc, 3), "hand_set_auc": round(hand_auc, 3),
        "leave_one_out_auc_range": r_learned, "hand_set_auc_range": r_hand,
        "auc_difference": round(learned_auc - hand_auc, 3), "auc_difference_range": r_diff,
        "agreement": agree, "max_ndvi": args.max_ndvi, "trained": datetime.date.today().isoformat(),
        "measurement_sets": {
            "original": {"count": len(cols_basic), "auc": round(auc_basic, 3)},
            "with_landforms": {"count": D.shape[1], "auc": round(auc_all, 3)},
            "difference_range": r_all_vs_basic,
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)

    print(f"Labels used: {n_pos} alike, {n_neg} different, on {model['pairs']} pairs, "
          f"from {model['labellers']} labeller(s) in {len(model['label_files'])} file(s).")
    n_all = D.shape[1]
    print(f"Measurements tried: the original {len(cols_basic)}, and {n_all} with the landform ones.")
    print("How well each sorts pairs it didn't train on (0.5 = guessing, 1.0 = perfect):")
    print(f"  learned, {len(cols_basic)} measurements   {auc_basic:.2f}")
    print(f"  learned, {n_all} measurements  {auc_all:.2f}   difference {auc_all - auc_basic:+.2f} "
          f"({r_all_vs_basic[0]:+.2f} to {r_all_vs_basic[1]:+.2f})")
    if len(cols) == len(cols_basic):
        print(f"Kept the {len(cols)}-measurement model: the landform measurements didn't help "
              "on unseen pairs.")
    elif r_all_vs_basic[0] > 0:
        print(f"Kept the {len(cols)}-measurement model: the landform measurements clearly helped.")
    else:
        print(f"Kept the {len(cols)}-measurement model: it did slightly better on unseen pairs, "
              "though the gain may be chance.")
    print("Kept model against the hand-set weights, with the likely range:")
    print(f"  learned weights   {learned_auc:.2f}  ({r_learned[0]:.2f} to {r_learned[1]:.2f})")
    print(f"  hand-set weights  {hand_auc:.2f}  ({r_hand[0]:.2f} to {r_hand[1]:.2f})")
    print(f"  difference       {learned_auc - hand_auc:+.2f}  ({r_diff[0]:+.2f} to {r_diff[1]:+.2f})")
    print(verdict(r_diff))
    if agree["pairs_shared"]:
        print(f"People agreed on {agree['agreed']:.0%} of the {agree['pairs_shared']} pairs "
              "that more than one person labelled.")
    print("Learned weights (bigger = that measure matters more):")
    for c, v in sorted(weights.items(), key=lambda kv: -kv[1]):
        print(f"  {c:18s} {v:.2f}")
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
