"""Learn how much each terrain measurement should count, from labelled pairs.

    python train_model.py

Reads labels/label_sheet.csv (rows labelled 1 or 0) and the data bundle, fits a
small logistic regression, tests it on pairs it didn't see (leave-one-out), and
writes models/learned_weights.json.

The model has one weight per measurement, and a weight can't be negative, so a
bigger difference never makes two places more alike. That keeps every learned
number readable.
"""
import argparse
import datetime
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
        p = _sigmoid(z)
        grad_z = p - y
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


def leave_one_out(D, y, l2=1.0):
    preds = np.empty(len(y))
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        b, w = fit(D[keep], y[keep], l2)
        preds[i] = predict(D[i:i + 1], b, w)[0]
    return preds


def hand_set_scores(D, cols):
    """The app's current score, from the hand-set weights."""
    w = np.array([FEATURE_WEIGHTS[c] for c in cols])
    return 1.0 / (1.0 + np.sqrt((D ** 2) @ w / w.sum()))


def labelled_pairs(sheet_path):
    sheet = pd.read_csv(sheet_path, encoding="utf-8-sig")
    sheet["label"] = pd.to_numeric(sheet["label"], errors="coerce")
    return sheet[sheet["label"].isin([0, 1])].copy()


def differences(mars_tiles, earth_tiles, pairs, max_ndvi):
    m_df = feature_table(mars_tiles).reset_index(drop=True)
    e_df = feature_table(earth_tiles).reset_index(drop=True)
    if "ndvi" in e_df.columns:
        e_df = e_df[e_df["ndvi"] <= max_ndvi].reset_index(drop=True)
    cols = feature_columns(m_df, e_df)
    m_s, e_s = prepare(m_df, e_df, cols)
    m_s.index, e_s.index = m_df["name"], e_df["name"]
    ok = pairs["mars_tile"].isin(m_s.index) & pairs["earth_tile"].isin(e_s.index)
    used = pairs[ok]
    D = np.abs(m_s.loc[used["mars_tile"]].to_numpy() - e_s.loc[used["earth_tile"]].to_numpy())
    return D, used["label"].to_numpy(int), cols, int((~ok).sum())


def main(argv=None):
    ap = argparse.ArgumentParser(description="Learn measurement weights from labelled pairs.")
    ap.add_argument("--labels", default=os.path.join("labels", "label_sheet.csv"))
    ap.add_argument("--bundle", default="bundle")
    ap.add_argument("--out", default=os.path.join("models", "learned_weights.json"))
    ap.add_argument("--max-ndvi", type=float, default=0.15)
    args = ap.parse_args(argv)

    pairs = labelled_pairs(args.labels)
    mars, earth = load_bundle(args.bundle)
    D, y, cols, missing = differences(mars, earth, pairs, args.max_ndvi)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if missing:
        print(f"{missing} labelled pairs were skipped because their tiles aren't in the bundle.")
    if n_pos < MIN_PER_CLASS or n_neg < MIN_PER_CLASS:
        print(f"Not enough labels yet: {n_pos} marked 1 and {n_neg} marked 0. "
              f"The model needs at least {MIN_PER_CLASS} of each.")
        return 1

    b, w = fit(D, y)
    learned_auc = auc(leave_one_out(D, y), y)
    baseline_auc = auc(hand_set_scores(D, cols), y)
    weights = {c: round(float(v), 4) for c, v in zip(cols, w)}
    model = {
        "features": cols, "weights": weights, "intercept": round(b, 4),
        "labels_similar": n_pos, "labels_different": n_neg,
        "leave_one_out_auc": round(learned_auc, 3), "hand_set_auc": round(baseline_auc, 3),
        "max_ndvi": args.max_ndvi, "trained": datetime.date.today().isoformat(),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)

    print(f"Labelled pairs used: {n_pos} alike, {n_neg} different.")
    print("How well each method sorts pairs it didn't train on "
          "(0.5 = guessing, 1.0 = perfect):")
    print(f"  learned weights   {learned_auc:.2f}")
    print(f"  hand-set weights  {baseline_auc:.2f}")
    print("Learned weights (bigger = that measure matters more):")
    for c, v in sorted(weights.items(), key=lambda kv: -kv[1]):
        print(f"  {c:18s} {v:.2f}")
    if len(y) < 100:
        print("With fewer than about 100 labels, treat these numbers as rough.")
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
