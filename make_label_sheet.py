"""Make a sheet of Mars-Earth pairs for people to label.

    python make_label_sheet.py

Reads the data bundle and config.json, then writes labels/label_sheet.csv.
Each row pairs the Mars target with one Earth tile. Fill in the label column:
    1 = the landforms look alike, 0 = they clearly differ, blank = not sure.

    python make_label_sheet.py --copy-for AMY
        copies the pairs of labels/label_sheet.csv, without answers, to
        labels/label_sheet_AMY.csv, so a teammate can label the same pairs

    python make_label_sheet.py --next
        a new sheet (labels/label_sheet_2.csv, then _3, ...) for several Mars spots:
        half are pairs the learned model is least sure about, half are random,
        and none repeats a pair already in a sheet

The sheet leaves out the app's scores on purpose. The labels must come from
people looking at the places, or the model would only learn the app's own opinion.
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd

from mars_analog.bundle import load_bundle
from mars_analog.regions import place_name
from mars_analog.water import is_water

# Sites from the Catalogue of Planetary Analogues (Preston, Grady and Barber, 2012,
# ESA contract 4000104716/11/NL/AF), with the Mars features its directory lists for them.
CATALOGUE_SITES = [
    ("Yungay, Atacama Desert, Chile", -24.062, -69.923,
     "deltas and fans, dunes, soils, volcanic features (chapter 2)"),
    ("Central Australia, near Alice Springs", -23.70, 133.87,
     "lakes, channels, deltas and fans, gullies, dunes (chapter 3)"),
    ("Rissani, near Erfoud, Morocco", 31.294, -4.403,
     "dunes, soils, mud volcanoes, hydrothermal deposits (chapter 11)"),
    ("Chebbi, near Erfoud, Morocco", 31.214, -3.995,
     "dunes, soils, mud volcanoes, hydrothermal deposits (chapter 11)"),
]
NEAR_KM = 75
COLUMNS = ["pair_id", "mars_tile", "mars_lat", "mars_lon", "earth_tile", "earth_lat", "earth_lon",
           "place", "nearest_catalogue_site", "km_to_site", "catalogue_mars_features",
           "satellite_view", "label", "reason", "labeller"]


def km_between(lat1, lon1, lat2, lon2):
    """Great-circle distance on Earth in km."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def nearest_site(lat, lon):
    best = min(CATALOGUE_SITES, key=lambda s: km_between(lat, lon, s[1], s[2]))
    return best, km_between(lat, lon, best[1], best[2])


def make_row(pair_id, m, e):
    (site, _, _, features), km = nearest_site(e.lat, e.lon)
    close = km <= NEAR_KM
    return {
        "pair_id": pair_id, "mars_tile": m.name, "mars_lat": round(m.lat, 4), "mars_lon": round(m.lon, 4),
        "earth_tile": e.name, "earth_lat": round(e.lat, 4), "earth_lon": round(e.lon, 4),
        "place": place_name(e.lat, e.lon),
        "nearest_catalogue_site": site if close else "",
        "km_to_site": round(km) if close else "",
        "catalogue_mars_features": features if close else "",
        "satellite_view": ("https://www.google.com/maps/@?api=1&map_action=map"
                           f"&center={e.lat:.4f},{e.lon:.4f}&zoom=11&basemap=satellite"),
        "label": "", "reason": "", "labeller": "",
    }


def build_sheet(mars_tiles, earth_tiles, target, size=60, max_ndvi=0.15, seed=0):
    t = min(mars_tiles, key=lambda m: (m.lat - target[0]) ** 2 + (m.lon - target[1]) ** 2)
    earth = [e for e in earth_tiles if (e.ndvi is None or e.ndvi <= max_ndvi) and not is_water(e)]
    near = [e for e in earth if nearest_site(e.lat, e.lon)[1] <= NEAR_KM]
    rest = [e for e in earth if e not in near]
    rng = np.random.default_rng(seed)
    extra = max(0, size - len(near))
    picked = near + [rest[i] for i in rng.permutation(len(rest))[:extra]]
    picked = [picked[i] for i in rng.permutation(len(picked))]     # mix catalogue and other tiles
    rows = [make_row(i, t, e) for i, e in enumerate(picked, 1)]
    return pd.DataFrame(rows, columns=COLUMNS)


def pairs_in_sheets(label_dir):
    """(mars_tile, earth_tile) for every pair already in any sheet."""
    seen = set()
    for p in glob.glob(os.path.join(label_dir, "*.csv")):
        df = pd.read_csv(p, encoding="utf-8-sig", encoding_errors="replace", dtype=str,
                         keep_default_na=False)
        if {"mars_tile", "earth_tile"} <= set(df.columns):
            seen.update(zip(df["mars_tile"], df["earth_tile"]))
    return seen


def spread_spots(mars_tiles, spots):
    """`spots` Mars tiles spread evenly across the area."""
    ordered = sorted(mars_tiles, key=lambda m: (m.lat, m.lon))
    idx = np.unique(np.linspace(0, len(ordered) - 1, min(spots, len(ordered))).round().astype(int))
    return [ordered[i] for i in idx]


def next_sheet(mars_tiles, earth_tiles, model, seen, spots=4, size=60, max_ndvi=0.15, seed=0):
    """Half the pairs the model is least sure about, half random; none already in a sheet."""
    rng = np.random.default_rng(seed)
    chosen_mars = spread_spots(mars_tiles, spots)
    earth = [e for e in earth_tiles if (e.ndvi is None or e.ndvi <= max_ndvi) and not is_water(e)]
    candidates = [(m, e) for m in chosen_mars for e in earth if (m.name, e.name) not in seen]
    if not candidates:
        return pd.DataFrame(columns=COLUMNS)

    unsure = []
    if model:
        from mars_analog.matching import learned_similarity_matrix
        from train_model import scaled
        m_s, e_s, cols = scaled(mars_tiles, earth_tiles, max_ndvi)
        if all(f in cols for f in model["features"]):
            S = pd.DataFrame(learned_similarity_matrix(m_s, e_s, model),
                             index=m_s.index, columns=e_s.index)
            by_spot = {m.name: sorted((abs(S.loc[m.name, e.name] - 0.5), i)
                                      for i, (mm, e) in enumerate(candidates) if mm is m)
                       for m in chosen_mars}
            # take the most uncertain pair from each spot in turn
            while len(unsure) < size // 2 and any(by_spot.values()):
                for m in chosen_mars:
                    if by_spot[m.name] and len(unsure) < size // 2:
                        unsure.append(by_spot[m.name].pop(0)[1])
    rest = [i for i in range(len(candidates)) if i not in set(unsure)]
    random_pick = [rest[i] for i in rng.permutation(len(rest))[:max(0, size - len(unsure))]]
    picked = [candidates[i] for i in unsure + random_pick]
    picked = [picked[i] for i in rng.permutation(len(picked))]
    rows = [make_row(i, m, e) for i, (m, e) in enumerate(picked, 1)]
    return pd.DataFrame(rows, columns=COLUMNS)


def copy_for(src, initials, out=None):
    df = pd.read_csv(src, encoding="utf-8-sig", encoding_errors="replace", dtype=str,
                     keep_default_na=False)
    for c in ("label", "reason", "labeller"):
        df[c] = ""
    stem, _ = os.path.splitext(src)
    return df, out or f"{stem}_{initials}.csv"


def free_name(label_dir):
    n = 2
    while os.path.exists(os.path.join(label_dir, f"label_sheet_{n}.csv")):
        n += 1
    return os.path.join(label_dir, f"label_sheet_{n}.csv")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Make a sheet of Mars-Earth pairs to label.")
    ap.add_argument("--bundle", default="bundle")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--labels", default="labels", help="the folder of label sheets")
    ap.add_argument("--model", default=os.path.join("models", "learned_weights.json"))
    ap.add_argument("--out", help="where to write the sheet")
    ap.add_argument("--size", type=int, default=60)
    ap.add_argument("--spots", type=int, default=4, help="Mars spots in a --next sheet")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--copy-for", metavar="INITIALS",
                      help="copy a sheet's pairs, without answers, for a teammate")
    mode.add_argument("--next", action="store_true",
                      help="a new sheet chosen with the learned model")
    ap.add_argument("--from", dest="source", default=None,
                    help="the sheet to copy with --copy-for (default labels/label_sheet.csv)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing sheet, including any labels in it")
    args = ap.parse_args(argv)

    if args.copy_for:
        src = args.source or os.path.join(args.labels, "label_sheet.csv")
        sheet, out = copy_for(src, args.copy_for, args.out)
        message = f"Copied the {len(sheet)} pairs of {src} for {args.copy_for}."
    elif args.next:
        model = None
        if os.path.exists(args.model):
            with open(args.model, encoding="utf-8") as f:
                model = json.load(f)
        mars, earth = load_bundle(args.bundle)
        sheet = next_sheet(mars, earth, model, pairs_in_sheets(args.labels),
                           spots=args.spots, size=args.size)
        out = args.out or free_name(args.labels)
        how = "half chosen where the model is least sure" if model else "chosen at random (no model yet)"
        message = (f"Made {len(sheet)} new pairs for {sheet['mars_tile'].nunique()} Mars spots, "
                   f"{how}.")
    else:
        with open(args.config, encoding="utf-8") as f:
            target = tuple(json.load(f).get("target") or (18.38, 77.58))
        mars, earth = load_bundle(args.bundle)
        sheet = build_sheet(mars, earth, target, size=args.size)
        out = args.out or os.path.join(args.labels, "label_sheet.csv")
        near = int((sheet["nearest_catalogue_site"] != "").sum())
        message = f"Made {len(sheet)} pairs, {near} of them near a catalogue site."

    if os.path.exists(out) and not args.overwrite:
        print(f"{out} already exists, so it was left alone to protect any labels in it.")
        print("Use --overwrite to replace it, or --out to write a new file.")
        return 1
    if sheet.empty:
        print("No new pairs to make: every possible pair is already in a sheet.")
        return 1
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    sheet.to_csv(out, index=False, encoding="utf-8-sig")   # utf-8-sig so Excel shows ° correctly
    print(message)
    print(f"Wrote {out}. Label it with: python -m streamlit run label_app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
