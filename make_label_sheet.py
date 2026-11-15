"""Make a sheet of Mars-Earth pairs for people to label.

    python make_label_sheet.py

Reads the data bundle and config.json, then writes labels/label_sheet.csv.
Each row pairs the Mars target with one Earth tile. Fill in the label column:
    1 = the landforms look alike, 0 = they clearly differ, blank = not sure.

The sheet leaves out the app's scores on purpose. The labels must come from
people looking at the places, or the model would only learn the app's own opinion.
"""
import argparse
import json
import math
import os

import numpy as np
import pandas as pd

from mars_analog.bundle import load_bundle
from mars_analog.regions import place_name

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


def build_sheet(mars_tiles, earth_tiles, target, size=60, max_ndvi=0.15, seed=0):
    t = min(mars_tiles, key=lambda m: (m.lat - target[0]) ** 2 + (m.lon - target[1]) ** 2)
    earth = [e for e in earth_tiles if e.ndvi is None or e.ndvi <= max_ndvi]
    near = [e for e in earth if nearest_site(e.lat, e.lon)[1] <= NEAR_KM]
    rest = [e for e in earth if e not in near]
    rng = np.random.default_rng(seed)
    extra = max(0, size - len(near))
    picked = near + [rest[i] for i in rng.permutation(len(rest))[:extra]]
    picked = [picked[i] for i in rng.permutation(len(picked))]     # mix catalogue and other tiles

    rows = []
    for i, e in enumerate(picked, 1):
        (site, _, _, features), km = nearest_site(e.lat, e.lon)
        close = km <= NEAR_KM
        rows.append({
            "pair_id": i, "mars_tile": t.name, "mars_lat": round(t.lat, 4), "mars_lon": round(t.lon, 4),
            "earth_tile": e.name, "earth_lat": round(e.lat, 4), "earth_lon": round(e.lon, 4),
            "place": place_name(e.lat, e.lon),
            "nearest_catalogue_site": site if close else "",
            "km_to_site": round(km) if close else "",
            "catalogue_mars_features": features if close else "",
            "satellite_view": ("https://www.google.com/maps/@?api=1&map_action=map"
                               f"&center={e.lat:.4f},{e.lon:.4f}&zoom=11&basemap=satellite"),
            "label": "", "reason": "", "labeller": "",
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Make a sheet of Mars-Earth pairs to label.")
    ap.add_argument("--bundle", default="bundle")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--out", default=os.path.join("labels", "label_sheet.csv"))
    ap.add_argument("--size", type=int, default=60)
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing sheet, including any labels in it")
    args = ap.parse_args(argv)

    if os.path.exists(args.out) and not args.overwrite:
        print(f"{args.out} already exists, so it was left alone to protect any labels in it.")
        print("Use --overwrite to replace it, or --out to write a new file.")
        return 1
    with open(args.config, encoding="utf-8") as f:
        target = tuple(json.load(f).get("target") or (18.38, 77.58))
    mars, earth = load_bundle(args.bundle)
    sheet = build_sheet(mars, earth, target, size=args.size)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sheet.to_csv(args.out, index=False, encoding="utf-8-sig")   # utf-8-sig so Excel shows ° correctly
    near = int((sheet["nearest_catalogue_site"] != "").sum())
    print(f"Wrote {args.out}: {len(sheet)} pairs, {near} of them near a catalogue site.")
    print("Fill in the label column (1 = look alike, 0 = clearly differ, blank = not sure).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
