"""Build the web app's data bundle from the files on your PC.

Run this from the project folder, after the Mars files exist and the Earth
comparison has run at least once:

    python bundle_data.py

It writes the bundle folder that app.py reads. The Earth cache is a pickle file
written by this project, so only load pickle files you created yourself.
"""
import json
import pickle

from mars_analog.bundle import save_bundle
from mars_analog.geo import grid_centers
from mars_analog.sources import EARTH_CACHE_VERSION, mars_tiles_from_geotiff


def earth_keys(boxes, step_deg, tile_km, n, year):
    """The cache key of every Earth tile the config asks for."""
    keys = []
    for box in boxes:
        for lat, lon in grid_centers(box, step_deg):
            keys.append((EARTH_CACHE_VERSION, round(lat, 4), round(lon, 4), tile_km, n, year))
    return keys


def select_earth(cache, keys):
    """Tiles found in the cache, plus the keys that are missing."""
    tiles = [cache[k] for k in keys if k in cache]
    missing = [k for k in keys if k not in cache]
    return tiles, missing


def main(config_path="config.json", bundle_dir="bundle"):
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    tile_km, n = cfg["tile_km"], cfg.get("grid_px", 64)
    m, e = cfg["mars"], cfg["earth"]

    mars = mars_tiles_from_geotiff(
        grid_centers(m["bbox"], m["step_deg"]), m["dem_geotiff"], m["image_geotiff"],
        m.get("chemistry_csv"), tile_km, n, max_gap=m.get("max_gap_fraction", 0.02))

    with open(e["cache"], "rb") as f:
        cache = pickle.load(f)
    keys = earth_keys(e["boxes"], e["step_deg"], tile_km, n, e.get("year", 2023))
    earth, missing = select_earth(cache, keys)
    if missing:
        print(f"[bundle] {len(missing)} Earth tiles are not in the cache and were left out.")
        print("[bundle] Run the comparison once to download them, then run this again.")

    save_bundle(bundle_dir, mars, earth)
    print(f"[bundle] wrote {len(mars)} Mars tiles and {len(earth)} Earth tiles to {bundle_dir}/")


if __name__ == "__main__":
    main()
