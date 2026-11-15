"""Save and load a small data bundle for the web app.

The bundle stores tile numbers as NumPy .npz files and their labels as JSON.
Unlike the pickle cache, loading it cannot run code.
"""
import json
import os

import numpy as np

from .tile import Tile

LABELS = ("name", "planet", "lat", "lon", "tile_km", "ndvi", "snow_fraction")


def _finite_or_none(v):
    v = float(v)
    return v if np.isfinite(v) else None


def _save_side(folder, side, tiles):
    if not tiles:
        raise ValueError(f"no {side} tiles to save")
    dem = np.stack([t.dem for t in tiles]).astype(np.float32)
    rgb = np.stack([t.rgb for t in tiles]).astype(np.float32)
    np.savez_compressed(os.path.join(folder, f"{side}.npz"), dem=dem, rgb=rgb)
    labels = []
    for t in tiles:
        row = {k: getattr(t, k) for k in LABELS}
        row["ndvi"] = None if t.ndvi is None else float(t.ndvi)
        row["snow_fraction"] = None if t.snow_fraction is None else float(t.snow_fraction)
        row["chem"] = {k: _finite_or_none(v) for k, v in t.chem.items()}
        labels.append(row)
    with open(os.path.join(folder, f"{side}.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=1)


def save_bundle(folder, mars_tiles, earth_tiles):
    os.makedirs(folder, exist_ok=True)
    _save_side(folder, "mars", mars_tiles)
    _save_side(folder, "earth", earth_tiles)


def _load_side(folder, side):
    with np.load(os.path.join(folder, f"{side}.npz")) as arrays:
        dem, rgb = arrays["dem"], arrays["rgb"]
    with open(os.path.join(folder, f"{side}.json"), encoding="utf-8") as f:
        labels = json.load(f)
    tiles = []
    for i, row in enumerate(labels):
        chem = {k: (np.nan if v is None else float(v)) for k, v in row["chem"].items()}
        tiles.append(Tile(name=row["name"], planet=row["planet"], lat=row["lat"], lon=row["lon"],
                          tile_km=row["tile_km"], dem=dem[i].astype(float),
                          rgb=rgb[i].astype(float), ndvi=row["ndvi"], chem=chem,
                          snow_fraction=row.get("snow_fraction")))
    return tiles


def load_bundle(folder):
    return _load_side(folder, "mars"), _load_side(folder, "earth")
