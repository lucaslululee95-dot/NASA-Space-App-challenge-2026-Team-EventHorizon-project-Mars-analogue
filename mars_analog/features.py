"""Features computed the same way for Mars and Earth tiles.

Every tile covers the same physical area on both planets (tile_km) and is
resampled to the same grid (n x n pixels), so slopes, roughness and texture
are directly comparable.
"""
import numpy as np
import pandas as pd
from scipy.ndimage import laplace, uniform_filter

# Terrain carries the most weight because it is the most comparable across planets.
# Colour is down-weighted because Mars and Earth colour calibration differ.
FEATURE_WEIGHTS = {
    "slope_mean_deg": 1.0,
    "slope_p90_deg": 1.0,
    "relief_m": 1.0,
    "roughness_m": 1.5,
    "curvature_per_m": 0.5,
    "texture_rel": 1.0,
    "grad_rel": 1.0,
    "red_blue_ratio": 0.25,
}
# Skewed, strictly positive features are log-transformed before scaling.
LOG_FEATURES = {"relief_m", "roughness_m", "curvature_per_m"}


def stretch(rgb):
    """Map all bands of one tile to 0-1 using the tile's own 2nd and 98th percentiles."""
    lo, hi = np.nanpercentile(rgb, [2, 98])
    return np.clip((rgb - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def _local_std(a, size=3):
    m = uniform_filter(a, size)
    m2 = uniform_filter(a * a, size)
    return np.sqrt(np.clip(m2 - m * m, 0.0, None))


def terrain_features(dem, tile_km):
    n = dem.shape[0]
    px_m = tile_km * 1000.0 / n
    gy, gx = np.gradient(dem, px_m)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    curvature = np.abs(laplace(dem)) / px_m ** 2
    return {
        "slope_mean_deg": float(slope.mean()),
        "slope_p90_deg": float(np.percentile(slope, 90)),
        "relief_m": float(np.percentile(dem, 95) - np.percentile(dem, 5)),
        "roughness_m": float(np.median(_local_std(dem, 3))),
        "curvature_per_m": float(curvature.mean()),
    }


def surface_features(rgb):
    """Texture features that do not depend on brightness, gain or stretch.

    Each texture measure is divided by the tile's own global spread, so an
    affine change in brightness (what differs between Mars and Earth imagery)
    cancels out. Colour ratios are not affine-invariant, so they get low weight.
    """
    if rgb.shape[0] >= 3:
        r, g, b = rgb[0], rgb[1], rgb[2]
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        red_blue = float(r.mean() / (b.mean() + 1e-6))
    else:
        lum = rgb[0]
        red_blue = np.nan  # greyscale image: no colour information
    spread = float(lum.std()) + 1e-9
    gy, gx = np.gradient(lum)
    return {
        "texture_rel": float(_local_std(lum, 3).mean() / spread),
        "grad_rel": float(np.hypot(gx, gy).mean() / spread),
        "red_blue_ratio": red_blue,
    }


def feature_table(tiles):
    """One row per tile: identifiers, terrain/surface features, optional NDVI and chemistry."""
    rows = []
    for t in tiles:
        row = {"name": t.name, "planet": t.planet, "lat": t.lat, "lon": t.lon}
        row.update(terrain_features(t.dem, t.tile_km))
        if t.rgb is not None:
            row.update(surface_features(t.rgb))
        if t.ndvi is not None:
            row["ndvi"] = t.ndvi
        row.update({f"chem_{k}": v for k, v in t.chem.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def feature_columns(mars_df, earth_df):
    """Features that exist and are finite on both planets."""
    return [
        c for c in FEATURE_WEIGHTS
        if c in mars_df and c in earth_df
        and mars_df[c].notna().all() and earth_df[c].notna().all()
    ]
