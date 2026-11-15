"""Features computed the same way for Mars and Earth tiles.

Every tile covers the same physical area on both planets (tile_km) and is
resampled to the same grid (n x n pixels), so slopes, roughness and texture
are directly comparable.
"""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, laplace, maximum_filter, minimum_filter, uniform_filter

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
# Extra terrain measurements for the learned model only. They describe landforms at
# different sizes. The hand-set score above never uses them, so it stays unchanged.
EXTRA_FEATURES = [
    "roughness_large_m",    # bumpiness over about 1.4 km, against 0.5 km for roughness_m
    "slope_std_deg",        # how much the slope varies across the tile
    "aspect_spread",        # 0 = slopes all face one way, 1 = they face every way
    "peaks_per_100km2",     # hills and hollows at about the 1 km scale
    "hypsometric",          # 0 = mostly low ground with a few highs, 1 = mostly high ground
    "slope_coarse_deg",     # slope after smoothing over about 0.5 km
]
# Skewed, strictly positive features are log-transformed before scaling.
LOG_FEATURES = {"relief_m", "roughness_m", "curvature_per_m", "roughness_large_m", "peaks_per_100km2"}


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
        **landform_features(dem, px_m, slope, gx, gy),
    }


def landform_features(dem, px_m, slope, gx, gy, prominence_m=1.0):
    """The EXTRA_FEATURES, from the same elevation grid."""
    n = dem.shape[0]
    mag = np.hypot(gx, gy)
    total = mag.sum()
    spread = 0.0 if total == 0 else 1.0 - np.hypot(gx.sum(), gy.sum()) / total

    smooth = gaussian_filter(dem, 1.0)
    hi, lo = maximum_filter(smooth, size=5), minimum_filter(smooth, size=5)
    peaks = (smooth == hi) & (smooth - lo > prominence_m)
    pits = (smooth == lo) & (hi - smooth > prominence_m)
    inside = np.zeros_like(peaks)
    inside[2:-2, 2:-2] = True                          # ignore the edges, where windows are cut off
    area_km2 = (px_m * (n - 4) / 1000.0) ** 2

    p2, p98 = np.percentile(dem, [2, 98])
    hyps = 0.5 if p98 - p2 < 1e-6 else float(np.clip((dem.mean() - p2) / (p98 - p2), 0.0, 1.0))

    cy, cx = np.gradient(gaussian_filter(dem, 3.0), px_m)
    return {
        "roughness_large_m": float(np.median(_local_std(dem, 9))),
        "slope_std_deg": float(slope.std()),
        "aspect_spread": float(spread),
        "peaks_per_100km2": float(((peaks | pits) & inside).sum() / area_km2 * 100.0),
        "hypsometric": hyps,
        "slope_coarse_deg": float(np.degrees(np.arctan(np.hypot(cx, cy))).mean()),
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


def feature_columns(mars_df, earth_df, extra=False):
    """Features that exist and are finite on both planets. extra=True adds EXTRA_FEATURES."""
    names = list(FEATURE_WEIGHTS) + (EXTRA_FEATURES if extra else [])
    return [
        c for c in names
        if c in mars_df and c in earth_df
        and mars_df[c].notna().all() and earth_df[c].notna().all()
    ]
