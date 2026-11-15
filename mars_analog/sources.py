"""Loaders that turn raw data into Tile objects.

rasterio and earthengine-api are imported inside the functions that need
them, so the demo and tests run with only numpy, scipy and pandas.
"""
import math
import os
import pickle

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from .features import stretch
from .geo import box_from_center, utm_epsg
from .tile import Tile


# ------------------------------------------------------------------ Mars

MARS_SPHERE_RADIUS_M = 3396190.0  # Mars (2015) sphere, used by the USGS CTX DTMs


def _sphere_radius(crs):
    """The sphere radius of a projected Mars raster, or the USGS default."""
    try:
        d = crs.to_dict()
        return float(d.get("R") or d.get("a") or MARS_SPHERE_RADIUS_M)
    except Exception:
        return MARS_SPHERE_RADIUS_M


def _to_raster_bounds(src, bbox_deg):
    """Express a lat/lon box in the raster's own coordinates.

    Geographic rasters are used as they are. Projected rasters (such as the
    USGS equirectangular Mars mosaics, whose true-scale latitude varies) are
    converted with PROJ, using a lat/lon system on the raster's own sphere.
    """
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds

    if src.crs is None:
        raise ValueError(f"{src.name} has no coordinate system. Set one before loading.")
    if src.crs.is_geographic:
        return bbox_deg
    lonlat = CRS.from_proj4(f"+proj=longlat +R={_sphere_radius(src.crs)} +no_defs")
    return transform_bounds(lonlat, src.crs, *bbox_deg, densify_pts=21)


def _read_window(src, bbox, n, resampling):
    from rasterio.windows import from_bounds

    west, south, east, north = bbox
    # rasterio silently fills out-of-raster reads with zeros, so check first.
    b = src.bounds
    if west < b.left or east > b.right or south < b.bottom or north > b.top:
        raise ValueError("tile extends beyond the raster")
    win = from_bounds(west, south, east, north, transform=src.transform)
    # masked=True keeps declared gap values out of the resampling. Without it,
    # gap values (e.g. -32767) blend into real heights at the edges of gaps.
    arr = src.read(out_shape=(src.count, n, n), window=win,
                   resampling=resampling, masked=True)
    return arr.astype(float).filled(np.nan)


def nearest_chem(df, lat, lon):
    """Element values from the grid cell nearest to (lat, lon)."""
    d = (df["lat"] - lat) ** 2 + ((df["lon"] - lon) * np.cos(np.radians(lat))) ** 2
    row = df.loc[d.idxmin()]
    fields = [c for c in df.columns if c not in ("lat", "lon")]
    return {f: float(row[f]) for f in fields}


def mars_tiles_from_geotiff(centers, dem_path, image_path, chem_csv, tile_km, n=64,
                            max_gap=0.02):
    """Build Mars tiles from a DEM and an image mosaic that cover the same area.

    Both rasters must use a geographic (lat/lon) CRS. Reproject with gdalwarp
    first if they do not. chem_csv needs columns lat, lon and one column per
    element, for example Fe, Si, S, Cl, K (wt %).
    """
    import rasterio
    from rasterio.enums import Resampling

    chem = pd.read_csv(chem_csv) if chem_csv else None
    tiles, skipped, gapped = [], 0, 0
    with rasterio.open(dem_path) as dem_src, rasterio.open(image_path) as img_src:
        dem_bounds = dem_src.bounds
        for lat, lon in centers:
            bbox = box_from_center(lat, lon, tile_km, "mars")
            try:
                dem = _read_window(dem_src, _to_raster_bounds(dem_src, bbox), n, Resampling.bilinear)[0]
                img = _read_window(img_src, _to_raster_bounds(img_src, bbox), n, Resampling.bilinear)
            except Exception:
                skipped += 1
                continue
            if np.isnan(dem).mean() > max_gap or np.isnan(img).mean() > max_gap:
                gapped += 1
                continue
            dem = np.where(np.isnan(dem), np.nanmedian(dem), dem)
            img = np.where(np.isnan(img), np.nanmedian(img), img)
            tiles.append(Tile(
                name=f"mars_{lat:.3f}_{lon:.3f}", planet="mars", lat=lat, lon=lon,
                tile_km=tile_km, dem=dem, rgb=stretch(img),
                chem=nearest_chem(chem, lat, lon) if chem is not None else {},
            ))
    print(f"[mars] kept {len(tiles)} tiles, skipped {gapped} with more than {max_gap:.0%} gaps")
    if skipped:
        print(f"[mars] skipped {skipped} tiles (outside the rasters or unreadable)")
        print(f"[mars] DEM covers {dem_bounds} in its own units. Check your bbox against it.")
    return tiles


# ----------------------------------------------------------------- Earth

EARTH_CACHE_VERSION = "avg-v1"  # bump this if the Earth tile method changes


def _load_cache(path):
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def _save_cache(path, cache):
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(cache, f)


def earth_tiles_from_gee(centers, tile_km, n=64, project=None, year=2023, cache_path=None):
    """Build Earth tiles from Sentinel-2 surface reflectance and the Copernicus 30 m DEM.

    Needs `pip install earthengine-api` and a one-time `earthengine authenticate`.
    Each tile is a separate Earth Engine request, so hundreds of tiles can take
    a while. Save the returned tiles if you plan to re-run the comparison.
    """
    import ee

    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()

    s2_year = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
               .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)))
    dem_img = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").mosaic().select("DEM")

    cache = _load_cache(cache_path)
    tiles, failed, from_cache = [], 0, 0
    for i, (lat, lon) in enumerate(centers):
        key = (EARTH_CACHE_VERSION, round(lat, 4), round(lon, 4), tile_km, n, year)
        if key in cache:
            tiles.append(cache[key])
            from_cache += 1
            continue
        bbox = box_from_center(lat, lon, tile_km, "earth")
        try:
            geom = ee.Geometry.Rectangle(list(bbox))
            # Average the 10 m and 30 m pixels into each output cell before resampling,
            # so each tile cell is an average, not one sampled pixel.
            s2 = (s2_year.filterBounds(geom).median()
                  .select(["B2", "B3", "B4", "B8"]).multiply(0.0001).setDefaultProjection(crs=f"EPSG:{utm_epsg(lon, lat)}", scale=10)
                  .reduceResolution(ee.Reducer.mean(), False, 1024))
            dem_avg = dem_img.setDefaultProjection(crs=f"EPSG:{utm_epsg(lon, lat)}", scale=30).reduceResolution(ee.Reducer.mean(), False, 1024)
            img = (s2.addBands(dem_avg)
                   .reproject(crs=f"EPSG:{utm_epsg(lon, lat)}",
                              scale=tile_km * 1000.0 / n))
            data = img.sampleRectangle(region=geom, defaultValue=0).getInfo()["properties"]
            band = {k: np.asarray(v, dtype=float)[:n, :n] for k, v in data.items()}
            if any(band[k].shape != (n, n) for k in ("B2", "B3", "B4", "B8", "DEM")):
                failed += 1
                continue
        except Exception:
            failed += 1
            continue

        b2, b3, b4, b8 = band["B2"], band["B3"], band["B4"], band["B8"]
        ndvi = float(np.mean((b8 - b4) / (b8 + b4 + 1e-6)))
        ferric = float(np.mean(b4 / (b2 + 1e-6)))
        tiles.append(Tile(
            name=f"earth_{lat:.3f}_{lon:.3f}", planet="earth", lat=lat, lon=lon,
            tile_km=tile_km, dem=band["DEM"], rgb=stretch(np.stack([b4, b3, b2])),
            ndvi=ndvi, chem={"ferric_index": ferric},
        ))
        cache[key] = tiles[-1]          # save after every tile, so a stopped run keeps its progress
        _save_cache(cache_path, cache)
        if (i + 1) % 25 == 0:
            print(f"[earth] {i + 1}/{len(centers)} tiles requested")
    if from_cache:
        print(f"[earth] {from_cache} tiles loaded from cache")
    if failed:
        print(f"[earth] {failed} tiles failed (no imagery, quota or network error)")
    return tiles


# ------------------------------------------------------------- synthetic

_STYLES = {
    "smooth": dict(amp=8.0, sigma=10.0, tex_amp=0.02, tex_sigma=6.0),   # dunes, playas
    "medium": dict(amp=80.0, sigma=4.0, tex_amp=0.06, tex_sigma=3.0),   # lava plains, fans
    "rugged": dict(amp=300.0, sigma=1.5, tex_amp=0.15, tex_sigma=1.5),  # badlands, craters
}


def _noise(rng, n, sigma):
    x = gaussian_filter(rng.standard_normal((n, n)), sigma)
    return (x - x.mean()) / (x.std() + 1e-9)


def synthetic_tiles(planet, count, n=64, tile_km=10.0, seed=0):
    """Made-up tiles in three terrain styles. For demos and tests only, not real data."""
    rng = np.random.default_rng(seed)
    styles = list(_STYLES)
    tiles = []
    for i in range(count):
        style = styles[i % len(styles)]
        p = _STYLES[style]
        dem = 2000.0 + p["amp"] * rng.uniform(0.7, 1.3) * _noise(rng, n, p["sigma"])
        lum = np.clip(0.4 + p["tex_amp"] * _noise(rng, n, p["tex_sigma"]), 0.01, 1.0)
        if planet == "earth":
            rgb = np.stack([lum * 1.2, lum, lum * 0.8])   # reddish, like iron-oxide soil
            ndvi = 0.4 if i % 7 == 0 else float(rng.uniform(0.02, 0.12))
            chem = {"ferric_index": float(rng.normal(1.2, 0.1))}
            lat, lon = float(rng.uniform(-30, 30)), float(rng.uniform(-70, 70))
        else:
            rgb = lum[None]                                 # greyscale, like CTX mosaics
            ndvi = None
            chem = {"Fe": float(rng.normal(17, 3))}
            lat, lon = float(rng.uniform(10, 25)), float(rng.uniform(70, 85))
        tiles.append(Tile(
            name=f"{planet}_{style}_{i:03d}", planet=planet, lat=lat, lon=lon,
            tile_km=tile_km, dem=dem, rgb=stretch(rgb), ndvi=ndvi, chem=chem,
        ))
    return tiles
