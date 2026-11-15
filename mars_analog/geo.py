"""Geometry helpers shared by all loaders."""
import math

import numpy as np

RADIUS_KM = {"earth": 6371.0, "mars": 3389.5}


def km_per_degree(planet: str) -> float:
    return 2 * math.pi * RADIUS_KM[planet] / 360.0


def box_from_center(lat, lon, tile_km, planet="earth"):
    """(west, south, east, north) in degrees for a square tile tile_km on a side."""
    kpd = km_per_degree(planet)
    dlat = (tile_km / 2) / kpd
    dlon = (tile_km / 2) / (kpd * max(math.cos(math.radians(lat)), 1e-6))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def grid_centers(bbox, step_deg):
    """Tile centres on a regular lat/lon grid. bbox = (west, south, east, north)."""
    west, south, east, north = bbox
    lats = np.arange(south, north + 1e-9, step_deg)
    lons = np.arange(west, east + 1e-9, step_deg)
    return [(float(a), float(b)) for a in lats for b in lons]


def utm_epsg(lon, lat):
    """EPSG code of the UTM zone containing (lon, lat). Used for metric resampling."""
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def km_between(lat1, lon1, lat2, lon2, planet="earth"):
    """Great-circle distance in km on Earth or Mars."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * RADIUS_KM[planet] * math.asin(min(1.0, math.sqrt(a)))


def nearest_tile(tiles, lat, lon, planet):
    """The tile whose centre is closest to (lat, lon), and how far away it is in km."""
    best = min(tiles, key=lambda t: km_between(lat, lon, t.lat, t.lon, planet))
    return best, km_between(lat, lon, best.lat, best.lon, planet)
