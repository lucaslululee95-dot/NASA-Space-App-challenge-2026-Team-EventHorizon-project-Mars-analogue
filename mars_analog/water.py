"""Spot Earth squares that are mostly sea or other water.

Over open sea the elevation model reads about 0 m, plant cover (NDVI) reads
below zero, and sometimes there's no satellite image at all. Any of these on
most of a square means it isn't dry land, so it shouldn't be compared with Mars.
"""
import numpy as np

SEA_LEVEL_M = 1.0           # within this of 0 m counts as sea level
SEA_LEVEL_SHARE = 0.5       # more than this share of the square at sea level means water
WATER_NDVI = -0.05          # open water has clearly negative NDVI; dry ground sits just above 0


def water_reason(tile):
    """Why an Earth square looks like water, in plain words, or None if it looks like land."""
    if tile.planet != "earth":
        return None
    if tile.rgb is not None and float(np.std(tile.rgb)) < 1e-6:
        return "no satellite image"
    if float(np.mean(np.abs(tile.dem) <= SEA_LEVEL_M)) > SEA_LEVEL_SHARE:
        return "mostly at sea level"
    if tile.ndvi is not None and tile.ndvi < WATER_NDVI:
        return "plant cover reads as water"
    return None


def is_water(tile):
    return water_reason(tile) is not None
