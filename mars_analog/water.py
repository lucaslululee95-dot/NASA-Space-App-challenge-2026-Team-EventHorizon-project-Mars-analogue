"""Spot Earth squares that aren't dry land: sea, lakes, snow and ice.

Over open sea the elevation model reads about 0 m and sometimes there's no
satellite image at all. Water and snow both make plant cover (NDVI) read below
zero. Snow and ice are told apart from rock with the normalised difference snow
index (NDSI), from the green and short-wave infrared bands; squares downloaded
before that band was added have snow_fraction None, so for them a negative NDVI
can only be called "water or ice".
"""
import numpy as np

SEA_LEVEL_M = 1.0           # within this of 0 m counts as sea level
SEA_LEVEL_SHARE = 0.5       # more than this share of the square at sea level means water
WATER_NDVI = -0.05          # water and snow read clearly below 0; dry ground sits just above
SNOW_SHARE = 0.5            # more than this share of snow or ice means it isn't comparable
NDSI_SNOW = 0.4             # standard snow threshold (as for MODIS and Landsat snow maps)
NIR_SNOW = 0.11             # snow is also bright in near-infrared; water isn't


def snow_fraction(b3, b8, b11):
    """Share of pixels that are snow or ice, from green, near-infrared and SWIR reflectance."""
    ndsi = (b3 - b11) / (b3 + b11 + 1e-6)
    return float(np.mean((ndsi > NDSI_SNOW) & (b8 > NIR_SNOW)))


def water_reason(tile):
    """Why an Earth square isn't dry land, in plain words, or None if it is."""
    if tile.planet != "earth":
        return None
    if tile.rgb is not None and float(np.std(tile.rgb)) < 1e-6:
        return "no satellite image"
    if float(np.mean(np.abs(tile.dem) <= SEA_LEVEL_M)) > SEA_LEVEL_SHARE:
        return "mostly at sea level"
    if tile.snow_fraction is not None and tile.snow_fraction > SNOW_SHARE:
        return "mostly snow or ice"
    if tile.ndvi is not None and tile.ndvi < WATER_NDVI:
        return "plant cover reads as water" if tile.snow_fraction is not None \
            else "plant cover reads as water or ice"
    return None


def is_water(tile):
    """True for anything that isn't dry land, including snow and ice."""
    return water_reason(tile) is not None
