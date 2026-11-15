"""The unit of comparison: one square patch of ground, the same size on both planets."""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Tile:
    name: str
    planet: str                              # "earth" or "mars"
    lat: float
    lon: float
    tile_km: float                           # physical side length, must match across planets
    dem: np.ndarray                          # (n, n) elevation in metres
    rgb: Optional[np.ndarray]                # (bands, n, n), values 0-1; 1 band = greyscale
    ndvi: Optional[float] = None             # Earth only: mean NDVI, used to drop vegetated tiles
    chem: dict = field(default_factory=dict) # scalar chemistry values, e.g. {"Fe": 17.2}
    snow_fraction: Optional[float] = None    # Earth only: share of the square that is snow or ice
