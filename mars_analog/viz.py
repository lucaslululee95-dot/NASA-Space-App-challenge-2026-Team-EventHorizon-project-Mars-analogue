"""Turn tile arrays into images for the web app.

Both planets are drawn with the same lighting and no vertical exaggeration,
so a steep slope looks equally steep on Mars and on Earth.
"""
import numpy as np
from scipy.ndimage import zoom


def hillshade(dem, tile_km, azimuth=315.0, altitude=45.0, size=256):
    """Shaded relief as a size x size uint8 image. Light comes from the northwest."""
    n = dem.shape[0]
    px_m = tile_km * 1000.0 / n
    gy, gx = np.gradient(dem.astype(float), px_m)      # gy points south (row order), gx east
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(gy, -gx)
    zenith = np.radians(90.0 - altitude)
    sun = np.radians(360.0 - azimuth + 90.0)
    shade = (np.cos(zenith) * np.cos(slope)
             + np.sin(zenith) * np.sin(slope) * np.cos(sun - aspect))
    shade = zoom(np.clip(shade, 0.0, 1.0), size / n, order=1)
    return (np.clip(shade, 0.0, 1.0) * 255).astype(np.uint8)


def surface_image(rgb, size=256):
    """The tile's satellite image as uint8: colour (H, W, 3) or greyscale (H, W)."""
    n = rgb.shape[-1]
    if rgb.shape[0] >= 3:
        img = np.stack([zoom(rgb[i], size / n, order=1) for i in range(3)], axis=-1)
    else:
        img = zoom(rgb[0], size / n, order=1)
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
