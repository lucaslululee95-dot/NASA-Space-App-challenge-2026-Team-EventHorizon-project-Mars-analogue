import numpy as np
import pandas as pd
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from mars_analog.geo import grid_centers  # noqa: E402
from mars_analog.sources import mars_tiles_from_geotiff  # noqa: E402


def _write_fake_mars(tmp_path):
    """A 2-degree geographic raster around (lat 18, lon 77) at 0.005 deg per pixel."""
    res = 0.005
    west, north = 76.0, 19.0
    h = w = int(2.0 / res)
    rng = np.random.default_rng(0)
    dem = (1000 + 50 * rng.standard_normal((h, w))).astype("float32")
    img = (0.4 + 0.05 * rng.standard_normal((h, w))).astype("float32")
    tf = from_origin(west, north, res, res)
    profile = dict(driver="GTiff", height=h, width=w, count=1, dtype="float32",
                   crs="EPSG:4326", transform=tf, nodata=-9999)
    dem_path, img_path = tmp_path / "dem.tif", tmp_path / "img.tif"
    for path, arr in [(dem_path, dem), (img_path, img)]:
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(arr, 1)
    chem_path = tmp_path / "chem.csv"
    pd.DataFrame({"lat": [18.0, 18.5], "lon": [77.0, 77.5],
                  "Fe": [15.0, 19.0], "Si": [22.0, 20.0]}).to_csv(chem_path, index=False)
    return dem_path, img_path, chem_path


def test_geotiff_tiles_load_with_chemistry(tmp_path):
    dem_path, img_path, chem_path = _write_fake_mars(tmp_path)
    centers = grid_centers((77.0, 18.0, 77.5, 18.5), 0.25)
    tiles = mars_tiles_from_geotiff(centers, str(dem_path), str(img_path), str(chem_path),
                                    tile_km=10, n=32)
    assert len(tiles) == len(centers) == 9
    t = tiles[0]
    assert t.dem.shape == (32, 32) and t.rgb.shape == (1, 32, 32)
    assert np.isfinite(t.dem).all()
    assert 0.0 <= t.rgb.min() and t.rgb.max() <= 1.0
    assert set(t.chem) == {"Fe", "Si"}


def test_tiles_outside_raster_are_skipped(tmp_path):
    dem_path, img_path, _ = _write_fake_mars(tmp_path)
    tiles = mars_tiles_from_geotiff([(50.0, 10.0)], str(dem_path), str(img_path),
                                    None, tile_km=10, n=32)
    assert tiles == []


def test_projected_mars_raster_is_read_in_its_own_coordinates(tmp_path):
    """USGS CTX DTMs are equirectangular metres on the Mars sphere, not lat/lon."""
    from rasterio.crs import CRS
    from mars_analog.sources import MARS_SPHERE_RADIUS_M, mars_tiles_from_geotiff

    R = MARS_SPHERE_RADIUS_M
    res = 200.0                                    # metres per pixel
    x0 = R * np.radians(76.5)                      # west edge, lon 76.5 E
    y0 = R * np.radians(19.5)                      # north edge, lat 19.5 N
    w = h = int(R * np.radians(3.0) / res)         # 3 degrees square
    xs = x0 + (np.arange(w) + 0.5) * res
    dem = np.tile(xs, (h, 1))                      # elevation = easting, so the mean reveals position
    img = np.full((h, w), 0.5, dtype="float32")

    crs = CRS.from_string("+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 "
                          "+R=3396190 +units=m +no_defs")
    tf = from_origin(x0, y0, res, res)
    dem_path, img_path = tmp_path / "dtm.tif", tmp_path / "ortho.tif"
    for path, arr, dt in [(dem_path, dem, "float64"), (img_path, img, "float32")]:
        with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1,
                           dtype=dt, crs=crs, transform=tf) as dst:
            dst.write(arr, 1)

    tiles = mars_tiles_from_geotiff([(18.0, 77.5)], str(dem_path), str(img_path),
                                    None, tile_km=10, n=32)
    assert len(tiles) == 1, "tile inside the projected raster should not be skipped"
    expected_easting = R * np.radians(77.5)
    assert abs(tiles[0].dem.mean() - expected_easting) < 50.0
    assert np.isfinite(tiles[0].dem).all()


def _write_gapped_raster(path, value, dtype, gap_from_col, nodata=-32767.0):
    """1 x 1 degree lat/lon raster. Columns from gap_from_col onward are a gap."""
    res = 0.001
    h = w = 1000
    arr = np.full((h, w), value, dtype=dtype)
    arr[:, gap_from_col:] = nodata
    profile = dict(driver="GTiff", height=h, width=w, count=1, dtype=dtype,
                   crs="EPSG:4326", transform=from_origin(76.0, 19.0, res, res), nodata=nodata)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
    return path


def test_edge_tile_has_no_gap_values_in_heights(tmp_path):
    """A tile across a gap edge keeps gap values out of its heights (regression guard)."""
    from mars_analog.sources import mars_tiles_from_geotiff
    dem = _write_gapped_raster(tmp_path / "dem.tif", 1000.0, "float64", gap_from_col=500)
    img = _write_gapped_raster(tmp_path / "img.tif", 0.5, "float32", gap_from_col=500)
    # Tile centred on the gap edge (lon 76.5), about half in gap. max_gap=1.0 keeps it,
    # so this checks the resampling itself, not the gap limit.
    tiles = mars_tiles_from_geotiff([(18.5, 76.5)], str(dem), str(img), None,
                                    tile_km=10, n=32, max_gap=1.0)
    assert len(tiles) == 1
    assert tiles[0].dem.min() > 0, "gap values (-32767) appeared in the heights"
    assert np.allclose(tiles[0].dem, 1000.0)


def test_tiles_with_too_many_gaps_are_skipped(tmp_path):
    from mars_analog.sources import mars_tiles_from_geotiff
    dem = _write_gapped_raster(tmp_path / "dem.tif", 1000.0, "float64", gap_from_col=500)
    img = _write_gapped_raster(tmp_path / "img.tif", 0.5, "float32", gap_from_col=500)
    # Tile centred inside the gap (lon 76.8) and tile straddling it (lon 76.5) are both dropped.
    tiles = mars_tiles_from_geotiff([(18.5, 76.8), (18.5, 76.5)], str(dem), str(img),
                                    None, tile_km=10, n=32, max_gap=0.02)
    assert tiles == []
