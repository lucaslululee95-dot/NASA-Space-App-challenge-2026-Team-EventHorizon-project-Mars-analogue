# mars_analog

Finds where Mars looks most like Earth, and where Earth looks most like a chosen
Mars site. Terrain is compared first, then iron chemistry is used to re-rank the matches.

## How it works

1. **Tiles.** Mars and Earth are cut into square patches of the same physical size
   (default 10 km), each resampled to 64 x 64 pixels.
2. **Features.** Every tile gets the same measurements: slope, relief, roughness,
   curvature, and two texture ratios. Colour ratio is included with low weight.
3. **Shared scale.** Both planets are scaled together, so one ruler applies to both.
4. **Step 1, Mars to Earth.** For each Mars tile, find the closest Earth tile.
   The Mars tiles with the closest matches are the ones whose terrain looks most Earth-like.
5. **Step 2, pick a target.** Use a Mars site you choose (for example Jezero
   Crater), or the most Earth-like Mars tile from step 1.
6. **Step 3, Earth to Mars.** Rank Earth tiles by terrain similarity to the target,
   then blend in an iron-chemistry score. Each match lists the features where it
   agrees and disagrees with the target.

Output is two CSV files, `mars_most_earthlike.csv` and `earth_matches_for_target.csv`,
written to a `results` folder.

## Project layout

```
mars_analog/
  geo.py          tile boxes, grids, UTM zones
  tile.py         Tile dataclass
  features.py     terrain and surface features
  chemistry.py    chemistry comparison (percentile based, explicit axis coverage)
  matching.py     shared scaling, similarity matrix, explanations
  sources.py      loaders: Mars GeoTIFF + element CSV, Earth Engine, synthetic
  pipeline.py     the three steps
  cli.py          command line
tests/            pytest suite
config.example.json
```

## Setup on Windows (PowerShell)

Run every command from inside the project folder. If you run `pytest` from
somewhere else, such as your home folder, it searches every file under that
folder, which is slow and produces errors from unrelated files.

```powershell
cd C:\Users\lucas\Downloads\mars_analog
python -m pip install numpy pandas scipy pytest
python -m pytest tests
python -m mars_analog.cli --demo
```

The demo uses made-up terrain, so its numbers mean nothing scientifically. It
exists to show the pipeline and the output format. The "Defaulting to user
installation" message from pip is normal.

## Running on real data

You need three Mars inputs and one Earth source. The paths go in `config.json`,
which you make by copying `config.example.json`.

### 1. Mars image: the CTX global mosaic

The Murray Lab CTX mosaic is a 5 m per pixel grayscale global map, published as
4 x 4 degree GeoTIFF tiles through USGS Astrogeology:
https://astrogeology.usgs.gov/search/map/mars-mro-ctx-global-mosaic-murray-lab-v1

The page asks you to download one tile at a time. Pick the tile that covers your
study area, such as Jezero (about 18.4 N, 77.6 E).

### 2. Mars elevation

Two options:

- **CTX targeted DTMs** (NASA/USGS, controlled to MOLA, public domain). These are
  high-resolution elevation models for specific sites, so check coverage for your
  area: https://registry.opendata.aws/nasa-usgs-controlled-mro-ctx-dtms
- **MOLA gridded elevation** from the PDS Geosciences Node. It is global but coarse,
  about 463 m per pixel: https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/
  With MOLA, raise `tile_km` to 40-50 km. A 10 km tile would be only about 22 pixels across.

Both rasters must use a geographic lat/lon coordinate system. If yours do not,
reproject them first (for example with `gdalwarp -t_srs EPSG:4326`).

### 3. Mars chemistry: GRS element maps

The Odyssey Gamma Ray Spectrometer element maps are on the PDS Geosciences Node:
https://pds-geosciences.wustl.edu/missions/odyssey/grs_elements.html

The data is in ASCII `.TAB` tables with PDS label files. They cover roughly
±45° latitude and mask the polar regions. Each file is binned at 2, 5, or 10
degrees. Convert the tables into a CSV with the columns `lat, lon, Fe, Si, Cl, K`.
The product does not include sulfur, so that axis is skipped.

To build the chemistry file, save `fe_5x5.tab` into `data/mars` and run
`python convert_grs.py`. It writes `data/mars/elements_grid.csv` with the iron value
and its rank among all valid Mars cells.

**Resolution warning.** A 5 degree cell is roughly 300 km across on Mars. Every
10 km tile inside the same cell gets the same chemistry value, so treat the
chemistry score as regional. The 2 degree product gives a finer picture if you
can get it.

### 4. Earth: Google Earth Engine

1. Go to https://earthengine.google.com/signup, sign in with a Google account, and
   choose the noncommercial option. This creates a Google Cloud project. Google
   verifies noncommercial projects, so registration may take a while.
2. Note your project ID.
3. Install the client and log in once:
   ```powershell
   python -m pip install earthengine-api rasterio
   python -c "import ee; ee.Authenticate()"
   ```
   A browser window opens for the Google login. The `earthengine` command is not
   on PATH for Windows Store installs of Python, so use the `python -c` form.
4. Put your project ID in `config.json` and run:
   ```powershell
   python -m mars_analog.cli --config config.json
   ```

Earth tiles come from Sentinel-2 surface reflectance (`COPERNICUS/S2_SR_HARMONIZED`)
and the Copernicus 30 m DEM (`COPERNICUS/DEM/GLO30_2024_1`, mosaicked). Tiles with mean NDVI above
`max_ndvi` (default 0.15) are dropped, because the Mars side has no vegetation.
Each tile is one Earth Engine request, so the first run is slow. Downloaded tiles are
saved to `data/cache/earth_tiles.pkl`, and later runs load them from there. Delete that file
to download everything again, for example after changing the Earth tile method.

## Covering the whole planet

The current code handles a few hundred tiles. A whole-planet run needs a different
approach, for three reasons:

- **Mars imagery is too large for a home computer.** The full CTX mosaic is about
  5.6 TB compressed (11.5 TB uncompressed), so download only the tiles you need.
- **The code keeps every tile in memory.** A 40 km grid over all of Mars is about
  90,000 tiles, which would need several gigabytes. A whole-planet run must compute
  features tile by tile and keep only the numbers.
- **Earth Engine requests are one at a time.** Requesting pixels for each tile works
  for hundreds of tiles but not for the whole Earth. Whole-planet Earth work has to
  run inside Earth Engine: compute the features for a grid of cells there and export
  a table. Noncommercial projects have a monthly compute quota, so plan exports carefully.

A practical order:

1. **Mars chemistry:** download the full GRS element tables. They are small ASCII files.
2. **Mars terrain, global pass:** use the global MOLA elevation with 40-50 km tiles,
   computing terrain features only.
3. **Mars candidates:** take the best few hundred regions from step 2, then download
   CTX tiles only for those and add the texture features.
4. **Earth, global pass:** compute terrain features in Earth Engine over dry land and
   export a table.
5. **Compare:** merge the two tables and score them with the same method. The code
   already uses only the features both sides have, so terrain-only runs work.

## Scoring

- **Terrain similarity** is a weighted distance in a shared, scaled feature space,
  converted to a 0 to 1 score. Roughness and texture have the most weight. Colour
  has the least, because Mars and Earth colour calibration differ.
- **Chemistry** is currently iron only. Mars Fe and the Earth iron-oxide ratio
  have different units, so each value is converted to a percentile within its own
  planet, and the two percentiles are compared. Two sites score high when they sit
  at a similar relative iron level on their own planet.
- **Final score** is a weighted average (default 0.7 terrain, 0.3 chemistry). If a
  tile has no chemistry data, the terrain score is used alone.

Each match lists the features where it agrees and disagrees with the target. Use
those to judge a match, not just the number.

## Limits

- **Not yet tested on real Mars or Earth Engine data.** The development environment
  could not reach those services. The synthetic demo, the feature code, and the
  GeoTIFF loader (tested on a small fake raster) all run. The Earth Engine loader
  is written against the documented API but has not been run. Expect to fix small
  issues on the first real run.
- **Chemistry covers iron only, at regional resolution.** The GRS cells are 2 to 10
  degrees. Silicon, chlorine, and potassium are loaded when present but are not
  compared, because Earth has no matching proxy yet. The output lists which axes
  were compared and which were skipped.
- **Earth DEM is a surface model.** It includes buildings and tree canopy. The dry
  filter helps, but urban tiles can still get through.
- **Mars colour and lighting are not calibrated against Earth.** Colour is
  down-weighted for this reason. Check colour-driven matches by eye.
- **Gaps in Mars elevation data.** Tiles with more than `max_gap_fraction` (default 2%) of
  gaps are skipped, not filled. Filling large gaps with a median height makes whole tiles
  look artificially flat, so the run reports how many tiles were kept and how many were dropped.
- **Dry Earth is not Mars.** Mars has about 0.6% of Earth's surface pressure, a
  mean temperature near -60 C, and no stable surface liquid water. Matches show
  shape and chemistry overlap, not a full environmental match.

## Next steps

- Add DINOv2 image embeddings as an extra terrain and texture score.
- Add Earth mineral maps (NASA EMIT) for a better iron and mineral comparison.
- Add a folium map of the top matches.
- Cache tile features to disk so re-runs do not re-query Earth Engine.
