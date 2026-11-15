# Mars Analog Finder

**Try the web app:** https://dvbxnltnopvsnkwnxekywb.streamlit.app/

This tool looks for places on Earth whose terrain and iron content resemble a chosen Mars site. By default the site is Jezero Crater, where NASA's Perseverance rover landed. It compares terrain first, then re-ranks the matches using an iron comparison.

## What it does

1. **Tiles.** Mars and Earth are cut into square tiles of the same size, 10 km by 10 km by default. Each tile is resampled to a 64 by 64 grid.
2. **Measurements.** Each tile gets the same measurements: slope, relief, roughness, curvature, and image texture.
3. **Shared scale.** Mars and Earth numbers are scaled together, so one ruler applies to both.
4. **Mars ranking.** For each Mars tile, the program finds the closest Earth tile. This shows which Mars patches look most Earth-like.
5. **Target.** The program uses the Mars tile nearest to your chosen position, normally Jezero.
6. **Earth ranking.** Each Earth tile gets a terrain score and an iron score. The final score combines them, and each match lists the terrain features it agrees and disagrees with the target on.

The output is two CSV files in a `results` folder: `mars_most_earthlike.csv` and `earth_matches_for_target.csv`.

## Project layout

```
mars_analog/
  config.example.json    template for your settings
  config.json            your settings and Earth regions
  requirements.txt       packages for the web app
  requirements-local.txt packages for the comparison and tests
  app.py                 the web app
  bundle_data.py         builds the web app's data bundle
  bundle/                the bundled data the web app reads
  convert_grs.py         turns the GRS iron and chlorine tables into a CSV
  check_earth_engine.py  optional check that Earth Engine works
  WEB_APP.md             how to build, run, and publish the web app
  make_label_sheet.py    makes a sheet of Mars-Earth pairs for people to label
  label_app.py           a page for labelling pairs side by side
  train_model.py         learns measurement weights from the labelled pairs
  LABELLING.md           how to label pairs
  labels/                the label sheet
  models/                the learned weights
  mars_analog/           the comparison code
    cli.py               the command you run
    sources.py           reads Mars files, downloads Earth tiles, cache
    features.py          measures each tile
    matching.py          scales numbers and computes similarity
    chemistry.py         iron comparison
    pipeline.py          the three comparison steps
    bundle.py            saves and loads the web app's data bundle
    regions.py           names of the Earth regions
    viz.py               shaded relief and images for the web app
    geo.py, tile.py      tile sizes, grids, and the tile structure
  tests/                 automatic tests
  data/                  your Mars files and Earth cache (not in GitHub)
  results/               output tables (not in GitHub)
```

## Set up (Windows, PowerShell)

Run every command from the project folder. Running `pytest` from another folder searches unrelated files, which is slow and gives confusing errors.

```powershell
cd "path\to\mars_analog"
python -m pip install -r requirements-local.txt
python -m pytest tests
python -m mars_analog.cli --demo
```

The demo uses made-up terrain, so its numbers have no scientific meaning. It shows the pipeline and the output format. You don't need the real data to run the tests or the demo.

## Get the data

For the real comparison you need three things: Mars terrain and images, the Mars iron and chlorine tables, and a Google Earth Engine account. Create the data folder first:

```powershell
New-Item -ItemType Directory -Force data\mars
```

### 1. Mars elevation and image (CTX)

The Mars side uses two USGS mosaics of Jezero crater, both public domain and both
from the server `planetarymaps.usgs.gov`:

- **Elevation:** the Mars 2020 Science Investigation CTX DEM Mosaic, 20 m per pixel,
  covering the whole crater (about 87 MB). It includes the hand-edited elevation made
  for the rover's landing. https://astrogeology.usgs.gov/search/map/mars_2020_science_investigation_ctx_dem_mosaic
- **Image:** the Mars 2020 Terrain Relative Navigation CTX orthomosaic, 6 m per pixel,
  covering the middle of the crater (about 35 MB).
  https://astrogeology.usgs.gov/search/map/mars_2020_terrain_relative_navigation_context_camera_orthorectified_image_mosaic

```powershell
$ProgressPreference = 'SilentlyContinue'
Invoke-WebRequest -UseBasicParsing "https://planetarymaps.usgs.gov/mosaic/mars2020_trn/CTX/ScienceInvestigationMaps_JPL/M20_JezeroCrater_CTXDEM_20m.tif" -OutFile data\mars\jezero_dem_20m.tif
Invoke-WebRequest -UseBasicParsing "https://planetarymaps.usgs.gov/mosaic/mars2020_trn/CTX/JEZ_ctx_B_soc_008_orthoMosaic_6m_Eqc_latTs0_lon0.tif" -OutFile data\mars\jezero_ortho_6m.tif
```

The image covers less ground than the elevation, so the Mars area in `config.json`
is set to the middle of the crater: 16 spots, including the landing site. A 5 m image
of the whole crater also exists, but its server (astropedia) wasn't reachable when this
was written. If it comes back, it would let the Mars area cover the whole crater.

Both mosaics use an equirectangular projection with its true scale away from the
equator. The loader converts coordinates with PROJ, so any such projection works.

### 2. Mars iron and chlorine (GRS)

The Odyssey gamma-ray spectrometer element tables are on the PDS website:
https://pds-geosciences.wustl.edu/missions/odyssey/grs_elements.html

Open the `unsmoothed/5x5` folder (`odgm_xxxx` is the folder name shown on the page). Save these four files into `data\mars`, using **Save link as**:

- `fe_5x5.tab` and `fe_5x5.lbl` (iron)
- `cl_5x5.tab` and `cl_5x5.lbl` (chlorine)

Then build the chemistry file:

```powershell
python convert_grs.py data\mars\fe_5x5.tab Fe
python convert_grs.py data\mars\cl_5x5.tab Cl
```

Each run adds one element to `data\mars\elements_grid.csv`. Each element is in 5-degree cells, which are about 300 km across on Mars. Every tile inside a cell gets the same value.

### 3. Earth (Google Earth Engine)

1. Sign up at https://earthengine.google.com/signup with a Google account. Choose the noncommercial option, which creates a Google Cloud project. Registration can take a while.
2. Note your project ID.
3. Install the client and log in once:

```powershell
python -c "import ee; ee.Authenticate()"
```

A browser window opens for the Google login.

Earth tiles come from Sentinel-2 satellite images (`COPERNICUS/S2_SR_HARMONIZED`) and the Copernicus 30 m elevation model (`COPERNICUS/DEM/GLO30_2024_1`). Tiles with mean NDVI above 0.15 are dropped, since vegetated ground doesn't resemble Mars.

## Configure

Copy the template, then edit your copy:

```powershell
copy config.example.json config.json
notepad config.json
```

The main settings are:

- `tile_km` and `grid_px`: tile size and grid resolution.
- `mars`: the file paths, the Mars area (`bbox`), the spacing between tiles (`step_deg`), and `max_gap_fraction`.
- `earth`: the `boxes` of Earth regions to search, the spacing (`step_deg`), your `gee_project` ID, the imagery `year`, and the `cache` file.
- `target`: the Mars position to compare, as `[latitude, longitude]`.
- `weights`: how much terrain (`morph`) and iron (`chem`) count. The two numbers should add up to 1.
- `max_ndvi` and `top_k`: the vegetation cut-off and how many Earth rows to save.

Each region in `boxes` is written as `[west, south, east, north]`. Negative numbers are west of Greenwich or south of the equator.

## Run the comparison

```powershell
python -m mars_analog.cli --config config.json
```

The first run downloads every Earth tile, which takes a long time. Each tile is saved to `data\cache\earth_tiles.pkl` as it arrives, so a stopped run keeps its progress. Later runs load tiles from the cache and only download new regions. Delete the cache file to start the Earth download again, for example after changing how tiles are built.

The output shows which terrain features were used and which chemistry was compared. It then prints the Mars tiles that look most Earth-like and the best Earth matches for the target. Full results are saved to `results`.

To read the Earth table, sort by `final_score`. The columns mean:

- `morph_similarity`: how closely the terrain matches the target, from 0 to 1.
- `chem_score`: how closely the iron rank matches, from 0 to 1.
- `final_score`: the weighted combination of the two, so `0.7 × terrain + 0.3 × iron` with the default weights.
- `agrees_on` and `differs_on`: the three terrain features where the tile and the target are most alike and most different.

Scores only compare within one run. Differences of about 0.01 are noise, so treat the results as a shortlist to check on satellite imagery.

## Web app

The web app runs the comparison on the bundled data, so visitors don't need Google Earth Engine or your data files. It uses the bundle in the `bundle` folder. To build or run it yourself, see [WEB_APP.md](WEB_APP.md).

It works in both directions. Start from a spot on Mars to find the closest places on Earth, or from a place on Earth to find the closest spots on Mars. Pick the place from a list, or type its coordinates: the app uses the nearest place it has data for and says how far away that is. If the coordinates are outside the data (more than about 12 km from a Mars spot, or 50 km from an Earth place), it says so and names the nearest covered place instead of guessing.

## Learning from labelled pairs

The scoring weights in `features.py` were set by hand. The project can also learn them
from examples that people label:

1. `python make_label_sheet.py` writes `labels/label_sheet.csv`: about 60 pairs, each
   the Mars target with one Earth tile. Pairs near sites from the Catalogue of Planetary
   Analogues are included and marked. The app's scores are left out on purpose, so the
   labels come from people rather than from the program.
2. People mark each pair `1` (alike), `0` (clearly different) or leave it blank, with a
   reason. [LABELLING.md](LABELLING.md) explains how.
3. `python train_model.py` reads every sheet in `labels/` and fits a small logistic
   regression (a single neuron with one input per measurement and one output). It tries
   the original 7 measurements and 13 that add six landform ones (large-scale roughness,
   slope variety, slope directions, hills and hollows, share of high ground, and broad
   slope), and keeps whichever does better on pairs it didn't train on. It compares the
   kept model with the
   hand-set weights, gives a likely range for each result to allow for chance, reports
   how often labellers agreed, and saves `models/learned_weights.json`.
4. `python make_label_sheet.py --copy-for AMY` gives a teammate their own copy of the
   pairs, and `python make_label_sheet.py --next` makes a new sheet over several Mars
   spots, half of it pairs the model is least sure about (active learning).

The model is trained from scratch on this project's own data. With fewer than about
100 labels its results are rough.

When `models/learned_weights.json` exists, the web app uses it for the terrain score: the
chance a person would call two places alike. A switch in the sidebar goes back to the
hand-set weights, and the About tab shows what the model learned. To use the learned
weights on the command line too, add `"terrain_model": "models/learned_weights.json"` to
`config.json`.

Latest result: 88 labelled pairs over 5 Mars spots (61 alike, 27 different), labelled
by one person. Some pairs were chosen where an earlier model was least sure. On pairs it
hadn't trained on, the learned model scored 0.91 (likely range 0.82 to 0.98), against
0.77 (0.66 to 0.86) for the hand-set weights, where 1.0 is perfect and 0.5 is guessing.
The difference, +0.14, has a likely range of +0.06 to +0.23, so on these labels the
learned weights are better. The model relies on average slope, steepest slopes and
curvature, and gives roughness, height range, image texture and edge sharpness almost
no weight.

Two cautions. The labeller was asked to judge the shape of the land, not its colour, so
the model partly learns those instructions; the hand-set weights include image texture,
which the labeller ignored. And 88 pairs from one person is still a small sample. Labels
from teammates on the same pairs would show how much people agree.

## Limits

- **One Mars area.** The Mars side covers 16 spots in the middle of Jezero crater. The results are a shortlist to check, not a confirmed answer.
- **Iron only.** The Earth side uses a red-to-blue ratio as a stand-in for iron, which is an approximation.
- **Chlorine is converted but not scored.** No satellite measure of chlorine on Earth has been validated for this comparison.
- **Coarse chemistry.** Each 5-degree cell covers about 300 km, so every tile inside it gets the same chemistry value.
- **Different instruments.** Mars and Earth images come from different instruments, so the texture features are less reliable than the elevation features.
- **Earth elevation includes buildings and trees.** The dry filter helps, but built-up tiles can still get through.
- **Sea is left out.** A square counts as water when most of it sits at sea level, its plant-cover reading (NDVI) is negative, or it has no satellite image. Water squares are never matches or starting places, and typed sea coordinates get a message naming the nearest land.
- **Lakes can look flat.** Dry salt lakes in dry regions can pass these checks and look like smooth ground.
- **Gaps in Mars elevation.** Tiles with more than 2% gaps are skipped rather than filled.
- **Not for the whole planet.** The code keeps tiles in memory and requests Earth tiles one at a time, so a whole-planet search needs a different approach.

## Next steps

- Get more pairs labelled, by more people and for more Mars spots, and retrain.
- Cover the whole crater once the 5 m image is reachable.
- Find a chlorine or salt measure that can be checked against known sites.
- Add image embeddings from a pretrained vision model as an extra terrain score.
- Add a map of the top matches.

## Data sources

- Mars elevation: Mars 2020 Science Investigation CTX DEM Mosaic, USGS Astrogeology (F. Calef); see Mangold et al. 2021, *Science*, https://doi.org/10.1126/science.abl4051.
- Mars image: Mars 2020 Terrain Relative Navigation CTX orthomosaic, USGS Astrogeology; Fergason et al. 2020, https://doi.org/10.5066/P9GV1ND3.
- Mars analogue sites: Catalogue of Planetary Analogues, Preston, Grady and Barber, The Open University, 2012 (ESA contract 4000104716/11/NL/AF).
- Mars iron and chlorine: NASA Mars Odyssey Gamma Ray Spectrometer, via the NASA Planetary Data System.
- Earth imagery and elevation: Copernicus Sentinel-2 and Copernicus DEM, accessed through Google Earth Engine. Earth elevation: produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved. Earth imagery: contains modified Copernicus Sentinel data 2023.

## AI assistance

Claude, an AI assistant made by Anthropic, helped with this project. It explained how to connect to Google Earth Engine, helped write the comparison code, and explained how to build the converter for the NASA iron and chlorine tables. The team ran the code, fixed the errors it hit, checked the output files, and can explain how each part works.
