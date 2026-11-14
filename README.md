# Mars Analog Finder



This tool looks for places on Earth whose terrain and iron content resemble a chosen Mars site. By default the site is Jezero Crater, where NASA's Perseverance rover landed. It compares terrain first, then re-ranks the matches using an iron comparison.



## What it does



1. **Tiles.** Mars and Earth are cut into square tiles of the same size, 10 km by 10 km by default. Each tile is resampled to a 64 by 64 grid.

2. **Measurements.** Each tile gets the same measurements: slope, relief, roughness, curvature, and image texture.

3. **Shared scale.** Mars and Earth numbers are scaled together, so one ruler applies to both.

4. **Mars ranking.** For each Mars tile, the program finds the closest Earth tile. This shows which Mars patches look most Earth-like.

5. **Target.** The program uses the Mars tile nearest to your chosen position, normally Jezero.

6. **Earth ranking.** Each Earth tile gets a terrain score and an iron score. The final score combines them, and each match lists the terrain features it agrees and disagrees with the target on.



The output is two CSV files in a `results` folder: `mars\_most\_earthlike.csv` and `earth\_matches\_for\_target.csv`.



## Project layout



```

mars\_analog/

&#x20; config.example.json   template for your settings

&#x20; config.json           your settings and Earth regions

&#x20; requirements.txt      Python packages

&#x20; convert\_grs.py        turns the GRS iron and chlorine tables into a CSV

&#x20; check\_earth\_engine.py optional check that Earth Engine works

&#x20; mars\_analog/          the comparison code

&#x20;   cli.py              the command you run

&#x20;   sources.py          reads Mars files, downloads Earth tiles, cache

&#x20;   features.py         measures each tile

&#x20;   matching.py         scales numbers and computes similarity

&#x20;   chemistry.py        iron comparison

&#x20;   pipeline.py         the three comparison steps

&#x20;   geo.py, tile.py     tile sizes, grids, and the tile structure

&#x20; tests/                automatic tests

&#x20; data/                 your Mars files and Earth cache (not in GitHub)

&#x20; results/              output tables (not in GitHub)

```



## Set up (Windows, PowerShell)



Run every command from the project folder. Running `pytest` from another folder searches unrelated files, which is slow and gives confusing errors.



```powershell

cd "path\\to\\mars\_analog"

python -m pip install -r requirements.txt

python -m pytest tests

python -m mars\_analog.cli --demo

```



The demo uses made-up terrain, so its numbers have no scientific meaning. It shows the pipeline and the output format. You don't need the real data to run the tests or the demo.



## Get the data



For the real comparison you need three things: Mars terrain and images, the Mars iron and chlorine tables, and a Google Earth Engine account. Create the data folder first:



```powershell

New-Item -ItemType Directory -Force data\\mars

```



### 1. Mars elevation and image (CTX)



The test site uses one USGS stereo pair over Jezero. It has an elevation model (DTM) and a matching orthoimage. Both are public on AWS, and no account is needed:

https://registry.opendata.aws/nasa-usgs-controlled-mro-ctx-dtms



Download them:



```powershell

Invoke-WebRequest -UseBasicParsing "https://astrogeo-ard.s3-us-west-2.amazonaws.com/mars/mro/ctx/controlled/usgs/P18\_007925\_1987\_XN\_18N282W\_\_P19\_008650\_1987\_XI\_18N282W/P18\_007925\_1987\_XN\_18N282W\_\_P19\_008650\_1987\_XI\_18N282W\_dtm.tif" -OutFile data\\mars\\ctx\_dtm.tif

Invoke-WebRequest -UseBasicParsing "https://astrogeo-ard.s3-us-west-2.amazonaws.com/mars/mro/ctx/controlled/usgs/P18\_007925\_1987\_XN\_18N282W\_\_P19\_008650\_1987\_XI\_18N282W/P18\_007925\_1987\_XN\_18N282W\_\_P19\_008650\_1987\_XI\_18N282W\_orthoimage.tif" -OutFile data\\mars\\ctx\_ortho.tif

```



To see the other pairs in this area, list the files in a pair's folder:



```powershell

$r = Invoke-WebRequest -UseBasicParsing "https://astrogeo-ard.s3-us-west-2.amazonaws.com/?list-type=2\&prefix=mars/mro/ctx/controlled/usgs/P18\_007925\_1987\_XN\_18N282W\_\_P19\_008650\_1987\_XI\_18N282W/"

\[regex]::Matches($r.Content, "<Key>(.\*?)</Key>") | ForEach-Object { $\_.Groups\[1].Value }

```



The elevation model has gaps, about 30% in this pair. The program skips any tile with more than 2% gaps, so the number of Mars tiles kept is printed at the start of each run.



### 2. Mars iron and chlorine (GRS)



The Odyssey gamma-ray spectrometer element tables are on the PDS website:

https://pds-geosciences.wustl.edu/missions/odyssey/grs\_elements.html



Open the `unsmoothed/5x5` folder (`odgm\_xxxx` is the folder name shown on the page). Save these four files into `data\\mars`, using \*\*Save link as\*\*:



- `fe\_5x5.tab` and `fe\_5x5.lbl` (iron)

- `cl\_5x5.tab` and `cl\_5x5.lbl` (chlorine)



Then build the chemistry file:



```powershell

python convert\_grs.py data\\mars\\fe\_5x5.tab Fe

python convert\_grs.py data\\mars\\cl\_5x5.tab Cl

```



Each run adds one element to `data\\mars\\elements\_grid.csv`. Each element is in 5-degree cells, which are about 300 km across on Mars. Every tile inside a cell gets the same value.



### 3. Earth (Google Earth Engine)



1. Sign up at https://earthengine.google.com/signup with a Google account. Choose the noncommercial option, which creates a Google Cloud project. Registration can take a while.

2. Note your project ID.

3. Install the client and log in once:



```powershell

python -c "import ee; ee.Authenticate()"

```



A browser window opens for the Google login.



Earth tiles come from Sentinel-2 satellite images (`COPERNICUS/S2\_SR\_HARMONIZED`) and the Copernicus 30 m elevation model (`COPERNICUS/DEM/GLO30\_2024\_1`). Tiles with mean NDVI above 0.15 are dropped, since vegetated ground doesn't resemble Mars.



## Configure



Copy the template, then edit your copy:



```powershell

copy config.example.json config.json

notepad config.json

```



The main settings are:



\- `tile\_km` and `grid\_px`: tile size and grid resolution.

\- `mars`: the file paths, the Mars area (`bbox`), the spacing between tiles (`step\_deg`), and `max\_gap\_fraction`.

\- `earth`: the `boxes` of Earth regions to search, the spacing (`step\_deg`), your `gee\_project` ID, the imagery `year`, and the `cache` file.

\- `target`: the Mars position to compare, as `\[latitude, longitude]`.

\- `weights`: how much terrain (`morph`) and iron (`chem`) count. The two numbers should add up to 1.

\- `max\_ndvi` and `top\_k`: the vegetation cut-off and how many Earth rows to save.



Each region in `boxes` is written as `\[west, south, east, north]`. Negative numbers are west of Greenwich or south of the equator.



## Run the comparison



```powershell

python -m mars\_analog.cli --config config.json

```



The first run downloads every Earth tile, which takes a long time. Each tile is saved to `data\\cache\\earth\_tiles.pkl` as it arrives, so a stopped run keeps its progress. Later runs load tiles from the cache and only download new regions. Delete the cache file to start the Earth download again, for example after changing how tiles are built.



The output shows which terrain features were used and which chemistry was compared. It then prints the Mars tiles that look most Earth-like and the best Earth matches for the target. Full results are saved to `results`.



To read the Earth table, sort by `final\_score`. The columns mean:



\- `morph\_similarity`: how closely the terrain matches the target, from 0 to 1.

\- `chem\_score`: how closely the iron rank matches, from 0 to 1.

\- `final\_score`: the weighted combination of the two, so `0.7 × terrain + 0.3 × iron` with the default weights.

\- `agrees\_on` and `differs\_on`: the three terrain features where the tile and the target are most alike and most different.



Scores only compare within one run. Differences of about 0.01 are noise, so treat the results as a shortlist to check on satellite imagery.



## Limits



\- \*\*Tested on real data for one Mars site and several dry Earth regions.\*\* The results are a shortlist to check, not a confirmed answer.

\- \*\*Iron only.\*\* The Earth side uses a red-to-blue ratio as a stand-in for iron, which is an approximation.

\- \*\*Chlorine is converted but not scored.\*\* No satellite measure of chlorine on Earth has been validated for this comparison.

\- \*\*Coarse chemistry.\*\* Each 5-degree cell covers about 300 km, so every tile inside it gets the same chemistry value.

\- \*\*Different instruments.\*\* Mars and Earth images come from different instruments, so the texture features are less reliable than the elevation features.

\- \*\*Earth elevation includes buildings and trees.\*\* The dry filter helps, but built-up tiles can still get through.

\- \*\*Lakes can look flat.\*\* Salt lakes in dry regions can pass the filter and look like smooth ground.

\- \*\*Gaps in Mars elevation.\*\* Tiles with more than 2% gaps are skipped rather than filled.

\- \*\*Not for the whole planet.\*\* The code keeps tiles in memory and requests Earth tiles one at a time, so a whole-planet search needs a different approach.



## Next steps



\- Find a chlorine or salt measure that can be checked against known sites.

\- Add more dry regions, such as the Gobi, Tibet, and the Lut desert.

\- Add image embeddings from a pretrained vision model as an extra terrain score.

\- Add a map of the top matches.



## Data sources



\- Mars elevation and images: NASA and USGS CTX stereo products (public domain).

\- Mars iron and chlorine: NASA Mars Odyssey Gamma Ray Spectrometer, via the NASA Planetary Data System.

\- Earth imagery and elevation: Copernicus Sentinel-2 and Copernicus DEM, accessed through Google Earth Engine.



## AI assistance



Claude, an AI assistant made by Anthropic, helped with this project. It explained how to connect to Google Earth Engine, helped write the comparison code, and explained how to build the converter for the NASA iron and chlorine tables. The team ran the code, fixed the errors it hit, checked the output files, and can explain how each part works.

