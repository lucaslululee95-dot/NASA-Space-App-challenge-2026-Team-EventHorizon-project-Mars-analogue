# Web app

The web app lets anyone compare the bundled Mars and Earth tiles from a browser. Nothing needs installing, and no Google Earth Engine account is needed. It runs the same comparison as the command line, but only on the data in the bundle.

## What's in the bundle

The `bundle` folder holds the tiles the app needs, about 12 MB in total:

- `mars.npz` and `earth.npz`: the elevation and image numbers for each tile.
- `mars.json` and `earth.json`: each tile's name, position, NDVI, and iron values.

Loading the bundle can't run code, unlike the pickle cache.

## Build the bundle (on your PC)

Run this from the project folder, after the comparison has run at least once so the Earth cache exists:

```powershell
python bundle_data.py
```

It prints how many Mars and Earth tiles it wrote. If it reports Earth tiles missing, run the comparison again to download them, then build the bundle again.

## Run the app on your PC

```powershell
python -m pip install -r requirements-local.txt
python -m streamlit run app.py
```

A browser tab opens. Choose a target tile and the weights in the left panel, then click **Run comparison**.

## Publish the app (free, Streamlit Community Cloud)

1. Commit the `bundle` folder and the new files to GitHub:

```powershell
git add bundle app.py bundle_data.py requirements.txt WEB_APP.md mars_analog tests
git commit -m "Add web app and data bundle"
git push
```

2. Go to https://share.streamlit.io and sign in with your GitHub account.
3. Click **Create app**, choose the repository, the `main` branch, and `app.py` as the main file, then click **Deploy**.

The app reads `requirements.txt`, which lists only the packages the web app needs. The first start takes a few minutes.

## Limits

- The Mars side covers one USGS stereo pair over Jezero.
- The Earth side covers only the dry regions in the bundle. To add a region, add it to `config.json`, run the comparison, then run `bundle_data.py` again.
- The app is a shortlist tool. It doesn't replace checking the matches on satellite imagery.
