"""Web app: compare the bundled Mars and Earth tiles and show the matches."""
import os

import streamlit as st

from mars_analog.bundle import load_bundle
from mars_analog.pipeline import run

BUNDLE_DIR = os.environ.get("MARS_BUNDLE_DIR", "bundle")
JEZERO = (18.38, 77.58)

st.set_page_config(page_title="Mars Analog Finder", layout="wide")


@st.cache_resource
def get_tiles(folder):
    return load_bundle(folder)


st.title("Mars Analog Finder")
st.write(
    "Finds dry places on Earth whose terrain and iron content resemble a Mars site, "
    "by default Jezero Crater. Terrain is compared first, then iron re-ranks the matches."
)

if not os.path.exists(os.path.join(BUNDLE_DIR, "earth.npz")):
    st.error(f"No data bundle found in '{BUNDLE_DIR}'. Run `python bundle_data.py` first.")
    st.stop()

mars_tiles, earth_tiles = get_tiles(BUNDLE_DIR)
mars_by_name = {t.name: t for t in mars_tiles}
names = sorted(mars_by_name)
start = min(range(len(names)), key=lambda i: (mars_by_name[names[i]].lat - JEZERO[0]) ** 2
            + (mars_by_name[names[i]].lon - JEZERO[1]) ** 2)

with st.sidebar:
    st.header("Settings")
    target_name = st.selectbox("Mars target tile", names, index=start)
    terrain_weight = st.slider("Weight on terrain (the rest is iron)", 0.0, 1.0, 0.7, 0.05)
    max_ndvi = st.slider("Vegetation cut-off (NDVI)", 0.0, 0.5, 0.15, 0.01)
    top_k = st.slider("Earth matches to show", 5, 50, 25)
    go = st.button("Run comparison", type="primary")

if go:
    target = mars_by_name[target_name]
    try:
        st.session_state["result"] = run(
            mars_tiles, earth_tiles, w_morph=terrain_weight, w_chem=1 - terrain_weight,
            max_ndvi=max_ndvi, target=(target.lat, target.lon), top_k=top_k)
    except ValueError as exc:
        st.session_state.pop("result", None)
        st.error(str(exc))

res = st.session_state.get("result")
if res:
    st.subheader(f"Target: {res['target']}")
    st.caption("Terrain features used: " + ", ".join(res["feature_cols"]))
    st.caption(f"Iron compared: {', '.join(res['chem_compared']) or 'none'}. "
               f"Not compared: {', '.join(res['chem_skipped'])}.")

    st.subheader("Earth matches")
    st.write("Sorted by final score. Each row is a 10 km square of Earth.")
    st.dataframe(res["earth_matches"].round(3))
    st.download_button("Download Earth matches (CSV)", res["earth_matches"].to_csv(index=False),
                       "earth_matches.csv", "text/csv")

    st.subheader("Mars tiles that look most Earth-like")
    st.dataframe(res["mars_ranking"].round(3))
    st.download_button("Download Mars ranking (CSV)", res["mars_ranking"].to_csv(index=False),
                       "mars_most_earthlike.csv", "text/csv")

with st.expander("Read this before using the results"):
    st.markdown(
        "- The results are a shortlist to check on satellite imagery, not a confirmed answer.\n"
        "- The Mars side covers one site (one USGS stereo pair over Jezero).\n"
        "- Iron is a rough stand-in. The Earth side uses a red-to-blue ratio.\n"
        "- Each iron cell covers about 300 km, so tiles in a cell share one value.\n"
        "- Earth matches come only from the dry regions in the bundle.\n"
        "- Scores only compare within one run. Differences of about 0.01 are noise.")

st.caption(
    "Earth elevation: produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and "
    "© Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European "
    "Union and ESA; all rights reserved. Earth imagery: contains modified Copernicus "
    "Sentinel data 2023."
)
