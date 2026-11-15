"""Web app: find the places on Earth that measure most like a spot on Mars."""
import os

import altair as alt
import pandas as pd
import streamlit as st

from mars_analog.bundle import load_bundle
from mars_analog.features import feature_columns, feature_table
from mars_analog.matching import prepare
from mars_analog.pipeline import run
from mars_analog.viz import hillshade, surface_image

BUNDLE_DIR = os.environ.get("MARS_BUNDLE_DIR", "bundle")
JEZERO = (18.38, 77.58)
MARS_COLOUR = "#A14A2A"     # iron oxide
EARTH_COLOUR = "#2F6B7C"    # slate

FEATURE_NAMES = {
    "slope_mean_deg": "Average slope",
    "slope_p90_deg": "Steepest slopes",
    "relief_m": "Height range",
    "roughness_m": "Roughness",
    "curvature_per_m": "Curvature",
    "texture_rel": "Image texture",
    "grad_rel": "Edge sharpness",
    "red_blue_ratio": "Red-to-blue colour",
}

# Named Earth regions, as [west, south, east, north]. Tiles outside them show coordinates only.
REGIONS = [
    ("Atacama Desert, Chile", [-70.5, -25.0, -68.0, -22.0]),
    ("Namib Desert, Namibia", [14.0, -25.0, 15.5, -23.0]),
    ("Mojave Desert, USA", [-117.5, 34.0, -115.5, 36.0]),
    ("Libyan Desert, Sahara", [24.0, 20.0, 26.0, 22.0]),
    ("Western Desert, Australia", [130.0, -26.0, 132.0, -24.0]),
    ("Lut Desert, Iran", [57.5, 30.0, 59.5, 32.0]),
    ("Gobi Desert", [102.0, 41.0, 104.0, 43.0]),
    ("Changtang Plateau, Tibet", [80.0, 33.0, 82.0, 35.0]),
    ("Taklamakan Desert, China", [80.0, 38.0, 82.0, 40.0]),
    ("Central Australia, near Alice Springs", [133.0, -24.5, 134.5, -23.0]),
    ("Erfoud, Morocco", [-5.0, 30.5, -3.5, 31.5]),
]

st.set_page_config(page_title="Mars Analog Finder", page_icon="🪨", layout="wide")


def fmt_place(lat, lon):
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{abs(lat):.2f}° {ns}, {abs(lon):.2f}° {ew}"


def region_of(lat, lon):
    for name, (w, s, e, n) in REGIONS:
        if w - 1e-6 <= lon <= e + 1e-6 and s - 1e-6 <= lat <= n + 1e-6:
            return name
    return None


def place_name(lat, lon):
    """The region's name, or its coordinates when it isn't in a named region."""
    return region_of(lat, lon) or fmt_place(lat, lon)


def readable(features):
    """'slope_mean_deg, relief_m' -> 'average slope, height range'"""
    names = [FEATURE_NAMES.get(f.strip(), f.strip()) for f in str(features).split(",") if f.strip()]
    return ", ".join(n.lower() for n in names)


@st.cache_resource
def get_tiles(folder):
    return load_bundle(folder)


@st.cache_data
def compare(folder, target_name, terrain_weight, max_ndvi, top_k):
    mars, earth = get_tiles(folder)
    target = next(t for t in mars if t.name == target_name)
    return run(mars, earth, w_morph=terrain_weight, w_chem=1 - terrain_weight,
               max_ndvi=max_ndvi, target=(target.lat, target.lon), top_k=top_k)


@st.cache_data
def scaled_features(folder, max_ndvi):
    """Feature values on the shared Mars-and-Earth scale, the same one the scores use."""
    mars, earth = get_tiles(folder)
    m_df = feature_table(mars).reset_index(drop=True)
    e_df = feature_table(earth).reset_index(drop=True)
    if "ndvi" in e_df.columns:
        e_df = e_df[e_df["ndvi"] <= max_ndvi].reset_index(drop=True)
    cols = feature_columns(m_df, e_df)
    m_s, e_s = prepare(m_df, e_df, cols)
    m_s.index, e_s.index = m_df["name"], e_df["name"]
    return m_s, e_s


def twin(mars_tile, earth_tile, view):
    """The Mars tile and the Earth tile side by side, drawn the same way."""
    left, right = st.columns(2)
    for col, tile, planet in ((left, mars_tile, "Mars"), (right, earth_tile, "Earth")):
        if view == "Shaded relief":
            img = hillshade(tile.dem, tile.tile_km)
        else:
            img = surface_image(tile.rgb)
        with col:
            st.image(img, width="stretch")
            if planet == "Mars":
                st.markdown(f"**Mars**, Jezero area  \n{fmt_place(tile.lat, tile.lon)}")
            else:
                region = region_of(tile.lat, tile.lon)
                heading = f"**Earth**, {region}" if region else "**Earth**"
                st.markdown(f"{heading}  \n{fmt_place(tile.lat, tile.lon)}")


def feature_chart(m_s, e_s, mars_name, earth_name):
    rows = []
    for col in m_s.columns:
        label = FEATURE_NAMES.get(col, col)
        rows.append({"Measure": label, "Planet": "Mars", "Value": float(m_s.loc[mars_name, col])})
        rows.append({"Measure": label, "Planet": "Earth", "Value": float(e_s.loc[earth_name, col])})
    data = pd.DataFrame(rows)
    return (alt.Chart(data)
            .mark_bar()
            .encode(
                y=alt.Y("Measure:N", title=None, sort=None),
                yOffset=alt.YOffset("Planet:N", sort=["Mars", "Earth"]),
                x=alt.X("Value:Q", title="Value on the shared scale (0 is the typical tile)"),
                color=alt.Color("Planet:N", sort=["Mars", "Earth"],
                                scale=alt.Scale(domain=["Mars", "Earth"],
                                                range=[MARS_COLOUR, EARTH_COLOUR]),
                                legend=alt.Legend(title=None, orient="top")),
                tooltip=["Planet", "Measure", alt.Tooltip("Value:Q", format=".2f")])
            .properties(height=36 * len(m_s.columns)))


# ---------------------------------------------------------------- page

st.title("Where on Earth looks like this part of Mars?")
st.write("Choose a spot near Jezero crater on Mars. The app measures its terrain and compares it "
         "with 10 km squares of dry land on Earth, then shows the closest matches.")

if not os.path.exists(os.path.join(BUNDLE_DIR, "earth.npz")):
    st.error(f"The data bundle is missing from the '{BUNDLE_DIR}' folder. "
             "Build it with `python bundle_data.py`, then reload this page.")
    st.stop()

mars_tiles, earth_tiles = get_tiles(BUNDLE_DIR)
mars_by_name = {t.name: t for t in mars_tiles}
earth_by_name = {t.name: t for t in earth_tiles}
names = sorted(mars_by_name, key=lambda n: (-mars_by_name[n].lat, mars_by_name[n].lon))
start = min(range(len(names)), key=lambda i: (mars_by_name[names[i]].lat - JEZERO[0]) ** 2
            + (mars_by_name[names[i]].lon - JEZERO[1]) ** 2)


def mars_label(name):
    t = mars_by_name[name]
    label = fmt_place(t.lat, t.lon)
    return label + " (landing site)" if name == names[start] else label


with st.sidebar:
    st.header("Your Mars spot")
    target_name = st.selectbox("Spot on Mars", names, index=start, format_func=mars_label)
    st.header("How to score")
    terrain_weight = st.slider("Weight on terrain", 0.0, 1.0, 0.7, 0.05,
                               help="The rest of the score comes from the iron comparison.")
    st.caption(f"Terrain {terrain_weight:.0%}, iron {1 - terrain_weight:.0%}")
    max_ndvi = st.slider("Leave out greener land above NDVI", 0.0, 0.5, 0.15, 0.01,
                         help="NDVI measures plant cover. Mars has none, so green tiles are dropped.")
    top_k = st.slider("Matches to list", 5, 50, 25)
    view = st.radio("Picture style", ["Shaded relief", "Satellite image"],
                    help="Shaded relief uses the same lighting on both planets.")

try:
    res = compare(BUNDLE_DIR, target_name, terrain_weight, max_ndvi, top_k)
except ValueError as exc:
    st.error(f"{exc}")
    st.stop()

matches = res["earth_matches"].copy()
matches["Place"] = [place_name(a, b) for a, b in zip(matches["lat"], matches["lon"])]
best = matches.iloc[0]
m_s, e_s = scaled_features(BUNDLE_DIR, max_ndvi)
target_tile = mars_by_name[res["target"]]

# The twin view: the target and its closest match, drawn the same way.
st.subheader(f"Closest match: {best['Place']}")
twin(target_tile, earth_by_name[best["earth_tile"]], view)
chem = "" if pd.isna(best["chem_score"]) else f", iron {best['chem_score']:.2f}"
st.write(f"Overall score {best['final_score']:.2f} (terrain {best['morph_similarity']:.2f}{chem}). "
         f"Most alike in {readable(best['agrees_on'])}. "
         f"Most different in {readable(best['differs_on'])}.")
st.caption("A high score means the two places measure alike. It doesn't show they formed the same way.")

tab_map, tab_pair, tab_about = st.tabs(["Map and list", "Compare any match", "About the data"])

with tab_map:
    pts = matches[["lat", "lon", "final_score"]].copy()
    pts["size"] = 20000 + 60000 * (pts["final_score"] - pts["final_score"].min()) / \
        max(pts["final_score"].max() - pts["final_score"].min(), 1e-9)
    pts["color"] = EARTH_COLOUR
    st.map(pts, latitude="lat", longitude="lon", size="size", color="color", zoom=1)
    st.caption("Larger dots are closer matches.")

    table = pd.DataFrame({
        "Rank": range(1, len(matches) + 1),
        "Place": matches["Place"],
        "Location": [fmt_place(a, b) for a, b in zip(matches["lat"], matches["lon"])],
        "Overall": matches["final_score"].round(2),
        "Terrain": matches["morph_similarity"].round(2),
        "Iron": matches["chem_score"].round(2),
        "Most alike in": [readable(x) for x in matches["agrees_on"]],
        "Most different in": [readable(x) for x in matches["differs_on"]],
    })
    st.dataframe(table, hide_index=True)
    st.download_button("Download matches (CSV)", res["earth_matches"].to_csv(index=False),
                       "earth_matches.csv", "text/csv")

    with st.expander("Which parts of Mars look most like Earth?"):
        ranking = res["mars_ranking"]
        st.dataframe(pd.DataFrame({
            "Mars spot": [fmt_place(a, b) for a, b in zip(ranking["lat"], ranking["lon"])],
            "Best Earth score": ranking["best_earth_similarity"].round(2),
        }), hide_index=True)
        st.download_button("Download Mars ranking (CSV)", ranking.to_csv(index=False),
                           "mars_most_earthlike.csv", "text/csv")

with tab_pair:
    options = list(matches.index)
    pick = st.selectbox(
        "Earth match", options,
        format_func=lambda i: (f"{matches.loc[i, 'Place']}: {fmt_place(matches.loc[i, 'lat'], matches.loc[i, 'lon'])}, "
                               f"score {matches.loc[i, 'final_score']:.2f}"))
    row = matches.loc[pick]
    twin(target_tile, earth_by_name[row["earth_tile"]], view)
    st.write(f"Most alike in {readable(row['agrees_on'])}. "
             f"Most different in {readable(row['differs_on'])}.")
    st.altair_chart(feature_chart(m_s, e_s, res["target"], row["earth_tile"]))
    st.caption("Both planets are measured on one shared scale. Bars of similar length mean "
               "the two places agree on that measure.")

with tab_about:
    st.markdown(
        "**How it works.** Mars and Earth are cut into 10 km squares. Each square gets the same "
        "measurements: slope, height range, roughness, curvature, and image texture. The score is "
        "mostly terrain, plus an iron comparison that ranks each place against its own planet.\n\n"
        "**What the results can and can't tell you.**\n"
        "- The results are a shortlist to check on satellite imagery, not a confirmed answer.\n"
        "- The Mars side covers one area around Jezero crater.\n"
        "- The Earth side covers only the dry regions in the data bundle.\n"
        "- Iron on Earth is estimated from colour, which is a rough stand-in.\n"
        "- Each iron value on Mars covers about 300 km, so nearby spots share one value.\n"
        "- Scores only compare within one run. Differences of about 0.01 are noise.\n\n"
        "**Data.** Mars elevation and images: NASA and USGS CTX stereo products. Mars iron: "
        "NASA Mars Odyssey Gamma Ray Spectrometer, via the Planetary Data System. Earth imagery "
        "and elevation: Copernicus Sentinel-2 and Copernicus DEM, through Google Earth Engine.\n\n"
        "**AI assistance.** Claude, an AI assistant made by Anthropic, helped write the code. "
        "The team ran, tested, and checked it.")

st.caption(
    "Earth elevation: produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and "
    "© Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European "
    "Union and ESA; all rights reserved. Earth imagery: contains modified Copernicus "
    "Sentinel data 2023."
)
