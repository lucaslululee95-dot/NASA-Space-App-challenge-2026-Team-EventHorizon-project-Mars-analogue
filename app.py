"""Web app: find the places on Earth that measure most like a spot on Mars."""
import json
import os

import altair as alt
import pandas as pd
import streamlit as st

from mars_analog.bundle import load_bundle
from mars_analog.features import feature_columns, feature_table
from mars_analog.matching import prepare
from mars_analog.geo import nearest_tile
from mars_analog.pipeline import run, run_from_earth
from mars_analog.regions import REGIONS, fmt_place, place_name, region_of
from mars_analog.viz import hillshade, surface_image

BUNDLE_DIR = os.environ.get("MARS_BUNDLE_DIR", "bundle")
MODEL_PATH = os.environ.get("MARS_MODEL_PATH", os.path.join("models", "learned_weights.json"))
JEZERO = (18.38, 77.58)
MARS_REACH_KM = 12      # a typed Mars spot farther than this from our data is outside it
EARTH_REACH_KM = 50     # the same for Earth, where our squares are about 55 km apart
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
    "roughness_large_m": "Large-scale roughness",
    "slope_std_deg": "Slope variety",
    "aspect_spread": "Slope directions",
    "peaks_per_100km2": "Hills and hollows",
    "hypsometric": "Share of high ground",
    "slope_coarse_deg": "Broad slope",
}

st.set_page_config(page_title="Mars Analog Finder", page_icon="🪨", layout="wide")


def readable(features):
    """'slope_mean_deg, relief_m' -> 'average slope, height range'"""
    names = [FEATURE_NAMES.get(f.strip(), f.strip()) for f in str(features).split(",") if f.strip()]
    return ", ".join(n.lower() for n in names)


@st.cache_resource
def get_tiles(folder):
    return load_bundle(folder)


@st.cache_data
def load_model(path):
    """Weights learned by train_model.py, or None if there aren't any yet."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def compare(folder, target_name, terrain_weight, max_ndvi, top_k, use_model):
    mars, earth = get_tiles(folder)
    target = next(t for t in mars if t.name == target_name)
    return run(mars, earth, w_morph=terrain_weight, w_chem=1 - terrain_weight,
               max_ndvi=max_ndvi, target=(target.lat, target.lon), top_k=top_k,
               terrain_model=load_model(MODEL_PATH) if use_model else None)


@st.cache_data
def compare_from_earth(folder, earth_name, terrain_weight, max_ndvi, top_k, use_model):
    mars, earth = get_tiles(folder)
    return run_from_earth(mars, earth, earth_name, w_morph=terrain_weight,
                          w_chem=1 - terrain_weight, max_ndvi=max_ndvi, top_k=top_k,
                          terrain_model=load_model(MODEL_PATH) if use_model else None)


@st.cache_data
def scaled_features(folder, max_ndvi, keep=None):
    """Feature values on the shared Mars-and-Earth scale, the same one the scores use."""
    mars, earth = get_tiles(folder)
    m_df = feature_table(mars).reset_index(drop=True)
    e_df = feature_table(earth).reset_index(drop=True)
    if "ndvi" in e_df.columns:
        e_df = e_df[(e_df["ndvi"] <= max_ndvi) | (e_df["name"] == keep)].reset_index(drop=True)
    cols = feature_columns(m_df, e_df, extra=True)
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

if not os.path.exists(os.path.join(BUNDLE_DIR, "earth.npz")):
    st.title("Mars Analog Finder")
    st.error(f"The data bundle is missing from the '{BUNDLE_DIR}' folder. "
             "Build it with `python bundle_data.py`, then reload this page.")
    st.stop()

mars_tiles, earth_tiles = get_tiles(BUNDLE_DIR)
mars_by_name = {t.name: t for t in mars_tiles}
earth_by_name = {t.name: t for t in earth_tiles}
names = sorted(mars_by_name, key=lambda n: (-mars_by_name[n].lat, mars_by_name[n].lon))
start = min(range(len(names)), key=lambda i: (mars_by_name[names[i]].lat - JEZERO[0]) ** 2
            + (mars_by_name[names[i]].lon - JEZERO[1]) ** 2)

# Earth places grouped by region, in the order of the region list
by_region = {}
for t in sorted(earth_tiles, key=lambda t: (-t.lat, t.lon)):
    by_region.setdefault(region_of(t.lat, t.lon) or "Other places", []).append(t.name)
region_names = [r for r, _ in REGIONS if r in by_region] + \
    (["Other places"] if "Other places" in by_region else [])


def mars_label(name):
    t = mars_by_name[name]
    label = fmt_place(t.lat, t.lon)
    return label + " (landing site)" if name == names[start] else label


def mars_extent():
    lats = [t.lat for t in mars_tiles]
    lons = [t.lon for t in mars_tiles]
    return f"{min(lats):.2f}° to {max(lats):.2f}° N, {min(lons):.2f}° to {max(lons):.2f}° E"


with st.sidebar:
    st.header("Start from")
    side = st.radio("A place on", ["Mars", "Earth"], horizontal=True)
    how = st.radio("Choose it", ["From the list", "By coordinates"], horizontal=True)
    target_name, earth_name, too_far = None, None, None
    if side == "Mars" and how == "From the list":
        target_name = st.selectbox("Spot on Mars", names, index=start, format_func=mars_label)
    elif side == "Mars":
        lat = st.number_input("Latitude, degrees north", -90.0, 90.0, 18.40, 0.05, format="%.2f")
        lon = st.number_input("Longitude, degrees east", -180.0, 360.0, 77.55, 0.05, format="%.2f")
        lon = lon - 360.0 if lon > 180 else lon
        tile, km = nearest_tile(mars_tiles, lat, lon, "mars")
        if km > MARS_REACH_KM:
            too_far = (f"That spot is outside our Mars data. We have {len(mars_tiles)} spots in "
                       f"the middle of Jezero crater ({mars_extent()}), and the nearest is "
                       f"{km:,.0f} km away. Try a spot inside that area.")
        else:
            target_name = tile.name
            st.caption(f"Using our nearest spot, {fmt_place(tile.lat, tile.lon)}, {km:.0f} km away.")
    elif how == "From the list":
        region = st.selectbox("Region", region_names)
        earth_name = st.selectbox("Place", by_region[region],
                                  format_func=lambda n: fmt_place(earth_by_name[n].lat,
                                                                  earth_by_name[n].lon))
    else:
        lat = st.number_input("Latitude, degrees (south is negative)", -90.0, 90.0, -24.0, 0.25,
                              format="%.2f")
        lon = st.number_input("Longitude, degrees (west is negative)", -180.0, 180.0, 15.5, 0.25,
                              format="%.2f")
        tile, km = nearest_tile(earth_tiles, lat, lon, "earth")
        if km > EARTH_REACH_KM:
            region = region_of(tile.lat, tile.lon)
            where = (f"in {region} ({fmt_place(tile.lat, tile.lon)})" if region
                     else f"at {fmt_place(tile.lat, tile.lon)}")
            named = [r for r in region_names if r != "Other places"]
            covers = (f" Our Earth data covers: {', '.join(named)}. Try coordinates inside one "
                      "of them." if named else "")
            too_far = (f"We don't have data there yet. The nearest place we have is {where}, "
                       f"{km:,.0f} km away.{covers}")
        else:
            earth_name = tile.name
            st.caption(f"Using our nearest place, {place_name(tile.lat, tile.lon)} "
                       f"({fmt_place(tile.lat, tile.lon)}), {km:.0f} km away.")

    st.header("How to score")
    model = load_model(MODEL_PATH)
    use_model = False
    if model:
        worse = model.get("auc_difference_range", [0, 0])[1] < 0     # clearly worse than hand-set
        source = st.radio("Terrain score", ["Learned from labels", "Hand-set weights"],
                          index=1 if worse else 0,
                          help="Learned weights come from pairs that people labelled as alike "
                               "or different. See About the data.")
        use_model = source == "Learned from labels"
    terrain_weight = st.slider("Weight on terrain", 0.0, 1.0, 0.7, 0.05,
                               help="The rest of the score comes from the iron comparison.")
    st.caption(f"Terrain {terrain_weight:.0%}, iron {1 - terrain_weight:.0%}")
    max_ndvi = st.slider("Leave out greener land above NDVI", 0.0, 0.5, 0.15, 0.01,
                         help="NDVI measures plant cover. Mars has none, so green tiles are dropped.")
    top_k = st.slider("Matches to list", 5, 50, 25)
    view = st.radio("Picture style", ["Shaded relief", "Satellite image"],
                    help="Shaded relief uses the same lighting on both planets.")

if side == "Mars":
    st.title("Where on Earth looks like this part of Mars?")
    st.write("Choose a spot near Jezero crater on Mars. The app measures its terrain and compares "
             "it with 10 km squares of dry land on Earth, then shows the closest matches.")
else:
    st.title("Where on Mars looks like this place on Earth?")
    st.write("Choose a place on Earth. The app measures its terrain and compares it with the "
             f"{len(mars_tiles)} spots we have in Jezero crater on Mars, then shows the closest.")

if too_far:
    st.info(too_far)
    st.stop()

score_note = ("The terrain score is the chance a person would call the two places alike, learned "
              f"from {model['labels_similar'] + model['labels_different']} labelled pairs. "
              "It doesn't show they formed the same way." if use_model else
              "A high score means the two places measure alike. "
              "It doesn't show they formed the same way.")


def score_line(row):
    chem = "" if pd.isna(row["chem_score"]) else f", iron {row['chem_score']:.2f}"
    return (f"Overall score {row['final_score']:.2f} (terrain {row['morph_similarity']:.2f}{chem}). "
            f"Most alike in {readable(row['agrees_on'])}. "
            f"Most different in {readable(row['differs_on'])}.")


try:
    if side == "Mars":
        res = compare(BUNDLE_DIR, target_name, terrain_weight, max_ndvi, top_k, use_model)
    else:
        res = compare_from_earth(BUNDLE_DIR, earth_name, terrain_weight, max_ndvi, top_k, use_model)
except ValueError as exc:
    st.error(f"{exc}")
    st.stop()

m_s, e_s = scaled_features(BUNDLE_DIR, max_ndvi, keep=earth_name)
shown = [c for c in res["feature_cols"] if c in m_s.columns]
chart_note = ("Both planets are measured on one shared scale. Bars of similar length mean "
              "the two places agree on that measure.")

if side == "Mars":
    matches = res["earth_matches"].copy()
    matches["Place"] = [place_name(a, b) for a, b in zip(matches["lat"], matches["lon"])]
    best = matches.iloc[0]
    target_tile = mars_by_name[res["target"]]

    # The twin view: the target and its closest match, drawn the same way.
    st.subheader(f"Closest match: {best['Place']}")
    twin(target_tile, earth_by_name[best["earth_tile"]], view)
    st.write(score_line(best))
    st.caption(score_note)

    tab_map, tab_pair, tab_about = st.tabs(["Map and list", "Compare any match", "About the data"])
    with tab_map:
        pts = matches[["lat", "lon", "final_score"]].copy()
        pts["size"] = 20000 + 60000 * (pts["final_score"] - pts["final_score"].min()) / \
            max(pts["final_score"].max() - pts["final_score"].min(), 1e-9)
        pts["color"] = EARTH_COLOUR
        st.map(pts, latitude="lat", longitude="lon", size="size", color="color", zoom=1)
        st.caption("Larger dots are closer matches.")
        st.dataframe(pd.DataFrame({
            "Rank": range(1, len(matches) + 1),
            "Place": matches["Place"],
            "Location": [fmt_place(a, b) for a, b in zip(matches["lat"], matches["lon"])],
            "Overall": matches["final_score"].round(2),
            "Terrain": matches["morph_similarity"].round(2),
            "Iron": matches["chem_score"].round(2),
            "Most alike in": [readable(x) for x in matches["agrees_on"]],
            "Most different in": [readable(x) for x in matches["differs_on"]],
        }), hide_index=True)
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
        pick = st.selectbox(
            "Earth match", list(matches.index),
            format_func=lambda i: (f"{matches.loc[i, 'Place']}: "
                                   f"{fmt_place(matches.loc[i, 'lat'], matches.loc[i, 'lon'])}, "
                                   f"score {matches.loc[i, 'final_score']:.2f}"))
        row = matches.loc[pick]
        twin(target_tile, earth_by_name[row["earth_tile"]], view)
        st.write(f"Most alike in {readable(row['agrees_on'])}. "
                 f"Most different in {readable(row['differs_on'])}.")
        st.altair_chart(feature_chart(m_s[shown], e_s[shown], res["target"], row["earth_tile"]))
        st.caption(chart_note)
else:
    spots = res["mars_matches"].copy()
    spots["Spot"] = [fmt_place(a, b) for a, b in zip(spots["lat"], spots["lon"])]
    best = spots.iloc[0]
    start_tile = earth_by_name[earth_name]

    st.subheader(f"Closest spot on Mars: {best['Spot']}")
    twin(mars_by_name[best["mars_tile"]], start_tile, view)
    st.write(score_line(best))
    st.caption(score_note)

    tab_map, tab_pair, tab_about = st.tabs(["Mars spots", "Compare any spot", "About the data"])
    with tab_map:
        st.altair_chart(
            alt.Chart(spots).mark_circle(color=MARS_COLOUR, opacity=0.85)
            .encode(x=alt.X("lon:Q", title="Longitude, degrees east", scale=alt.Scale(zero=False)),
                    y=alt.Y("lat:Q", title="Latitude, degrees north", scale=alt.Scale(zero=False)),
                    size=alt.Size("final_score:Q", title="Overall score",
                                  scale=alt.Scale(range=[40, 600])),
                    tooltip=["Spot", alt.Tooltip("final_score:Q", title="Overall", format=".2f")])
            .properties(height=320))
        st.caption("Each dot is a 10 km spot in Jezero crater. Larger dots are closer matches.")
        st.dataframe(pd.DataFrame({
            "Rank": range(1, len(spots) + 1),
            "Mars spot": spots["Spot"],
            "Overall": spots["final_score"].round(2),
            "Terrain": spots["morph_similarity"].round(2),
            "Iron": spots["chem_score"].round(2),
            "Most alike in": [readable(x) for x in spots["agrees_on"]],
            "Most different in": [readable(x) for x in spots["differs_on"]],
        }), hide_index=True)
        st.download_button("Download Mars matches (CSV)", res["mars_matches"].to_csv(index=False),
                           "mars_matches.csv", "text/csv")
    with tab_pair:
        pick = st.selectbox(
            "Mars spot", list(spots.index),
            format_func=lambda i: f"{spots.loc[i, 'Spot']}, score {spots.loc[i, 'final_score']:.2f}")
        row = spots.loc[pick]
        twin(mars_by_name[row["mars_tile"]], start_tile, view)
        st.write(f"Most alike in {readable(row['agrees_on'])}. "
                 f"Most different in {readable(row['differs_on'])}.")
        st.altair_chart(feature_chart(m_s[shown], e_s[shown], row["mars_tile"], earth_name))
        st.caption(chart_note)

with tab_about:
    st.markdown(
        "**How it works.** Mars and Earth are cut into 10 km squares. Each square gets the same "
        "measurements: slope, height range, roughness, curvature, and image texture. The score is "
        "mostly terrain, plus an iron comparison that ranks each place against its own planet.\n\n"
        "**What the results can and can't tell you.**\n"
        "- The results are a shortlist to check on satellite imagery, not a confirmed answer.\n"
        "- The Mars side covers 16 spots in the middle of Jezero crater.\n"
        "- The Earth side covers only the dry regions in the data bundle.\n"
        "- Iron on Earth is estimated from colour, which is a rough stand-in.\n"
        "- Each iron value on Mars covers about 300 km, so nearby spots share one value.\n"
        "- Scores only compare within one run. Differences of about 0.01 are noise.\n\n"
        "**Data.** Mars elevation: Mars 2020 Science Investigation CTX DEM Mosaic (USGS, "
        "F. Calef; Mangold et al. 2021). Mars image: Mars 2020 Terrain Relative Navigation "
        "CTX orthomosaic (USGS; Fergason et al. 2020). Mars iron: "
        "NASA Mars Odyssey Gamma Ray Spectrometer, via the Planetary Data System. Earth imagery "
        "and elevation: Copernicus Sentinel-2 and Copernicus DEM, through Google Earth Engine.\n\n"
        "**AI assistance.** Claude, an AI assistant made by Anthropic, helped write the code. "
        "The team ran, tested, and checked it.")

    if model:
        st.markdown("**What the model learned.**")
        n_labels = model["labels_similar"] + model["labels_different"]
        st.write(
            f"People labelled {n_labels} Mars-Earth pairs: {model['labels_similar']} alike and "
            f"{model['labels_different']} different. A small model learned from them how much each "
            f"measurement should count. On pairs it hadn't trained on, it sorted alike from "
            f"different with a score of {model['leave_one_out_auc']:.2f}, against "
            f"{model['hand_set_auc']:.2f} for the hand-set weights (1.0 is perfect, 0.5 is "
            f"guessing).")
        if "auc_difference_range" in model:
            lo, hi = model["auc_difference_range"]
            if lo > 0:
                judgement = "so the learned weights are better"
            elif hi < 0:
                judgement = "so the hand-set weights do better, and the app uses them by default"
            else:
                judgement = "so the two aren't clearly different yet. More labels will tell"
            people = (f" People agreed on {model['agreement']['agreed']:.0%} of the pairs that more "
                      f"than one person labelled." if model.get("agreement", {}).get("pairs_shared")
                      else "")
            st.write(f"Allowing for chance, the difference is likely between {lo:+.2f} and "
                     f"{hi:+.2f}, {judgement}.{people}")
        weights = pd.DataFrame({
            "Measure": [FEATURE_NAMES.get(f, f) for f in model["features"]],
            "Weight": [model["weights"][f] for f in model["features"]],
        }).sort_values("Weight", ascending=False)
        st.altair_chart(
            alt.Chart(weights).mark_bar(color="#5F6B73")
            .encode(y=alt.Y("Measure:N", sort=None, title=None),
                    x=alt.X("Weight:Q", title="Learned weight (bigger counts more)"),
                    tooltip=["Measure", alt.Tooltip("Weight:Q", format=".2f")])
            .properties(height=36 * len(weights)))
        unused = [m.lower() for m, w in zip(weights["Measure"], weights["Weight"]) if w == 0]
        if unused:
            st.caption(f"Weight zero: {', '.join(unused)}. In these labels, they didn't help tell "
                       "alike from different.")

st.caption(
    "Earth elevation: produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and "
    "© Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European "
    "Union and ESA; all rights reserved. Earth imagery: contains modified Copernicus "
    "Sentinel data 2023."
)
