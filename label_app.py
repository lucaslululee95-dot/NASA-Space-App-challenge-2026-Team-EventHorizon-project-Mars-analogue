"""Label Mars-Earth pairs in the browser.

    python -m streamlit run label_app.py

Shows the Mars target and one Earth tile side by side, drawn the same way, and
saves each answer to labels/label_sheet.csv straight away. Run it on your own
PC: the public web app can't save files.
"""
import os

import pandas as pd
import streamlit as st

from mars_analog.bundle import load_bundle
from mars_analog.regions import fmt_place
from mars_analog.viz import hillshade, surface_image

SHEET = os.environ.get("MARS_LABEL_SHEET", os.path.join("labels", "label_sheet.csv"))
BUNDLE_DIR = os.environ.get("MARS_BUNDLE_DIR", "bundle")

st.set_page_config(page_title="Label pairs", page_icon="🪨", layout="wide")


@st.cache_resource
def get_tiles(folder):
    mars, earth = load_bundle(folder)
    return {t.name: t for t in mars}, {t.name: t for t in earth}


def load_sheet(path):
    df = pd.read_csv(path, encoding="utf-8-sig", encoding_errors="replace",
                     dtype=str, keep_default_na=False)
    for col in ("label", "reason", "labeller"):
        if col not in df.columns:
            df[col] = ""
    df["label"] = df["label"].str.strip().replace({"1.0": "1", "0.0": "0"})
    return df


def save_sheet(df, path):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def next_unlabelled(df, after):
    """The next row after `after` without a 1 or 0, wrapping round; `after` if none."""
    n = len(df)
    for step in range(1, n + 1):
        j = (after + step) % n
        if df.loc[j, "label"] not in ("1", "0"):
            return j
    return after


def answer(value):
    df, i = st.session_state.df, st.session_state.i
    df.loc[i, "label"] = value
    df.loc[i, "reason"] = st.session_state.get(f"reason_{i}", "").strip()
    df.loc[i, "labeller"] = st.session_state.get("labeller", "").strip()
    try:
        save_sheet(df, SHEET)
    except PermissionError:
        st.session_state.problem = ("Couldn't save, because another program has label_sheet.csv "
                                    "open. Close it in Excel, then click the answer again.")
        return
    st.session_state.problem = ""
    st.session_state.i = next_unlabelled(df, i)


def move(step):
    st.session_state.i = (st.session_state.i + step) % len(st.session_state.df)


# ---------------------------------------------------------------- page

st.title("Label pairs")

if not os.path.exists(SHEET):
    st.error(f"The label sheet {SHEET} is missing. Make it with `python make_label_sheet.py`.")
    st.stop()
if not os.path.exists(os.path.join(BUNDLE_DIR, "earth.npz")):
    st.error(f"The data bundle is missing from '{BUNDLE_DIR}'. Build it with `python bundle_data.py`.")
    st.stop()

mars_by_name, earth_by_name = get_tiles(BUNDLE_DIR)
if "df" not in st.session_state:
    st.session_state.df = load_sheet(SHEET)
    st.session_state.i = next_unlabelled(st.session_state.df, len(st.session_state.df) - 1)
    st.session_state.problem = ""
df, i = st.session_state.df, st.session_state.i
row = df.loc[i]

alike = int((df["label"] == "1").sum())
different = int((df["label"] == "0").sum())
st.progress((alike + different) / len(df),
            text=f"{alike + different} of {len(df)} labelled: {alike} alike, {different} different. "
                 "Training needs at least 10 of each.")

with st.sidebar:
    st.text_input("Your initials", key="labeller")
    st.number_input("Go to pair", 1, len(df), int(i) + 1, key="goto",
                    on_change=lambda: st.session_state.update(i=st.session_state.goto - 1))
    st.caption("Judge the shape of the land, not its colour: how flat or steep it is, "
               "how much it rises and falls, and its surface texture.")

if st.session_state.problem:
    st.error(st.session_state.problem)

st.subheader(f"Pair {row['pair_id']}: do these show the same kind of land?")
st.caption("Each picture is a 10 km square. Shaded relief uses the same lighting on both planets.")

mars = mars_by_name.get(row["mars_tile"])
earth = earth_by_name.get(row["earth_tile"])
left, right = st.columns(2)
for col, tile, title in ((left, mars, "Mars, Jezero crater"), (right, earth, f"Earth, {row['place']}")):
    with col:
        st.markdown(f"**{title}**")
        if tile is None:
            st.warning("This tile isn't in the data bundle. Skip it, or open it in Google Maps.")
            continue
        st.image(hillshade(tile.dem, tile.tile_km), caption="Shaded relief", width="stretch")
        st.image(surface_image(tile.rgb), caption="Satellite image", width="stretch")
        st.caption(fmt_place(tile.lat, tile.lon))

if row.get("nearest_catalogue_site", ""):
    st.info(f"Hint: {row['km_to_site']} km from {row['nearest_catalogue_site']}, which a catalogue "
            f"lists as an analogue for {row['catalogue_mars_features']}. Judge what you see.")
if row.get("satellite_view", ""):
    st.link_button("See the Earth place in Google Maps", row["satellite_view"])

st.text_input("Reason (a few words)", value=row["reason"], key=f"reason_{i}")
if row["label"] in ("1", "0"):
    st.caption(f"Current answer: {'alike' if row['label'] == '1' else 'different'}. "
               "Click again to change it.")
a, b, c = st.columns(3)
a.button("Alike (1)", key="alike", on_click=answer, args=("1",), type="primary", width="stretch")
b.button("Different (0)", key="different", on_click=answer, args=("0",), width="stretch")
c.button("Not sure, skip", key="skip", on_click=move, args=(1,), width="stretch")
p, n = st.columns(2)
p.button("Previous pair", key="prev", on_click=move, args=(-1,), width="stretch")
n.button("Next pair", key="next", on_click=move, args=(1,), width="stretch")
