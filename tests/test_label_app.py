import os

import pandas as pd
from streamlit.testing.v1 import AppTest

import make_label_sheet
from mars_analog.bundle import save_bundle
from mars_analog.sources import synthetic_tiles

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "label_app.py")


def _setup(tmp_path, monkeypatch):
    save_bundle(str(tmp_path / "b"), synthetic_tiles("mars", 8, seed=1), synthetic_tiles("earth", 40, seed=2))
    (tmp_path / "config.json").write_text('{"target": [17.0, 77.0]}')
    sheet = tmp_path / "sheet.csv"
    make_label_sheet.main(["--bundle", str(tmp_path / "b"), "--config", str(tmp_path / "config.json"),
                           "--out", str(sheet), "--size", "6"])
    monkeypatch.setenv("MARS_BUNDLE_DIR", str(tmp_path / "b"))
    monkeypatch.setenv("MARS_LABEL_SHEET", str(sheet))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at, sheet


def _labels(sheet):
    return pd.read_csv(sheet, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def test_answers_are_saved_and_move_on(tmp_path, monkeypatch):
    at, sheet = _setup(tmp_path, monkeypatch)
    assert not at.exception
    at.sidebar.text_input[0].set_value("LL").run()
    at.text_input(key="reason_0_0").set_value("flat rocky plain").run()
    at.button(key="alike").click().run()
    assert not at.exception
    at.button(key="different").click().run()
    df = _labels(sheet)
    assert list(df["label"][:2]) == ["1", "0"]
    assert df.loc[0, "reason"] == "flat rocky plain" and df.loc[0, "labeller"] == "LL"
    assert at.subheader[0].value.startswith(f"Pair {df.loc[2, 'pair_id']}")


def test_skip_leaves_the_pair_blank(tmp_path, monkeypatch):
    at, sheet = _setup(tmp_path, monkeypatch)
    at.button(key="skip").click().run()
    at.button(key="alike").click().run()
    df = _labels(sheet)
    assert df.loc[0, "label"] == "" and df.loc[1, "label"] == "1"


def test_missing_sheet_is_explained(tmp_path, monkeypatch):
    monkeypatch.setenv("MARS_LABEL_SHEET", str(tmp_path / "nothing.csv"))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert any("make_label_sheet.py" in e.value for e in at.error)


def test_sheet_picker_switches_between_teammates_sheets(tmp_path, monkeypatch):
    save_bundle(str(tmp_path / "b"), synthetic_tiles("mars", 8, seed=1), synthetic_tiles("earth", 40, seed=2))
    (tmp_path / "config.json").write_text('{"target": [17.0, 77.0]}')
    labels = tmp_path / "labels"
    make_label_sheet.main(["--bundle", str(tmp_path / "b"), "--config", str(tmp_path / "config.json"),
                           "--labels", str(labels), "--size", "6"])
    make_label_sheet.main(["--labels", str(labels), "--copy-for", "AMY"])
    monkeypatch.delenv("MARS_LABEL_SHEET", raising=False)
    monkeypatch.setenv("MARS_LABEL_DIR", str(labels))
    monkeypatch.setenv("MARS_BUNDLE_DIR", str(tmp_path / "b"))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    picker = at.sidebar.selectbox[0]
    assert picker.label == "Label sheet" and len(picker.options) == 2
    picker.select_index(1).run()                     # label_sheet_AMY.csv
    at.button(key="different").click().run()
    assert not at.exception
    amy = _labels(labels / "label_sheet_AMY.csv")
    mine = _labels(labels / "label_sheet.csv")
    assert amy.loc[0, "label"] == "0" and mine.loc[0, "label"] == ""


def test_initials_are_remembered_from_the_sheet(tmp_path, monkeypatch):
    at, sheet = _setup(tmp_path, monkeypatch)
    at.sidebar.text_input[0].set_value("LL").run()
    at.button(key="alike").click().run()
    fresh = AppTest.from_file(APP, default_timeout=60)    # as if the page were restarted
    fresh.run()
    assert fresh.sidebar.text_input[0].value == "LL"
