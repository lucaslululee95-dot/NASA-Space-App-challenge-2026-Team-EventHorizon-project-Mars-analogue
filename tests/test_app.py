import os

from streamlit.testing.v1 import AppTest

from mars_analog.bundle import save_bundle
from mars_analog.sources import synthetic_tiles

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def _app(tmp_path, monkeypatch):
    save_bundle(str(tmp_path), synthetic_tiles("mars", 30, seed=1), synthetic_tiles("earth", 90, seed=2))
    monkeypatch.setenv("MARS_BUNDLE_DIR", str(tmp_path))
    if "MARS_MODEL_PATH" not in os.environ:          # don't pick up a real model from models/
        monkeypatch.setenv("MARS_MODEL_PATH", str(tmp_path / "no_model.json"))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def test_app_shows_results_on_load(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    assert not at.exception
    assert not at.error
    assert len(at.tabs) == 3
    assert len(at.dataframe) >= 1
    assert at.subheader[0].value.startswith("Closest match")


def test_changing_settings_and_picking_another_match(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    at.sidebar.slider[0].set_value(0.5).run()
    assert not at.exception
    pair = [s for s in at.selectbox if s.label == "Earth match"][0]
    pair.select_index(1).run()
    assert not at.exception
    assert not at.error


def test_app_explains_a_missing_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("MARS_BUNDLE_DIR", str(tmp_path / "nothing_here"))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert any("bundle_data.py" in e.value for e in at.error)


def test_app_offers_the_learned_score_when_a_model_exists(tmp_path, monkeypatch):
    import json
    cols = ["slope_mean_deg", "slope_p90_deg", "relief_m", "roughness_m", "curvature_per_m",
            "texture_rel", "grad_rel"]
    model = {"features": cols, "weights": {c: (1.0 if i < 3 else 0.0) for i, c in enumerate(cols)},
             "intercept": 2.0, "labels_similar": 41, "labels_different": 17,
             "leave_one_out_auc": 0.93, "hand_set_auc": 0.88, "max_ndvi": 0.15}
    (tmp_path / "model.json").write_text(json.dumps(model))
    monkeypatch.setenv("MARS_MODEL_PATH", str(tmp_path / "model.json"))
    at = _app(tmp_path, monkeypatch)
    assert not at.exception
    radio = [r for r in at.sidebar.radio if r.label == "Terrain score"][0]
    assert radio.value == "Learned from labels"
    assert any("learned" in c.value for c in at.caption)
    radio.set_value("Hand-set weights").run()
    assert not at.exception and not at.error


def test_app_without_a_model_has_no_score_choice(tmp_path, monkeypatch):
    monkeypatch.setenv("MARS_MODEL_PATH", str(tmp_path / "missing.json"))
    at = _app(tmp_path, monkeypatch)
    assert not at.exception
    assert not [r for r in at.sidebar.radio if r.label == "Terrain score"]


def test_app_defaults_to_hand_set_when_the_model_is_clearly_worse(tmp_path, monkeypatch):
    import json
    cols = ["slope_mean_deg", "slope_p90_deg", "relief_m", "roughness_m", "curvature_per_m",
            "texture_rel", "grad_rel"]
    model = {"features": cols, "weights": {c: 1.0 for c in cols}, "intercept": 2.0,
             "labels_similar": 30, "labels_different": 30, "leave_one_out_auc": 0.80,
             "hand_set_auc": 0.86, "auc_difference_range": [-0.09, -0.02], "max_ndvi": 0.15}
    (tmp_path / "model.json").write_text(json.dumps(model))
    monkeypatch.setenv("MARS_MODEL_PATH", str(tmp_path / "model.json"))
    at = _app(tmp_path, monkeypatch)
    radio = [r for r in at.sidebar.radio if r.label == "Terrain score"][0]
    assert radio.value == "Hand-set weights"
    assert any("hand-set weights do better" in m.value for m in at.markdown)


def _radio(at, label):
    return [r for r in at.sidebar.radio if r.label == label][0]


def _number(at, label_start):
    return [n for n in at.sidebar.number_input if n.label.startswith(label_start)][0]


def test_starting_from_earth_lists_mars_spots(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    _radio(at, "A place on").set_value("Earth").run()
    assert not at.exception and not at.error
    assert at.title[0].value.startswith("Where on Mars")
    assert at.subheader[0].value.startswith("Closest spot on Mars")
    assert [s.label for s in at.sidebar.selectbox][:2] == ["Region", "Place"]
    pick = [s for s in at.selectbox if s.label == "Mars spot"][0]
    pick.select_index(1).run()
    assert not at.exception and not at.error


def test_typed_earth_coordinates_snap_to_the_nearest_place(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    near = synthetic_tiles("earth", 90, seed=2)[1]
    _radio(at, "A place on").set_value("Earth").run()
    _radio(at, "Choose it").set_value("By coordinates").run()
    _number(at, "Latitude").set_value(round(near.lat + 0.1, 2)).run()
    _number(at, "Longitude").set_value(round(near.lon - 0.1, 2)).run()
    assert not at.exception
    assert any(c.value.startswith("Using our nearest place") for c in at.sidebar.caption)
    assert at.subheader[0].value.startswith("Closest spot on Mars")


def test_typed_coordinates_outside_the_data_are_explained(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    _radio(at, "Choose it").set_value("By coordinates").run()        # Mars side
    _number(at, "Latitude").set_value(-60.0).run()
    assert not at.exception
    assert any("outside our Mars data" in i.value for i in at.info)
    assert not at.subheader
    _radio(at, "A place on").set_value("Earth").run()
    _number(at, "Latitude").set_value(89.0).run()
    assert not at.exception
    assert any("don't have data there yet" in i.value for i in at.info)
