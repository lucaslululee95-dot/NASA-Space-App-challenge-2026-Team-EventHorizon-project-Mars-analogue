import os

from streamlit.testing.v1 import AppTest

from mars_analog.bundle import save_bundle
from mars_analog.sources import synthetic_tiles

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def _app(tmp_path, monkeypatch):
    save_bundle(str(tmp_path), synthetic_tiles("mars", 30, seed=1), synthetic_tiles("earth", 90, seed=2))
    monkeypatch.setenv("MARS_BUNDLE_DIR", str(tmp_path))
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
