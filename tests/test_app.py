import os

from streamlit.testing.v1 import AppTest

from mars_analog.bundle import save_bundle
from mars_analog.sources import synthetic_tiles

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def test_app_runs_and_shows_results(tmp_path, monkeypatch):
    save_bundle(str(tmp_path), synthetic_tiles("mars", 30, seed=1), synthetic_tiles("earth", 90, seed=2))
    monkeypatch.setenv("MARS_BUNDLE_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert len(at.dataframe) == 2


def test_app_explains_a_missing_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("MARS_BUNDLE_DIR", str(tmp_path / "nothing_here"))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert any("bundle_data.py" in e.value for e in at.error)
