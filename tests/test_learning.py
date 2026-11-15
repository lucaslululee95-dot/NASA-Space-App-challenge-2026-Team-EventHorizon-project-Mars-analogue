import json

import numpy as np
import pandas as pd

import make_label_sheet
import train_model
from mars_analog.bundle import save_bundle
from mars_analog.sources import synthetic_tiles


def test_auc_values():
    y = np.array([1, 1, 0, 0])
    assert train_model.auc([0.9, 0.8, 0.2, 0.1], y) == 1.0
    assert train_model.auc([0.1, 0.2, 0.8, 0.9], y) == 0.0
    assert train_model.auc([0.5, 0.5, 0.5, 0.5], y) == 0.5


def test_fit_finds_the_measure_that_matters():
    rng = np.random.default_rng(0)
    D = np.abs(rng.normal(size=(300, 3)))
    y = (rng.random(300) < 1 / (1 + np.exp(-(2.0 - 3.0 * D[:, 0])))).astype(int)
    b, w = train_model.fit(D, y)
    assert w[0] > 1.0 and w[0] > 3 * max(w[1], w[2])
    assert (w >= 0).all()


def test_sheet_has_no_scores_and_is_not_overwritten(tmp_path):
    save_bundle(str(tmp_path / "b"), synthetic_tiles("mars", 12, seed=1), synthetic_tiles("earth", 60, seed=2))
    (tmp_path / "config.json").write_text(json.dumps({"target": [17.0, 77.0]}))
    out = tmp_path / "labels" / "sheet.csv"
    args = ["--bundle", str(tmp_path / "b"), "--config", str(tmp_path / "config.json"), "--out", str(out)]
    assert make_label_sheet.main(args + ["--size", "30"]) == 0
    sheet = pd.read_csv(out, encoding="utf-8-sig")
    assert len(sheet) == 30 and sheet["mars_tile"].nunique() == 1
    assert not any("score" in c or "similarity" in c for c in sheet.columns)
    out.write_text(out.read_text(encoding="utf-8-sig").replace(",,,\n", ",1,test,me\n", 1), encoding="utf-8-sig")
    before = out.read_text(encoding="utf-8-sig")
    assert make_label_sheet.main(args) == 1                      # refuses to replace labels
    assert out.read_text(encoding="utf-8-sig") == before


def test_training_end_to_end_learns_terrain_style(tmp_path):
    save_bundle(str(tmp_path / "b"), synthetic_tiles("mars", 12, seed=1), synthetic_tiles("earth", 120, seed=2))
    (tmp_path / "config.json").write_text(json.dumps({"target": [17.0, 77.0]}))
    sheet_path = tmp_path / "sheet.csv"
    make_label_sheet.main(["--bundle", str(tmp_path / "b"), "--config", str(tmp_path / "config.json"),
                           "--out", str(sheet_path), "--size", "80"])
    sheet = pd.read_csv(sheet_path, encoding="utf-8-sig")
    style = lambda name: name.split("_")[1]                      # synthetic names carry the terrain style
    sheet["label"] = [int(style(m) == style(e)) for m, e in zip(sheet["mars_tile"], sheet["earth_tile"])]
    sheet.to_csv(sheet_path, index=False, encoding="utf-8-sig")

    out = tmp_path / "model.json"
    assert train_model.main(["--labels", str(sheet_path), "--bundle", str(tmp_path / "b"),
                             "--out", str(out)]) == 0
    model = json.loads(out.read_text())
    assert model["leave_one_out_auc"] > 0.8
    assert set(model["weights"]) == set(model["features"])


def test_training_stops_with_too_few_labels(tmp_path, capsys):
    save_bundle(str(tmp_path / "b"), synthetic_tiles("mars", 6, seed=1), synthetic_tiles("earth", 20, seed=2))
    sheet = pd.DataFrame({"mars_tile": ["mars_smooth_000"] * 4, "earth_tile": ["earth_smooth_003"] * 4,
                          "label": [1, 0, 1, ""]})
    sheet.to_csv(tmp_path / "s.csv", index=False)
    assert train_model.main(["--labels", str(tmp_path / "s.csv"), "--bundle", str(tmp_path / "b"),
                             "--out", str(tmp_path / "m.json")]) == 1
    assert "Not enough labels" in capsys.readouterr().out
