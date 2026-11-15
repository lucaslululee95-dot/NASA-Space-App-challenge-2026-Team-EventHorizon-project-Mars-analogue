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


def test_leave_one_out_leaves_whole_pairs_out(monkeypatch):
    sizes = []
    real_fit = train_model.fit
    monkeypatch.setattr(train_model, "fit", lambda D, y, l2=1.0: (sizes.append(len(y)) or real_fit(D, y, l2)))
    D = np.abs(np.random.default_rng(0).normal(size=(10, 2)))
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 0])
    groups = np.repeat(["a", "b", "c", "d", "e"], 2)        # each pair labelled twice
    train_model.leave_one_out(D, y, groups=groups)
    assert sizes == [8] * 5


def test_likely_range_brackets_the_scores():
    y = np.array([1] * 15 + [0] * 15)
    learned = np.linspace(1, 0, 30)                      # perfect
    hand = np.random.default_rng(1).random(30)
    groups = np.arange(30)
    r_l, r_h, r_d = train_model.likely_range(learned, hand, y, groups, n=300)
    assert r_l[0] <= 1.0 <= r_l[1] + 1e-9 and r_l[0] > 0.9
    assert r_h[0] <= r_h[1] and r_d[0] <= r_d[1]


def _two_labellers(tmp_path, flip=1):
    save_bundle(str(tmp_path / "b"), synthetic_tiles("mars", 12, seed=1), synthetic_tiles("earth", 120, seed=2))
    (tmp_path / "config.json").write_text(json.dumps({"target": [17.0, 77.0]}))
    labels = tmp_path / "labels"
    make_label_sheet.main(["--bundle", str(tmp_path / "b"), "--config", str(tmp_path / "config.json"),
                           "--labels", str(labels), "--size", "70"])
    assert make_label_sheet.main(["--labels", str(labels), "--copy-for", "AMY"]) == 0
    style = lambda name: name.split("_")[1]
    for name, who, n_flip in (("label_sheet.csv", "LL", 0), ("label_sheet_AMY.csv", "AMY", flip)):
        s = pd.read_csv(labels / name, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        s["label"] = [str(int(style(m) == style(e))) for m, e in zip(s["mars_tile"], s["earth_tile"])]
        s.loc[: n_flip - 1, "label"] = ["1" if v == "0" else "0" for v in s.loc[: n_flip - 1, "label"]]
        s["labeller"] = who
        s.to_csv(labels / name, index=False, encoding="utf-8-sig")
    return labels


def test_copy_for_has_the_same_pairs_and_no_answers(tmp_path):
    labels = _two_labellers(tmp_path)
    mine = pd.read_csv(labels / "label_sheet.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    assert make_label_sheet.main(["--labels", str(labels), "--copy-for", "BEN"]) == 0
    ben = pd.read_csv(labels / "label_sheet_BEN.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    assert list(ben["earth_tile"]) == list(mine["earth_tile"])
    assert (ben["label"] == "").all() and (ben["labeller"] == "").all()
    assert make_label_sheet.main(["--labels", str(labels), "--copy-for", "BEN"]) == 1   # won't overwrite


def test_training_combines_labellers_and_reports_agreement(tmp_path):
    labels = _two_labellers(tmp_path, flip=7)
    out = tmp_path / "model.json"
    assert train_model.main(["--labels", str(labels), "--bundle", str(tmp_path / "b"), "--out", str(out)]) == 0
    model = json.loads(out.read_text())
    assert model["labellers"] == 2 and model["pairs"] == 70
    assert model["labels_similar"] + model["labels_different"] == 140
    assert model["agreement"]["pairs_shared"] == 70
    assert abs(model["agreement"]["agreed"] - 63 / 70) < 1e-3
    lo, hi = model["auc_difference_range"]
    assert lo <= model["auc_difference"] <= hi


def test_next_sheet_spreads_over_spots_and_skips_labelled_pairs(tmp_path):
    labels = _two_labellers(tmp_path)
    model_path = tmp_path / "model.json"
    train_model.main(["--labels", str(labels), "--bundle", str(tmp_path / "b"), "--out", str(model_path)])
    seen = make_label_sheet.pairs_in_sheets(str(labels))
    assert make_label_sheet.main(["--bundle", str(tmp_path / "b"), "--labels", str(labels),
                                  "--model", str(model_path), "--next", "--spots", "3", "--size", "30"]) == 0
    new = pd.read_csv(labels / "label_sheet_2.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    assert len(new) == 30 and new["mars_tile"].nunique() == 3
    assert not set(zip(new["mars_tile"], new["earth_tile"])) & seen
    assert (new["label"] == "").all()


def test_next_sheet_works_before_any_model_exists(tmp_path):
    labels = _two_labellers(tmp_path)
    assert make_label_sheet.main(["--bundle", str(tmp_path / "b"), "--labels", str(labels),
                                  "--model", str(tmp_path / "none.json"), "--next", "--size", "20"]) == 0
    assert (labels / "label_sheet_2.csv").exists()


def test_verdict_covers_better_worse_and_unclear():
    assert "learned weights are better" in train_model.verdict([0.01, 0.08])
    assert "hand-set weights do better" in train_model.verdict([-0.04, -0.01])
    assert "Not clearly different" in train_model.verdict([-0.03, 0.09])


def test_rows_without_initials_belong_to_the_sheets_labeller(tmp_path):
    sheet = pd.DataFrame({"mars_tile": ["m"] * 4, "earth_tile": ["a", "b", "c", "d"],
                          "label": ["1", "0", "1", "1"], "labeller": ["LL", "LL", "", ""]})
    sheet.to_csv(tmp_path / "s.csv", index=False)
    pairs = train_model.read_labels([str(tmp_path / "s.csv")])
    assert pairs["who"].nunique() == 1 and set(pairs["who"]) == {"LL"}


def test_training_compares_measurement_sets_and_keeps_the_better(tmp_path):
    from mars_analog.features import EXTRA_FEATURES, FEATURE_WEIGHTS
    labels = _two_labellers(tmp_path)
    out = tmp_path / "model.json"
    assert train_model.main(["--labels", str(labels), "--bundle", str(tmp_path / "b"), "--out", str(out)]) == 0
    model = json.loads(out.read_text())
    sets = model["measurement_sets"]
    assert sets["with_landforms"]["count"] == sets["original"]["count"] + len(EXTRA_FEATURES)
    kept_all = len(model["features"]) == sets["with_landforms"]["count"]
    better = "with_landforms" if kept_all else "original"
    assert model["leave_one_out_auc"] == sets[better]["auc"]
    assert sets[better]["auc"] >= min(sets["original"]["auc"], sets["with_landforms"]["auc"])
    if not kept_all:
        assert set(model["features"]) <= set(FEATURE_WEIGHTS)
