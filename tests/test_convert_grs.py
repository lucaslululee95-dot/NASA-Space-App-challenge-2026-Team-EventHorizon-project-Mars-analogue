import numpy as np

from convert_grs import read_grs_table, update_grid


def test_reads_fixed_layout_and_blanks_invalid_cells(tmp_path):
    p = tmp_path / "fe.tab"
    p.write_text(
        "-87.5     2.5   9999.999        9999.999        9999.999\n"
        "17.5    77.5     13.751           1.820           1.969\n"
        "-57.5     2.5     11.629           3.449           3.595\n"
    )
    df = read_grs_table(str(p))
    assert len(df) == 3
    assert np.isnan(df["Fe"].iloc[0])
    assert df["Fe"].iloc[1] == 13.751
    assert df.iloc[1][["lat", "lon"]].tolist() == [17.5, 77.5]


def test_adding_an_element_keeps_the_others(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fe = tmp_path / "fe.tab"
    cl = tmp_path / "cl.tab"
    fe.write_text("17.5    77.5     13.751           1.820           1.969\n"
                  "-57.5     2.5     11.629           3.449           3.595\n")
    cl.write_text("17.5    77.5      0.520           0.100           0.110\n"
                  "-57.5     2.5   9999.999        9999.999        9999.999\n")
    update_grid(str(fe), "Fe")
    df = update_grid(str(cl), "Cl")
    assert {"Fe", "Fe_pct", "Cl", "Cl_pct"} <= set(df.columns)
    jezero = df[(df["lat"] == 17.5) & (df["lon"] == 77.5)].iloc[0]
    assert jezero["Fe"] == 13.751 and jezero["Cl"] == 0.52
    blank = df[(df["lat"] == -57.5) & (df["lon"] == 2.5)]["Cl"].iloc[0]
    assert np.isnan(blank)
