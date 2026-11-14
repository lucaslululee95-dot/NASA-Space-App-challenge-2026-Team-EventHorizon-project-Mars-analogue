import numpy as np

from convert_grs import read_grs_table


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
