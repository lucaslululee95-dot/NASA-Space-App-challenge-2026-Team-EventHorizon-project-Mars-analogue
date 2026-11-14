import numpy as np
import pandas as pd

from mars_analog.chemistry import chemistry_scores


def test_iron_uses_global_percentile_when_given():
    mars = pd.DataFrame({"chem_Fe": [13.0, 13.0, 13.0], "chem_Fe_pct": [0.30, 0.30, 0.30]})
    earth = pd.DataFrame({"chem_ferric_index": [1.0, 2.0, 3.0]})   # percentiles 1/3, 2/3, 1
    res = chemistry_scores(mars, 0, earth)
    assert res["compared"] == ["iron"]
    e_pct = np.array([1 / 3, 2 / 3, 1.0])
    assert np.allclose(res["score"], 1 - np.abs(e_pct - 0.30))


def test_tied_mars_values_sit_at_middle_rank():
    mars = pd.DataFrame({"chem_Fe": [13.0, 13.0, 13.0]})           # no global percentile column
    earth = pd.DataFrame({"chem_ferric_index": [1.0, 2.0, 3.0]})
    res = chemistry_scores(mars, 0, earth)
    e_pct = np.array([1 / 3, 2 / 3, 1.0])
    assert np.allclose(res["score"], 1 - np.abs(e_pct - 0.5))


def test_nearest_cell_returns_iron_and_percentile():
    from mars_analog.sources import nearest_chem
    df = pd.DataFrame({"lat": [17.5, 22.5], "lon": [77.5, 77.5],
                       "Fe": [13.751, 9.0], "Fe_pct": [0.67, 0.2]})
    v = nearest_chem(df, 18.4, 77.55)          # Jezero sits in the 17.5 N, 77.5 E cell
    assert v["Fe"] == 13.751 and v["Fe_pct"] == 0.67
