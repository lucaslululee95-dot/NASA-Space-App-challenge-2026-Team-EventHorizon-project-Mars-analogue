"""Convert a GRS element table into the chemistry CSV.

    python convert_grs.py data\\mars\\fe_5x5.tab Fe
    python convert_grs.py data\\mars\\cl_5x5.tab Cl

Each run adds one element to data/mars/elements_grid.csv and keeps the others.
For each element it writes two columns:
    <element>       weight percent (blank where the table has 9999.999)
    <element>_pct   rank of the cell among all valid cells for that element, 0 to 1

The five-degree element tables share one layout, described in their .LBL files.
"""
import os
import sys

import pandas as pd

INVALID = 9999.0  # the label's NOT_APPLICABLE_CONSTANT is 9999.999
OUT = os.path.join("data", "mars", "elements_grid.csv")


def read_grs_table(path, element="Fe"):
    # Split on whitespace. The fields are fixed-width, but this still works if
    # leading spaces were lost when the file was copied.
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            lat, lon, conc = float(parts[0]), float(parts[1]), float(parts[2])
            rows.append((lat, lon, conc if conc < INVALID else None))
    return pd.DataFrame(rows, columns=["lat", "lon", element])


def update_grid(src, element, out=OUT):
    df = read_grs_table(src, element)
    df[element + "_pct"] = df[element].rank(pct=True)   # NaN stays NaN
    if os.path.exists(out):
        base = pd.read_csv(out)
        base = base.drop(columns=[c for c in (element, element + "_pct") if c in base.columns])
        df = base.merge(df, on=["lat", "lon"], how="outer")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    return df


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "mars", "fe_5x5.tab")
    element = sys.argv[2] if len(sys.argv) > 2 else "Fe"
    df = update_grid(src, element)
    valid = int(df[element].notna().sum())
    print(f"wrote {OUT}: {len(df)} cells, {valid} with valid {element}")
    jezero = df[(df["lat"] == 17.5) & (df["lon"] == 77.5)]
    if len(jezero):
        print(f"Jezero cell (17.5 N, 77.5 E): {element} = {jezero[element].iloc[0]}, "
              f"rank {round(float(jezero[element + '_pct'].iloc[0]), 3)}")


if __name__ == "__main__":
    main()