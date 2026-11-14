"""Smoke test: pull a few real Earth tiles from Google Earth Engine.

    python check_earth_engine.py --project YOUR-PROJECT-ID

Uses the Atacama Desert, which should be dry (low NDVI). If this prints a
table with NDVI well below 0.15, the Earth side of the pipeline works.
"""
import argparse
import sys

from mars_analog.features import feature_table
from mars_analog.geo import grid_centers
from mars_analog.sources import earth_tiles_from_gee

ATACAMA_BOX = (-70.0, -24.0, -68.0, -22.0)  # (west, south, east, north), degrees


def main():
    ap = argparse.ArgumentParser(description="Pull a few real Earth tiles from Earth Engine.")
    ap.add_argument("--project", required=True, help="your Earth Engine Cloud project ID")
    ap.add_argument("--tiles", type=int, default=10, help="how many tiles to request")
    args = ap.parse_args()

    centers = grid_centers(ATACAMA_BOX, step_deg=0.25)   # 81 candidate tiles
    step = max(1, len(centers) // args.tiles)
    sample = centers[::step][:args.tiles]                 # spread across the box

    print(f"Requesting {len(sample)} Atacama tiles from project '{args.project}'...")
    try:
        tiles = earth_tiles_from_gee(sample, tile_km=10, n=64, project=args.project)
    except Exception as exc:
        print("\nEarth Engine did not connect. The error was:")
        print(f"  {exc}\n")
        print("Check these in order:")
        print("  1. The project is registered for Earth Engine:")
        print(f"     https://code.earthengine.google.com/register?project={args.project}")
        print("  2. You logged in with the same Google account:")
        print('     python -c "import ee; ee.Authenticate()"')
        print("  3. The project ID is spelled exactly as in Google Cloud Console.")
        sys.exit(1)

    if not tiles:
        print("\nNo tiles came back. Look for the [earth] messages above.")
        sys.exit(1)

    df = feature_table(tiles)
    cols = ["name", "lat", "lon", "ndvi", "chem_ferric_index",
            "slope_mean_deg", "relief_m", "roughness_m", "texture_rel"]
    print()
    print(df[[c for c in cols if c in df.columns]].round(3).to_string(index=False))

    dry = (df["ndvi"] < 0.15).sum()
    print(f"\n{len(tiles)} of {len(sample)} tiles returned. {dry} have NDVI below 0.15 (expected: most).")


if __name__ == "__main__":
    main()
