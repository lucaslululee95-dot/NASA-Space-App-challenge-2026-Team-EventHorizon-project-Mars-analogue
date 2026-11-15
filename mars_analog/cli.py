"""Command-line entry point.

    python -m mars_analog.cli --demo                   # synthetic data, runs offline
    python -m mars_analog.cli --config config.json     # real Mars and Earth data
"""
import argparse
import json
import os

from .pipeline import run
from .sources import synthetic_tiles


def _build_real(cfg):
    from .geo import grid_centers
    from .sources import earth_tiles_from_gee, mars_tiles_from_geotiff

    tile_km, n = cfg["tile_km"], cfg.get("grid_px", 64)
    m, e = cfg["mars"], cfg["earth"]

    mars_centers = grid_centers(m["bbox"], m["step_deg"])
    earth_centers = [c for box in e["boxes"] for c in grid_centers(box, e["step_deg"])]
    earth_centers = list(dict.fromkeys((round(a, 4), round(b, 4)) for a, b in earth_centers))
    print(f"Candidate tiles: {len(mars_centers)} on Mars, {len(earth_centers)} on Earth")

    mars = mars_tiles_from_geotiff(mars_centers, m["dem_geotiff"], m["image_geotiff"],
                                   m.get("chemistry_csv"), tile_km, n,
                                   max_gap=m.get("max_gap_fraction", 0.02))
    earth = earth_tiles_from_gee(earth_centers, tile_km, n,
                                 project=e.get("gee_project"), year=e.get("year", 2023),
                                 cache_path=e.get("cache"))
    return mars, earth


def _print(res):
    print(f"\nTerrain features used: {', '.join(res['feature_cols'])}")
    print(f"Chemistry compared:    {', '.join(res['chem_compared']) or 'none'}")
    print(f"Chemistry not compared: {', '.join(res['chem_skipped'])}")

    print("\nMars tiles whose terrain looks most like Earth:")
    print(res["mars_ranking"].head(10).round(3).to_string(index=False))

    print(f"\nEarth tiles most like Mars target {res['target']}:")
    print(res["earth_matches"].round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description="Find where Mars looks like Earth, and vice versa.")
    ap.add_argument("--config", help="JSON config file (see config.example.json)")
    ap.add_argument("--demo", action="store_true", help="run on synthetic data")
    ap.add_argument("--out", default="results", help="folder for CSV output")
    args = ap.parse_args()

    if not args.demo and not args.config:
        ap.error("pass --demo or --config")

    if args.demo:
        cfg = {"top_k": 15}
        mars = synthetic_tiles("mars", 30, seed=1)
        earth = synthetic_tiles("earth", 90, seed=2)
    else:
        with open(args.config) as f:
            cfg = json.load(f)
        mars, earth = _build_real(cfg)

    w = cfg.get("weights", {})
    target = tuple(cfg["target"]) if cfg.get("target") else None
    res = run(
        mars, earth,
        w_morph=w.get("morph", 0.7),
        w_chem=w.get("chem", 0.3),
        max_ndvi=cfg.get("max_ndvi", 0.15),
        target=target,
        top_k=cfg.get("top_k", 25),
    )

    os.makedirs(args.out, exist_ok=True)
    res["mars_ranking"].to_csv(os.path.join(args.out, "mars_most_earthlike.csv"), index=False)
    res["earth_matches"].to_csv(os.path.join(args.out, "earth_matches_for_target.csv"), index=False)
    _print(res)
    print(f"\nCSV files written to {args.out}/")


if __name__ == "__main__":
    main()
