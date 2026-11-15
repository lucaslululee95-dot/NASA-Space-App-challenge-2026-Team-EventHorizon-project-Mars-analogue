"""Clear some Earth squares from the cache, so the next comparison downloads them again.

    python refresh_tiles.py                       # the snowy regions (the default)
    python refresh_tiles.py --region "Iceland highlands"

Use it after the download changes, for example when a new satellite band is
added. Only the squares in the named regions are cleared; everything else stays.
"""
import argparse
import json

from mars_analog.regions import REGIONS, region_of
from mars_analog.sources import _load_cache, _save_cache

SNOWY = ["Antarctic Dry Valleys", "Devon Island, Canadian Arctic", "Iceland highlands",
         "Changtang Plateau, Tibet"]


def clear(cache, regions):
    """The cache without squares in `regions`, and how many were removed."""
    keep = {k: t for k, t in cache.items() if region_of(t.lat, t.lon) not in regions}
    return keep, len(cache) - len(keep)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Clear Earth squares from the cache.")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--region", action="append",
                    help="a region name (repeat for several). Default: the snowy regions")
    args = ap.parse_args(argv)
    regions = args.region or SNOWY
    known = {name for name, _ in REGIONS}
    unknown = [r for r in regions if r not in known]
    if unknown:
        print("Unknown region(s): " + ", ".join(unknown))
        print("Known regions: " + ", ".join(sorted(known)))
        return 1
    with open(args.config, encoding="utf-8") as f:
        path = json.load(f)["earth"]["cache"]
    cache, removed = clear(_load_cache(path), set(regions))
    _save_cache(path, cache)
    print(f"Cleared {removed} squares in: {', '.join(regions)}. {len(cache)} squares kept.")
    print("Run the comparison to download them again: python -m mars_analog.cli --config config.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
