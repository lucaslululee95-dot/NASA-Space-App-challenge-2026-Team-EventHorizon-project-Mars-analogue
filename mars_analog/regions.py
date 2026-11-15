"""Named Earth regions, shared by the web app and the labelling tools."""

# [west, south, east, north] in degrees. Tiles outside them are shown by coordinates.
REGIONS = [
    ("Atacama Desert, Chile", [-70.5, -25.0, -68.0, -22.0]),
    ("Namib Desert, Namibia", [14.0, -25.0, 15.5, -23.0]),
    ("Mojave Desert, USA", [-117.5, 34.0, -115.5, 36.0]),
    ("Libyan Desert, Sahara", [24.0, 20.0, 26.0, 22.0]),
    ("Western Desert, Australia", [130.0, -26.0, 132.0, -24.0]),
    ("Lut Desert, Iran", [57.5, 30.0, 59.5, 32.0]),
    ("Gobi Desert", [102.0, 41.0, 104.0, 43.0]),
    ("Changtang Plateau, Tibet", [80.0, 33.0, 82.0, 35.0]),
    ("Taklamakan Desert, China", [80.0, 38.0, 82.0, 40.0]),
    ("Central Australia, near Alice Springs", [133.0, -24.5, 134.5, -23.0]),
    ("Erfoud, Morocco", [-5.0, 30.5, -3.5, 31.5]),
]


def fmt_place(lat, lon):
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{abs(lat):.2f}° {ns}, {abs(lon):.2f}° {ew}"


def region_of(lat, lon):
    for name, (w, s, e, n) in REGIONS:
        if w - 1e-6 <= lon <= e + 1e-6 and s - 1e-6 <= lat <= n + 1e-6:
            return name
    return None


def place_name(lat, lon):
    """The region's name, or its coordinates when it isn't in a named region."""
    return region_of(lat, lon) or fmt_place(lat, lon)
