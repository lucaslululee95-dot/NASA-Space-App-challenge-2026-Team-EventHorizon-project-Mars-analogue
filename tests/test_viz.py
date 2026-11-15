import numpy as np

from mars_analog.viz import hillshade, surface_image


def test_hillshade_shape_and_type():
    img = hillshade(np.random.default_rng(0).normal(size=(64, 64)), 10)
    assert img.shape == (256, 256) and img.dtype == np.uint8


def test_flat_ground_is_evenly_lit():
    img = hillshade(np.full((64, 64), 500.0), 10)
    assert img.min() == img.max()


def test_light_from_northwest_lights_west_facing_slopes():
    x = np.tile(np.arange(64, dtype=float), (64, 1)) * 50.0   # ground rises to the east
    west_facing = hillshade(x, 10).mean()
    east_facing = hillshade(-x, 10).mean()
    assert west_facing > east_facing


def test_surface_image_colour_and_grey():
    colour = surface_image(np.full((3, 64, 64), 0.5))
    grey = surface_image(np.full((1, 64, 64), 0.5))
    assert colour.shape == (256, 256, 3) and grey.shape == (256, 256)
    assert colour.dtype == np.uint8 and grey.dtype == np.uint8
