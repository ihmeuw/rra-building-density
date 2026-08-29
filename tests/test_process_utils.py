"""Tests for the Microsoft processing nodata semantics and reprojection path."""

from typing import Any

import numpy as np
import numpy.typing as npt
import rasterra as rt
from affine import Affine

from rra_building_density.process import utils

EQUAL_AREA = "ESRI:54034"
WEB_MERCATOR = "EPSG:3857"
RES = 40.0


def make_template(width: int = 100, height: int = 100) -> rt.RasterArray:
    """An empty 40m equal-area template with its origin at (0, 4000)."""
    data = np.full((height, width), np.nan, dtype=np.float32)
    transform = Affine(RES, 0, 0, 0, -RES, 4000)
    return rt.RasterArray(data, transform, crs=EQUAL_AREA, no_data_value=np.nan)


def make_tile(
    x0: float, y0: float, n: int = 32, res: float = 38.2185, fill: float | None = None
) -> rt.RasterArray:
    """A Web Mercator tile with a value gradient (misalignment shows in values)."""
    if fill is None:
        data = (np.arange(n * n, dtype=np.float32).reshape(n, n) % 97) / 97
    else:
        data = np.full((n, n), fill, dtype=np.float32)
    transform = Affine(res, 0, x0, 0, -res, y0)
    return rt.RasterArray(data, transform, crs=WEB_MERCATOR, no_data_value=np.nan)


def window_slice(
    template: rt.RasterArray, window: rt.RasterArray, arr: npt.NDArray[Any]
) -> npt.NDArray[Any]:
    """The window's cells cut from an array on the template grid."""
    r0 = int((template.y_max - window.y_max) / RES)
    c0 = int((window.x_min - template.x_min) / RES)
    return arr[r0 : r0 + window.height, c0 : c0 + window.width]


def test_set_no_data_value_converts_the_tagged_value() -> None:
    # The raw Microsoft tiles tag -1 (water). set_no_data_value converts pixels
    # matching the tagged value; unsetting first must therefore come after the
    # water conversion, not before it.
    data = np.array([[0.5, -1.0], [-1.0, 0.2]], dtype=np.float32)
    n_water = int((data == -1).sum())
    tile = rt.RasterArray(
        data, Affine(RES, 0, 0, 0, -RES, 0), crs=WEB_MERCATOR, no_data_value=-1.0
    )

    broken = tile.unset_no_data_value().set_no_data_value(np.nan)
    assert (broken.to_numpy() == -1).sum() == n_water  # the original bug

    fixed = tile.set_no_data_value(0.0).unset_no_data_value().set_no_data_value(np.nan)
    result = fixed.to_numpy()
    assert not np.isnan(result).any()
    assert (result == 0.0).sum() == n_water
    assert np.isnan(fixed.no_data_value)


def test_suppress_noise_zeroes_small_values_and_keeps_nan() -> None:
    data = np.array([[0.005, 0.02], [np.nan, 0.0]], dtype=np.float32)
    raster = rt.RasterArray(
        data, Affine(RES, 0, 0, 0, -RES, 0), crs=EQUAL_AREA, no_data_value=np.nan
    )
    result = utils.suppress_noise(raster).to_numpy()
    assert result[0, 0] == 0.0
    assert result[0, 1] == np.float32(0.02)
    assert np.isnan(result[1, 0])
    assert result[1, 1] == 0.0


def test_make_template_window_returns_none_outside_template() -> None:
    template = make_template()
    far = 10_000_000
    for x0, y0 in [(-far, 2000), (far, 2000), (500, far), (500, -far)]:
        assert utils.make_template_window(template, make_tile(x0, y0)) is None


def test_make_template_window_is_grid_aligned_and_clipped() -> None:
    template = make_template()
    window = utils.make_template_window(template, make_tile(-600, 4600))
    assert window is not None
    assert (window.x_min - template.x_min) % RES == 0
    assert (template.y_max - window.y_max) % RES == 0
    # clipped at the template's top-left corner
    assert window.x_min == template.x_min
    assert window.y_max == template.y_max


def test_window_warp_matches_direct_warp_onto_full_template() -> None:
    # The core property of the single-pass reprojection: warping a tile onto
    # its template window is identical to warping it onto the full template.
    template = make_template()
    for tile in [make_tile(403.7, 3611.3), make_tile(-600, 4600)]:
        window = utils.make_template_window(template, tile)
        assert window is not None
        windowed = tile.resample_to(window, "average").to_numpy()
        direct = window_slice(
            template, window, tile.resample_to(template, "average").to_numpy()
        )
        assert (np.isfinite(windowed) == np.isfinite(direct)).all()
        finite = np.isfinite(windowed)
        assert np.allclose(windowed[finite], direct[finite], atol=1e-6)


def test_merge_and_crop_equals_first_wins_composition() -> None:
    # Adjacent tiles merged on the shared grid and cropped to the template
    # must equal a first-wins composition of direct warps: the merge is exact
    # and the final resample is a pixel copy.
    template = make_template()
    tile_a = make_tile(0, 3000)
    tile_b = make_tile(32 * 38.2185, 3000)
    warped = []
    for tile in (tile_a, tile_b):
        window = utils.make_template_window(template, tile)
        assert window is not None
        warped.append(tile.resample_to(window, "average"))
    merged = rt.merge(warped, method="first").resample_to(template, "nearest")
    result = merged.to_numpy()

    direct_a = tile_a.resample_to(template, "average").to_numpy()
    direct_b = tile_b.resample_to(template, "average").to_numpy()
    composed = np.where(np.isfinite(direct_a), direct_a, direct_b)

    assert (np.isfinite(result) == np.isfinite(composed)).all()
    finite = np.isfinite(result)
    assert np.array_equal(result[finite], composed[finite])
