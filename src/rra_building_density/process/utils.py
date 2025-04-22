from typing import Any

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pyproj
import rasterra as rt
import shapely
from affine import Affine
from pyproj import Transformer

from rra_building_density import constants as bdc
from rra_building_density.data import BuildingDensityData


def make_raster_template(
    bounding_box: shapely.Polygon,
    resolution: int,
    crs: bdc.CRS = bdc.CRSES["equal_area"],
) -> rt.RasterArray:
    """Build a template raster for a given tile.

    A raster template is an empty raster with a fixed resolution and extent. It is
    used to provide a resampling target for other rasters, so they end up with
    exactly the same extent and resolution.

    Parameters
    ----------
    bounding_box
        The polygon defining the extent of the raster.
    resolution
        The resolution of the tile in the units of the CRS, ie how wide/tall is a pixel.
    crs
        The CRS of the raster.

    Returns
    -------
    rt.RasterArray
        The template raster.
    """
    xmin, ymin, xmax, ymax = bounding_box.bounds

    # We can overlap a bit in the x-direction. This is because the tiles are
    # not perfectly aligned with the antimeridian, so we need to allow for a
    # bit of overlap to avoid gaps in the final output.
    x_pixels = (xmax - xmin) // resolution + ((xmax - xmin) % resolution and 1)
    # We don't overlap in the y-direction, we just ignore the last row of pixels.
    y_pixels = (ymax - ymin) // resolution

    data = np.nan * np.ones((int(y_pixels), int(x_pixels)), dtype=np.float32)
    transform = Affine(
        a=resolution,
        b=0,
        c=np.round(xmin, 2),
        d=0,
        e=-resolution,
        f=np.round(ymax, 2),
    )
    return rt.RasterArray(data, transform, crs=crs.to_pyproj(), no_data_value=np.nan)


def suppress_noise(
    raster: rt.RasterArray,
    noise_threshold: float = 0.01,
    fill_value: float = 0.0,
) -> rt.RasterArray:
    """Suppress small values in a raster.

    Parameters
    ----------
    raster
        The raster to suppress noise in.
    noise_threshold
        The threshold below which values are considered noise.

    Returns
    -------
    rt.RasterArray
        The raster with small values suppressed
    """
    raster._ndarray[raster._ndarray < noise_threshold] = fill_value  # noqa: SLF001
    return raster


def bbox_safe_buffer(geom: gpd.GeoSeries, distance: float) -> gpd.GeoSeries:
    """Buffer a geometry, but ensure it stays within the CRS bounds.

    Parameters
    ----------
    geom
        The geometry to buffer.
    distance
        The distance to buffer the geometry.
    """
    crs = geom.crs
    transformer = Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True)
    hard_bounds = shapely.box(*transformer.transform_bounds(*crs.area_of_use.bounds))
    buffered_geom = geom.buffer(distance).intersection(hard_bounds)
    return buffered_geom


def precise_floor(a: float, precision: int = 0) -> float:
    """Round a number down to a given precision.

    Parameters
    ----------
    a
        The number to round down.
    precision
        The number of decimal places to round down to.

    Returns
    -------
    float
        The rounded down number.
    """
    return float(np.true_divide(np.floor(a * 10**precision), 10**precision))


def get_block_polys(
    block_index: gpd.GeoDataFrame,
    native_crs: pyproj.CRS,
) -> tuple[shapely.Polygon, shapely.Polygon]:
    """Get polygons representing the block boundary in the target and native CRSs.

    The target CRS is the CRS of the block index and will be the CRS of the
    final tiles. The native CRS is the CRS of the input data, which may come
    in a variety of projections.

    Parameters
    ----------
    block_index
        The subset of the tile index for a particular block.
    native_crs
        The CRS of the working data.

    Returns
    -------
    tuple[shapely.Polygon, shapely.Polygon]
        The block polygon in the native and target CRSs.
    """
    block_poly_series = block_index.dissolve("block_key").geometry
    block_poly = block_poly_series.iloc[0]
    block_poly_native = block_poly_series.to_crs(native_crs).iloc[0]
    return block_poly, block_poly_native


def get_provider_tile_keys(
    provider_index: gpd.GeoDataFrame,
    block_poly: shapely.Polygon,
    provider_version: bdc.BuiltVersion,
    bd_data: BuildingDensityData,
    **kwargs: Any,
) -> list[str]:
    """Get the provider tiles that overlap with the block.

    Parameters
    ----------
    provider_index
        The provider index.
    block_poly
        The block polygon.
    provider_version
        The provider version.
    **kwargs
        Additional keyword arguments to pass to the provider index.

    Returns
    -------
    list[str]
        The provider tile keys that overlap with the block.
    """
    overlapping = provider_index.loc[
        provider_index.intersects(block_poly), "quad_name"
    ].tolist()

    provider_tile_keys = [
        tile_key
        for tile_key in overlapping
        if bd_data.provider_tile_exists(provider_version, tile_key=tile_key, **kwargs)
    ]
    return provider_tile_keys


def fix_microsoft_tile(
    tile: rt.RasterArray,
) -> rt.RasterArray:
    # The resolution of the MSFT tiles has too many decimal points.
    # This causes tiles slightly west of the antimeridian to cross
    # over and really mucks up reprojection. We'll clip the values
    # here to 5 decimal places (ie to 100 microns), explicitly
    # rounding down. This reduces the width of the tile by
    # 512*0.0001 = 0.05m or 50cm, enough to fix roundoff issues.
    x_res, y_res = tile.resolution
    xmin, xmax, ymin, ymax = tile.bounds
    tile._transform = Affine(  # noqa: SLF001
        a=precise_floor(x_res, 4),
        b=0.0,
        c=xmin,
        d=0.0,
        e=-precise_floor(-y_res, 4),
        f=ymax,
    )
    return tile


def load_and_resample_ghsl_data(
    measure: str,
    time_point: str,
    ghsl_version: bdc.GHSLVersion,
    bounds: shapely.Polygon | shapely.MultiPolygon,
    reference_block: rt.RasterArray,
    bd_data: BuildingDensityData,
) -> rt.RasterArray:
    ghsl_measure = ghsl_version.prefix_and_measure(measure)[1]
    raw_tile = bd_data.load_provider_tile(
        ghsl_version,
        bounds=bounds,
        measure=ghsl_measure,
        time_point=time_point,
        year=time_point[:4],
    )
    raw_tile = raw_tile.astype(np.float32) / 10000.0
    tile = raw_tile.set_no_data_value(np.nan).resample_to(reference_block, "average")
    tile._ndarray[reference_block.no_data_mask] = np.nan  # noqa: SLF001
    tile._ndarray[np.isnan(tile._ndarray) & ~reference_block.no_data_mask] = 0.0  # noqa: SLF001
    tile = suppress_noise(tile)
    return tile


POSITIVE_THRESHOLD = 0.01
HEIGHT_MIN = 2.4384  # 8ft
HEIGHT_MAX = 828  # Burj Khalifa


def generate_height_array(
    density_arr: npt.NDArray[np.floating[Any]],
    volume_arr: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    density_is_positive = density_arr > POSITIVE_THRESHOLD
    volume_is_positive = volume_arr > POSITIVE_THRESHOLD

    height_arr = np.zeros_like(density_arr)
    height_arr[density_is_positive] = (
        volume_arr[density_is_positive] / density_arr[density_is_positive]
    )

    height_min = HEIGHT_MIN
    if np.any(density_is_positive & volume_is_positive):
        height_min = max(
            height_arr[density_is_positive & volume_is_positive].min(), HEIGHT_MIN
        )

    height_arr = np.where(
        density_is_positive & (height_arr < height_min),
        height_min,
        height_arr,
    )

    height_too_large = height_arr > HEIGHT_MAX
    if np.any(height_too_large):
        msg = f"Height array has values greater than {HEIGHT_MAX}"
        raise ValueError(msg)

    density_is_nan = np.isnan(density_arr)
    height_arr[density_is_nan] = np.nan

    return height_arr


def generate_proportion_residential_array(
    density_arr: npt.NDArray[np.floating[Any]],
    nonresidential_density_arr: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    # We'll get the nan mask from the density array
    nonresidential_density_arr = np.nan_to_num(nonresidential_density_arr, 0.0)  # type: ignore[call-overload]

    density_is_positive = density_arr > POSITIVE_THRESHOLD

    proportion_residential_arr = np.zeros_like(density_arr)
    proportion_residential_arr[density_is_positive] = (
        density_arr[density_is_positive]
        - nonresidential_density_arr[density_is_positive]
    ) / density_arr[density_is_positive]

    # If we have actual negative values, that's bad
    tolerance = 1e-6
    if np.any(proportion_residential_arr < -tolerance):
        msg = "Proportion residential has values less than 0.0"
        raise ValueError(msg)

    # As long as it's not very negative (ie as long as it's negative due to
    # floating point error), we'll set it to 0.0
    if np.any(proportion_residential_arr < 0):
        proportion_residential_arr[proportion_residential_arr < 0] = 0.0

    if np.any(proportion_residential_arr > 1.0):
        msg = "Proportion residential has values greater than 1.0"
        raise ValueError(msg)

    density_is_nan = np.isnan(density_arr)
    proportion_residential_arr[density_is_nan] = np.nan

    return proportion_residential_arr
