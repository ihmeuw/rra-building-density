import rasterra as rt
import numpy as np
import shapely
from affine import Affine
from pyproj import Transformer
import geopandas as gpd

from rra_building_density import constants as bdc

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
