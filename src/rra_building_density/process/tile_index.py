import itertools

import click
import geopandas as gpd
import tqdm
from rra_tools import jobmon
from shapely import box

from rra_building_density import cli_options as clio
from rra_building_density import constants as bdc
from rra_building_density.data import BuildingDensityData, TileIndexInfo


def tile_index_main(
    tile_size: int,
    block_size: int,
    resolution: int | str,
    output_dir: str,
    *,
    progress_bar: bool,
) -> None:
    """Build the global tile index.

    We want a geographic reference frame modeling and covariate development. The
    representation of the reference frame (ie the object produced and saved by this
    function) is a shapefile mapping a tile key to a tile bounding box. This
    tile index can then be easily intersected with other geometries (e.g. input
    tile indices, admin geometries, etc.) in order to find the tiles associated
    with the intersected geometry.
    """
    bd_data = BuildingDensityData(output_dir)

    # Bounds found at https://epsg.io/
    crs = bdc.CRSES["equal_area"]
    xmin, ymin, xmax, ymax = crs.bounds

    tile_span = int(resolution) * tile_size  # Size of the tile in the units of the crs
    height, width = (ymax - ymin), (xmax - xmin)
    nx = int(width // tile_span + (width % tile_span and 1))
    ny = int(height // tile_span + (width % tile_span and 1))
    tile_nos = list(itertools.product(range(nx), range(ny)))

    print("Building Tile map")
    data = []
    # This takes ~15 minutes to run.  The vectorized version of this operation
    # is comparable in runtime (implying the underlying intersection is a python
    # loop, boo) so we keep the loop to get a progress bar.
    for xi, yi in tqdm.tqdm(tile_nos, disable=not progress_bar):
        block_key = f"B-{xi // block_size:>04}X-{yi // block_size:>04}Y"
        tile_key = f"T-{xi:>04}X-{yi:>04}Y"

        # Compute tile bounds, respecting the world bounds.
        t_xmin = xmin + xi * tile_span
        t_xmax = min(t_xmin + tile_span, xmax)
        t_ymax = ymax - yi * tile_span
        t_ymin = max(t_ymax - tile_span, ymin)

        tile_poly = box(t_xmin, t_ymin, t_xmax, t_ymax)

        data.append([block_key, tile_key, tile_poly])

    columns = ["block_key", "tile_key", "geometry"]
    modeling_frame = gpd.GeoDataFrame(data, columns=columns, crs=crs.to_pyproj())

    print("Saving")
    modeling_frame_info = TileIndexInfo(
        tile_size=tile_size,
        tile_resolution=int(resolution),
        block_size=block_size,
        crs=crs.code,
    )
    bd_data.save_tile_index(modeling_frame, modeling_frame_info)


@click.command()  # type: ignore[arg-type]
@clio.with_tile_size()
@clio.with_block_size()
@clio.with_resolution(bdc.RESOLUTIONS)
@clio.with_output_directory(bdc.MODEL_ROOT)
@clio.with_progress_bar()
def tile_index_task(
    tile_size: int,
    block_size: int,
    resolution: str,
    output_dir: str,
    progress_bar: bool,  # noqa: FBT001
) -> None:
    """Build the global tile index."""
    tile_index_main(
        tile_size,
        block_size,
        resolution,
        output_dir,
        progress_bar=progress_bar,
    )


@click.command()  # type: ignore[arg-type]
@clio.with_tile_size()
@clio.with_block_size()
@clio.with_resolution(bdc.RESOLUTIONS, allow_all=True)
@clio.with_output_directory(bdc.MODEL_ROOT)
@clio.with_queue()
def tile_index(
    tile_size: int,
    block_size: int,
    resolution: list[str],
    output_dir: str,
    queue: str,
) -> None:
    """Build the global tile index."""
    bd_data = BuildingDensityData(output_dir)
    jobmon.run_parallel(
        task_name="tile_index",
        runner="bdtask process",
        task_args={
            "output-dir": output_dir,
        },
        node_args={
            "tile-size": [tile_size],
            "block-size": [block_size],
            "resolution": resolution,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "25G",
            "runtime": "60m",
            "project": "proj_rapidresponse",
        },
        log_root=bd_data.tiles,
        max_attempts=1,
    )
