import click
import numpy as np
from pyproj import CRS
from rra_tools import jobmon

from rra_building_density import cli_options as clio
from rra_building_density import constants as bdc
from rra_building_density import utils
from rra_building_density.data import BuildingDensityData


def format_ghsl_main(
    block_key: str,
    measure: str,
    time_point: str,
    crs: str,
    resolution: int | str,
    output_dir: str,
) -> None:
    ghsl_measure = bdc.GHSL_MEASURE_MAP[measure][1]
    ghsl_resolution = bdc.GHSL_CRS_MAP[crs]
    crs_pyproj = bdc.CRSES[crs].to_pyproj()

    bd_data = BuildingDensityData(output_dir)
    tile_index = bd_data.load_tile_index(resolution)
    tile_index_info = bd_data.load_tile_index_info(resolution)

    block_index = tile_index[tile_index.block_key == block_key]
    block_poly_series = block_index.dissolve("block_key").geometry
    block_poly = block_poly_series.iloc[0]
    block_poly_ghsl = utils.bbox_safe_buffer(block_poly_series, 5000).to_crs(crs_pyproj).iloc[0]

    block_template = utils.make_raster_template(
        block_poly,
        resolution=tile_index_info.tile_resolution,
        crs=bdc.CRSES["equal_area"].to_pyproj(),
    )

    year = int(time_point[:4])
    start = year - year % 5
    if year % 5 == 0:
        end = start
        w = 1.0
    else:
        end = start + 5
        t = float(time_point[:4]) + float(time_point[-1:]) / 4
        w = (t - start) / (end - start)

    start_tile = bd_data.load_provider_tile(
        "ghsl_r2023a",
        bounds=block_poly_ghsl,
        measure=ghsl_measure,
        resolution=ghsl_resolution,
        year=str(start),
    )
    start_tile = start_tile.astype(np.float32) / 10000.0
    end_tile = bd_data.load_provider_tile(
        "ghsl_r2023a",
        bounds=block_poly_ghsl,
        measure=ghsl_measure,
        resolution=ghsl_resolution,
        year=str(end),
    )
    end_tile = end_tile.astype(np.float32) / 10000.0

    raw_tile = start_tile * (1 - w) + end_tile * w
    tile = raw_tile.set_no_data_value(np.nan).resample_to(block_template, "average")
    tile = utils.suppress_noise(tile)
    bd_data.save_tile(
        tile,
        resolution,
        provider="ghsl_r2023a",
        block_key=block_key,
        time_point=time_point,
        measure=measure,
    )


@click.command()  # type: ignore[arg-type]
@clio.with_block_key()
@clio.with_measure(bdc.GHSL_MEASURE_MAP)
@clio.with_time_point(bdc.ALL_TIME_POINTS)
@clio.with_crs(bdc.GHSL_CRS_MAP)
@clio.with_resolution(bdc.RESOLUTIONS)
@clio.with_output_directory(bdc.MODEL_ROOT)
def format_ghsl_task(
    block_key: str,
    measure: str,
    time_point: str,
    crs: str,
    resolution: str,
    output_dir: str,
) -> None:
    """Build predictors for a given tile and time point."""
    format_ghsl_main(block_key, measure, time_point, crs, resolution, output_dir)


@click.command()  # type: ignore[arg-type]
@clio.with_measure(bdc.GHSL_MEASURE_MAP, allow_all=True)
@clio.with_time_point(bdc.ALL_TIME_POINTS, allow_all=True)
@clio.with_crs(bdc.GHSL_CRS_MAP)
@clio.with_resolution(bdc.RESOLUTIONS)
@clio.with_output_directory(bdc.MODEL_ROOT)
@clio.with_queue()
def format_ghsl(
    measure: list[str],
    time_point: list[str],
    crs: str,
    resolution: str,
    output_dir: str,
    queue: str,
) -> None:
    """Format GHSL building density data."""
    bd_data = BuildingDensityData(output_dir)

    print("Loading the tile index")
    tile_index = bd_data.load_tile_index(resolution)
    block_keys = tile_index.block_key.unique().tolist()
    njobs = len(block_keys) * len(time_point) * len(measure)
    print(f"Formating building density for {njobs} block-times")

    jobmon.run_parallel(
        task_name="ghsl",
        runner="bdtask format",
        task_args={
            "output-dir": output_dir,
            "resolution": resolution,
        },
        node_args={
            "block-key": block_keys,
            "measure": measure,
            "time-point": time_point,
            "crs": crs,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "8G",
            "runtime": "20m",
            "project": "proj_rapidresponse",
        },
        log_root=bd_data.tiles,
        max_attempts=1,
    )
