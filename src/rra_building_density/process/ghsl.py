import click
import numpy as np
from rra_tools import jobmon

from rra_building_density import cli_options as clio
from rra_building_density import constants as bdc
from rra_building_density import utils
from rra_building_density.data import BuildingDensityData


def format_ghsl_main(
    block_key: str,
    measure: str,
    time_point: str,
    resolution: int | str,
    output_dir: str,
) -> None:
    ghsl_version = bdc.GHSL_VERSIONS["r2023a"]
    ghsl_measure = ghsl_version.prefix_and_measure(measure)[1]
    crs = bdc.CRSES["wgs84"].to_pyproj()

    bd_data = BuildingDensityData(output_dir)
    print("Loading the tile index")
    tile_index = bd_data.load_tile_index(resolution)
    tile_index_info = bd_data.load_tile_index_info(resolution)

    print("Building template")
    block_index = tile_index[tile_index.block_key == block_key]
    block_poly_series = block_index.dissolve("block_key").geometry
    block_poly = block_poly_series.iloc[0]
    block_poly_ghsl = (
        utils.bbox_safe_buffer(block_poly_series, 5000).to_crs(crs).iloc[0]
    )

    block_template = utils.make_raster_template(
        block_poly,
        resolution=tile_index_info.tile_resolution,
        crs=bdc.CRSES["equal_area"],
    )

    print("Loading GHSL data")
    raw_tile = bd_data.load_provider_tile(
        ghsl_version,
        bounds=block_poly_ghsl,
        measure=ghsl_measure,
        time_point=time_point,
        year=time_point[:4],
    )
    raw_tile = raw_tile.astype(np.float32) / 10000.0

    print("Resampling")
    tile = raw_tile.set_no_data_value(np.nan).resample_to(block_template, "average")
    tile = utils.suppress_noise(tile)
    print("Saving")
    bd_data.save_tile(
        tile,
        resolution,
        provider="ghsl_r2023a",
        block_key=block_key,
        time_point=time_point,
        measure=measure,
    )


@click.command()
@clio.with_measure(bdc.GHSLVersion.measure_map)
@clio.with_block_key()
@clio.with_time_point()
@clio.with_resolution(bdc.RESOLUTIONS)
@clio.with_output_directory(bdc.MODEL_ROOT)
def format_ghsl_task(
    block_key: str,
    measure: str,
    time_point: str,
    resolution: str,
    output_dir: str,
) -> None:
    """Build predictors for a given tile and time point."""
    format_ghsl_main(block_key, measure, time_point, resolution, output_dir)


@click.command()
@clio.with_measure(bdc.GHSLVersion.measure_map, allow_all=True)
@clio.with_time_point(allow_all=True)
@clio.with_resolution(bdc.RESOLUTIONS)
@clio.with_output_directory(bdc.MODEL_ROOT)
@clio.with_queue()
def format_ghsl(
    measure: list[str],
    time_point: str,
    resolution: str,
    output_dir: str,
    queue: str,
) -> None:
    """Format GHSL building density data."""
    ghsl_version = bdc.GHSL_VERSIONS["r2023a"]
    time_points = clio.convert_choice(time_point, ghsl_version.time_points)
    bd_data = BuildingDensityData(output_dir)

    print("Loading the tile index")
    tile_index = bd_data.load_tile_index(resolution)
    block_keys = tile_index.block_key.unique().tolist()
    njobs = len(block_keys) * len(time_point) * len(measure)
    print(f"Formating building density for {njobs} block-times")

    memory, runtime = ghsl_version.process_resources(resolution)

    jobmon.run_parallel(
        task_name="ghsl",
        runner="bdtask process",
        task_args={
            "output-dir": output_dir,
            "resolution": resolution,
        },
        node_args={
            "block-key": block_keys,
            "measure": measure,
            "time-point": time_points,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": memory,
            "runtime": runtime,
            "project": "proj_rapidresponse",
        },
        log_root=bd_data.log_dir("process_ghsl"),
        max_attempts=1,
    )
