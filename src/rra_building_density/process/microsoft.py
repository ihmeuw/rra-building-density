import click
import numpy as np
import rasterra as rt
from rra_tools import jobmon

from rra_building_density import cli_options as clio
from rra_building_density import constants as bdc
from rra_building_density.data import BuildingDensityData
from rra_building_density.process import utils


def format_microsoft_main(
    block_key: str,
    time_point: str,
    version: str,
    resolution: int | str,
    output_dir: str,
) -> None:
    msft_version = bdc.MICROSOFT_VERSIONS[version]
    bd_data = BuildingDensityData(output_dir)
    tile_index = bd_data.load_tile_index(resolution)
    tile_index_info = bd_data.load_tile_index_info(resolution)
    msft_index = bd_data.load_provider_index(msft_version, "union")

    block_poly, block_poly_msft = utils.get_block_polys(
        tile_index[tile_index.block_key == block_key], msft_index.crs
    )

    block_template = utils.make_raster_template(
        block_poly,
        resolution=tile_index_info.tile_resolution,
        crs=bdc.CRSES["equal_area"],
    )

    msft_tile_keys = utils.get_provider_tile_keys(
        msft_index, block_poly_msft, msft_version, bd_data, time_point=time_point
    )

    if not msft_tile_keys:
        print("No overlapping building tiles, likely open ocean.")
        bd_data.save_tile(
            block_template,
            resolution,
            provider=msft_version.name,
            block_key=block_key,
            time_point=time_point,
            measure="density",
        )
        return

    bd_tiles = []
    for tile_key in msft_tile_keys:
        bd_tile = bd_data.load_provider_tile(
            msft_version, tile_key=tile_key, time_point=time_point
        )
        bd_tile = utils.fix_microsoft_tile(bd_tile)
        bd_tile = bd_tile.unset_no_data_value().set_no_data_value(np.nan)

        reprojected_tile = bd_tile.reproject(
            dst_resolution=block_template.x_resolution,
            dst_crs=block_template.crs,
            resampling="average",
        )
        bd_tiles.append(reprojected_tile)

    building_density = rt.merge(bd_tiles, method="first")
    building_density = building_density.resample_to(block_template, "average")
    building_density = utils.suppress_noise(building_density)
    bd_data.save_tile(
        building_density,
        resolution,
        provider=msft_version.name,
        block_key=block_key,
        time_point=time_point,
        measure="density",
    )


@click.command()
@clio.with_block_key()
@clio.with_time_point()
@clio.with_version(bdc.MICROSOFT_VERSIONS)
@clio.with_resolution(bdc.RESOLUTIONS)
@clio.with_output_directory(bdc.MODEL_ROOT)
def format_microsoft_task(
    block_key: str,
    time_point: str,
    version: str,
    resolution: str,
    output_dir: str,
) -> None:
    """Format building density for a given tile and time point."""
    format_microsoft_main(block_key, time_point, version, resolution, output_dir)


@click.command()
@clio.with_time_point(allow_all=True)
@clio.with_version(bdc.MICROSOFT_VERSIONS)
@clio.with_resolution(bdc.RESOLUTIONS)
@clio.with_output_directory(bdc.MODEL_ROOT)
@clio.with_queue()
def format_microsoft(
    time_point: str,
    version: str,
    resolution: str,
    output_dir: str,
    queue: str,
) -> None:
    """Format Microsoft building density data."""
    if version == "water_mask":
        msg = "Formatting can't be run on water mask"
        raise NotImplementedError(msg)

    msft_version = bdc.MICROSOFT_VERSIONS[version]
    time_points = clio.convert_choice(time_point, msft_version.time_points)

    bd_data = BuildingDensityData(output_dir)

    print("Loading the tile index")
    tile_index = bd_data.load_tile_index(resolution)
    block_keys = tile_index.block_key.unique().tolist()

    njobs = len(block_keys) * len(time_points)
    print(f"Formating building density for {njobs} block-times")

    memory, runtime = msft_version.process_resources(resolution)

    jobmon.run_parallel(
        task_name="microsoft",
        runner="bdtask process",
        task_args={
            "version": version,
            "resolution": resolution,
            "output-dir": output_dir,
        },
        node_args={
            "block-key": block_keys,
            "time-point": time_points,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": memory,
            "runtime": runtime,
            "project": "proj_rapidresponse",
        },
        log_root=bd_data.log_dir("process_microsoft"),
        max_attempts=3,
    )
