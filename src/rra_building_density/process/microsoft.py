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
    print("Loading indices")
    tile_index = bd_data.load_tile_index(resolution)
    tile_index_info = bd_data.load_tile_index_info(resolution)
    msft_index = bd_data.load_provider_index(msft_version, "union")

    print("Extracting block polygons")
    block_poly, block_poly_msft = utils.get_block_polys(
        tile_index[tile_index.block_key == block_key], msft_index.crs
    )

    print("Making block template")
    block_template = utils.make_raster_template(
        block_poly,
        resolution=tile_index_info.tile_resolution,
        crs=bdc.CRSES["equal_area"],
    )

    print("Getting provider tile keys")
    msft_tile_keys = utils.get_provider_tile_keys(
        msft_index, block_poly_msft, msft_version, bd_data, time_point=time_point
    )

    print("Checking for no overlapping building tiles")
    if not msft_tile_keys:
        print("No overlapping building tiles, likely open ocean.")
        for measure in msft_version.bands:
            bd_data.save_tile(
                block_template,
                resolution,
                provider=msft_version.name,
                block_key=block_key,
                time_point=time_point,
                measure=measure,
            )
        return

    for measure, band in msft_version.bands.items():
        print(f"Loading and processing {len(msft_tile_keys)} {measure} tiles")
        tiles = []
        for tile_key in msft_tile_keys:
            tile = bd_data.load_provider_tile(
                msft_version, tile_key=tile_key, time_point=time_point, band=band
            )
            tile = utils.fix_microsoft_tile(tile)
            tile = tile.unset_no_data_value().set_no_data_value(np.nan)

            reprojected_tile = tile.reproject(
                dst_resolution=block_template.x_resolution,
                dst_crs=block_template.crs,
                resampling="average",
            )
            tiles.append(reprojected_tile)

        print("Merging and resampling tiles")
        full_tile = rt.merge(tiles, method="first")
        full_tile = full_tile.resample_to(block_template, "average")
        full_tile = utils.suppress_noise(full_tile)
        print(f"Saving {measure} tile")
        bd_data.save_tile(
            full_tile,
            resolution,
            provider=msft_version.name,
            block_key=block_key,
            time_point=time_point,
            measure=measure,
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
