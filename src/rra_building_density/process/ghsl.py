import click
import numpy as np
import rasterra as rt
from rra_tools import jobmon

from rra_building_density import cli_options as clio
from rra_building_density import constants as bdc
from rra_building_density.data import BuildingDensityData
from rra_building_density.process import utils


def format_ghsl_main(
    block_key: str,
    time_point: str,
    resolution: int | str,
    output_dir: str,
) -> None:
    version = bdc.GHSL_VERSIONS["r2023a"]
    crs = bdc.CRSES["wgs84"].to_pyproj()
    bd_data = BuildingDensityData(output_dir)

    print("Loading the reference block")
    mask_version = bdc.LATEST_MICROSOFT_VERSION
    reference_block = bd_data.load_tile(
        resolution,
        provider=mask_version.name,
        block_key=block_key,
        # This will be the transition time point
        time_point=mask_version.time_points[0],
        measure="density",
    )

    print("Loading the tile index")
    tile_index = bd_data.load_tile_index(resolution)
    block_index = tile_index[tile_index.block_key == block_key]

    block_poly_series = block_index.dissolve("block_key").geometry
    block_poly_ghsl = (
        utils.bbox_safe_buffer(block_poly_series, 5000).to_crs(crs).iloc[0]
    )

    print("Loading the GHSL data")
    density_arr, raw_volume_arr, raw_nonresidential_density_arr = (
        utils.load_and_resample_ghsl_data(  # noqa: SLF001
            measure=measure,
            time_point=time_point,
            ghsl_version=version,
            bounds=block_poly_ghsl,
            reference_block=reference_block,
            bd_data=bd_data,
        )._ndarray
        for measure in ["density", "volume", "nonresidential_density"]
    )
    print("Generating height and proportion residential arrays")
    height_arr = utils.generate_height_array(density_arr, raw_volume_arr)  # type: ignore[arg-type]
    proportion_residential_arr = utils.generate_proportion_residential_array(
        density_arr,  # type: ignore[arg-type]
        raw_nonresidential_density_arr,  # type: ignore[arg-type]
    )

    print("Generating and saving output arrays")
    out_ops = {
        "height": lambda _, h, __: h,
        "proportion_residential": lambda _, __, p: p,
        "density": lambda d, _, __: d,
        "residential_density": lambda d, _, p: d * p,
        "nonresidential_density": lambda d, _, p: d * (1 - p),
        "volume": lambda d, h, _: h * d,
        "residential_volume": lambda d, h, p: h * d * p,
        "nonresidential_volume": lambda d, h, p: h * d * (1 - p),
    }
    for measure, op in out_ops.items():
        out = rt.RasterArray(
            data=op(density_arr, height_arr, proportion_residential_arr),  # type: ignore[no-untyped-call]
            transform=reference_block.transform,
            crs=reference_block.crs,
            no_data_value=np.nan,
        )
        bd_data.save_tile(
            out,
            resolution,
            provider="ghsl_r2023a",
            block_key=block_key,
            time_point=time_point,
            measure=measure,
        )


@click.command()
@clio.with_block_key()
@clio.with_time_point()
@clio.with_resolution(bdc.RESOLUTIONS)
@clio.with_output_directory(bdc.MODEL_ROOT)
def format_ghsl_task(
    block_key: str,
    time_point: str,
    resolution: str,
    output_dir: str,
) -> None:
    """Build predictors for a given tile and time point."""
    format_ghsl_main(block_key, time_point, resolution, output_dir)


@click.command()
@clio.with_time_point(allow_all=True)
@clio.with_resolution(bdc.RESOLUTIONS)
@clio.with_output_directory(bdc.MODEL_ROOT)
@clio.with_queue()
def format_ghsl(
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
    njobs = len(block_keys) * len(time_points)
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
