import click
import numpy as np
import rasterra as rt
from affine import Affine
from rra_tools import jobmon

from rra_building_density import cli_options as clio
from rra_building_density import constants as bdc
from rra_building_density import utils
from rra_building_density.data import BuildingDensityData


def format_microsoft_main(
    block_key: str,
    time_point: str,
    version: str,
    resolution: int | str,
    output_dir: str,
) -> None:
    provider = f"microsoft_v{version}"
    bd_data = BuildingDensityData(output_dir)
    tile_index = bd_data.load_tile_index(resolution)
    tile_index_info = bd_data.load_tile_index_info(resolution)
    msft_index = bd_data.load_provider_index(provider, "union")

    block_index = tile_index[tile_index.block_key == block_key]
    block_poly_series = block_index.dissolve("block_key").geometry
    block_poly = block_poly_series.iloc[0]
    block_poly_msft = block_poly_series.to_crs(msft_index.crs).iloc[0]

    block_template = utils.make_raster_template(
        block_poly,
        resolution=tile_index_info.tile_resolution,
        crs=bdc.CRSES["equal_area"],
    )

    overlapping = msft_index.intersects(block_poly_msft)
    msft_tile_keys = [
        tile_key
        for tile_key in msft_index.loc[overlapping, "quad_name"].tolist()
        if bd_data.provider_tile_exists(
            provider, tile_key=tile_key, time_point=time_point
        )
    ]

    if not msft_tile_keys:
        print("No overlapping building tiles, likely open ocean.")
        bd_data.save_tile(
            block_template,
            resolution,
            provider=provider,
            block_key=block_key,
            time_point=time_point,
            measure="density",
        )
        return

    bd_tiles = []
    for tile_key in msft_tile_keys:
        bd_tile = bd_data.load_provider_tile(
            provider, tile_key=tile_key, time_point=time_point
        )
        # The resolution of the MSFT tiles has too many decimal points.
        # This causes tiles slightly west of the antimeridian to cross
        # over and really mucks up reprojection. We'll clip the values
        # here to 5 decimal places (ie to 100 microns), explicitly
        # rounding down. This reduces the width of the tile by
        # 512*0.0001 = 0.05m or 50cm, enough to fix roundoff issues.
        x_res, y_res = bd_tile.resolution
        xmin, xmax, ymin, ymax = bd_tile.bounds
        bd_tile._transform = Affine(  # noqa: SLF001
            a=utils.precise_floor(x_res, 4),
            b=0.0,
            c=xmin,
            d=0.0,
            e=-utils.precise_floor(-y_res, 4),
            f=ymax,
        )
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
        provider=provider,
        block_key=block_key,
        time_point=time_point,
        measure="density",
    )


def link_time_points(
    bd_data: BuildingDensityData,
    version: str,
    time_points: list[str],
    resolution: int | str,
) -> None:
    def _to_float(tp: str) -> float:
        year, quarter = tp.split("q")
        return float(year) + float(quarter) / 4

    all_time_points = [(tp, _to_float(tp)) for tp in bdc.ALL_TIME_POINTS]
    src_time_points = [(tp, _to_float(tp)) for tp in time_points]
    src_dest_time_points = []
    for dest_tp, dest_tpf in all_time_points:
        src_tp = sorted(src_time_points, key=lambda x: abs(x[1] - dest_tpf))[0][0]
        if src_tp != dest_tp:
            src_dest_time_points.append((src_tp, dest_tp))

    for src_tp, dest_tp in src_dest_time_points:
        src = bd_data.tile_path(
            resolution, f"microsoft_v{version}", "test_block", src_tp, "density"
        ).parent
        dest = bd_data.tile_path(
            resolution, f"microsoft_v{version}", "test_block", dest_tp, "density"
        ).parent
        if dest.exists():
            if not dest.is_symlink():
                msg = f"Destination {dest} exists and is not a symlink"
                raise RuntimeError(msg)
            dest.unlink()
        dest.symlink_to(src)


@click.command()  # type: ignore[arg-type]
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


@click.command()  # type: ignore[arg-type]
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
    valid_time_points = bdc.MICROSOFT_TIME_POINTS[version]
    time_points = clio.convert_choice(time_point, valid_time_points)

    bd_data = BuildingDensityData(output_dir)

    print("Loading the tile index")
    tile_index = bd_data.load_tile_index(resolution)
    block_keys = tile_index.block_key.unique().tolist()

    njobs = len(block_keys) * len(time_points)
    print(f"Formating building density for {njobs} block-times")

    memory, runtime = {
        "40": ("3G", "5m"),
        "100": ("3G", "20m"),
        "250": ("3G", "60m"),
        "500": ("3G", "120m"),
    }[resolution]

    status = jobmon.run_parallel(
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

    if status != "D":
        msg = f"Workflow failed with status {status}"
        raise RuntimeError(msg)

    if time_point == clio.RUN_ALL:
        print("Workflow complete, linking time points")
        link_time_points(bd_data, version, time_points, resolution)
