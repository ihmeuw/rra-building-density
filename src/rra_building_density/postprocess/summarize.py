# mypy: ignore-errors
from pathlib import Path

import click
import numpy as np
import numpy.typing as npt
import pandas as pd
from rra_tools import jobmon, parallel

from rra_building_density import cli_options as clio
from rra_building_density import constants as bdc
from rra_building_density.data import BuildingDensityData


def get_quad_array(
    bd_data: BuildingDensityData,
    tile_key: str,
    time_point: str,
) -> npt.NDArray:  # type: ignore[type-arg]
    if bd_data.provider_tile_exists("microsoft_v4", tile_key, time_point):
        return bd_data.load_provider_tile("microsoft_v4", tile_key, time_point)
    else:
        return np.nan * np.ones((512, 512))


def compute_quad_stats(tile_key: str) -> pd.DataFrame:
    bd_data = BuildingDensityData()

    idx_cols = ["tile_key", "time_point"]
    data_cols = [
        "built_area_total",
        "built_pixel_total",
        "built_area_growth",
        "built_area_reduction",
        "built_area_change",
        "built_pixel_growth",
        "built_pixel_reduction",
        "built_pixel_change",
    ]
    bd_threshold = 0.01
    data = []
    last = get_quad_array(bd_data, tile_key, "no_time")
    for time_point in bdc.ALL_TIME_POINTS:
        current = get_quad_array(bd_data, tile_key, time_point)

        current_nans = np.isnan(current).sum()
        current_all_nan = current_nans == 512**2
        last_nans = np.isnan(last).sum()
        last_all_nan = last_nans.sum() == 512**2
        if current_all_nan:
            built_area, built_pix = np.nan, np.nan
            built_area_plus, built_area_minus, built_area_change = (
                np.nan,
                np.nan,
                np.nan,
            )
            built_pix_plus, built_pix_minus, built_pix_change = np.nan, np.nan, np.nan
        else:
            current_pix = np.where(current > bd_threshold, 1.0, current)
            built_area = np.nansum(current)
            built_pix = np.nansum(current_pix)
            if last_all_nan:
                built_area_plus, built_area_minus, built_area_change = (
                    np.nan,
                    np.nan,
                    np.nan,
                )
                built_pix_plus, built_pix_minus, built_pix_change = (
                    np.nan,
                    np.nan,
                    np.nan,
                )
            else:
                difference = current - last

                built_area_plus = np.nansum(np.where(difference > 0, difference, 0))
                built_area_minus = np.nansum(np.where(difference < 0, -difference, 0))
                built_area_change = built_area_plus + built_area_minus

                last_pix = np.where(last > bd_threshold, 1.0, last)
                built_pix_plus = np.nansum(
                    np.where((current_pix == 1) & (last_pix == 0), 1, 0)
                )
                built_pix_minus = np.nansum(
                    np.where((current_pix == 0) & (last_pix == 1), 1, 0)
                )
                built_pix_change = built_pix_plus + built_pix_minus

        data.append(
            (
                tile_key,
                time_point,
                built_area,
                built_pix,
                built_area_plus,
                built_area_minus,
                built_area_change,
                built_pix_plus,
                built_pix_minus,
                built_pix_change,
            )
        )
        last = current

    df = pd.DataFrame(data, columns=idx_cols + data_cols).set_index(idx_cols)
    df = df * 100 / 512**2
    return df.reset_index()


def summarize_main(
    block_key: str,
    output_dir: str | Path,
    num_cores: int,
    *,
    progress_bar: bool,
) -> pd.DataFrame:
    bd_data = BuildingDensityData(output_dir)
    tile_index = bd_data.load_tile_index()
    tile_keys = tile_index[tile_index.block_key == block_key].tile_key.tolist()

    print(f"Summarizing building density for {len(tile_keys)} in {block_key}.")
    stats = parallel.run_parallel(
        compute_quad_stats,
        tile_keys,
        num_cores=num_cores,
        progress_bar=progress_bar,
    )
    stats_df = pd.concat(stats).reset_index(drop=True)
    return stats_df


@click.command()  # type: ignore[arg-type]
@clio.with_block_key()
@clio.with_output_directory(bdc.MODEL_ROOT)
@clio.with_num_cores(default=20)
@clio.with_progress_bar()
def summarize_task(
    block_key: str,
    output_dir: str,
    num_cores: int,
    progress_bar: bool,  # noqa: FBT001
) -> None:
    """Summarize building density by tile for a block of tiles."""
    summarize_main(
        block_key,
        output_dir,
        num_cores,
        progress_bar=progress_bar,
    )


@click.command()  # type: ignore[arg-type]
@clio.with_resolution(bdc.RESOLUTIONS)
@clio.with_output_directory(bdc.MODEL_ROOT)
@clio.with_num_cores(default=20)
@clio.with_queue()
def summarize(
    output_dir: str,
    num_cores: int,
    queue: str,
) -> None:
    """Summarize building density by tile."""
    bd_data = BuildingDensityData(output_dir)
    tile_index = bd_data.load_tile_index()
    block_keys = tile_index["block_key"].unique().tolist()

    jobmon.run_parallel(
        task_name="summarize",
        runner="bdtask postprocess",
        task_args={
            "output-dir": output_dir,
            "num-cores": num_cores,
        },
        task_resources={
            "queue": queue,
            "cores": num_cores,
            "memory": "10G",
            "runtime": "240m",
            "project": "proj_rapidresponse",
        },
        node_args={
            "block-key": block_keys,
        },
        max_attempts=1,
        log_root=bd_data.summaries,
    )
