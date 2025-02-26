import shlex
import subprocess
import sys
from pathlib import Path

import click
import geopandas as gpd
from rra_tools import jobmon
from rra_tools.shell_tools import mkdir

from rra_building_density import cli_options as clio
from rra_building_density import constants as bdc
from rra_building_density.data import BuildingDensityData


def extract_microsoft_indices_main(
    version: str,
    output_dir: str,
    *,
    overwrite: bool,
) -> None:
    msft_version = bdc.MICROSOFT_VERSIONS[version]
    bd_data = BuildingDensityData(output_dir)

    blob_url, blob_key = bd_data.blob_credentials
    azcopy = bd_data.azcopy_binary_path
    overwrite_flag = "true" if overwrite else "false"

    print("Caching building density tile indices.")
    index_files = {
        "intersection": "intersection_tile_index.gpkg",
        "union": "union_tile_index.gpkg",
        "difference": "difference_index.gpkg",
        "land_cover": "land_cover_index.gpkg",
        "land_cover_v2": "land_cover_index_v2.gpkg",
        "land_cover_v3": "land_cover_index_v3.gpkg",
    }
    for index_type, index_file in index_files.items():
        print(f"Caching {index_type} index.")
        index_url = f"{blob_url}/{index_file}?{blob_key}"
        cache_path = bd_data.provider_index_cache_path(msft_version, index_type)
        temp_cache_path = cache_path.with_suffix(".gpkg")
        mkdir(cache_path.parent, exist_ok=True)

        command = (
            f"{azcopy} copy {index_url} {temp_cache_path} "
            f"--overwrite={overwrite_flag} "
            f"--check-md5 FailIfDifferent "
            f"--from-to=BlobLocal "
            f"--log-level=INFO"
        )
        _run_azcopy_subprocess(command, verbose=True)

        index = gpd.read_file(temp_cache_path)
        temp_cache_path.unlink()
        index["quad_name"] = index["quad"].str.replace(".tif", "")
        keep = [c for c in ["quad_name", "layers", "geometry"] if c in index]
        index = index.loc[:, keep]
        bd_data.cache_provider_index(index, msft_version, index_type)


@click.command()
@clio.with_version(bdc.MICROSOFT_VERSIONS)
@clio.with_output_directory(bdc.MODEL_ROOT)
@clio.with_overwrite()
def extract_microsoft_indices_task(
    version: str,
    output_dir: str,
    overwrite: bool,  # noqa: FBT001
) -> None:
    """Cache building density indices."""
    extract_microsoft_indices_main(version, output_dir, overwrite=overwrite)


def extract_microsoft_tiles_main(
    time_point: str,
    version: str,
    output_dir: str | Path,
    *,
    overwrite: bool,
    verbose: bool,
) -> None:
    msft_version = bdc.MICROSOFT_VERSIONS[version]

    bd_data = BuildingDensityData(output_dir)
    azcopy = bd_data.azcopy_binary_path
    blob_url, blob_key = bd_data.blob_credentials

    input_stem = msft_version.input_template.format(time_point=time_point)
    input_root = f"{blob_url}/{input_stem}?{blob_key}"

    output_root = bd_data.provider_root(msft_version)
    if time_point:
        output_root = output_root / time_point
    mkdir(output_root, exist_ok=True, parents=True)

    overwrite_flag = "true" if overwrite else "false"

    command = (
        f"{azcopy} copy {input_root} {output_root} "
        f"--overwrite={overwrite_flag} "
        f"--check-md5 FailIfDifferent "
        f"--from-to=BlobLocal "
        f"--recursive "
        f"--log-level=INFO"
    )

    _run_azcopy_subprocess(command, verbose=verbose)


def _run_azcopy_subprocess(azcopy_command_str: str, *, verbose: bool = False) -> None:
    if verbose:
        process = subprocess.Popen(  # noqa: S603
            shlex.split(azcopy_command_str),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Read and print the output in real-time, updating the progress bar
        while True:
            in_progress = False
            output = process.stdout.readline()  # type: ignore[union-attr]
            if output == "" and process.poll() is not None:
                break
            if output:
                if not in_progress:
                    in_progress = "Done" in output
                # Use '\r' to return the cursor to the beginning of the line
                end = "\r" if in_progress else "\n"
                print(output.strip(), end=end, flush=True)

        # Print any remaining error messages
        stderr_output, _ = process.communicate()
        print(stderr_output, file=sys.stderr)
    else:
        subprocess.run(shlex.split(azcopy_command_str), check=True)  # noqa: S603


@click.command()
@clio.with_time_point()
@clio.with_version(bdc.MICROSOFT_VERSIONS)
@clio.with_output_directory(bdc.MODEL_ROOT)
@clio.with_overwrite()
@clio.with_verbose()
def extract_microsoft_tiles_task(
    time_point: str,
    version: str,
    output_dir: str,
    overwrite: bool,  # noqa: FBT001
    verbose: bool,  # noqa: FBT001
) -> None:
    """Cache building density tiles for a particular year and quarter."""
    extract_microsoft_tiles_main(
        time_point, version, output_dir, overwrite=overwrite, verbose=verbose
    )


@click.command()
@clio.with_time_point(allow_all=True)
@clio.with_version(bdc.MICROSOFT_VERSIONS)
@clio.with_output_directory(bdc.MODEL_ROOT)
@clio.with_overwrite()
@clio.with_queue()
def extract_microsoft(
    time_point: str,
    version: str,
    output_dir: str,
    overwrite: bool,  # noqa: FBT001
    queue: str,
) -> None:
    """Cache building density tiles and indices."""

    print("Extracting Microsoft Indices...")
    extract_microsoft_indices_main(version, output_dir, overwrite=overwrite)

    print("Extracting Microsoft Tiles...")
    bd_data = BuildingDensityData(output_dir)
    msft_version = bdc.MICROSOFT_VERSIONS[version]
    time_points = clio.convert_choice(time_point, msft_version.time_points)

    task_args: dict[str, str | None] = {
        "output-dir": output_dir,
        "version": version,
    }
    if overwrite:
        task_args["overwrite"] = None

    jobmon.run_parallel(
        runner="bdtask extract",
        task_name="microsoft",
        node_args={
            "time-point": time_points,
        },
        task_args=task_args,
        task_resources={
            "queue": queue,
            "cores": 3,
            "memory": "10G",
            "runtime": "240m",
            "project": "proj_rapidresponse",
            "stdout": str(bdc.MODEL_ROOT / "output"),
            "stderr": str(bdc.MODEL_ROOT / "error"),
        },
        max_attempts=1,
        log_root=bd_data.log_dir("extract_msft"),
    )
