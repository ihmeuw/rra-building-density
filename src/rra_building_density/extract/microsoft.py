import shlex
import subprocess
import sys
import uuid
from pathlib import Path

import click
import geopandas as gpd
from rra_tools import jobmon
from rra_tools.shell_tools import mkdir

from rra_building_density.data import BuildingDensityData
from rra_building_density import cli_options as clio
from rra_building_density import constants as bdc


def extract_microsoft_indices_main(
    version: str,
    output_dir: str,
    overwrite: bool,  
) -> None:
    bd_data = BuildingDensityData(output_dir)
    blob_url, blob_key = bd_data.blob_credentials
    provider = f"microsoft_v{version}"

    print("Caching building density tile indices.")
    index_files = {
        "intersection": "intersection_tile_index.gpkg",
        "union": "union_tile_index.gpkg",
    }
    for index_type, index_file in index_files.items():
        print(f"Caching {index_type} index.")
        index_url = f"{blob_url}/{index_file}?{blob_key}"
        out_name = f"{index_type.replace('_', '-')}"
        cache_path = bd_data.provider_index_cache_path(provider, out_name)
        if cache_path.exists() and not overwrite:
            continue
        index = gpd.read_file(index_url)
        index["quad_name"] = index["quad"].str.replace(".tif", "")
        keep = [c for c in ["quad_name", "layers", "geometry"] if c in index]
        index = index.loc[:, keep]
        bd_data.cache_provider_index(index, provider, out_name)


def extract_microsoft_tiles_main(
    time_point: str,
    version: str,
    output_dir: str | Path,
    overwrite: bool,
    verbose: bool,
) -> None:
    bd_data = BuildingDensityData(output_dir)
    azcopy = bd_data.azcopy_binary_path

    blob_url, blob_key = bd_data.blob_credentials
    provider = f"microsoft_v{version}"
    version_key = {
        "2": "postprocess_v2",
        "3": "ensemble_v3_pp",
        "4": "v45_ensemble",
    }[version]
    input_root = (
        f"{blob_url}/predictions/{time_point}/predictions/{version_key}/*?{blob_key}"
    )

    output_root = bd_data.provider_root(provider) / time_point / "tiles"
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
        process = subprocess.Popen(
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
        subprocess.run(shlex.split(azcopy_command_str), check=True)


@click.command()  # type: ignore[arg-type]
@click.option("--dummy")
@clio.with_version(bdc.MICROSOFT_VERSIONS)
@clio.with_output_directory(bdc.MODEL_ROOT)
@clio.with_overwrite()
def extract_microsoft_indices_task(
    dummy: int,  # noqa: ARG001
    version: str,
    output_dir: str,
    overwrite: bool,
) -> None:
    """Cache building density indices."""
    extract_microsoft_indices_main(version, output_dir, overwrite)


@click.command()  # type: ignore[arg-type]
@clio.with_time_point()
@clio.with_version(bdc.MICROSOFT_VERSIONS)
@clio.with_output_directory(bdc.MODEL_ROOT)
@clio.with_overwrite()
@clio.with_verbose()
def extract_microsoft_tiles_task(
    time_point: str,
    version: str,
    output_dir: str,
    overwrite: bool,
    verbose: bool,
) -> None:
    """Cache building density tiles for a particular year and quarter."""
    extract_microsoft_tiles_main(time_point, version, output_dir, overwrite, verbose)


@click.command()  # type: ignore[arg-type]
@clio.with_time_point(allow_all=True)
@clio.with_version(bdc.MICROSOFT_VERSIONS)
@clio.with_output_directory(bdc.MODEL_ROOT)
@clio.with_overwrite()
@clio.with_queue()
def extract_microsoft(
    time_point: str,
    version: str,
    output_dir: str,
    overwrite: bool,
    queue: str,
) -> None:
    """Cache building density tiles and indices."""
    valid_time_points = bdc.MICROSOFT_TIME_POINTS[version]
    time_point = clio.convert_choice(time_point, valid_time_points)

    bd_data = BuildingDensityData(output_dir)
    provider_root = bd_data.provider_root(f"microsoft_v{version}")
    log_root = jobmon.make_log_dir(provider_root)
    tool = jobmon.get_jobmon_tool("extract_microsoft_tiles")
    workflow = tool.create_workflow(f"extract_microsoft_tiles_{uuid.uuid4()}")
    task_args: dict[str, str | None] = {
        "output-dir": output_dir,
        "version": version,
    }
    if overwrite:
        task_args["overwrite"] = None
    index_task = jobmon.build_parallel_task_graph(
        jobmon_tool=tool,
        runner="bdtask extract",
        task_name="microsoft_indices",
        node_args={"dummy": [0]},
        task_args=task_args,
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "10G",
            "runtime": "60m",
            "project": "proj_rapidresponse",
            "stdout": str(log_root / "output"),
            "stderr": str(log_root / "error"),
        },
    )


    tile_tasks = jobmon.build_parallel_task_graph(
        jobmon_tool=tool,
        runner="bdtask extract",
        task_name="microsoft",
        node_args={
            "time-point": time_point,
        },
        task_args=task_args,
        task_resources={
            "queue": queue,
            "cores": 3,
            "memory": "10G",
            "runtime": "240m",
            "project": "proj_rapidresponse",
            "stdout": str(log_root / "output"),
            "stderr": str(log_root / "error"),
        },
        max_attempts=1,
    )

    workflow.add_tasks(index_task + tile_tasks)
    jobmon.run_workflow(workflow)
