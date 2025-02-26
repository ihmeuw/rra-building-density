import math
import zipfile
from pathlib import Path

import click
import requests
import tqdm
from rra_tools import jobmon
from rra_tools.shell_tools import mkdir

from rra_building_density import cli_options as clio
from rra_building_density import constants as bdc
from rra_building_density.data import BuildingDensityData

URL_ROOT = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL"


def download_ghsl_zipfile(url: str, out_zipfile: Path, *, progress_bar: bool) -> None:
    response = requests.get(url, stream=True, timeout=10)
    scale, unit = 1024**2, "MB"
    file_size_mb = math.ceil(int(response.headers["content-length"]) / scale)

    with out_zipfile.open("wb") as handle:
        for data in tqdm.tqdm(
            response.iter_content(chunk_size=scale),
            unit=unit,
            total=file_size_mb,
            disable=not progress_bar,
        ):
            handle.write(data)


def extract_ghsl_main(
    raw_measure: str,
    time_point: str,
    output_dir: str,
    *,
    progress_bar: bool,
) -> None:
    ghsl_version = bdc.GHSL_VERSIONS["r2023a"]
    year = int(time_point[:4])
    measure_prefix, measure = ghsl_version.prefix_and_measure(raw_measure)
    template_kwargs = {
        "measure_prefix": measure_prefix,
        "measure": measure,
        "year": year,
        "time_point": time_point,
    }

    bd_data = BuildingDensityData(output_dir)
    output_root = bd_data.provider_root(ghsl_version) / time_point
    mkdir(output_root, exist_ok=True, parents=True)

    print("Downloading GHSL data...")
    url = f"{URL_ROOT}/{ghsl_version.input_template.format(**template_kwargs)}"
    out_zipfile = output_root / f"{raw_measure}_{year}.zip"
    download_ghsl_zipfile(url, out_zipfile, progress_bar=progress_bar)

    print("Extracting GHSL data...")
    # Time point is already included in the output_root
    out_file = ghsl_version.raw_output_template.format(**template_kwargs).split("/")[-1]
    with zipfile.ZipFile(out_zipfile, "r") as zip_ref:
        zip_ref.extract(out_file, output_root)

    out_zipfile.unlink()


@click.command()
@clio.with_measure(bdc.GHSLVersion.measure_map)
@clio.with_time_point()
@clio.with_output_directory(bdc.MODEL_ROOT)
@clio.with_progress_bar()
def extract_ghsl_task(
    measure: str,
    time_point: str,
    output_dir: str,
    progress_bar: bool,  # noqa: FBT001
) -> None:
    """Extract GHSL data for a given time_point and measure."""
    extract_ghsl_main(measure, time_point, output_dir, progress_bar=progress_bar)


@click.command()
@clio.with_measure(bdc.GHSLVersion.measure_map, allow_all=True)
@clio.with_time_point(allow_all=True)
@clio.with_output_directory(bdc.MODEL_ROOT)
@clio.with_queue()
def extract_ghsl(
    measure: list[str],
    time_point: str,
    output_dir: str,
    queue: str,
) -> None:
    """Extract GHSL data."""
    bd_data = BuildingDensityData(output_dir)

    ghsl_version = bdc.GHSL_VERSIONS["r2023a"]
    time_points = clio.convert_choice(time_point, ghsl_version.time_points)

    jobmon.run_parallel(
        task_name="ghsl",
        runner="bdtask extract",
        task_args={
            "output-dir": output_dir,
        },
        node_args={
            "measure": measure,
            "time-point": time_points,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "2G",
            "runtime": "20m",
            "project": "proj_rapidresponse",
            "constraints": "archive",
        },
        log_root=bd_data.log_dir("extract_ghsl"),
        max_attempts=1,
    )
