import math
import zipfile

import click
import requests
import tqdm
from rra_tools import jobmon
from rra_tools.shell_tools import mkdir

from rra_building_density import cli_options as clio
from rra_building_density import constants as bdc
from rra_building_density.data import BuildingDensityData


def extract_ghsl_main(
    crs: str,
    raw_measure: str,
    year: str,
    output_dir: str,
    *,
    progress_bar: bool,
) -> None:
    bd_data = BuildingDensityData(output_dir)
    provider_root = bd_data.provider_root("ghsl_r2023a")
    mkdir(provider_root, exist_ok=True)
    out_zipfile = provider_root / f"{crs}_{raw_measure}_{year}.zip"

    resolution = bdc.GHSL_CRS_MAP[crs]
    measure_prefix, measure = bdc.GHSL_MEASURE_MAP[raw_measure]

    url_root = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL"
    url = f"{url_root}/GHS_{measure_prefix}_GLOBE_R2023A/GHS_{measure}_E{year}_GLOBE_R2023A_{resolution}/V1-0/GHS_{measure}_E{year}_GLOBE_R2023A_{resolution}_V1_0.zip"

    response = requests.get(url, stream=True, timeout=10)
    scale, unit = 1024**2, "MB"
    file_size_mb = math.ceil(int(response.headers["content-length"]) / scale)

    print("Downloading GHSL data...")
    with out_zipfile.open("wb") as handle:
        for data in tqdm.tqdm(
            response.iter_content(chunk_size=scale),
            unit=unit,
            total=file_size_mb,
            disable=not progress_bar,
        ):
            handle.write(data)

    print("Extracting GHSL data...")
    with zipfile.ZipFile(out_zipfile, "r") as zip_ref:
        zip_ref.extract(
            f"GHS_{measure}_E{year}_GLOBE_R2023A_{resolution}_V1_0.tif", provider_root
        )

    out_zipfile.unlink()


@click.command()  # type: ignore[arg-type]
@clio.with_crs(bdc.GHSL_CRS_MAP)
@clio.with_measure(bdc.GHSL_MEASURE_MAP)
@clio.with_year(bdc.GHSL_YEARS)
@clio.with_output_directory(bdc.MODEL_ROOT)
@clio.with_progress_bar()
def extract_ghsl_task(
    crs: str,
    measure: str,
    year: str,
    output_dir: str,
    progress_bar: bool,  # noqa: FBT001
) -> None:
    """Extract GHSL data for a given year and measure."""
    extract_ghsl_main(crs, measure, year, output_dir, progress_bar=progress_bar)


@click.command()  # type: ignore[arg-type]
@clio.with_crs(bdc.GHSL_CRS_MAP, allow_all=True)
@clio.with_measure(bdc.GHSL_MEASURE_MAP, allow_all=True)
@clio.with_year(bdc.GHSL_YEARS, allow_all=True)
@clio.with_output_directory(bdc.MODEL_ROOT)
@clio.with_queue()
def extract_ghsl(
    crs: list[str],
    measure: list[str],
    year: list[str],
    output_dir: str,
    queue: str,
) -> None:
    """Extract GHSL data."""
    bd_data = BuildingDensityData(output_dir)
    provider_root = bd_data.provider_root("ghsl_r2023a")
    mkdir(provider_root, exist_ok=True)
    log_dir = bd_data.log_dir("extract_ghsl")

    jobmon.run_parallel(
        task_name="ghsl",
        runner="bdtask extract",
        task_args={
            "output-dir": output_dir,
        },
        node_args={
            "crs": crs,
            "measure": measure,
            "year": year,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "2G",
            "runtime": "20m",
            "project": "proj_rapidresponse",
            "constraints": "archive",
        },
        log_root=log_dir,
        max_attempts=1,
    )
