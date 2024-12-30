from collections.abc import Collection
from typing import ParamSpec, TypeVar

import click
from rra_tools.cli_tools import (
    RUN_ALL,
    ClickOption,
    convert_choice,
    with_choice,
    with_debugger,
    with_dry_run,
    with_input_directory,
    with_num_cores,
    with_output_directory,
    with_overwrite,
    with_progress_bar,
    with_queue,
    with_verbose,
)

_T = TypeVar("_T")
_P = ParamSpec("_P")


def with_crs(
    choices: Collection[str],
    *,
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "crs",
        allow_all=allow_all,
        choices=choices,
        help="The coordinate reference system of the data to extract.",
        convert=allow_all,
    )


def with_measure(
    choices: Collection[str],
    *,
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "measure",
        "m",
        allow_all=allow_all,
        choices=choices,
        help="The measure to extract.",
        convert=allow_all,
    )


def with_year(
    choices: Collection[str],
    *,
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "year",
        "y",
        allow_all=allow_all,
        choices=choices,
        help="The year to extract.",
        convert=allow_all,
    )


def with_time_point(
    choices: Collection[str] | None = None,
    *,
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "time_point",
        "t",
        allow_all=allow_all,
        choices=choices,
        help="Time point to run.",
        convert=choices is not None and allow_all,
    )

def with_version(
    choices: Collection[str],
    *,
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "version",
        allow_all=allow_all,
        choices=choices,
        help="The version of the data to extract.",
        convert=allow_all,
    )

def with_tile_size() -> ClickOption[_P, _T]:
    return ClickOption(
        "--tile-size",
        type=int,
        default=512,
        help="The number of pixels in each tile dimension.",
        show_default=True,
    )

def with_block_size() -> ClickOption[_P, _T]:
    return ClickOption(
        "--block-size",
        type=int,
        default=16,
        help="The number of tiles in each block dimension",
        show_default=True,
    )

def with_block_key() -> ClickOption[_P, _T]:
    return click.option(
        "--block-key",
        "-b",
        type=click.STRING,
        required=True,
        help="Block key of block to run.",
    )

def with_resolution(
    choices: Collection[int],
    *,
    allow_all: bool = False,
) -> ClickOption[_P, _T]:
    return with_choice(
        "resolution",
        allow_all=allow_all,
        choices=choices,
        help="Resolution of each pixel in the units of the selected CRS.",
    )


__all__ = [
    "RUN_ALL",
    "ClickOption",
    "convert_choice",
    "with_block_key",
    "with_block_size",
    "with_choice",
    "with_crs",
    "with_debugger",
    "with_dry_run",
    "with_input_directory",
    "with_measure",
    "with_num_cores",
    "with_output_directory",
    "with_overwrite",
    "with_progress_bar",
    "with_queue",
    "with_resolution",
    "with_tile_size",
    "with_time_point",
    "with_verbose",
    "with_version",
    "with_year",
]
