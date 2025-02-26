from collections.abc import Callable, Collection

import click
from rra_tools.cli_tools import (
    RUN_ALL,
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


def with_crs[**P, T](
    choices: Collection[str],
    *,
    allow_all: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    return with_choice(
        "crs",
        allow_all=allow_all,
        choices=choices,
        help="The coordinate reference system of the data to extract.",
        convert=allow_all,
        required=True,
    )


def with_measure[**P, T](
    choices: Collection[str],
    *,
    allow_all: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    return with_choice(
        "measure",
        "m",
        allow_all=allow_all,
        choices=choices,
        help="The measure to extract.",
        convert=allow_all,
    )


def with_time_point[**P, T](
    choices: Collection[str] | None = None,
    *,
    allow_all: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    return with_choice(
        "time_point",
        "t",
        allow_all=allow_all,
        choices=choices,
        help="Time point to run.",
        convert=choices is not None and allow_all,
    )


def with_version[**P, T](
    choices: Collection[str],
    *,
    allow_all: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    return with_choice(
        "version",
        allow_all=allow_all,
        choices=choices,
        help="The version of the data to extract.",
        convert=allow_all,
    )


def with_tile_size[**P, T]() -> Callable[[Callable[P, T]], Callable[P, T]]:
    return click.option(
        "--tile-size",
        type=int,
        default=512,
        help="The number of pixels in each tile dimension.",
        show_default=True,
    )


def with_block_size[**P, T]() -> Callable[[Callable[P, T]], Callable[P, T]]:
    return click.option(
        "--block-size",
        type=int,
        default=16,
        help="The number of tiles in each block dimension",
        show_default=True,
    )


def with_block_key[**P, T]() -> Callable[[Callable[P, T]], Callable[P, T]]:
    return click.option(
        "--block-key",
        "-b",
        type=click.STRING,
        required=True,
        help="Block key of block to run.",
    )


def with_resolution[**P, T](
    choices: Collection[str],
    *,
    allow_all: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    return with_choice(
        "resolution",
        allow_all=allow_all,
        choices=choices,
        help="Resolution of each pixel in the units of the selected CRS.",
    )


__all__ = [
    "RUN_ALL",
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
]
