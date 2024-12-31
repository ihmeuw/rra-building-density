# ruff: noqa
# mypy: ignore-errors
import uuid
from pathlib import Path
from typing import Any

import click
import geopandas as gpd
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rra_tools import jobmon

TITLE_FONTSIZE = 24
AX_TITLE_FONTSIZE = 20
AX_LABEL_FONTSIZE = 16
FIG_SIZE = (30, 20)
GRID_SPEC_MARGINS = {"top": 0.92, "bottom": 0.08}


def safe_divide(
    x: Any,
    y: Any,
) -> Any:
    z = x / y
    z[(x == 0) & (y == 0)] = 0
    z[np.isinf(z)] = np.nan
    return z


def prepare_block_summaries(
    block_data: pd.DataFrame, urban_threshold: float = 2.5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    urban_quads = (
        block_data[block_data.built_area_total > urban_threshold]
        .quad_name.unique()
        .tolist()
    )
    urban_mask = block_data.quad_name.isin(urban_quads)

    ur_masks = [
        ("combined", pd.Series(data=True, index=block_data.index)),
        ("urban", urban_mask),
        ("rural", ~urban_mask),
    ]

    space_summaries = []
    time_summaries = []
    for ur_cat, ur_mask in ur_masks:
        ss_df = (
            block_data.loc[ur_mask]
            .drop(columns="quad_name")
            .groupby("quarter")
            .mean()
            .reset_index()
            .assign(urban_rural=ur_cat)
        )
        space_summaries.append(ss_df)

        ts_df = (
            block_data.loc[ur_mask]
            .drop(columns="quarter")
            .groupby("quad_name")
            .mean()
            .reset_index()
            .assign(urban_rural=ur_cat)
        )
        time_summaries.append(ts_df)

    space_summary_df = pd.concat(space_summaries).reset_index(drop=True)
    time_summary_df = pd.concat(time_summaries).set_index("quad_name")
    empty_mask = (
        block_data.drop(columns="quarter")
        .groupby("quad_name")
        .built_area_total.apply(lambda s: s.isna().all())
    )
    empty_quads = empty_mask[empty_mask].index.tolist()
    time_summary_df.loc[empty_quads] = np.nan
    time_summary_df = time_summary_df.reset_index()
    time_summary_df["east"] = time_summary_df.quad_name.apply(
        lambda s: int(s.split("-")[1].split("E")[0])
    )
    time_summary_df["north"] = time_summary_df.quad_name.apply(
        lambda s: int(s.split("-")[2].split("N")[0])
    )
    time_summary_df["built_area_volatility"] = safe_divide(
        time_summary_df["built_area_change"], time_summary_df["built_area_total"]
    )
    time_summary_df["built_pixel_volatility"] = safe_divide(
        time_summary_df["built_pixel_change"],
        time_summary_df["built_pixel_total"] + 0.1,
    )

    return space_summary_df, time_summary_df


def make_title_and_legend(
    fig: plt.Figure,  # type: ignore[name-defined]
    title: str,
    names_and_colors: list[tuple[str, str]],
) -> None:
    fig.suptitle(title, x=0.5, fontsize=TITLE_FONTSIZE, ha="center")
    fig.legend(
        handles=make_legend_handles(names_and_colors),
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        fontsize=AX_LABEL_FONTSIZE,
        frameon=False,
        ncol=len(names_and_colors),
    )


def write_or_show(
    fig: plt.Figure,  # type: ignore[name-defined]
    plot_file: str | Path | None,
) -> None:
    if plot_file:
        fig.savefig(plot_file, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def make_legend_handles(names_and_colors: list[tuple[str, str]]) -> list[mlines.Line2D]:
    handles = [
        mlines.Line2D([], [], color=color, label=name, linewidth=2.5)
        for name, color in names_and_colors
    ]
    return handles


def plot_building_density_block_main(
    block_name: str,
    output_dir: str,
    *,
    write: bool = True,
) -> None:
    pps_data = PeoplePerStructureData(output_dir)
    tile_index = pps_data.load_full_tile_index()
    block_gdf = tile_index[tile_index.block_name == block_name]
    global_a0 = gpd.read_file(
        GEOSPATIAL.admin_shapefiles / "current" / "lbd_standard_admin_0_simplified.shp"
    )

    block_data = pps_data.load_building_density_summary(block_name)

    space_summary_df, time_summary_df = prepare_block_summaries(block_data)

    fig = plt.figure(figsize=FIG_SIZE)
    grid_spec = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=[2, 4, 2],
        wspace=0.2,
    )
    grid_spec.update(**GRID_SPEC_MARGINS)

    gs_area = grid_spec[0, 0].subgridspec(nrows=3, ncols=1)
    gs_center = grid_spec[0, 1].subgridspec(
        nrows=2, ncols=1, height_ratios=[1.5, 4], hspace=0.1
    )
    gs_line = gs_center[1, 0].subgridspec(nrows=5, ncols=2)
    gs_pix = grid_spec[0, 2].subgridspec(nrows=3, ncols=1)

    ax_map = fig.add_subplot(gs_center[0])
    block_gdf.dissolve().boundary.plot(ax=ax_map, color="r")
    global_a0.plot(ax=ax_map)
    global_a0.boundary.plot(ax=ax_map, color="k", linewidth=0.2)
    strip_axes(ax_map)

    heatmap_df = (
        time_summary_df[time_summary_df.urban_rural == "combined"]
        .set_index(["north", "east"])
        .sort_index()
    )

    names_and_colors = [
        ("Urban", "tab:purple"),
        ("Rural", "tab:green"),
        ("Combined", "black"),
    ]
    for metric, gs in zip(["area", "pixel"], [gs_area, gs_pix], strict=False):
        measure_map = [
            (
                f"built_{metric}_total",
                f"Average Built {metric.title()}",
                f"% of {metric.title()}",
                "GnBu",
                None,
            ),
            (
                f"built_{metric}_change",
                f"Cumulative Built {metric.title()} Change",
                f"% of {metric.title()}",
                "YlGn",
                None,
            ),
            (
                f"built_{metric}_volatility",
                f"Built {metric.title()} Volatility",
                f"Fraction of {metric.title()}",
                "PuRd",
                None,
            ),
        ]

        for row, (measure, title, cbar_title, cmap, vmax) in enumerate(measure_map):
            ax = fig.add_subplot(gs[row])
            g = sns.heatmap(
                heatmap_df[measure].unstack().sort_index(ascending=False),
                cmap=cmap,
                ax=ax,
                vmax=vmax,
                cbar_kws={"label": cbar_title},
            )
            g.set_facecolor("lightgrey")
            g.figure.axes[-1].yaxis.label.set_size(AX_LABEL_FONTSIZE)
            ax.set_title(title, fontsize=AX_TITLE_FONTSIZE)

    for col, measure in enumerate(["area", "pixel"]):
        row_map = {
            0: "total",
            2: "change",
            3: "growth",
            4: "reduction",
        }

        ax_group = []
        for row, metric in row_map.items():
            ax = fig.add_subplot(gs_line[row, col])
            for ur_cat, color in names_and_colors:
                mask = space_summary_df.urban_rural == ur_cat.lower()
                (
                    space_summary_df.loc[mask]
                    .set_index("quarter")[f"built_{measure}_{metric}"]
                    .plot(ax=ax, color=color, linewidth=2)
                )

            if row == 0:
                ax.set_title(f"{measure.title()}", fontsize=AX_TITLE_FONTSIZE)
            if row != 4:  # noqa: PLR2004
                ax.set_xlabel(None)  # type: ignore[arg-type]
            if col == 0:
                ax.set_ylabel(f"{metric.capitalize()} (%)", fontsize=AX_LABEL_FONTSIZE)

            ax_group.append(ax)

        ax = fig.add_subplot(gs_line[1, col])
        stacked_df = (
            space_summary_df.loc[space_summary_df.urban_rural == "combined"]
            .iloc[1:]
            .set_index("quarter")
        )
        stacked_df.rename(
            columns={
                f"built_{measure}_growth": "growth",
                f"built_{measure}_reduction": "reduction",
            }
        )[["growth", "reduction"]].plot.area(
            ax=ax,
            color={"growth": "slateblue", "reduction": "indianred"},
            legend=col == 0,
        )
        stacked_df[f"built_{measure}_change"].plot(ax=ax, color="k", linewidth=2)
        ax.set_xlabel(None)  # type: ignore[arg-type]
        if col == 0:
            ax.set_ylabel("Total change (%)", fontsize=AX_LABEL_FONTSIZE)
        ax_group.append(ax)

        fig.align_ylabels(ax_group)

    make_title_and_legend(
        fig,
        block_name,
        names_and_colors,
    )
    plot_file = (
        str(pps_data.building_density_plots / f"{block_name}.pdf") if write else ""
    )
    write_or_show(fig, plot_file)


def plot_building_density_global_main(output_dir: str) -> None:
    pps_data = PeoplePerStructureData(output_dir)
    tile_index = pps_data.load_full_tile_index()
    block_names = tile_index["block_name"].unique().tolist()
    global_data = pd.concat(
        [
            pps_data.load_building_density_summary(block_name)
            for block_name in block_names
        ]
    )
    print(global_data)


@click.command()  # type: ignore[arg-type]
@clio.with_block_name()
@clio.with_output_directory(DEFAULT_MODEL_ROOT)
def plot_building_density_block_task(
    block_name: str,
    output_dir: str,
) -> None:
    """Plot comprehensive building density diagnostics for a block of tiles."""
    plot_building_density_block_main(block_name, output_dir)


@click.command()  # type: ignore[arg-type]
@click.option("--dummy", type=int, default=0)
@clio.with_output_directory(DEFAULT_MODEL_ROOT)
def plot_building_density_global_task(
    dummy: int,  # noqa: ARG001
    output_dir: str,
) -> None:
    """Plot a comprehensive global summary of building density."""
    plot_building_density_global_main(output_dir)


@click.command()  # type: ignore[arg-type]
@clio.with_output_directory(DEFAULT_MODEL_ROOT)
@clio.with_queue()
def plot_building_density(
    output_dir: str,
    queue: str,
) -> None:
    """Plot comprehensive building density diagnostics."""
    pps_data = PeoplePerStructureData(output_dir)
    tile_index = pps_data.load_full_tile_index()
    block_names = tile_index["block_name"].unique().tolist()

    log_root = pps_data.building_density_plots
    log_dir = jobmon.make_log_dir(log_root)
    tool = jobmon.get_jobmon_tool("plot_building_density")
    workflow = tool.create_workflow(
        name=f"plot_building_density_{uuid.uuid4()}",
    )
    block_tasks = jobmon.build_parallel_task_graph(
        jobmon_tool=tool,
        runner="ppstask preprocess",
        task_name="plot_building_density_block",
        node_args={"block-name": block_names},
        task_args={"output-dir": output_dir},
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "10G",
            "runtime": "60m",
            "project": "proj_rapidresponse",
            "stdout": str(log_dir / "output"),
            "stderr": str(log_dir / "error"),
        },
    )

    global_task = jobmon.build_parallel_task_graph(
        jobmon_tool=tool,
        runner="ppstask preprocess",
        task_name="plot_building_density_global",
        node_args={"dummy": [0]},
        task_args={"output-dir": output_dir},
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "10G",
            "runtime": "60m",
            "project": "proj_rapidresponse",
            "stdout": str(log_dir / "output"),
            "stderr": str(log_dir / "error"),
        },
    )

    workflow.add_tasks(block_tasks + global_task)
    jobmon.run_workflow(workflow)
