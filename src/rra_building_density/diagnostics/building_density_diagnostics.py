# ruff: noqa
# mypy: ignore-errors
import math
import itertools
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterra as rt
import matplotlib.pyplot as plt
from matplotlib import patches
from pathlib import Path

import seaborn as sns
from rra_population_pipelines.shared.data.team import RRAPopulationData
from rra_population_pipelines.shared.plot_utils import strip_axes
from shapely import box

from rra_tools import parallel

TIME_POINTS = [
    f"{year}q{quarter}"
    for year, quarter in list(itertools.product(range(2018, 2024), range(1, 5)))
][:-1]


def count_quarters(quad):
    pop_data = RRAPopulationData()
    return sum(
        [
            (
                pop_data.building_density_data
                / t
                / "postprocess_v2"
                / f"{t}_{quad}.tif"
            ).exists()
            for t in TIME_POINTS
        ]
    )


def load_full_tile_index(block_resolution=50):
    index_path = Path("full_tile_index.parquet")
    if not index_path.exists():
        pop_data = RRAPopulationData()
        tile_index = pop_data.load_building_density_index()
        tile_index["east"] = tile_index.quad_name.apply(
            lambda s: int(s.split("-")[1].split("E")[0])
        )
        tile_index["north"] = tile_index.quad_name.apply(
            lambda s: int(s.split("-")[2].split("N")[0])
        )
        tile_index = pd.concat([tile_index, tile_index.bounds], axis=1)
        east_west = (
            tile_index[["east", "minx", "maxx"]].drop_duplicates().sort_values("east")
        )
        north_south = (
            tile_index[["north", "miny", "maxy"]].drop_duplicates().sort_values("north")
        )
        full = north_south.merge(east_west, how="cross")
        full["quad_name"] = full.apply(
            lambda r: f"L15-{int(r['east']):0>4}E-{int(r['north']):0>4}N", axis=1
        )
        full["geometry"] = full.apply(
            lambda r: box(r["minx"], r["miny"], r["maxx"], r["maxy"]), axis=1
        )
        full = gpd.GeoDataFrame(full, crs="EPSG:4326")
        quarter_count = parallel.run_parallel(
            count_quarters,
            full["quad_name"].tolist(),
            num_cores=25,
            progress_bar=True,
        )
        full["quarter_count"] = quarter_count
        full = full[["quad_name", "north", "east", "quarter_count", "geometry"]]
        full.to_parquet(index_path)

    full = gpd.read_parquet(index_path)

    full["block_x"] = full["east"] // block_resolution
    full["block_y"] = full["north"] // block_resolution
    full["block_name"] = full.apply(
        lambda x: f"B-{x['block_x']:0>3}X-{x['block_y']:0>3}Y", axis=1
    )
    full["all_empty"] = full.groupby("block_name").quarter_count.transform("sum") == 0
    full = full.sort_values(
        ["block_y", "block_x"], ascending=[False, True]
    ).reset_index(drop=True)
    return full


pop_data = RRAPopulationData()
global_a0 = gpd.read_file(
    "/snfs1/WORK/11_geospatial/admin_shapefiles/current/lbd_standard_admin_0_simplified.shp"
)
tile_index = load_full_tile_index()

blocks = tile_index[~tile_index.all_empty]["block_name"].unique().tolist()
block = blocks[458]

block_gdf = tile_index[tile_index.block_name == block]


def get_quad_array(quad_name, time_point):
    pop_data = RRAPopulationData()
    path = (
        pop_data.building_density_data
        / time_point
        / "postprocess_v2"
        / f"{time_point}_{quad_name}.tif"
    )
    # import pdb; pdb.set_trace()
    if path.exists():
        return rt.load_raster(path).to_numpy()
    else:
        return np.nan * np.ones((512, 512))


def quad_stats(quad_name):
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
    time_points = [
        f"{year}q{quarter}"
        for year, quarter in list(itertools.product(range(2018, 2024), range(1, 5)))
    ][:-1]

    data = []
    last = get_quad_array(quad_name, "no_time")
    for tp in time_points:
        current = get_quad_array(quad_name, tp)

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
            current_pix = np.where(current > 0.01, 1.0, current)
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

                last_pix = np.where(last > 0.01, 1.0, last)
                built_pix_plus = np.nansum(
                    np.where((current_pix == 1) & (last_pix == 0), 1, 0)
                )
                built_pix_minus = np.nansum(
                    np.where((current_pix == 0) & (last_pix == 1), 1, 0)
                )
                built_pix_change = built_pix_plus + built_pix_minus

        data.append(
            (
                quad_name,
                tp,
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
    return pd.DataFrame(data, columns=["quad_name", "time_point"] + data_cols)


stats = parallel.run_parallel(
    quad_stats,
    quads,
    num_cores=25,
    progress_bar=True,
)


stats_df = pd.concat(stats).reset_index(drop=True)
stats_df = (
    stats_df.set_index(["quad_name", "time_point"]) * 100 / 512**2
).reset_index()


def safe_divide(x, y):
    z = x / y
    z[(x == 0) & (y == 0)] = 0
    z[np.isinf(z)] = np.nan
    return z


pct_data = []
for measure, change_type in itertools.product(
    ["area", "pixel"], ["growth", "reduction", "change"]
):
    pct_data.append(
        100
        * stats_df.groupby("quad_name")
        .apply(
            lambda g: safe_divide(
                g[f"built_{measure}_{change_type}"],
                10 + g[f"built_{measure}_total"].shift(1),
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
        .rename(f"built_{measure}_{change_type}")
    )
pct_df = pd.concat(
    [
        stats_df[["quad_name", "time_point", "built_area_total", "built_pixel_total"]],
        *pct_data,
    ],
    axis=1,
)

urban_quads = stats_df[stats_df.built_area_total > 5].quad_name.unique().tolist()
print(len(urban_quads))
urban_mask = stats_df.quad_name.isin(urban_quads)


def format_df(df):
    df = df.stack().reset_index()
    df.columns = ["time_point", "measure", "value"]


space_summaries = []
time_summaries = []
for ur_cat, ur_mask in [
    ("combined", pd.Series(True, index=stats_df.index)),
    ("urban", urban_mask),
    ("rural", ~urban_mask),
]:
    ss_df = (
        stats_df.loc[ur_mask]
        .drop(columns="quad_name")
        .groupby("time_point")
        .mean()
        .reset_index()
        .assign(urban_rural=ur_cat)
    )
    space_summaries.append(ss_df)

    ts_df = (
        stats_df.loc[ur_mask]
        .drop(columns="time_point")
        .groupby("quad_name")
        .mean()
        .reset_index()
        .assign(urban_rural=ur_cat)
    )
    time_summaries.append(ts_df)


space_summary_df = pd.concat(space_summaries).reset_index(drop=True)

time_summary_df = pd.concat(time_summaries).set_index("quad_name")
empty_mask = (
    stats_df.drop(columns="time_point")
    .groupby("quad_name")
    .built_area_total.apply(lambda s: s.isnull().all())
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
    time_summary_df["built_pixel_change"], time_summary_df["built_pixel_total"] + 0.1
)


# mask = time_summary_df.notnull().all(axis=1) & (time_summary_df.built_pix > 100)
# biggest_area = time_summary_df[mask].sort_values('built_area', ascending=False)['quad_name'].tolist()
# biggest_change = time_summary_df[mask].sort_values('built_area_change', ascending=False)['quad_name'].tolist()
# biggest_pix_change = time_summary_df[mask].sort_values('built_pix_change', ascending=False)['quad_name'].tolist()

# q1 = biggest_area[0]
# q2 = [bc for bc in biggest_change[:2] if bc not in [q1]][0]
# q3 = [bpc for bpc in biggest_pix_change[:3] if bpc not in [q1, q2]][0]
# line_plot_data = {
#     q1: ('orangered', stats_df[stats_df.quad_name == q1].iloc[1:].set_index('time_point')),
#     q2: ('darkgoldenrod', stats_df[stats_df.quad_name == q2].iloc[1:].set_index('time_point')),
#     q3: ('indigo', stats_df[stats_df.quad_name == q3].iloc[1:].set_index('time_point')),
# }

fig = plt.figure(figsize=(30, 20))
grid_spec = fig.add_gridspec(
    nrows=1,
    ncols=3,
    width_ratios=[2, 4, 2],
    wspace=0.2,
)

gs_area = grid_spec[0, 0].subgridspec(nrows=3, ncols=1)
gs_center = grid_spec[0, 1].subgridspec(nrows=2, ncols=1, height_ratios=[1.5, 4])
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
for measure, gs in zip(["area", "pixel"], [gs_area, gs_pix]):
    measure_map = [
        (
            f"built_{measure}_total",
            f"Average Built {measure.title()}",
            f"% of {measure.title()}",
            "GnBu",
            None,
        ),
        (
            f"built_{measure}_change",
            f"Cumulative Built {measure.title()} Change",
            f"% of {measure.title()}",
            "YlGn",
            None,
        ),
        (
            f"built_{measure}_volatility",
            f"Built {measure.title()} Volatility",
            f"Fraction of {measure.title()}",
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
        ax.set_title(title)


for col, measure in enumerate(["area", "pixel"]):
    row_map = {
        0: "total",
        2: "change",
        3: "growth",
        4: "reduction",
    }

    for row, metric in row_map.items():
        ax = fig.add_subplot(gs_line[row, col])
        for color, ur_cat in [
            ("tab:purple", "urban"),
            ("tab:green", "rural"),
            ("black", "combined"),
        ]:
            mask = space_summary_df.urban_rural == ur_cat
            space_summary_df.loc[mask].set_index("time_point")[
                f"built_{measure}_{metric}"
            ].plot(ax=ax, color=color, linewidth=2)
        if row == 0:
            ax.set_title(f"{measure.title()}", fontsize=20)
        if row != 4:
            ax.set_xlabel(None)
        if col == 0:
            ax.set_ylabel(f"{metric.capitalize()} (%)", fontsize=15)

    ax = fig.add_subplot(gs_line[1, col])
    stacked_df = (
        space_summary_df.loc[space_summary_df.urban_rural == "combined"]
        .iloc[1:]
        .set_index("time_point")
    )
    stacked_df.rename(
        columns={
            f"built_{measure}_growth": "growth",
            f"built_{measure}_reduction": "reduction",
        }
    )[["growth", "reduction"]].plot.area(
        ax=ax, color={"growth": "slateblue", "reduction": "indianred"}, legend=col == 0
    )
    stacked_df[f"built_{measure}_change"].plot(ax=ax, color="k", linewidth=2)
    ax.set_xlabel(None)
    if col == 0:
        ax.set_ylabel(f"Total change (%)", fontsize=15)
