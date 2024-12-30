# mypy: ignore-errors
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns

from rra_population_pipelines.pipelines.models.people_per_structure.data import (
    DEFAULT_MODEL_ROOT,
    PeoplePerStructureData,
)
from rra_population_pipelines.shared.cli_tools import options as clio
from rra_population_pipelines.shared.data import (
    RRA_DATA_ROOT,
    RRAPopulationData,
)


def get_quad_array(
    pop_data: RRAPopulationData,
    quarter: str,
    quad_name: str,
) -> npt.NDArray[np.float64]:
    if pop_data.building_density_tile_exists(quarter, quad_name):  # type: ignore[attr-defined]
        return pop_data.load_building_density_tile(quarter, quad_name).to_numpy()  # type: ignore[no-any-return, attr-defined]
    else:
        return np.nan * np.ones((512, 512))


def compute_quad_stats(quad_spec: tuple[str, str | Path]) -> pd.DataFrame:
    quad_name, pop_data_root = quad_spec
    pop_data = RRAPopulationData(pop_data_root)

    idx_cols = ["quad_name", "quarter"]
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
    last = get_quad_array(pop_data, quad_name, "no_time")
    for quarter in clio.VALID_TIME_POINTS:
        current = get_quad_array(pop_data, quarter, quad_name)

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
            built_area = np.nansum(current)  # type: ignore[assignment]
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
                quad_name,
                quarter,
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


block_name = "B-012X-025Y"
pop_data_root = RRA_DATA_ROOT
model_root = DEFAULT_MODEL_ROOT
num_cores = 20
progress_bar = True

pps_data = PeoplePerStructureData(model_root)
tile_index = pps_data.load_full_tile_index()
quad_names = tile_index[tile_index.block_name == block_name].quad_name.tolist()

quad_spec = ("L15-0603E-1278N", pop_data_root)

quad_name, pop_data_root = quad_spec
pop_data = RRAPopulationData(pop_data_root)

data = np.stack(
    [get_quad_array(pop_data, tp, quad_name) for tp in clio.VALID_TIME_POINTS]
)

temporal_mean = data.mean(axis=0)
delta = data - temporal_mean

positive = delta >= 0
pos_delta = np.where(positive, delta, 0)
neg_delta = np.where(~positive, delta, 0)

pos_std_dev = np.sqrt(np.sum(pos_delta**2, axis=0) / np.sum(positive, axis=0))
neg_std_dev = np.sqrt(np.sum(neg_delta**2, axis=0) / np.sum(~positive, axis=0))
std_dev = np.sqrt(np.mean(delta**2, axis=0))

df = pd.DataFrame(
    {
        "Mean Built Area": temporal_mean.flatten(),
        "Positive Std. Dev.": pos_std_dev.flatten(),
        "Negative Std. Dev.": neg_std_dev.flatten(),
        "Std. Dev.": std_dev.flatten(),
    }
)


bins = 30
cut_df = pd.DataFrame(
    {
        "Mean Built Area": pd.cut(  # type: ignore[call-overload]
            df["Mean Built Area"], bins=np.linspace(0, 1, bins + 1)
        ),
        "Positive Std. Dev.": pd.cut(  # type: ignore[call-overload]
            df["Positive Std. Dev."], bins=np.linspace(0, 0.5, bins + 1)
        ),
        "Negative Std. Dev.": pd.cut(  # type: ignore[call-overload]
            df["Negative Std. Dev."], bins=np.linspace(0, 0.5, bins + 1)
        ),
        "Std. Dev.": pd.cut(df["Std. Dev."], bins=np.linspace(0, 0.5, bins + 1)),  # type: ignore[call-overload]
        "dummy": 1,
    }
)


plot_df = (
    cut_df.groupby(["Positive Std. Dev.", "Mean Built Area"])["dummy"]
    .count()
    .unstack()
    .sort_index(ascending=False)
)
plot_df.index = plot_df.index.rename(lambda s: s.right)
sns.heatmap(plot_df.divide(plot_df.sum(axis=1), axis=0), vmax=0.3)

fig, axes = plt.subplots(figsize=(20, 6), ncols=3)

for i, col in enumerate(df.filter(like="Std.")):
    plot_df = (
        cut_df.groupby([col, "Mean Built Area"])["dummy"]
        .count()
        .unstack()
        .sort_index(ascending=False)
    )
    sns.heatmap(
        plot_df.divide(plot_df.sum(axis=0), axis=1),
        vmax=0.25,
        ax=axes[i],
        xticklabels=[s.left for s in plot_df.columns],  # type: ignore[attr-defined]
        yticklabels=[s.left for s in plot_df.index],
    )

fig.tight_layout()


compute_quad_stats(quad_spec)
