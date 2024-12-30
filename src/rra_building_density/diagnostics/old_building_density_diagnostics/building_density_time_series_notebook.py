# ruff: noqa
# mypy: ignore-errors
import colorcet
import datashader
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterra as rt
import seaborn as sns
from rasterio.plot import plotting_extent

old = (
    rt.load_raster(
        "/mnt/team/rapidresponse/pub/population/data/03-processed-data/building-layers/KEN/kenya_global_quarterly_2017q3_mosaic_predictions_aligned_masked.tif"
    )
    .unset_no_data_value()
    .set_no_data_value(np.nan)
)
new = (
    rt.load_raster(
        "/mnt/team/rapidresponse/pub/population/data/03-processed-data/building-layers/KEN/kenya_global_quarterly_2023q2_mosaic_predictions_aligned_masked.tif"
    )
    .unset_no_data_value()
    .set_no_data_value(np.nan)
)
admin0 = gpd.read_file(
    "/mnt/team/rapidresponse/pub/population/data/01-raw-data/shapefiles/KEN/Unknown/gadm41_KEN_shp/gadm41_KEN_0.shp"
).to_crs(old.crs)
old = old.mask(admin0)
new = new.mask(admin0)

old.to_numpy()

change = new - old
rel_change = change / (old + 1e-6)
frac_built = change / (1 - old)
frac_destroyed = change / old
frac_change = np.wher

e(change > 0, frac_built, frac_destroyed)


def flatten(arr, keep_mask):
    arr = arr.to_numpy()

    return arr[np.abs(change.to_numpy()) > 0].flatten()


old_flat = flatten(old)
new_flat = flatten(new)
change_flat = flatten(change)
rel_change_flat = flatten(rel_change)
frac_change_flat = flatten(frac_change)


def make_hist(arr, xlabel, ax):
    ax.hist(
        arr[arr > 0],
        bins=100,
        range=(arr.min(), arr.max()),
        color="dodgerblue",
        align="right",
    )
    ax.hist(
        arr[arr < 0],
        bins=100,
        range=(arr.min(), arr.max()),
        color="firebrick",
        align="left",
    )
    ax.set_xlabel(xlabel, fontsize=13)
    sns.despine(ax=ax, left=True, bottom=True)
    return ax


fig, axes = plt.subplots(ncols=3, figsize=(15, 5))

ax = axes[0]
make_hist(change_flat, "Absolute Change", ax)
ax.set_ylabel("Pixel Count", fontsize=13)
make_hist(rel_change_flat[np.abs(rel_change_flat) < 10], "Relative Change", axes[1])
make_hist(frac_change_flat, "Realized Change", axes[2])
axes[2].text(
    -1,
    6e6,
    "Proportion of built\nstructure removed\n\n     built_end\n    ---------------\n     built_start",
    color="firebrick",
)
axes[2].text(
    0.1,
    6e6,
    "Proportion of\nunbuilt area filled\n\n(built_end - built_start)\n-------------------------------\n     (1 - built_start)",
    color="dodgerblue",
)

cvs = datashader.Canvas(plot_width=850, plot_height=500)
agg = cvs.points(
    pd.DataFrame({"density": old_flat, "change": frac_change_flat}), "density", "change"
)
img = datashader.tf.shade(agg, cmap=colorcet.fire, how="log")
img

cvs = datashader.Canvas(plot_width=850, plot_height=500)
mask = old_flat < 0.05
agg = cvs.points(
    pd.DataFrame({"density": old_flat[mask], "change": change_flat[mask]}),
    "density",
    "change",
)
img = datashader.tf.shade(agg, cmap=colorcet.fire, how="log")
img

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(old_flat, frac_change_flat, alpha=0.1, s=0.1)
ax.set_xlabel("Building Coverage at Start")
ax.set_ylabel("Realized Change")

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(old_flat, frac_change_flat, alpha=0.1, s=0.1)
ax.set_xlabel("Building Coverage at Start")
ax.set_ylabel("Realized Change out of Potential")
ax.set_ylim(-1.1, -0.9)

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(old_flat, change_flat, alpha=0.1, s=0.1)
ax.set_ylabel("Absolute Change in Building Coverage")
ax.set_xlabel("Building Coverage at start")
ax.set_xlim([0, 0.05])
ax.set_ylim([-0.05, 1])


def implot(data, ax, vmin=-0.25, vmax=0.25, **kwargs):
    im = ax.imshow(
        data,
        vmin=vmin,
        vmax=vmax,
        cmap="seismic_r",
        **kwargs,
    )
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel("Absolute Change in Density")
    return im


extent = plotting_extent(old_r._data, old_r.transform)
fig, ax = plt.subplots(figsize=(8, 8))

change_r = rt.RasterArray(
    change,
    crs=old_r.crs,
    transform=old_r.transform,
    nodata=np.nan,
).mask([admin0.unary_union])
im = implot(change_r._data, ax, extent=extent)
admin0.boundary.plot(ax=ax, color="k", linewidth=0.3)
sns.despine(ax=ax, left=True, bottom=True)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Change in Building Density 2017q3 to 2023q2")

fig, axes = plt.subplots(figsize=(20, 10), ncols=2)
xmin, xmax = 12000, 15000
ymin, ymax = 15000, 20000
min_structure = 0.025
max_threshold = 0.25
im_old = axes[0].imshow(
    np.where(old > min_structure, old, 0)[ymin:ymax, xmin:xmax],
    vmin=0,
    vmax=max_threshold,
)
im_new = axes[1].imshow(
    np.where(new > min_structure, new, 0)[ymin:ymax, xmin:xmax],
    vmin=0,
    vmax=max_threshold,
)
# plt.colorbar(im_new)
plt.show()
