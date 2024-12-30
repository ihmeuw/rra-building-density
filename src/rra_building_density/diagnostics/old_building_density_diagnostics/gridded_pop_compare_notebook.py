# ruff: noqa
# mypy: ignore-errors
import itertools
import time
import shutil
import tempfile
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import rasterio
import rasterio.plot
import rasterio.mask
import seaborn as sns
import tqdm
import wget

out_dir = Path("diagnostics/grid_pop_compare")
if out_dir.exists():
    shutil.rmtree(out_dir)
out_data_dir = out_dir / "data"

out_data_dir.mkdir(parents=True, exist_ok=True)

ihme_in_path = Path(
    "/mnt/team/rapidresponse/pub/population/modeling/prototype/20230605_james_repro/distributed_500m/raked_raster_population.tif"
)

bespoke_in_path = Path(
    "/mnt/team/rapidresponse/pub/population/data/01-raw-data/other-gridded-pop-projects/worldpop-bespoke/KEN/KEN_population_v2_0_gridded.tif"
)
constrained_in_path = Path(
    "/mnt/share/scratch/users/victorvt/spatial/KEN/ken_ppp_2020_constrained.tif"
)
unconstrained_in_path = Path(
    "/mnt/share/scratch/users/victorvt/spatial/KEN/ken_ppp_2020.tif"
)
in_paths = [unconstrained_in_path, constrained_in_path, bespoke_in_path]

ihme_out_path = out_data_dir / "ihme.tif"
ihme_lowres_path = out_data_dir / "ihme_lowres.tif"
ones_out_path = out_data_dir / "ones.tif"

bespoke_out_path = out_data_dir / "bespoke.tif"
constrained_out_path = out_data_dir / "constrained.tif"
unconstrained_out_path = out_data_dir / "unconstrained.tif"
out_paths = [unconstrained_out_path, constrained_out_path, bespoke_out_path]
hires_outpaths = [
    out_path.parent / f"{out_path.stem}_hires.tif" for out_path in out_paths
]

admin0_path = "/mnt/team/rapidresponse/pub/population/data/01-raw-data/shapefiles/KEN/Unknown/gadm41_KEN_shp/gadm41_KEN_0.shp"

projection = "ESRI:54009"
plot_projection = "EPSG:4326"


def plot(path):
    with rasterio.open(path) as src:
        data = src.read(1, masked=True)
        s = pd.Series(data.flatten())
        vmin = s.min()
        vmax = s.mean() + 2 * s.std()
        print(vmin, vmax)
        rasterio.plot.show(src, vmin=vmin, vmax=vmax, cmap="viridis")
    plt.show()


def mask_admin(in_path: Path, admin_path: Path, out_path: Path):
    with rasterio.open(in_path) as src:
        admin = gpd.read_file(admin_path).to_crs(src.crs)

        ones_meta = src.meta.copy()
        ones_meta["dtype"] = np.uint8
        ones_meta["nodata"] = 0

        # Make a dummy array of ones and write out
        ones = np.ones(
            shape=(1, ones_meta["width"], ones_meta["height"]), dtype=np.uint8
        )
        with rasterio.open(ones_out_path, "w", **ones_meta) as dest:
            dest.write(ones)

        # Mask ones array so it is 0 outside the admin boundary and 1 inside
        kwargs = {
            "all_touched": True,
            "nodata": 0,
        }
        with rasterio.open(ones_out_path) as ones_src:
            ones, _ = rasterio.mask.mask(ones_src, admin.geometry.tolist(), **kwargs)

        # Turn this into an actual numpy boolean mask in the shape of the admin unit
        admin_mask = ones[0].astype(bool)

        data = src.read(1)
        nodata_mask = src.read_masks(1).astype(bool)

        data[admin_mask & ~nodata_mask] = 0.0
        data[~admin_mask] = np.nan

        out_meta = src.meta.copy()

    out_meta.update(
        {
            "nodata": np.nan,
        }
    )

    with rasterio.open(out_path, "w", **out_meta) as dest:
        dest.write(data.reshape((1, *data.shape)))


def project_raster(in_path: Path, dst_crs: str, out_path: Path):
    with rasterio.open(in_path) as src:
        src_transform = src.transform

        # calculate the transform matrix for the output
        dst_transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,  # unpacks outer boundaries (left, bottom, right, top)
        )

        # set properties for output
        dst_kwargs = src.meta.copy()
        dst_kwargs.update(
            {
                "crs": dst_crs,
                "transform": dst_transform,
                "width": width,
                "height": height,
            }
        )

        with rasterio.open(out_path, "w", **dst_kwargs) as dst:
            for i in range(1, src.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=rasterio.warp.Resampling.nearest,
                )


def align_rasters(in_path, match, out_path):
    """Reproject a file to match the shape and projection of existing raster.

    Parameters
    ----------
    in_path : (string) path to input file to reproject
    match : (string) path to raster with desired shape and projection
    out_path : (string) path to output file tif
    """
    # open input
    with rasterio.open(in_path) as src:
        src_transform = src.transform

        # open input to match
        with rasterio.open(match) as match:
            dst_crs = match.crs

            # calculate the output transform matrix
            dst_transform, dst_width, dst_height = (
                rasterio.warp.calculate_default_transform(
                    src.crs,  # input CRS
                    dst_crs,  # output CRS
                    match.width,  # input width
                    match.height,  # input height
                    *match.bounds,  # unpacks input outer boundaries (left, bottom, right, top)
                )
            )

        # set properties for output
        dst_kwargs = src.meta.copy()
        dst_kwargs.update(
            {
                "crs": dst_crs,
                "transform": dst_transform,
                "width": dst_width,
                "height": dst_height,
            }
        )
        print(
            "Coregistered to shape:", dst_height, dst_width, "\n Affine", dst_transform
        )
        # open output
        with rasterio.open(out_path, "w", **dst_kwargs) as dst:
            # iterate through bands and write using reproject function
            for i in range(1, src.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=rasterio.warp.Resampling.nearest,
                )


def to_density(in_path, out_path):
    with rasterio.open(in_path) as src:
        if str(src.crs) != "ESRI:54009":
            raise ValueError(
                "Input source must be in Molleweide Equal Area Projection (ESRI:54009) to translate to density."
            )
        assert src.res[0] == src.res[1]
        out_data = src.read() / src.res[0] ** 2
        out_meta = src.meta.copy()

    with rasterio.open(out_path, "w", **out_meta) as dest:
        dest.write(out_data)


def to_count(in_path, out_path):
    with rasterio.open(in_path) as src:
        if str(src.crs) != "ESRI:54009":
            raise ValueError(
                "Input source must be in Molleweide Equal Area Projection (ESRI:54009) to translate to density."
            )
        assert src.res[0] == src.res[1]
        out_data = src.read() * src.res[0] ** 2
        out_meta = src.meta.copy()

    with rasterio.open(out_path, "w", **out_meta) as dest:
        dest.write(out_data)


project_raster(ihme_in_path, projection, ihme_out_path)
mask_admin(ihme_out_path, admin0_path, ihme_out_path)

for in_path, out_path, out_path_hires in tqdm.tqdm(
    list(zip(in_paths, out_paths, hires_outpaths))
):
    mask_admin(in_path, admin0_path, out_path)
    project_raster(out_path, projection, out_path)
    to_density(out_path, out_path_hires)
    align_rasters(out_path_hires, ihme_out_path, out_path_hires)
    to_count(out_path_hires, out_path_hires)
    project_raster(out_path_hires, plot_projection, out_path_hires)
    project_raster(out_path, plot_projection, out_path)

project_raster(ihme_out_path, plot_projection, ihme_out_path)


def map_pixels(data, transform, ax):
    s = pd.Series(data.flatten()).dropna()
    vmin = s.min()
    vmax = s.mean() + 3 * s.std()
    vmax = vmax if s.max() > 1 else s.max()
    extent = rasterio.plot.plotting_extent(data, transform)
    im = ax.imshow(data, cmap="viridis", vmin=vmin, vmax=vmax, extent=extent)

    # fig.colorbar(im, cax=cax, orientation='vertical', drawedges=False)
    # cax.yaxis.set_ticks_position('left')
    ax.axis("off")
    return ax


def make_hist(data, ax):
    s = pd.Series(data.flatten()).dropna()
    vmin = s.min()
    vmax = s.mean() + 3 * s.std()
    vmax = vmax if s.max() > 1 else s.max()
    s = s[(vmin < s) & (s <= vmax)]

    cm = mpl.colormaps["viridis"]
    n, bins, patches = ax.hist(s, orientation="horizontal")
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    col = bin_centers - min(bin_centers)
    col /= max(col)

    for c, p in zip(col, patches):
        plt.setp(p, "facecolor", cm(c))

    sns.despine(ax=ax, left=True, bottom=True)
    ax.set(xticklabels=[])
    # invert the order of x-axis values
    ax.set_xlim(ax.get_xlim()[::-1])

    return ax


# Find coordinates of bounding box of interest
with rasterio.open(ihme_out_path) as src:
    ulx, uly = src.index(40.21802903146841, 0.21972581200499225)
    brx, bry = src.index(40.42264938563206, -0.028495994491360935)
    ulx -= 50

data_paths = [
    ("IHME", ihme_out_path),
    *zip(["Unconstrained", "Constrained", "Bespoke"], hires_outpaths),
]
data = {}
for key, p in data_paths:
    with rasterio.open(p) as src:
        r = src.read(1)
        data[key] = r[ulx:brx, uly:bry]

FILL_ALPHA = 0.2
AX_LABEL_FONTSIZE = 14
TITLE_FONTSIZE = 20
HIST_BINS = 25
FIG_SIZE = (18, 6.5)
GRID_SPEC_MARGINS = {"top": 0.92, "bottom": 0.08}

fig = plt.figure(figsize=FIG_SIZE)
grid_spec = fig.add_gridspec(
    nrows=2,
    ncols=4,
    height_ratios=[3, 1],
    # width_ratios=[3, 1, 1, 1, 1],
)
grid_spec.update(**GRID_SPEC_MARGINS)


def map_pixels(data, ax):
    s = pd.Series(data.flatten()).dropna()
    vmin = s.min()
    vmax = s.mean() + 3 * s.std()
    vmax = vmax if s.max() > 1 else s.max()
    im = ax.imshow(
        data,
        cmap="viridis",
        vmin=0,
        vmax=15,
    )
    # fig.colorbar(im, cax=cax, orientation='vertical', drawedges=False)
    # cax.yaxis.set_ticks_position('left')
    sns.despine(ax=ax, left=True, bottom=True)
    ax.set_yticks([])
    ax.set_xticks([])
    return ax


for col, (label, arr) in enumerate(data.items()):
    map_ax = fig.add_subplot(grid_spec[0, col])
    map_pixels(arr, map_ax)
    map_ax.set_title(label, fontsize=18)

    raw_data = pd.Series(arr.flatten())

    q = 0.95

    zeros = len(raw_data[raw_data == 0])
    total_pop = np.sum(raw_data)
    filtered_data = raw_data[raw_data > 0]
    upper = filtered_data.quantile(q)
    filtered_data = filtered_data[(filtered_data < upper)]

    hist_ax = fig.add_subplot(grid_spec[1, col])
    hist_ax.hist(filtered_data, bins=50)
    if col == 0:
        hist_ax.set_ylabel("Pixel Count", fontsize=15)
    hist_ax.set_title(f"Total_population: {total_pop:.0f}")
    sns.despine(ax=hist_ax, left=True, bottom=True)


# plots = list(itertools.product(enumerate(data), enumerate(data)))
# for (row, row_label), (col, col_label) in plots:
#     ax = fig.add_subplot(grid_spec[row+1, col])
#     raw_row_data = pd.Series(data[row_label].flatten())
#     raw_col_data = pd.Series(data[col_label].flatten())

#     q = 0.95

#     zeros = len(raw_row_data[raw_row_data == 0])
#     row_data = raw_row_data[raw_row_data > 0]
#     upper = row_data.quantile(q)
#     row_data = row_data[(row_data < upper)]

#     col_data = raw_col_data[raw_col_data > 0]
#     upper = col_data.quantile(q)
#     col_data = col_data[(col_data < upper)]

#     if row_label == col_label:
#         ax.hist(col_data, bins=50)
#     else:
#         rs = np.random.RandomState(1234)
#         n_samples = 100000
#         draw = rs.randint(0, len(raw_col_data), n_samples)
#         x = raw_col_data.iloc[draw]
#         y = raw_row_data.iloc[draw]
#         non_zero = (x != 0) & (y != 0)
#         zero = ~non_zero
#         ax.scatter(x[zero], y[zero], color='firebrick', alpha=0.5)
#         ax.scatter(x[non_zero], y[non_zero], color='dodgerblue', alpha=0.3)
#         ax.set_xlim((0, col_data.max()))
#         ax.set_ylim((0, row_data.max()))

#     if col == 0:
#         print(row_label, np.sum(raw_row_data))
#         ax.set_ylabel(row_label.title(), fontsize=18)

fig.supxlabel("Population", fontsize=15)

plt.show()
