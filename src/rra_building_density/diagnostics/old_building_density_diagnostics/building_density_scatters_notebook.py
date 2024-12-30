# ruff: noqa
# mypy: ignore-errors
from contextlib import ExitStack
from pathlib import Path

import contextily as cx
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.coords
import rasterio.mask
import rasterio.merge
import rasterio.plot
import rasterio.warp
import rasterra as rt
import scipy.signal
import seaborn as sns
import tqdm

sns.set_style("white")

import pop_data

# Parameters
plot_dir = Path("./diagnostics")
plot_data_dir = plot_dir / "data"

census_in_path = Path(
    "/mnt/team/rapidresponse/pub/population/data/03-processed-data/census/KEN/2019/population.csv"
)
geotagged_in_path = Path(
    "/mnt/team/rapidresponse/pub/population/data/03-processed-data/census/KEN/2019/old_kenya_geotagged.csv"
)

wgs84_projection = "EPSG:4326"
mollweide_equal_area_projection = "ESRI:54009"
destination_crs = mollweide_equal_area_projection
country_iso3 = "KEN"

admin0 = pop_data.load_admin_boundaries(country_iso3, level=0).to_crs(destination_crs)

building_density = (
    pop_data.load_building_density(country_iso3)
    .to_crs(destination_crs)
    .mask(admin0, all_touched=True)
)

population_density = (
    pop_data.load_population_density(country_iso3)
    .to_crs(destination_crs)
    .resample_to(building_density)
    .mask(admin0, all_touched=True)
)

population = (
    pop_data.load_population(country_iso3)
    .to_crs(destination_crs)
    .mask(admin0, all_touched=True)
)

ntl = (
    pop_data.load_ntl(country_iso3)
    .to_crs(destination_crs)
    .resample_to(building_density)
    .mask(admin0, all_touched=True)
)

# 5. Build the census dataset

census = pd.read_csv(census_in_path)
geotagged = pd.read_csv(geotagged_in_path)
m = census.merge(
    geotagged, how="left", left_on="path_to_top_parent", right_on="Hierarchy"
)
m = m.loc[
    m.sex == "total",
    ["location_name", "path_to_top_parent", "latitude_y", "longitude_y", "value"],
].rename(columns={"latitude_y": "latitude", "longitude_y": "longitude"})
census_out = (
    gpd.GeoDataFrame(
        m, geometry=gpd.points_from_xy(m.longitude, m.latitude), crs="EPSG:4326"
    )
    .drop(columns=["latitude", "longitude"])
    .to_crs(destination_crs)
)
census_out["level"] = census_out.path_to_top_parent.apply(
    lambda x: len(x.split(",")) - 1
)
census_out.to_file(census_out_path)

# 6. Make building density covariates

radii = [100, 500, 1000, 2500, 5000, 10000]

for radius in radii[:1]:
    print("Radius = ", radius)
    kernel = make_kernel(building_density.x_resolution, radius)

    out_image = scipy.signal.oaconvolve(
        building_density.to_numpy(), kernel, mode="same"
    )
    out_image -= np.nanmin(out_image)
    out_image[out_image < 0.005] = 0.0
    out_image = out_image.reshape(building_density.shape)
    out_image = rt.RasterArray(
        out_image,
        transform=building_density.transform,
        crs=building_density.crs,
        no_data_value=building_density.no_data_value,
    )
    out_image.to_file(plot_data_dir / f"building_density_{radius}m.tif")

# Load data

admin3 = gpd.read_file(admin3_path).to_crs(destination_crs)
cols = ["shapeName", "shapeID", "geometry"]
# admin_units = list(admin3[cols].itertuples(index=False, name=None))
admin3.head()

results = []
paths = {
    "building_density": building_density_out_path,
    "population_density": population_density_out_path,
    "population": population_out_path,
    "ntl": ntl_out_path,
    **{
        f"building_density_{r}m": plot_data_dir / f"building_density_{r}m.tif"
        for r in radii
    },
}

census_data = gpd.read_file(census_out_path)

with ExitStack() as stack:
    sources = {
        src_name: stack.enter_context(rasterio.open(path))
        for src_name, path in paths.items()
    }
    all_data = {}
    for src_name, src in sources.items():
        all_data[src_name] = src.read(1).flatten()
    bd_src = sources.pop("building_density")
    for name, shape_id, shape in tqdm.tqdm(admin_units):
        kwargs = {"shapes": [shape], "all_touched": True, "crop": True}
        bd, transform = rasterio.mask.mask(bd_src, **kwargs)
        empty = bd.copy()
        empty[~np.isnan(empty)] = -1
        shape_data = {}
        for src_name, src in sources.items():
            try:
                shape_data[src_name] = rasterio.mask.mask(src, **kwargs)[0]
            except ValueError:
                shape_data[src_name] = empty

        out = {
            **bd_src.meta,
            "building_density": bd,
            "transform": transform,
            "name": name,
            "shape_id": shape_id,
            "geometry": shape,
            "census": census_data[census_data.geometry.within(shape)].copy(),
            **{src_name: data for src_name, data in shape_data.items()},
        }
        results.append(out)

pops = []
pds = []
for i in range(len(results)):
    pop = results[i]["population"]
    shape = results[i]["geometry"]
    pops.append(np.nansum(pop))
    pds.append(np.nansum(pop) / shape.area * 1000**2)
    census = results[i]["census"]
    if np.nansum(pop) > 100000 and not census.empty:
        print(i)


fig, axes = plt.subplots(figsize=(12, 5), ncols=2)
pops = pd.Series(pops)
pops.hist(ax=axes[0], bins=50)
pds = pd.Series(pds)
pds.hist(ax=axes[1], bins=50)

FILL_ALPHA = 0.2
AX_LABEL_FONTSIZE = 14
TITLE_FONTSIZE = 20
HIST_BINS = 25
FIG_SIZE = (30, 15)
GRID_SPEC_MARGINS = {"top": 0.92, "bottom": 0.08}


def map_pixels(data, ax):
    s = pd.Series(data.flatten()).dropna()
    vmin = s.min()
    vmax = s.mean() + 3 * s.std()
    vmax = vmax if s.max() > 1 else s.max()
    extent = rasterio.plot.plotting_extent(data, r["transform"])
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

    for c, p in zip(col, patches, strict=False):
        plt.setp(p, "facecolor", cm(c))

    sns.despine(ax=ax, left=True, bottom=True)
    ax.set(xticklabels=[])
    # invert the order of x-axis values
    ax.set_xlim(ax.get_xlim()[::-1])

    return ax


def make_base_map_plot(shape, census, ax):
    admin_shape.boundary.plot(ax=ax, color="k")
    census.to_crs("EPSG:3857").plot(
        ax=ax, markersize=np.sqrt(census.value), color="r", alpha=0.2
    )
    ax.axis("off")
    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik, attribution=False)
    return ax


radii_ = radii[:-1]
r = results[70]

r["people_per_building"] = r["population"] / (r["building_density"] + 1e-8)
admin_shape = gpd.GeoSeries([r["geometry"]], crs=destination_crs).to_crs("EPSG:3857")
census = r["census"]

fig = plt.figure(figsize=FIG_SIZE)
grid_spec = fig.add_gridspec(
    nrows=5,
    ncols=2 + len(radii),
)
grid_spec.update(**GRID_SPEC_MARGINS)

for col, density in enumerate(
    [None, "building_density"] + [f"building_density_{r}m" for r in radii_]
):
    for row, variable in enumerate(
        [None, "ntl", "population_density", "population", "people_per_building"]
    ):
        if row == col == 0:
            ax = fig.add_subplot(grid_spec[row, col])
            make_base_map_plot(admin_shape, census, ax)
        elif row == 0 or col == 0:
            var = density if row == 0 else variable
            gs = grid_spec[row, col].subgridspec(1, 2, width_ratios=[1, 5])
            ax = fig.add_subplot(gs[0])
            data = r[var][0]
            make_hist(data, ax)
            if col == 0:
                ax.set_ylabel(
                    variable.replace("_", " ").title(), fontsize=AX_LABEL_FONTSIZE
                )
            ax = fig.add_subplot(gs[1])
            map_pixels(data, ax)
            if row == 0:
                ax.set_title(
                    density.replace("_", " ").title(), fontsize=AX_LABEL_FONTSIZE
                )

        else:
            df = pd.DataFrame(
                {density: r[density].flatten(), variable: r[variable].flatten()}
            ).dropna()
            non_zero = df[(df > 0).all(axis=1)]
            if len(non_zero) > 10000:
                non_zero = non_zero.sample(n=10000, random_state=12345)

            ax = fig.add_subplot(grid_spec[row, col])
            ax.scatter(non_zero[density], non_zero[variable], alpha=0.1)
            if variable in ["population", "people_per_building"]:
                ax.set_yscale("log")

fig.suptitle(
    f"{r['name']}\n{r['shape_id']}", x=0.5, fontsize=TITLE_FONTSIZE, ha="center"
)
plt.show()

pd.Series(r["building_density_5000m"].flatten()).dropna()

sq_km_scalar = building_density.resolution[0] ** 2 / 1000**2

unbuilt_pixels = df.loc[df.building_density == 0, "building_density"]
any_built_pixels = df.loc[df.building_density > 0, "building_density"]
unoccupied_pixels = df.loc[
    (df.building_density != 0) & (df.population_density == 0), "building_density"
]

total_area = len(df) * sq_km_scalar

non_empty_pixel_area = len(any_built_pixels) * sq_km_scalar
empty_pixel_area = len(unbuilt_pixels) * sq_km_scalar

built_area = any_built_pixels.sum() * sq_km_scalar
unbuilt_area = (len(unbuilt_pixels) + (1 - any_built_pixels).sum()) * sq_km_scalar

unoccupied_area = unoccupied_pixels.sum() * sq_km_scalar

print(f"Total area {total_area:.2f} sq. km.")

print(f"Non-empty pixel area {non_empty_pixel_area:.2f} sq. km.")
print(f"Empty pixel area {empty_pixel_area:.2f} sq. km.")
print(f"Pixel building coverage {non_empty_pixel_area / total_area * 100:.4f}%")

print(f"Built area {built_area:.2f} sq. km.")
print(f"Unbuilt area {unbuilt_area:.2f} sq. km.")
print(f"Actual building coverage {built_area / total_area * 100:.4f}%")

print(f"Built, unoccupied area {unoccupied_area:.2f} sq. km.")

plt.imshow(make_kernel(1, 20, kernel_type="gaussian"))

len(
    built.building_density[
        (built.building_density > 0) & (built.building_density < 0.1)
    ]
) * resolution**2 / 1000 / 1000

b_l, b_u = 0.5, 1
ppb_l, ppb_u = 0, 1000
occupied = df[
    (b_l < df.building_density)
    & (df.building_density < b_u)
    & (ppb_l < df.people_per_building)
    & (df.people_per_building < ppb_u)
]


occupied.loc[:, "population_density_group"] = "<5"
occupied.loc[occupied.population_density > 5, "population_density_group"] = ">5, <25"
occupied.loc[occupied.population_density > 25, "population_density_group"] = (
    ">25, < 100"
)
occupied.loc[occupied.population_density > 100, "population_density_group"] = ">100"

occupied["log_people_per_building"] = np.log(occupied.people_per_building)

occupied_s = occupied

N = 50_000
rs = np.random.RandomState(12345)
idx = rs.randint(0, len(occupied) + 1, N)
occupied_s = occupied.iloc[idx]

fig, ax = plt.subplots(figsize=(5, 3))
occupied.people_per_building.hist(bins=50, ax=ax)
ax.set_xlabel("People per Building")

fig, ax = plt.subplots(figsize=(5, 3))
occupied_s.people_per_building.hist(bins=50, ax=ax)
ax.set_xlabel("People per Building")

y_vars = [
    "people_per_building",
    "log_people_per_building",
]

for kernel_type in ["uniform", "gaussian"]:
    x_vars = ["building_density"] + [
        f"{kernel_type}_kernel_{radius}_m" for radius in [500, 1000, 2500, 5000, 10000]
    ]
    fig, axes = plt.subplots(
        nrows=len(y_vars),
        ncols=len(x_vars),
        figsize=(5 * len(x_vars), 4.5 * len(y_vars)),
    )

    for row, y_var in enumerate(y_vars):
        for col, x_var in enumerate(x_vars):
            ax = axes[row, col]
            sns.regplot(
                x=x_var,
                y=y_var,
                data=occupied_s,
                scatter_kws={"alpha": 0.05},
                line_kws={"color": "red"},
                ax=ax,
            )
            ax.set_xlabel(None)
            if row == 0:
                ax.set_title(x_var, fontsize=15)
            if col == 0:
                ax.set_ylabel(y_var, fontsize=15)
            else:
                ax.set_ylabel(None)
    fig.suptitle(kernel_type.title(), fontsize=20)
    fig.show()
    s
