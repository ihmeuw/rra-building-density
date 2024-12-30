# ruff: noqa
# mypy: ignore-errors
from rra_population_pipelines.shared.plot_utils import strip_axes


def raster_and_gdf_plot(fig, raster_data, gdf, column):
    axes = fig.subplots(1, 2)

    ax = axes[0]
    raster_data.plot(ax=ax)
    strip_axes(ax)

    ax = axes[1]
    gdf.plot(ax=ax, column=f"admin_{column}")
    ax.set_xlim(raster_data.x_min, raster_data.x_max)
    ax.set_ylim(raster_data.y_min, raster_data.y_max)

    strip_axes(ax)
    fig.suptitle(column)


def make_tile_diagnostics(tile_key):
    features = PPS_DATA.load_features(tile_key, "2020q1")
    if "night_time_lights_alt" in features:
        del features["night_time_lights_alt"]

    gdf = PPS_DATA.load_admin_test_train(tile_key)

    bd_raster = features.pop("building_density")
    bd_raster * bd_raster.x_resolution**2
    features.pop("population")
    pd_raster = features.pop("population_density")

    fig = plt.figure(figsize=(30, 15), layout="constrained")
    sub_figs = fig.subfigures(
        nrows=1,
        ncols=2,
        wspace=0.05,
    )

    n_features = len(features)
    n_rows = min(4, n_features)
    n_cols = n_features // 4 + 1

    feature_figs = sub_figs[1].subfigures(n_rows, n_cols)

    bd_increments = sorted(
        [
            int(k.split("_")[-1].split("m")[0])
            for k in features
            if "building_density" in k
        ]
    )
    feature_order = [f"building_density_average_{inc}m" for inc in bd_increments]
    feature_order += sorted([f for f in features if "building_density" not in f])

    for i, f in enumerate(feature_order):
        row, col = i % 4, i // 4
        sf = feature_figs[row, col]
        raster_and_gdf_plot(sf, features[f], gdf, f)

    main_fig = sub_figs[0].subfigures(2, 1)

    top_fig = main_fig[0].subfigures(1, 2)

    raster_and_gdf_plot(top_fig[0], bd_raster, gdf, "building_density")
    raster_and_gdf_plot(top_fig[1], pd_raster, gdf, "population_density")

    axes = main_fig[1].subplots(1, 2)
    ax = axes[0]
    occ_rate = 100 * pd_raster / bd_raster
    occ_rate._ndarray[~np.isfinite(occ_rate._ndarray)] = 0
    occ_rate.plot(ax=ax, vmin=0.01, vmax=5, under_color="grey")
    strip_axes(ax)

    ax = axes[1]
    z = gdf[["geometry", "admin_occupancy_rate"]].copy()
    z["admin_occupancy_rate"] *= 100
    z.plot(ax=ax, column="admin_occupancy_rate", vmax=5, vmin=0)
    z[z.admin_occupancy_rate < 0].plot(ax=ax, color="red")
    ax.set_xlim(pd_raster.x_min, pd_raster.x_max)
    ax.set_ylim(pd_raster.y_min, pd_raster.y_max)
    strip_axes(ax)
    main_fig[1].suptitle("Occupancy Rate")


# plot_building_density_histogram("T-0366X-0292Y")
# plot_area_distribution(gdf)

### COME CANNIBALIZE HIST EQ PLOT

# for c in ['admin_night_time_lights_alt']:
#     fig, axes = plt.subplots(figsize=(15, 4), ncols=3)
#     for ax, (u_var, urban_mask) in zip(axes, urban_masks):
#         X_train, X_test, y_train, y_test = train_test_split(
#             X[urban_mask], y[urban_mask],
#             test_size=0.2,
#             random_state=0,
#         )
#         model.fit(X_train[[c]], y_train)

#         bins = 256
#         val, xx, yy = np.histogram2d(X_train[c], y_train, bins=bins)
#         val = val.T[::-1] / val.max()
#         mask = val == 0
#         val_eq = exposure.equalize_hist(1000*val, nbins=100000, mask=~mask)
#         ax.imshow(
#             np.ma.masked_array(val_eq, mask),
#             cmap=plt.cm.cividis,
#             extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#             aspect='auto'
#         )

#         xs = np.linspace(xx.min(), xx.max(), 1000).reshape(1000, 1)
#         ys = model.predict(xs)
#         ax.plot(xs, ys, color='r')

#         if 'night_time_lights' in c:
#             ax.set_xlim(0, 150)
#         # else:
#         #     #ax.set_xlim(0, 1)

#         ax.set_ylabel('log people per structure', fontsize=16)
#         ax.set_title(f"{u_var.title()}\nCoef: {model.coef_[0]:.4f}, Icept: {model.intercept_:.2f}\n$R^2$: {model.score(X_train[[c]], y_train):.4f}")

#     fig.suptitle(c)
#     fig.tight_layout()
