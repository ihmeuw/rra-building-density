# ruff: noqa
import geopandas as gpd
import rasterra as rt
from matplotlib.figure import Figure
from rra_tools.plotting import strip_axes


def raster_and_gdf_plot(
    fig: Figure,
    raster_data: rt.RasterArray,
    gdf: gpd.GeoDataFrame,
    column: str,
) -> None:
    """Plot a raster and a GeoDataFrame on the same figure."""
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
