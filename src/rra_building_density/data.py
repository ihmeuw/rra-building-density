from pathlib import Path
from typing import Any

import geopandas as gpd
import rasterra as rt
import shapely
import yaml
from pydantic import BaseModel
from rra_tools.shell_tools import mkdir, touch

from rra_building_density import constants as bdc


class TileIndexInfo(BaseModel):
    tile_size: int
    tile_resolution: int
    block_size: int
    crs: str


class BuildingDensityData:
    def __init__(
        self,
        root: str | Path = bdc.MODEL_ROOT,
        credentials_root: str | Path = bdc.RRA_CREDENTIALS_ROOT,
        binaries_root: str | Path = bdc.RRA_BINARIES_ROOT,
    ) -> None:
        self._root = Path(root)
        self._credentials_root = Path(credentials_root)
        self._binaries_root = Path(binaries_root)
        self._create_model_root()

    def _create_model_root(self) -> None:
        mkdir(self.root, exist_ok=True)
        mkdir(self.footprints, exist_ok=True)
        mkdir(self.raw_tiles, exist_ok=True)
        mkdir(self.tiles, exist_ok=True)
        mkdir(self.summaries, exist_ok=True)
        mkdir(self.diagnostics, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def log_dir(self, step_name: str) -> Path:
        return self.logs / step_name

    @property
    def credentials_root(self) -> Path:
        return self._credentials_root

    @property
    def blob_credentials(self) -> tuple[str, str]:
        with (self._credentials_root / "microsoft-blob.yaml").open("r") as f:
            credentials = yaml.safe_load(f)
        return credentials["root"], credentials["key"]

    @property
    def binaries_root(self) -> Path:
        return self._binaries_root

    @property
    def azcopy_binary_path(self) -> Path:
        return self.binaries_root / "azcopy"

    @property
    def footprints(self) -> Path:
        return self.root / "footprints"

    def load_microsoft_building_footprints(
        self, *, skip_cache: bool = False
    ) -> gpd.GeoDataFrame:
        """This is the coverage of Microsoft's building footprint dataset.

        This is a metadata file that contains the count of building polygons in ~300km^2
        tiles across the globe. It is not a footprint dataset itself, but can be used
        to estimate the coverage of the footprint dataset as well as to estimate the
        building count in a covered area.

        Parameters
        ----------
        skip_cache
            If True, skip the cache and load the data from the source.

        Returns
        -------
        gpd.GeoDataFrame
            The building footprints coverage.
        """
        cache_path = self.footprints / "microsoft.parquet"
        if skip_cache or not cache_path.exists():
            url = "https://minedbuildings.blob.core.windows.net/global-buildings/buildings-coverage.geojson"
            target_crs = bdc.CRSES["wgs84"].to_pyproj()
            data = gpd.read_file(url).to_crs(target_crs)
            data.to_parquet(cache_path)
            return data
        else:
            return gpd.read_parquet(cache_path)

    @property
    def raw_tiles(self) -> Path:
        return self.root / "raw_tiles"

    def provider_root(self, built_version: bdc.BuiltVersion) -> Path:
        return self.raw_tiles / built_version.name

    def provider_index_cache_path(
        self, built_version: bdc.BuiltVersion, index_name: str
    ) -> Path:
        return self.provider_root(built_version) / f"{index_name}_index.parquet"

    def provider_tile_path(
        self,
        built_version: bdc.BuiltVersion,
        **kwargs: str,
    ) -> Path:
        root = self.provider_root(built_version)
        stem = built_version.raw_output_template.format(**kwargs)
        return root / stem

    def cache_provider_index(
        self,
        index: gpd.GeoDataFrame,
        built_version: bdc.BuiltVersion,
        index_name: str,
    ) -> None:
        cache_path = self.provider_index_cache_path(built_version, index_name)
        mkdir(cache_path.parent, exist_ok=True)
        touch(cache_path, clobber=True)
        index.to_parquet(cache_path)

    def load_provider_index(
        self, built_version: bdc.BuiltVersion, index_name: str
    ) -> gpd.GeoDataFrame:
        cache_path = self.provider_index_cache_path(built_version, index_name)
        return gpd.read_parquet(cache_path)

    def provider_tile_exists(
        self,
        built_version: bdc.BuiltVersion,
        **kwargs: str,
    ) -> bool:
        tile_path = self.provider_tile_path(built_version, **kwargs)
        return tile_path.exists()

    def load_provider_tile(
        self,
        built_version: bdc.BuiltVersion,
        bounds: shapely.Polygon | None = None,
        **kwargs: str,
    ) -> rt.RasterArray:
        tile_path = self.provider_tile_path(built_version, **kwargs)
        return rt.load_raster(tile_path, bounds=bounds)

    @property
    def tiles(self) -> Path:
        return self.root / "tiles"

    def save_tile_index(
        self,
        tile_index: gpd.GeoDataFrame,
        tile_index_info: TileIndexInfo,
    ) -> None:
        resolution = tile_index_info.tile_resolution
        root = self.tiles / f"{resolution}m"
        mkdir(root, exist_ok=True)

        tile_index_path = root / "tile_index.parquet"
        touch(tile_index_path, clobber=True)
        tile_index.to_parquet(tile_index_path)

        tile_index_info_path = root / "tile_index_info.yaml"
        touch(tile_index_info_path, clobber=True)
        with tile_index_info_path.open("w") as f:
            yaml.dump(tile_index_info.model_dump(), f)

    def _check_resolution(self, resolution: int | str) -> None:
        available_resolutions = [
            p.name for p in self.tiles.iterdir() if p.is_dir() and p.name.endswith("m")
        ]
        if f"{resolution}m" not in available_resolutions:
            msg = f"Resolution {resolution} not available. Available resolutions: {available_resolutions}"
            raise ValueError(msg)

    def load_tile_index(self, resolution: int | str) -> gpd.GeoDataFrame:
        self._check_resolution(resolution)
        path = self.tiles / f"{resolution}m" / "tile_index.parquet"
        return gpd.read_parquet(path)

    def load_tile_index_info(self, resolution: int | str) -> TileIndexInfo:
        self._check_resolution(resolution)
        path = self.tiles / f"{resolution}m" / "tile_index_info.yaml"
        with path.open() as f:
            info = yaml.safe_load(f)
        return TileIndexInfo(**info)

    def tile_path(
        self,
        resolution: int | str,
        provider: str,
        block_key: str,
        time_point: str,
        measure: str,
    ) -> Path:
        return (
            self.tiles
            / f"{resolution}m"
            / provider
            / time_point
            / f"{block_key}_{measure}.tif"
        )

    def save_tile(
        self,
        tile: rt.RasterArray,
        resolution: int | str,
        provider: str,
        block_key: str,
        time_point: str,
        measure: str,
    ) -> None:
        self._check_resolution(resolution)
        tile_path = self.tile_path(resolution, provider, block_key, time_point, measure)
        mkdir(tile_path.parent, exist_ok=True, parents=True)
        touch(tile_path, clobber=True)
        save_raster(tile, tile_path)

    def link_tile(
        self,
        resolution: int | str,
        provider: str,
        block_key: str,
        time_point: str,
        measure: str,
        source_path: Path,
    ) -> None:
        self._check_resolution(resolution)
        dest = self.tile_path(resolution, provider, block_key, time_point, measure)
        mkdir(dest.parent, exist_ok=True, parents=True)
        if dest.exists():
            dest.unlink()
        dest.symlink_to(source_path)

    def load_tile(
        self,
        resolution: int | str,
        provider: str,
        block_key: str,
        time_point: str,
        measure: str,
        bounds: shapely.Polygon | None = None,
    ) -> rt.RasterArray:
        self._check_resolution(resolution)
        tile_path = self.tile_path(resolution, provider, block_key, time_point, measure)
        return rt.load_raster(tile_path, bounds=bounds)

    @property
    def summaries(self) -> Path:
        return self.root / "summaries"

    @property
    def diagnostics(self) -> Path:
        return self.root / "diagnostics"


def save_raster(
    raster: rt.RasterArray,
    output_path: str | Path,
    num_cores: int = 1,
    **kwargs: Any,
) -> None:
    """Save a raster to a file with standard parameters."""
    save_params = {
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "ZSTD",
        "predictor": 2,  # horizontal differencing
        "num_threads": num_cores,
        "bigtiff": "yes",
        **kwargs,
    }
    raster.to_file(output_path, **save_params)


def save_raster_to_cog(
    raster: rt.RasterArray,
    output_path: str | Path,
    num_cores: int = 1,
    resampling: str = "nearest",
) -> None:
    """Save a raster to a COG file."""
    cog_save_params = {
        "driver": "COG",
        "overview_resampling": resampling,
    }
    save_raster(raster, output_path, num_cores, **cog_save_params)
