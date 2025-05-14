import abc
import itertools
import warnings
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Literal

import pyproj
from pydantic import BaseModel, model_validator

RRA_ROOT = Path("/mnt/team/rapidresponse/")
RRA_CREDENTIALS_ROOT = RRA_ROOT / "priv" / "shared" / "credentials"
RRA_BINARIES_ROOT = RRA_ROOT / "priv" / "shared" / "bin"
MODEL_ROOT = RRA_ROOT / "pub" / "building-density"


class RESOLUTIONS(StrEnum):
    r40 = "40"
    r100 = "100"

    @classmethod
    def to_list(cls) -> list[str]:
        return [r.value for r in cls]


class BuiltVersion(BaseModel, abc.ABC):
    provider: str
    version: str
    time_points: list[str]
    input_template: str
    raw_output_template: str

    @abc.abstractmethod
    def process_resources(self, resolution: str) -> tuple[str, str]:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return f"{self.provider}_{self.version}"


class MicrosoftVersion(BuiltVersion):
    provider: Literal["microsoft"] = "microsoft"
    version: Literal["v2", "v3", "v4", "v5", "v6", "v7", "water_mask"]
    bands: dict[str, int]

    def process_resources(self, resolution: str) -> tuple[str, str]:
        return {
            RESOLUTIONS.r40: ("5G", "30m"),
            RESOLUTIONS.r100: ("8G", "60m"),
        }[RESOLUTIONS(resolution)]


MICROSOFT_VERSIONS = {
    "2": MicrosoftVersion(
        version="v2",
        time_points=[
            f"{y}q{q}" for q, y in itertools.product(range(1, 5), range(2018, 2024))
        ][:-1],
        input_template="predictions/{time_point}/predictions/postprocess_v2/*",
        raw_output_template="{time_point}/{time_point}_{tile_key}.tif",
        bands={
            "density": 1,
        },
    ),
    "3": MicrosoftVersion(
        version="v3",
        time_points=["2023q3"],
        input_template="predictions/{time_point}/predictions/ensemble_v3_pp/*",
        raw_output_template="{time_point}/{time_point}_{tile_key}.tif",
        bands={
            "density": 1,
        },
    ),
    "4": MicrosoftVersion(
        version="v4",
        time_points=["2023q4"],
        input_template="predictions/{time_point}/predictions/v45_ensemble/*",
        raw_output_template="{time_point}/{tile_key}.tif",
        bands={
            "density": 1,
        },
    ),
    "5": MicrosoftVersion(
        version="v5",
        time_points=[
            f"{y}q{q}" for y, q in itertools.product(range(2020, 2024), range(1, 5))
        ][1:],
        input_template="predictions/{time_point}/az_8_ensemble/*",
        raw_output_template="{time_point}/{tile_key}.tif",
        bands={
            "density": 1,
        },
    ),
    "6": MicrosoftVersion(
        version="v6",
        time_points=[
            f"{y}q{q}" for y, q in itertools.product(range(2020, 2024), range(1, 5))
        ][1:],
        input_template="predictions/{time_point}/az_8_ensemble_v6/*",
        raw_output_template="{time_point}/{tile_key}.tif",
        bands={
            "density": 1,
        },
    ),
    "7": MicrosoftVersion(
        version="v7",
        time_points=[
            f"{y}q{q}" for y, q in itertools.product(range(2020, 2024), range(1, 5))
        ][1:],
        input_template="predictions/{time_point}/9-37-best_practices_p3_ensemble/*",
        raw_output_template="{time_point}/{tile_key}.tif",
        bands={
            "density": 1,
            "height": 2,
        },
    ),
    "water_mask": MicrosoftVersion(
        version="water_mask",
        time_points=[""],
        input_template="permanent_or_seasonal_water/*",
        raw_output_template="{tile_key}.tif",
        bands={
            "water_mask": 1,
        },
    ),
}

LATEST_MICROSOFT_VERSION = MICROSOFT_VERSIONS["7"]


class GHSLVersion(BuiltVersion):
    measure_map: ClassVar[dict[str, tuple[str, str]]] = {
        "density": ("BUILT_S", "BUILT_S"),
        "nonresidential_density": ("BUILT_S", "BUILT_S_NRES"),
        "volume": ("BUILT_V", "BUILT_V"),
        "nonresidential_volume": ("BUILT_V", "BUILT_V_NRES"),
    }

    provider: Literal["ghsl"] = "ghsl"
    version: Literal["r2023a"]

    def prefix_and_measure(self, raw_measure: str) -> tuple[str, str]:
        return self.measure_map[raw_measure]

    def process_resources(self, resolution: str) -> tuple[str, str]:
        return {
            RESOLUTIONS.r40: ("8G", "20m"),
            RESOLUTIONS.r100: ("8G", "20m"),
        }[RESOLUTIONS(resolution)]


GHSL_VERSIONS = {
    "r2023a": GHSLVersion(
        version="r2023a",
        time_points=[f"{y}q1" for y in range(1975, 2030, 5)],
        input_template="GHS_{measure_prefix}_GLOBE_R2023A/GHS_{measure}_E{year}_GLOBE_R2023A_4326_3ss/V1-0/GHS_{measure}_E{year}_GLOBE_R2023A_4326_3ss_V1_0.zip",
        raw_output_template="{time_point}/GHS_{measure}_E{year}_GLOBE_R2023A_4326_3ss_V1_0.tif",
    ),
}


class CRS(BaseModel):
    name: str
    short_name: str
    bounds: tuple[float, float, float, float]
    code: str = ""
    proj_string: str = ""

    @model_validator(mode="after")
    def validate_code_or_proj_string(self) -> "CRS":
        if not self.code and not self.proj_string:
            msg = "Either code or proj_string must be provided."
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_code_and_proj_string(self) -> "CRS":
        if self.code and self.proj_string:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                code_proj = pyproj.CRS.from_user_input(self.code).to_proj4()
                proj_proj = pyproj.CRS.from_user_input(self.proj_string).to_proj4()
            if code_proj != proj_proj:
                msg = "code and proj_string must represent the same CRS."
                raise ValueError(msg)
        return self

    def to_string(self) -> str:
        if self.code:
            return self.code
        return self.proj_string

    def to_pyproj(self) -> pyproj.CRS:
        if self.code:
            return pyproj.CRS.from_user_input(self.code)
        return pyproj.CRS.from_user_input(self.proj_string)

    def __hash__(self) -> int:
        return hash(self.name)


CRSES: dict[str, CRS] = {
    "wgs84": CRS(
        name="WGS84",
        short_name="wgs84",
        code="EPSG:4326",
        proj_string="+proj=longlat +datum=WGS84 +no_defs +type=crs",
        bounds=(-180.0, -90.0, 180.0, 90.0),
    ),
    "wgs84_anti_meridian": CRS(
        name="WGS84 Anti-Meridian",
        short_name="wgs84_am",
        proj_string="+proj=longlat +lon_0=180 +datum=WGS84 +no_defs +type=crs",
        bounds=(-180.0, -90.0, 180.0, 90.0),
    ),
    "itu_anti_meridian": CRS(
        name="PDC Mercator",
        short_name="itu_am",
        code="EPSG:3832",
        proj_string="+proj=merc +lon_0=150 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs",
        bounds=(-5711803.07, -8362698.55, 15807367.69, 10023392.49),
    ),
    "mollweide": CRS(
        name="Mollweide",
        short_name="mollweide",
        code="ESRI:54009",
        proj_string="+proj=moll +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs",
        bounds=(-18040095.7, -9020047.85, 18040095.7, 9020047.85),
    ),
    "mollweide_anti_meridian": CRS(
        name="Mollweide Anti-Meridian",
        short_name="mollweide_am",
        proj_string="+proj=moll +lon_0=180 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs",
        bounds=(-18040095.7, -9020047.85, 18040095.7, 9020047.85),
    ),
    "world_cylindrical": CRS(
        name="World Cylindrical",
        short_name="world_cylindrical",
        code="ESRI:54034",
        proj_string="+proj=cea +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs",
        bounds=(-20037508.34, -6363885.33, 20037508.34, 6363885.33),
    ),
    "world_cylindrical_anti_meridian": CRS(
        name="World Cylindrical Anti-Meridian",
        short_name="world_cylindrical_am",
        proj_string="+proj=cea +lat_ts=0 +lon_0=180 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs",
        bounds=(-20037508.34, -6363885.33, 20037508.34, 6363885.33),
    ),
    "web_mercator": CRS(
        name="Web Mercator",
        short_name="web_mercator",
        code="EPSG:3857",
        proj_string="+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m +nadgrids=@null +wktext +no_defs +type=crs",
        bounds=(-20037508.34, -20048966.1, 20037508.34, 20048966.1),
    ),
}

# Add some aliases
CRSES["equal_area"] = CRSES["world_cylindrical"]
CRSES["equal_area_anti_meridian"] = CRSES["world_cylindrical_anti_meridian"]
