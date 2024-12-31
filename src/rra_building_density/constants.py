import itertools
from pathlib import Path

import pyproj
from pydantic import BaseModel, model_validator

RRA_ROOT = Path("/mnt/team/rapidresponse/")
RRA_CREDENTIALS_ROOT = RRA_ROOT / "priv" / "shared" / "credentials"
RRA_BINARIES_ROOT = RRA_ROOT / "priv" / "shared" / "bin"
MODEL_ROOT = RRA_ROOT / "pub" / "building-density"

GHSL_CRS_MAP = {
    "mollweide": "54009_100",
    "wgs84": "4326_3ss",
}

GHSL_MEASURE_MAP = {
    "density": ("BUILT_S", "BUILT_S"),
    "nonresidential_density": ("BUILT_S", "BUILT_S_NRES"),
    "volume": ("BUILT_V", "BUILT_V"),
    "nonresidential_volume": ("BUILT_V", "BUILT_V_NRES"),
}

GHSL_YEARS = [str(y) for y in range(1975, 2035, 5)]
GHSL_TIME_POINTS = [f"{y}q1" for y in GHSL_YEARS]

MICROSOFT_TIME_POINTS = {
    "2": [f"{y}q{q}" for q, y in itertools.product(range(1, 5), range(2018, 2024))][
        :-1
    ],
    "3": ["2023q3"],
    "4": ["2023q4"],
}
MICROSOFT_VERSIONS = list(MICROSOFT_TIME_POINTS)

ALL_TIME_POINTS = sorted(
    set(GHSL_TIME_POINTS) | set().union(*MICROSOFT_TIME_POINTS.values())
)


class CRS(BaseModel):
    name: str
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
            code_proj = pyproj.CRS.from_user_input(self.code)
            proj_proj = pyproj.CRS.from_user_input(self.proj_string)
            if code_proj != proj_proj:
                msg = "code and proj_string must represent the same CRS."
                raise ValueError(msg)
        return self

    def to_pyproj(self) -> pyproj.CRS:
        if self.proj_string:
            return pyproj.CRS.from_proj4(self.proj_string)
        return pyproj.CRS.from_user_input(self.code)


CRSES: dict[str, CRS] = {
    "wgs84": CRS(
        name="WGS84",
        code="EPSG:4326",
        proj_string="+proj=longlat +datum=WGS84 +no_defs +type=crs",
        bounds=(-180.0, -90.0, 180.0, 90.0),
    ),
    "wgs84_anti_meridian": CRS(
        name="WGS84 Anti-Meridian",
        proj_string="+proj=longlat +lon_0=180 +datum=WGS84 +no_defs +type=crs",
        bounds=(-180.0, -90.0, 180.0, 90.0),
    ),
    "itu_anti_meridian": CRS(
        name="PDC Mercator",
        code="EPSG:3832",
        proj_string="+proj=merc +lon_0=150 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs",
        bounds=(-5711803.07, -8362698.55, 15807367.69, 10023392.49),
    ),
    "mollweide": CRS(
        name="Mollweide",
        code="ESRI:54009",
        proj_string="+proj=moll +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs",
        bounds=(-18040095.7, -9020047.85, 18040095.7, 9020047.85),
    ),
    "mollweide_anti_meridian": CRS(
        name="Mollweide Anti-Meridian",
        proj_string="+proj=moll +lon_0=180 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs",
        bounds=(-18040095.7, -9020047.85, 18040095.7, 9020047.85),
    ),
    "world_cylindrical": CRS(
        name="World Cylindrical",
        code="ESRI:54034",
        proj_string="+proj=cea +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs",
        bounds=(-20037508.34, -6363885.33, 20037508.34, 6363885.33),
    ),
    "world_cylindrical_anti_meridian": CRS(
        name="World Cylindrical Anti-Meridian",
        proj_string="+proj=cea +lat_ts=0 +lon_0=180 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs",
        bounds=(-20037508.34, -6363885.33, 20037508.34, 6363885.33),
    ),
    "web_mercator": CRS(
        name="Web Mercator",
        code="EPSG:3857",
        proj_string="+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m +nadgrids=@null +wktext +no_defs +type=crs",
        bounds=(-20037508.34, -20048966.1, 20037508.34, 20048966.1),
    ),
}

# Add some aliases
CRSES["equal_area"] = CRSES["world_cylindrical"]
CRSES["equal_area_anti_meridian"] = CRSES["world_cylindrical_anti_meridian"]

RESOLUTIONS = ["40", "100", "1000", "5000"]
