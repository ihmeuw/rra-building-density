# How to add a new microsoft version

## Update `constants.py`

The constants module stores metadata used by the pipeline and includes small data-structures representing each version of Microsoft and GHSL building density.

1. Update allowed versions: In the `MicrosoftVersion` class add the new version in the format `vX` to the `version` Literal typing.
2. Add new version: In the `MICROSOFT_VERSIONS` dict add a new key value pair for the version. Say our new version is `X`, then:
    - The dict key is also `X`.
    - The `version` field of the value is `vX`
    - The `time_points` field is a list of time points in `YYYYqQ` format where `YYYY` is the year and `Q` is the quarter (1, 2, 3, or 4). Available time points vary by building density version.
    - The `input_template` is the Azure storage blob relative path from root (`predictions` is a top level directory) to the subdirectory with the actual tif files present.  We use `azcopy` to download these from blob storage to a subdirectory on the shared drives.
    - The `raw_output_template` is the file-pattern for saving the raw tiles. It is a relative path from `MODEL_ROOT/raw_tiles/VERSION_NAME/`. `MODEL_ROOT` is typically `/mnt/team/rapidresponse/pub/building-density/` but can be altered as an argument to all pipeline stages with the `--output-dir` flag.
    - `bands` is a dict mapping a measure to a band number (1-indexed int) in a tif.  Microsoft introduced a height layer in v7 and so we need to process multiple bands sometimes.  Our processing pipeline writes out flat (single-band) tifs for each measure.
3. If it makes sense, update the `LATEST_MICROSOFT_VERSION` to point at the new version you created. We use the `LATEST_MICROSOFT_VERSION` to generate the GHSL tiles to ensure they share the same no_data (ie water) mask.

## Extract new microsoft version

You generally shouldn't have to modify the extraction script unless MSFT has re-arranged their data in an unfortunate way. I usually just launch a run with `bdrun extract microsoft --version X` and presume it will work, then check on failures if they occur. This job is only parallelized over time-points and is cheap to run.

## Process the microsoft version

This step is parallelized by `block_key` and `time_point`. The block keys come from the tile index (`bdrun process tile_index --resolution R`) and are specific to the working resolution of the output tiles. This step does the following:

- Generates a block template, ie an empty raster representing the block data with the exact CRS, resolution, and pixel layout we want our data to be in.
- Crosswalks the Microsoft tile indexing system to our indexing system to find all Microsoft tiles in the respective block
- Writes out an empty block if no Microsoft tiles are found (they only provide data over land), presuming open ocean.
- For each measure/band:
    - For each tile in the block:
        - Load the tile
        - Fix some rounding issues in their coordinates
        - Fixes their nodata specification
        - Reprojects into our target CRS
    - Then we merge all tiles
    - And resample to our block template. This second resampling ensures the merge didn't do anything weird. It also clips the boundaries so the resulting raster exactly matches the block template.
    - Suppresses noise in the resulting raster (ie clips all density values to be above 1, all height values to be in a reasonable range).
    - Writes the measure

This step occasionally needs modification for a new version.
