from rra_building_density.format.prepare_tile_index import (
    prepare_tile_index_task,
    prepare_tile_index,
)
from rra_building_density.format.ghsl import (
    ghsl_task,
    ghsl,
)
from rra_building_density.format.microsoft import (
    microsoft_task,
    microsoft,
)

RUNNERS = {
    "prepare_tile_index": prepare_tile_index,
    "ghsl": ghsl,
    "microsoft": microsoft,
}

TASK_RUNNERS = {
    "prepare_tile_index": prepare_tile_index_task,
    "ghsl": ghsl_task,
    "microsoft": microsoft_task,
}