from rra_building_density.process.ghsl import (
    format_ghsl,
    format_ghsl_task,
)
from rra_building_density.process.microsoft import (
    format_microsoft,
    format_microsoft_task,
)
from rra_building_density.process.tile_index import (
    tile_index,
    tile_index_task,
)

RUNNERS = {
    "tile_index": tile_index,
    "ghsl": format_ghsl,
    "microsoft": format_microsoft,
}

TASK_RUNNERS = {
    "tile_index": tile_index_task,
    "ghsl": format_ghsl_task,
    "microsoft": format_microsoft_task,
}
