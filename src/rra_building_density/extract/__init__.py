from rra_building_density.extract.ghsl import (
    extract_ghsl_task,
    extract_ghsl,
)
from rra_building_density.extract.microsoft import (
    extract_microsoft_indices_task,
    extract_microsoft_tiles_task,
    extract_microsoft,
)

RUNNERS = {
    "ghsl": extract_ghsl,
    "microsoft": extract_microsoft,
}

TASK_RUNNERS = {
    "ghsl": extract_ghsl_task,
    "microsoft_indices": extract_microsoft_indices_task,
    "microsoft": extract_microsoft_tiles_task,
}
