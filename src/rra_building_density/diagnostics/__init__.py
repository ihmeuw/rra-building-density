from rra_population_pipelines.pipelines.models.building_density.diagnostics.plot_building_density import (
    plot_building_density,
    plot_building_density_block_task,
    plot_building_density_global_task,
)
from rra_population_pipelines.pipelines.models.building_density.diagnostics.summarize_building_density import (
    summarize_building_density,
    summarize_building_density_task,
)

RUNNERS = {
    "summarize_building_density": summarize_building_density,
    "plot_building_density": plot_building_density,
}

TASK_RUNNERS = {
    "summarize_building_density": summarize_building_density_task,
    "plot_building_density_block": plot_building_density_block_task,
    "plot_building_density_global": plot_building_density_global_task,
}
