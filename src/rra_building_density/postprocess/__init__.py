from rra_building_density.postprocess.summarize import (
    summarize,
    summarize_task,
)

RUNNERS = {
    "summarize": summarize,
}

TASK_RUNNERS = {
    "summarize": summarize_task,
}
