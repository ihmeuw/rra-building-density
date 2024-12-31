import click

from rra_building_density import (
    extract,
    postprocess,
    process,
)


@click.group()
def bdrun() -> None:
    """Run a stage of the building density pipeline."""


@click.group()
def bdtask() -> None:
    """Run an individual modeling task in the building density pipeline."""


for module in [extract, process, postprocess]:
    runners = getattr(module, "RUNNERS", {})
    task_runners = getattr(module, "TASK_RUNNERS", {})

    if not runners or not task_runners:
        continue

    command_name = module.__name__.split(".")[-1]

    @click.group(name=command_name)
    def workflow_runner() -> None:
        pass

    for name, runner in runners.items():
        workflow_runner.add_command(runner, name)

    bdrun.add_command(workflow_runner)

    @click.group(name=command_name)
    def task_runner() -> None:
        pass

    for name, runner in task_runners.items():
        task_runner.add_command(runner, name)

    bdtask.add_command(task_runner)
