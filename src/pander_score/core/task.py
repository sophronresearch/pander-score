"""The single public benchmark task."""

from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from pander_score.core.scorers import credence_scorer


@task
def pander_benchmark(dataset_path: str) -> Task:
    return Task(
        dataset=json_dataset(dataset_path),
        solver=[generate()],
        scorer=credence_scorer(),
        config=GenerateConfig(cache=True),
    )
