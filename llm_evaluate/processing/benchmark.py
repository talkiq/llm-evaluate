import logging
import pathlib
from dataclasses import dataclass
from typing import Any

import evaluate

from ..metrics import load_metric
from ..metrics import MetricSpec


@dataclass(kw_only=True)
class BenchmarkDatasetTaskSpec:
    task_name: str
    metrics: list[str] | None = None
    # loaded_metrics: Optional[List[evaluate.EvaluationModule]] = None

    def __init__(
        self, task_name: str,
        metrics: list[str] | None = None,
    ) -> None:
        self.task_name = task_name
        self.metrics = list(set(metrics or []))
        self.loaded_metrics: list[evaluate.EvaluationModule] = []

    def load_metrics(
        self, metrics_catalog: dict[str, MetricSpec],
        extension_filepath: pathlib.Path,
    ) -> None:
        if self.loaded_metrics:
            return
        if not self.metrics:
            return
        for metric in self.metrics:
            self.loaded_metrics.append(
                load_metric(
                    metrics_catalog.get(
                        metric,
                        MetricSpec(
                            name=metric, filename='', is_extension=False,
                        ),
                    ), extension_filepath,
                ),
            )

    def compute_metrics(
        self, references: list[str], predictions: list[str],
        metrics_catalog: dict[str, MetricSpec],
        extension_filepath: pathlib.Path,
    ) -> dict[str, float]:
        self.load_metrics(metrics_catalog, extension_filepath)
        overall = {}
        average_types = {'weighted', 'micro', 'macro'}
        for metric in self.loaded_metrics:
            if metric.name in {'precision', 'f1', 'recall', 'support'}:
                for avg_type in average_types:
                    computed = {
                        f'{name}-{avg_type}': val
                        for name, val in metric.compute(
                            references=references,
                            predictions=predictions,
                            average=avg_type,
                        ).items()
                    }
                    overall = {**overall, **computed}
            else:
                overall = {
                    **overall, **metric.compute(
                        references=references, predictions=predictions,
                    ),
                }
        return overall


@dataclass(kw_only=True)
class BenchmarkDatasetSpec:
    dataset_name: str
    prompt: str | None
    tasks: dict[str, BenchmarkDatasetTaskSpec]

    def __init__(
        self, dataset_name: str, tasks: dict[str, Any],
        prompt: str | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.prompt = prompt
        self.tasks = self.load_tasks(tasks)

    @staticmethod
    def load_tasks(spec: dict[str, Any]) -> dict[
        str,
        BenchmarkDatasetTaskSpec,
    ]:
        return {
            key: BenchmarkDatasetTaskSpec(task_name=key, **val)
            for key, val in spec.items()
        }


@dataclass(kw_only=True)
class BenchmarkSpec:
    benchmark_name: str
    datasets: dict[str, BenchmarkDatasetSpec]

    def __init__(
        self, benchmark_name: str,
        datasets: list[dict[str, Any]],
    ) -> None:
        self.benchmark_name = benchmark_name
        self.datasets = self.load_datasets(datasets)

    @staticmethod
    def load_datasets(
        spec: dict[str, dict[str, Any]],
    ) -> dict[str, BenchmarkDatasetSpec]:
        return {
            dataset: BenchmarkDatasetSpec(**db_specs)
            for dataset, db_specs in spec.items()
        }


def load_benchmark_catalog(data: dict[str, Any]) -> list[BenchmarkSpec]:
    logging.debug('loading benchmark catalog...')
    catalog = [
        {
            'benchmark_name': benchmark_name,
            'datasets': {
                dataset_name: {
                    'dataset_name': dataset_name,
                    **task_spec,
                }
                for dataset_name, task_spec in data[benchmark_name].items()
            },
        }
        for benchmark_name in data
    ]

    if not catalog:
        raise ValueError('No benchmarks available')
    return [BenchmarkSpec(**item) for item in catalog]
