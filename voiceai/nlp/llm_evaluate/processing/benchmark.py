import logging
import pathlib
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import evaluate

from ..metrics import load_metric
from ..metrics import MetricSpec

@dataclass(kw_only=True)
class BenchmarkDatasetTaskSpec:
    task_name: str
    metrics: Optional[List[str]] = None
    # loaded_metrics: Optional[List[evaluate.EvaluationModule]] = None

    def __init__(self, task_name: str, metrics: Optional[List[str]] = None,) -> None:
        self.task_name = task_name
        self.metrics = list(set(metrics or []))
        self.loaded_metrics: List[evaluate.EvaluationModule] = []

    def load_metrics(self, metrics_catalog: Dict[str, MetricSpec],
                     extension_filepath: pathlib.Path) -> None:
        if self.loaded_metrics:
            return
        if not self.metrics:
            return
        for metric in self.metrics:
            self.loaded_metrics.append(
                load_metric(metrics_catalog.get(
                    metric,
                    MetricSpec(name=metric, filename='', is_extension=False)
                ), extension_filepath))

    def compute_metrics(
        self, references: List[str], predictions: List[str],
        metrics_catalog: Dict[str, MetricSpec],
        extension_filepath: pathlib.Path,
    ) -> Dict[str, float]:
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
                            average=avg_type).items()
                    }
                    overall = {**overall, **computed}
            else:
                overall = {**overall, **metric.compute(
                    references=references, predictions=predictions)}
        return overall


@dataclass(kw_only=True)
class BenchmarkDatasetSpec:
    dataset_name: str
    prompt: Optional[str]
    tasks: Dict[str, BenchmarkDatasetTaskSpec]

    def __init__(self, dataset_name: str, tasks: Dict[str, Any],
                 prompt: Optional[str] = None) -> None:
        self.dataset_name = dataset_name
        self.prompt = prompt
        self.tasks = self.load_tasks(tasks)

    @staticmethod
    def load_tasks(spec: Dict[str, Any]) -> Dict[str,
                                                 BenchmarkDatasetTaskSpec]:
        return {
            key: BenchmarkDatasetTaskSpec(task_name=key, **val)
            for key, val in spec.items()
        }


@dataclass(kw_only=True)
class BenchmarkSpec:
    benchmark_name: str
    datasets: Dict[str, BenchmarkDatasetSpec]

    def __init__(self, benchmark_name: str,
                 datasets: List[Dict[str, Any]]) -> None:
        self.benchmark_name = benchmark_name
        self.datasets = self.load_datasets(datasets)

    @staticmethod
    def load_datasets(spec: Dict[str, Dict[str, Any]],
                      ) -> Dict[str, BenchmarkDatasetSpec]:
        return {
            dataset: BenchmarkDatasetSpec(**db_specs)
            for dataset, db_specs in spec.items()
        }


def load_benchmark_catalog(data: Dict[str, Any]) -> List[BenchmarkSpec]:
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
