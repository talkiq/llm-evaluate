import enum
import logging
import os
import pathlib

from ..datasets.spec import DatasetSpec
from ..metrics import MetricSpec
from ..models import ModelLoaderSpec
from ..models import ModelSpec
from ..parsers import ParserSpec
from ..processing.benchmark import BenchmarkSpec


class ExtensionType(enum.Enum):
    DATASETS = 'datasets'
    METRICS = 'metrics'
    MODELS = 'models'
    PARSERS = 'parsers'


class ConfigurationProfile:

    def __init__(
        self,
        extensions: dict[str, str],
        benchmarks: dict[str, BenchmarkSpec],
        datasets: dict[str, DatasetSpec],
        parsers: dict[str, ParserSpec],
        metrics: dict[str, MetricSpec],
        model_loaders: dict[str, ModelLoaderSpec],
        model: ModelSpec,
    ) -> None:
        self.extensions = {}
        for type_ in ExtensionType:
            path = extensions.get(type_.value)
            if not path:
                path = os.path.join(os.getcwd(), 'extensions', type_.value)

            if not os.path.isabs(path):
                path = os.path.join(os.getcwd(), path)
            self.extensions[type_] = pathlib.Path(path)

            if not os.path.exists(path):
                logging.warning(
                    'directory for %s extensions not available: %s',
                    type_.value, path,
                )
            else:
                logging.debug(
                    'extensions for %s available at %s',
                    type_.value,
                    path,
                )

        self.parsers = parsers
        self.benchmarks = benchmarks
        self.datasets = datasets
        self.metrics = metrics
        self.model_loaders = model_loaders
        self.model = model
