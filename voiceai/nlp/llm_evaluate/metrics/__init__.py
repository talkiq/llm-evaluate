import logging
import pathlib
from dataclasses import dataclass
from typing import Any
from typing import Dict

import evaluate

@dataclass(kw_only=True)
class MetricSpec:
    name: str
    filename: str
    is_extension: bool = True


def load_metric_catalog(data: Dict[str, Any]) -> Dict[str, MetricSpec]:
    logging.debug('loading metric catalog...')
    specs = [
        {
            'name': metric_name,
            **data[metric_name],
        }
        for metric_name in data
    ]
    if not specs:
        raise ValueError('No metrics available')
    return {spec['name']: MetricSpec(**spec) for spec in specs}


def load_metric(
        spec: MetricSpec,
        extensions_filepath: pathlib.Path,
    ) -> evaluate.Metric:
    load_name = str(extensions_filepath / spec.filename) if spec.is_extension else spec.name
    logging.debug('loading metric %s from %s', spec.name, load_name)
    return evaluate.load(load_name)
