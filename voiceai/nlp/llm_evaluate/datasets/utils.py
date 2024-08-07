import logging
from typing import Any
from typing import Dict
from typing import Optional

from ..helpers.profile import Profile
from .dataset import EvaluationDataset
from .spec import DatasetSpec


def load_dataset_catalog(data: Dict[str, Any]) -> Dict[str, DatasetSpec]:
    logging.debug('loading dataset catalog...')
    dataset_specs = [
        {
            'name': dataset_name,
            **data[dataset_name],
        }
        for dataset_name in data
    ]
    if not dataset_specs:
        raise ValueError('No datasets available')

    return {spec['name']: DatasetSpec(**spec) for spec in dataset_specs}


def load_dataset_from_spec(
    spec: DatasetSpec, profile: Profile,
    max_samples: Optional[int] = None, batch_size: int = 1,
) -> EvaluationDataset:
    assert spec.tasks, f'No tasks defined in dataset spec: {spec.name}!'

    logging.debug('loading dataset %s with total tasks %s',
                  spec.name, len(spec.tasks))

    # load task data
    return EvaluationDataset(spec,
                             batch_size=batch_size,
                             max_samples=max_samples,
                             profile=profile)


def load_dataset(
    name: str, profile: Profile,
    max_samples: Optional[int] = None, batch_size: int = 1,
) -> EvaluationDataset:
    logging.debug('loading dataset: %s', name)
    try:
        spec = profile.datasets[name]
    except KeyError as e:
        logging.exception(
            '`%s` not found in %s',
            name,
            profile.datasets.keys())
        raise KeyError(f'Dataset `{name}` not found!') from e

    return load_dataset_from_spec(
        spec, profile=profile, max_samples=max_samples, batch_size=batch_size)
