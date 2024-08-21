# pylint: disable=line-too-long
from typing import Any

import pandas
import pytest

from llm_evaluate.datasets import EvaluationDataset
from llm_evaluate.datasets import load_dataset
from llm_evaluate.datasets.spec import DatasetMetadata
from llm_evaluate.datasets.spec import DatasetTaskSpec
from llm_evaluate.datasets.spec import DatasetTaskType
from llm_evaluate.datasets.utils import load_dataset_from_spec
from llm_evaluate.helpers.configuration_profile import ConfigurationProfile
from llm_evaluate.helpers.configuration_profile import ExtensionType


def test_dataset_spec_load(default_profile: ConfigurationProfile) -> None:
    catalog = default_profile.datasets
    assert catalog

    # check for datasets
    for name, dataset_spec in catalog.items():
        assert dataset_spec.name == name
        assert dataset_spec.tasks
        assert isinstance(dataset_spec.tasks, dict)
        assert isinstance(
            list(
                dataset_spec.tasks.values(),
            )[0],
            DatasetTaskSpec,
        )
        assert dataset_spec.metadata
        assert isinstance(dataset_spec.metadata, DatasetMetadata)


def test_dataset_raw_load(default_profile: ConfigurationProfile) -> None:
    catalog = default_profile.datasets
    assert catalog

    # check for datasets
    for dataset_spec in catalog.values():
        df = dataset_spec.load_data(
            extensions_filepath=default_profile.extensions[
                ExtensionType.DATASETS
            ],
        )
        assert isinstance(df, pandas.DataFrame)
        assert len(df)


def test_dataset_task_load(default_profile: ConfigurationProfile) -> None:
    catalog = default_profile.datasets
    assert catalog

    # check for datasets
    for dataset_spec in catalog.values():
        load_dataset_from_spec(dataset_spec, default_profile)


def test_load_dataset_all(default_profile: ConfigurationProfile) -> None:
    catalog = default_profile.datasets

    # load every dataset
    for name in catalog:
        dataset = load_dataset(name, default_profile)
        assert isinstance(dataset, EvaluationDataset)


@pytest.mark.parametrize(
    'dataset_name, expected_row_data', [
        ('single_task', ('How to get to Vancouver from Seattle?', [])),
        ('single_task_numeric', ('1 + 1 = ?', [0])),
        (
            'multiple_tasks', (
                """Provide responses to the following questions in JSON with the key as the question number.
1. How to get to Vancouver from Seattle?
2. Classifiy the sentiment of the following statement into positive, negative or neutral.
  "Seattle and Vancouver are very similar cities."
""",
                ['2 hours and 30 minutes', 0],
            ),
        ),
        (
            'single_task_multi_y',
            ('Give the names of 3 biggest cities in Canada.', [[1, 3, 0]]),
        ),
        (
            'multiple_tasks_multiple_responses', (
                """Provide responses to the following questions in JSON with the key as the question number.
1. How to get to Vancouver from Seattle?
2. Classifiy the sentiment of the following statement into positive, negative or neutral.
  "Seattle and Vancouver are very similar cities."
""",
                ['2 hours and 30 minutes', 0],
            ),
        ),
    ],
)
def test_load_dataset_single(
    default_profile: ConfigurationProfile,
    dataset_name: str,
    expected_row_data: Any,
) -> None:
    dataset = load_dataset(dataset_name, default_profile)
    assert isinstance(dataset, EvaluationDataset)
    assert len(dataset)
    sample = dataset[0]
    assert sample == expected_row_data


def test_dataset_dataloader(default_profile: ConfigurationProfile) -> None:
    catalog = default_profile.datasets
    batch_size = 1

    # load every dataset
    for name in catalog:
        dataset = load_dataset(name, default_profile, batch_size=batch_size)
        assert isinstance(dataset, EvaluationDataset)
        dataloader = dataset.get_dataloader()
        for batch in dataloader:
            inputs, refs = batch
            assert len(inputs) == batch_size
            assert isinstance(inputs, (list, tuple))
            assert isinstance(inputs[0], str)
            assert isinstance(refs, (list, tuple))

            tasks = catalog[name].tasks
            if len(tasks) > 1 or list(tasks.values())[
                    0
            ].task_type == DatasetTaskType.CLASSIFICATION:
                assert len(refs[0]) == len(tasks)
