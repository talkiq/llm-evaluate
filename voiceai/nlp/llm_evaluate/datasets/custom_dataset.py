import abc
import logging
import pathlib
from typing import Any

import pandas

from ..helpers.utils import camel_to_snake
from ..helpers.utils import dynamic_load_class


class CustomDataset:
    @abc.abstractmethod
    def load_all(self, n_shots: int) -> pandas.DataFrame:
        '''Load all the data in the custom dataset.'''


def load_custom_dataset(extensions_filepath: pathlib.Path,
                        filename: str, dataset_class: str,
                        **kwargs: Any) -> CustomDataset:
    filepath = extensions_filepath / filename
    logging.debug('loading dataset: %s with kwargs: %s from: %s',
                  dataset_class, kwargs, filepath)
    module_name = f'voiceai.nlp.llm_evaluate.datasets.{camel_to_snake(dataset_class)}'
    dataset = dynamic_load_class(
        filepath=filepath, class_name=dataset_class, module_name=module_name)(**kwargs)
    assert isinstance(dataset, CustomDataset)
    return dataset
