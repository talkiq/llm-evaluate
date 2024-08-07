import abc
import enum
import logging
import pathlib
import warnings
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import pandas

from ..parsers import load_parser
from ..parsers import Parser
from ..parsers import ParserSpec
from .custom_dataset import load_custom_dataset


class DatasetSource(enum.Enum):
    GCS = 'gcs'
    LOCAL = 'local'
    CUSTOM = 'custom'


class FileFormat(enum.Enum):
    CSV = 'csv'
    JSON = 'json'
    JSONL = 'jsonl'


class DatasetTaskType(enum.Enum):
    CLASSIFICATION = 'classification'
    GENERATION = 'generation'


@dataclass(kw_only=True)
class DatasetMetadata:
    source: DatasetSource
    version: Union[int, str] = 0
    n_shots: int = 0
    extra_args: Dict[str, Any] = field(default_factory=dict)

    @abc.abstractmethod
    def load_data(
        self,
        extensions_filepath: Optional[pathlib.Path] = None,
    ) -> pandas.DataFrame:
        '''Load data from the specified source.'''


@dataclass(kw_only=True)
class GcsDatasetMetadata(DatasetMetadata):
    format: str
    path: str

    def load_data(
        self,
        _extensions_filepath: Optional[pathlib.Path] = None,
    ) -> pandas.DataFrame:
        if self.format == FileFormat.CSV.value:
            return pandas.read_csv(self.path)
        if self.format == FileFormat.JSON.value:
            return pandas.read_json(self.path, orient='records')
        if self.format == FileFormat.JSONL.value:
            return pandas.read_json(self.path, orient='records', lines=True)
        raise ValueError(f'Unknown format: {self.format}')


@dataclass(kw_only=True)
class CustomDatasetMetadata(DatasetMetadata):
    loader_classname: str
    loader_filename: str

    def load_data(
        self,
        extensions_filepath: Optional[pathlib.Path] = None,
    ) -> pandas.DataFrame:
        if not extensions_filepath:
            raise ValueError('No extensions available to load the custom dataset')
        custom_dataset = load_custom_dataset(
            extensions_filepath=extensions_filepath,
            filename=self.loader_filename,
            dataset_class=self.loader_classname,
            **{**self.extra_args, 'version': self.version})
        return custom_dataset.load_all(self.n_shots)


@dataclass(kw_only=True)
class LocalDatasetMetadata(GcsDatasetMetadata):
    format: Optional[str] = None
    path: Optional[str] = None
    data: List[Dict[str, str]] = field(default_factory=list)

    def load_data(
        self,
        extensions_filepath: Optional[pathlib.Path] = None,
    ) -> pandas.DataFrame:
        if self.path:
            return super().load_data(extensions_filepath)
        return pandas.DataFrame(self.data)


@dataclass(kw_only=True)
class DatasetTaskSpec:
    name: str
    task_type: str
    key: Optional[str]
    multi_y: bool
    none_value: Optional[str]
    case_sensitive: bool
    labels: List[Union[str, int]]
    task_reference_parser: str
    model_output_parser: str

    def __init__(
        self,
        name: str,
        task_type: str,
        key: Optional[str] = None,
        multi_y: bool = False,
        none_value: Optional[str] = 'Unknown',
        case_sensitive: bool = False,
        labels: Optional[List[Union[str, int]]] = None,
        task_reference_parser: str = 'DefaultParser',
        model_output_parser: str = 'DefaultParser',
    ) -> None:
        logging.debug('building DatasetTaskSpec: %s', name)
        self.name = name
        self.task_type = DatasetTaskType(task_type)
        self.multi_y = multi_y
        if none_value:
            self.none_value = none_value if case_sensitive else none_value.lower()
        else:
            self.none_value = None
        self.case_sensitive = case_sensitive
        if labels:
            self.labels = labels if case_sensitive else [
                str(lbl).lower() for lbl in labels]
        else:
            self.labels = []
        self.key = key
        self.task_reference_parser = task_reference_parser
        self.model_output_parser = model_output_parser

    def get_task_reference_parser(
        self, parser_spec: ParserSpec, extensions_path: pathlib.Path,
        **kwargs: Any,
    ) -> Parser:
        return load_parser(
            spec=parser_spec, extensions_path=extensions_path,
            none_value=self.none_value, case_sensitive=self.case_sensitive,
            multi_y=self.multi_y, **kwargs,
        )

    def get_model_output_parser(
        self, parser_spec: ParserSpec, extensions_path: pathlib.Path,
        **kwargs: Any,
    ) -> Parser:
        return load_parser(
            spec=parser_spec, extensions_path=extensions_path,
            none_value=self.none_value, case_sensitive=self.case_sensitive,
            multi_y=self.multi_y, labels=self.labels, **kwargs,
        )


@dataclass(kw_only=True)
class DatasetSpec:
    name: str
    column_input: str
    column_reference: Optional[str]
    metadata: DatasetMetadata
    description: Optional[str]
    tasks: Dict[str, DatasetTaskSpec]
    reference_split_parser: str = 'DefaultReferenceSplitParser'
    _reference_split_parser: Optional[Parser] = None

    def __init__(
        self,
        name: str,
        column_input: str,
        metadata: Dict[str, Any],
        tasks: Dict[str, Any],
        description: Optional[str] = None,
        column_reference: Optional[str] = None,
        reference_split_parser: str = 'DefaultReferenceSplitParser',
    ) -> None:
        logging.debug('building DatasetSpec: %s', name)
        self.name = name
        self.description = description
        self.column_input = column_input
        self.column_reference = column_reference
        self.metadata = self.build_metadata(metadata)
        self.tasks = self.build_tasks(tasks)
        self.reference_split_parser = reference_split_parser

    def get_reference_split_parser(
        self, parser_spec: ParserSpec, extensions_path: pathlib.Path,
    ) -> Parser:
        if not self._reference_split_parser:
            self._reference_split_parser = load_parser(
                parser_spec,
                extensions_path=extensions_path,
                keys=[task.key for task in self.tasks.values()])
        return self._reference_split_parser

    @staticmethod
    def build_metadata(meta: Dict[str, Any]) -> DatasetMetadata:
        source = DatasetSource(meta['source'])
        if source == DatasetSource.GCS:
            class_ = GcsDatasetMetadata
        elif source == DatasetSource.LOCAL:
            class_ = LocalDatasetMetadata
        elif source == DatasetSource.CUSTOM:
            class_ = CustomDatasetMetadata
        else:
            raise ValueError(f'Unknown source: {source}')
        meta['source'] = source
        return class_(**meta)

    @staticmethod
    def build_tasks(tasks: Dict[str, Dict[str, Any]],
                    ) -> Dict[str, DatasetTaskSpec]:
        return {key: DatasetTaskSpec(name=key, **value)
                for key, value in tasks.items()}

    def load_data(self, extensions_filepath: pathlib.Path) -> pandas.DataFrame:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            return self.metadata.load_data(extensions_filepath)
