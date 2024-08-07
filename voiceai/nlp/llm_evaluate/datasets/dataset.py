import abc
import logging
import warnings
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import pandas
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import default_collate

from ..helpers.profile import ExtensionType
from ..helpers.profile import Profile
from .spec import DatasetSpec
from .spec import DatasetTaskSpec
from .spec import DatasetTaskType


def collate_fn(data: Any) -> Any:
    refs = [elem[-1] for elem in data]
    inputs = [elem[0] for elem in data]
    inputs = default_collate(inputs)
    return inputs, refs


class ReferencesDataset(Dataset):
    def __init__(
        self, dataset_spec: DatasetSpec, task_spec: DatasetTaskSpec,
        data: pandas.DataFrame, task_data: List[str], profile: Profile,
    ) -> None:
        super().__init__()
        self.dataset_spec = dataset_spec
        self.task_spec = task_spec
        self.data = data
        self.task_data = task_data
        self.profile = profile
        self.parse_task_data()
        self.post_init_processing()

    def __len__(self) -> int:
        return len(self.data)

    def post_init_processing(self) -> None:
        pass

    def parse_task_data(self) -> None:
        self.data[f'{self.task_spec.name}_raw'] = self.task_data
        self.data[self.task_spec.name] = self.data.loc[
            :, [self.dataset_spec.column_input, f'{self.task_spec.name}_raw'],
        ].apply(
            lambda x: self.task_spec.get_task_reference_parser(
                parser_spec=self.profile.parsers[self.task_spec.task_reference_parser],
                extensions_path=self.profile.extensions[ExtensionType.PARSERS],
            ).parse(
                prompt=x.iloc[0], output=x.iloc[1]),
            axis=1,
        )

    @abc.abstractmethod
    def __getitem__(
        self, idx: int,
    ) -> Union[Optional[Union[str, int]], List[Union[str, int]]]:
        '''Returns the reference for the given sample index.'''

    def postprocess_references(self, raw_references: List[int]) -> List[Any]:
        '''Transform model references into a form that metrics can process.'''
        return raw_references

    def postprocess_outputs(self, raw_outputs: List[str]) -> List[Any]:
        '''Transform model outputs into a form that metrics can process.'''
        return raw_outputs


class GenerationReferencesDataset(ReferencesDataset):
    def __getitem__(
        self, idx: int,
    ) -> Optional[str]:
        return self.data[self.task_spec.name][idx]


class ClassificationReferencesDataset(ReferencesDataset):
    labels: List[str]
    labels_to_idx: Dict[str, int]
    idx_to_label: Dict[int, str]

    def post_init_processing(self) -> None:
        super().post_init_processing()
        if self.task_spec.labels:
            if not self.task_spec.case_sensitive:
                self.task_spec.labels = [
                    lbl.strip().lower() for lbl in self.task_spec.labels]
            labels = sorted(set(self.task_spec.labels))
        if self.task_spec.multi_y:
            labels = list(
                {lbl for label_tuple in self.data[self.task_spec.name].tolist()
                 for lbl in label_tuple},
            )
        else:
            labels = self.data[self.task_spec.name].unique().tolist()
        if self.task_spec.none_value not in labels:
            labels.append(self.task_spec.none_value)
        self.labels = sorted(labels)
        self.labels_to_idx = {lbl: idx for idx, lbl in enumerate(self.labels)}
        self.idx_to_label = dict(enumerate(self.labels))

    def __getitem__(self, idx: int) -> Union[int, List[int]]:
        if self.task_spec.multi_y:
            return [self.labels_to_idx[lbl]
                    for lbl in self.data[self.task_spec.name][idx]]
        return self.labels_to_idx[self.data[self.task_spec.name][idx]]

    def postprocess_references(
        self, raw_references: List[Union[int, List[int]]],
    ) -> List[Union[str, List[str]]]:
        references: List[str] = []
        for idx in raw_references:
            if self.task_spec.multi_y:
                labels = idx
                if not labels:
                    references.append([self.task_spec.none_value])
                else:
                    references.append([self.idx_to_label[id_] for id_ in idx])
            else:
                label = idx.cpu().item() if isinstance(idx, torch.Tensor) else idx
                if pandas.isna(label):
                    references.append(self.task_spec.none_value)
                else:
                    references.append(self.idx_to_label[label])
        return references

    def postprocess_outputs(
        self, raw_outputs: List[Union[str, List[str]]],
    ) -> List[Union[int, List[int]]]:
        outputs: List[int] = []
        for label in raw_outputs:
            if self.task_spec.multi_y and isinstance(
                    label, list) and not label:
                outputs.append([self.labels_to_idx[self.task_spec.none_value]])
            elif self.task_spec.multi_y and not isinstance(
                    label, list) and (not label or pandas.isna(label)):
                outputs.append(self.labels_to_idx[self.task_spec.none_value])
            elif self.task_spec.multi_y and isinstance(label, list):
                processed = [
                    self.labels_to_idx.get(
                        lbl, self.labels_to_idx[self.task_spec.none_value])
                    for lbl in label]
                outputs.append(processed)
            else:
                outputs.append(
                    self.labels_to_idx.get(
                        label, self.labels_to_idx[self.task_spec.none_value]))
        return outputs


class EvaluationDataset(Dataset):
    '''Base class for defining datasets.'''

    def __init__(
        self, spec: DatasetSpec, batch_size: int, profile: Profile,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.batch_size = batch_size
        self.data = spec.load_data(
            extensions_filepath=profile.extensions[
                ExtensionType.DATASETS])
        if max_samples:
            logging.info('Only loading %s out of %s samples',
                         max_samples, len(self.data))
            self.data = self.data[:max_samples]
        self.task_datasets: Dict[str, ReferencesDataset] = {}
        if self.spec.column_reference:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=DeprecationWarning)
                self.process_task_references(profile=profile)

    def process_task_references(self, profile: Profile) -> None:
        self.data['_reference_data_parsed'] = self.data.loc[
            :, [self.spec.column_input, self.spec.column_reference],
        ].apply(
            lambda x: self.spec.get_reference_split_parser(
                parser_spec=profile.parsers[self.spec.reference_split_parser],
                extensions_path=profile.extensions[ExtensionType.PARSERS],
            ).parse(prompt=x.iloc[0], output=x.iloc[1])[0],
            axis=1,
        )
        for task_idx, task in enumerate(self.spec.tasks.values()):
            split_task_data = self.data[task.name] = self.data[
                '_reference_data_parsed'].apply(
                    lambda x: x[task_idx]).tolist()  # pylint: disable=cell-var-from-loop

            # Create a dataset object based on task's type
            if task.task_type == DatasetTaskType.GENERATION:
                cls_ = GenerationReferencesDataset
            elif task.task_type == DatasetTaskType.CLASSIFICATION:
                cls_ = ClassificationReferencesDataset
            else:
                raise ValueError(f'Unknown task_type: {task.task_type}')
            self.task_datasets[task.name] = cls_(
                dataset_spec=self.spec, task_spec=task, data=self.data,
                task_data=split_task_data, profile=profile)

    @property
    def identifier(self) -> str:
        return f'{self.spec.name}@{self.spec.metadata.version}'

    def get_dataloader(self) -> DataLoader:
        '''Get a pytorch dataloader for the dataset.'''
        return DataLoader(
            self, batch_size=self.batch_size, shuffle=False,
            collate_fn=collate_fn if self.spec.column_reference else default_collate)

    def __getitem__(
            self, idx: int) -> Tuple[str, List[Union[str, int, List[int]]]]:
        prompt = self.data[self.spec.column_input][idx]
        if self.spec.column_reference:
            task_labels = [self.task_datasets[task_name][idx]
                           for task_name in self.spec.tasks]
            return prompt, task_labels
        return prompt, []

    def __len__(self) -> int:
        return len(self.data)
