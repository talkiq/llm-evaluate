from typing import Any

import datasets
import pandas

from .custom_dataset import CustomDataset


class HfHubMcqDataset(CustomDataset):
    def __init__(
            self,
            dataset_id: str,
            tasks: list[str],
            split: str = 'test',
            input_column: str = 'input',
            reference_column: str = 'target',
            **_kwargs: Any,
    ) -> None:
        super().__init__()
        self.dataset_id = dataset_id
        self.split = split
        self.tasks = tasks
        self.input_column = input_column
        self.reference_column = reference_column

    def load_subtask(self, task_name: str, n_shots: int) -> pandas.DataFrame:
        df = datasets.load_dataset(
            self.dataset_id, task_name, split=self.split,
        ).to_pandas()
        return self.format_data(df, n_shots)

    def format_data(
        self, df: pandas.DataFrame,
        n_shots: int,
    ) -> pandas.DataFrame:
        formatted = {
            'prompt': [],
            'target': [],
        }
        for idx, main_sample in df.iterrows():
            prompt_bag = []
            n_shot_samples = df.loc[
                ~df.index.isin([idx]), :,
            ].sample(n=n_shots)
            for context_idx, sample in [
                *n_shot_samples.iterrows(), (n_shots, main_sample),
            ]:
                options = {
                    'A': sample['A'],
                    'B': sample['B'],
                    'C': sample['C'],
                    'D': sample['D'],
                }

                if context_idx == n_shots:
                    prompt_bag.append(
                        self.format_sample(sample[self.input_column], options),
                    )
                else:
                    prompt_bag.append(
                        self.format_sample(
                            sample[self.input_column], options,
                            sample[self.reference_column].strip('() ').upper(),
                        ),
                    )
            prompt = '\n'.join(prompt_bag)
            formatted['prompt'].append(prompt)
            formatted['target'].append(
                main_sample[self.reference_column].strip('() '),
            )
        return pandas.DataFrame({
            self.input_column: formatted['prompt'],
            self.reference_column: formatted['target'],
        })

    @staticmethod
    def format_sample(
        question: str,
        options: dict[str, str],
        target: str | None = None,
    ) -> str:
        formatted = question
        for key in sorted(options.keys()):
            formatted += f'\n{key}: {options[key]}'
        if target:
            formatted += f'\nAnswer: {target}'
        return formatted

    def load_all(self, n_shots: int) -> pandas.DataFrame:
        return pandas.concat(
            [
                self.load_subtask(task, n_shots)
                for task in self.tasks
            ],
        ).reset_index(drop=True)
