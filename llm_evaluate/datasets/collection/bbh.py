import datasets
import pandas

from ..custom_dataset import CustomDataset


class BBH(CustomDataset):
    dataset_id = 'lukaemon/bbh'
    tasks = [
        'boolean_expressions',
        'causal_judgement',
        'date_understanding',
        'disambiguation_qa',
        'dyck_languages',
        'formal_fallacies',
        'geometric_shapes',
        'hyperbaton',
        'logical_deduction_five_objects',
        'logical_deduction_seven_objects',
        'logical_deduction_three_objects',
        'movie_recommendation',
        'multistep_arithmetic_two',
        'navigate',
        'object_counting',
        'penguins_in_a_table',
        'reasoning_about_colored_objects',
        'ruin_names',
        'salient_translation_error_detection',
        'snarks',
        'sports_understanding',
        'temporal_sequences',
        'tracking_shuffled_objects_five_objects',
        'tracking_shuffled_objects_seven_objects',
        'tracking_shuffled_objects_three_objects',
        'web_of_lies',
        'word_sorting',
    ]

    @classmethod
    def load_subtask(cls, task_name: str, n_shots: int) -> pandas.DataFrame:
        if task_name not in cls.tasks:
            raise ValueError(f'Task {task_name} not found in BBH')
        df = datasets.load_dataset(
            cls.dataset_id, task_name, split='test',
        ).to_pandas()
        return cls.format_data(df, n_shots)

    @classmethod
    def format_data(
        cls, df: pandas.DataFrame,
        n_shots: int,
    ) -> pandas.DataFrame:
        formatted = {
            'prompt': [],
            'target': [],
        }
        for idx, main_sample in df.iterrows():
            prompt_bag = []
            n_shot_samples = df.loc[~df.index.isin([idx]), :].sample(n=n_shots)
            for context_idx, sample in [
                *n_shot_samples.iterrows(), (n_shots, main_sample),
            ]:
                if context_idx == n_shots:
                    prompt_bag.append(
                        cls.format_sample(sample['input']),
                    )
                else:
                    prompt_bag.append(
                        cls.format_sample(
                            sample['input'],
                            sample['target'].strip('() ').upper(),
                        ),
                    )
            prompt = '\n'.join(prompt_bag) + '\nAnswer: '
            formatted['prompt'].append(prompt)
            formatted['target'].append(main_sample['target'].strip('() '))
        return pandas.DataFrame(formatted)

    @staticmethod
    def format_sample(question: str, target: str | None = None) -> str:
        formatted = question
        if target:
            formatted += f'\nAnswer: {target}'
        return formatted

    def load_all(self, n_shots: int) -> pandas.DataFrame:
        return pandas.concat([
            self.load_subtask(task, n_shots)
            for task in self.tasks
        ]).reset_index(drop=True)
