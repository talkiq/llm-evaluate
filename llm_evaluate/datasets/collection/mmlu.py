import datasets
import pandas

from ..custom_dataset import CustomDataset


class MMLU(CustomDataset):
    dataset_id = 'lukaemon/mmlu'
    tasks = [
        'high_school_european_history',
        'business_ethics',
        'clinical_knowledge',
        'medical_genetics',
        'high_school_us_history',
        'high_school_physics',
        'high_school_world_history',
        'virology',
        'high_school_microeconomics',
        'econometrics',
        'college_computer_science',
        'high_school_biology',
        'abstract_algebra',
        'professional_accounting',
        'philosophy',
        'professional_medicine',
        'nutrition',
        'global_facts',
        'machine_learning',
        'security_studies',
        'public_relations',
        'professional_psychology',
        'prehistory',
        'anatomy',
        'human_sexuality',
        'college_medicine',
        'high_school_government_and_politics',
        'college_chemistry',
        'logical_fallacies',
        'high_school_geography',
        'elementary_mathematics',
        'human_aging',
        'college_mathematics',
        'high_school_psychology',
        'formal_logic',
        'high_school_statistics',
        'international_law',
        'high_school_mathematics',
        'high_school_computer_science',
        'conceptual_physics',
        'miscellaneous',
        'high_school_chemistry',
        'marketing',
        'professional_law',
        'management',
        'college_physics',
        'jurisprudence',
        'world_religions',
        'sociology',
        'us_foreign_policy',
        'high_school_macroeconomics',
        'computer_security',
        'moral_scenarios',
        'moral_disputes',
        'electrical_engineering',
        'astronomy',
        'college_biology',
    ]

    @classmethod
    def load_subtask(cls, task_name: str, n_shots: int) -> pandas.DataFrame:
        if task_name not in cls.tasks:
            raise ValueError(f'Task {task_name} not found in MMLU')
        df = datasets.load_dataset(
            cls.dataset_id,
            task_name,
            split='test',
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
                options = {
                    'A': sample['A'],
                    'B': sample['B'],
                    'C': sample['C'],
                    'D': sample['D'],
                }

                if context_idx == n_shots:
                    prompt_bag.append(
                        cls.format_sample(sample['input'], options),
                    )
                else:
                    prompt_bag.append(
                        cls.format_sample(
                            sample['input'], options,
                            sample['target'].strip('() ').upper(),
                        ),
                    )
            prompt = '\n'.join(prompt_bag)
            formatted['prompt'].append(prompt)
            formatted['target'].append(main_sample['target'].strip('() '))
        return pandas.DataFrame(formatted)

    @staticmethod
    def format_sample(
            question: str, options: dict[str, str], target: str | None = None,
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
