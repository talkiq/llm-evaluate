import random
from typing import List
from typing import Optional

import datasets
import pandas

from ..custom_dataset import CustomDataset

class MsMarco(CustomDataset):
    @classmethod
    def load(cls, n_shots: int) -> pandas.DataFrame:
        df = datasets.load_dataset('ms_marco', 'v2.1', split='validation').to_pandas()
        df['selected_passages'] = df['passages'].apply(
            lambda x: [
                text.strip()
                for use, text in zip(x['is_selected'], x['passage_text'])
                if use
            ]
        )
        df['query'] = df['query'].apply(lambda x: x.strip())
        df['has_answer'] = df['selected_passages'].apply(bool)
        df = df[df['has_answer']]
        df['has_wellformed_answer'] = df['wellFormedAnswers'].apply(lambda x: bool(x.any()))
        df = df[df['has_wellformed_answer']]
        df['wellFormedAnswers'] = df['wellFormedAnswers'].apply(list)
        df['desired_answer'] = df['wellFormedAnswers'].apply(lambda x: random.choice(x).strip())
        return cls.format_data(df, n_shots)

    @classmethod
    def format_data(cls, df: pandas.DataFrame, n_shots: int) -> pandas.DataFrame:
        formatted = {
            'prompt': [],
            'target_single': [],
            'target_original': [],
        }
        for idx, main_sample in df.iterrows():
            prompt_bag = []
            n_shot_samples = df.loc[~df.index.isin([idx]), :].sample(n=n_shots)
            for context_idx, sample in [
                *n_shot_samples.iterrows(), (n_shots, main_sample)
            ]:
                if context_idx == n_shots:
                    prompt_bag.append(
                        cls.format_sample(
                            sample['query'], sample['selected_passages']))
                else:
                    prompt_bag.append(
                        cls.format_sample(
                            sample['query'],
                            sample['selected_passages'],
                            sample['desired_answer']))
            prompt = '\n'.join(prompt_bag)
            formatted['prompt'].append(prompt)
            formatted['target_single'].append(main_sample['desired_answer'])
            formatted['target_original'].append(main_sample['wellFormedAnswers'])
        return pandas.DataFrame(formatted).reset_index(drop=True)

    @staticmethod
    def format_sample(question: str, search_results: List[str],
                      target: Optional[str] = None) -> str:
        results = '\n*'.join(search_results)
        formatted = f'Query: {question}\nResults:\n*{results}'

        if target:
            formatted += f'\nResponse: {target}'
        else:
            formatted += '\nResponse: '
        return formatted

    def load_all(self, n_shots: int) -> pandas.DataFrame:
        return self.load(n_shots)
