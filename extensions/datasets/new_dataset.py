import pandas

from llm_evaluate.datasets.custom_dataset import CustomDataset


class NewDataset(CustomDataset):

    def load_all(self, n_shots: int) -> pandas.DataFrame:
        assert n_shots == 1, 'n_shots not supported by this dataset'
        return pandas.DataFrame([
            {
                'prompt': 'How to get to Vancouver from Seattle?',
                'response': '2 hours and 30 minutes',
            },
            {
                'prompt': 'Classifiy the sentiment of the following statement '
                'into positive, negative or neutral. '
                '"Seattle and Vancouver are very similar cities."',
                'response': 'neutral',
            },
            {
                'prompt': 'Convert 4.33 x 100 to scientific notation',
                'response': '4.33e2',
            },
        ])
