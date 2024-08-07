import datasets
import evaluate

# Disable pylint errors for this file
# pylint: disable=unnecessary-pass


# TODO: Add BibTeX citation
_CITATION = """\
@InProceedings{huggingface:module,
title = {A great new module},
authors={huggingface, Inc.},
year={2020}
}
"""

# TODO: Add description of the module here
_DESCRIPTION = """\
This new metric is designed to evaluate the model predictions in terms of x and y.
Share links to documents, papers, or repos that motivated the creation of this metric.
"""


# TODO: Add description of the arguments of the module here
_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of predictions to score. Each prediction
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
Returns:
    first_score: description of the first score,
    second_score: description of the second score,
Examples:
    Examples should be written in doctest format, and should illustrate how
    to use the function.

    >>> my_new_metric = evaluate.load("/path/to/new_metric.py")
    >>> results = my_new_metric.compute(references=['text1', 'paragraph1'],
                                        predictions=['text2', 'paragraph2'])
    >>> print(results)
    {'made_up_metric': 1.0}
"""

# TODO: Define external resources urls if needed
BAD_WORDS_URL = 'http://url/to/external/resource/bad_words.txt'


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class NewMetric(evaluate.Metric):
    """TODO: Rename class and provide a short description of evaluation module."""

    @staticmethod
    def _info():
        # TODO: Specifies the evaluate.EvaluationModuleInfo object
        return evaluate.MetricInfo(
            # This is the description that will appear on the modules page.
            module_type='metric',
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            # supported datasets.Value dtypes are listed here:
            # https://huggingface.co/docs/datasets/v1.10.2/features.html
            features=datasets.Features({
                'predictions': datasets.Value('string'),
                'references': datasets.Value('string'),
            }),
            # add links to papers or repos for the metric
            reference_urls=['http://path.to.reference.url/']
        )

    @staticmethod
    def _download_and_prepare(dl_manager):
        """Optional: download external resources useful to compute the scores"""
        # TODO: Download external resources if needed
        pass

    @staticmethod
    def _compute(predictions, references):
        """Returns the scores"""
        # TODO: Compute the different scores of the module
        equals = 0
        for i, j in zip(predictions, references):
            if i[0] == j[0]:
                equals += 1
        return {
            'made_up_metric': equals / len(predictions),
        }
