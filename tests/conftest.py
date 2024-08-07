import os
import pathlib
import warnings

import pytest

from voiceai.nlp.llm_evaluate.helpers.profile_utils import load_profile
from voiceai.nlp.llm_evaluate.helpers.profile_utils import Profile
from voiceai.nlp.llm_evaluate.models import load_model
from voiceai.nlp.llm_evaluate.models import Model
from voiceai.nlp.llm_evaluate.models import ModelLoaderSpec
from voiceai.nlp.llm_evaluate.models import ModelSpec
from voiceai.nlp.llm_evaluate.models.autoclass_model import AutoClassModel
from voiceai.nlp.llm_evaluate.models.gcp_model import GCPModel
from voiceai.nlp.llm_evaluate.models.openai_model import OpenAIModel
from voiceai.nlp.llm_evaluate.models.pipeline_model import PipelineModel


@pytest.fixture(scope='module')
def model_auto_left() -> AutoClassModel:
    return AutoClassModel('t5-small',
                          tokenizer_args={'truncation_side': 'left'},
                          model_load_args={'max_input_tokens': 3,
                                           'device_map': 'cpu',
                                           'task_type': 'T5ForConditionalGeneration'})


@pytest.fixture(scope='module')
def model_auto_right() -> AutoClassModel:
    return AutoClassModel('t5-small',
                          tokenizer_args={'truncation_side': 'right'},
                          model_load_args={'max_input_tokens': 3,
                                           'device_map': 'cpu',
                                           'task_type': 'T5ForConditionalGeneration'})


@pytest.fixture(scope='module')
def model_pipeline_left() -> PipelineModel:
    return PipelineModel('t5-small',
                         tokenizer_args={'truncation_side': 'left'},
                         model_load_args={'max_new_tokens': 3,
                                          'device_map': 'cpu',
                                          'task': 'text2text-generation'})


@pytest.fixture(scope='module')
def model_pipeline_right() -> PipelineModel:
    return PipelineModel('t5-small',
                         tokenizer_args={'truncation_side': 'right'},
                         model_load_args={'max_input_tokens': 3,
                                          'max_new_tokens': 3,
                                          'device_map': 'cpu',
                                          'task': 'text2text-generation'})


@pytest.fixture(scope='module')
def model_gcp() -> GCPModel:
    return GCPModel('text-bison@001')


@pytest.fixture(scope='module')
def model_openai() -> OpenAIModel:
    return OpenAIModel('gpt-3.5-turbo')


@pytest.fixture(scope='module')
def default_profile() -> Profile:
    path = (
        pathlib.Path(__file__).parents[0] / 'assets' / 'profile.yaml')
    return load_profile(str(path))


@pytest.fixture(scope='module')
def simple_model() -> Model:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=FutureWarning)
        # https://github.com/huggingface/transformers/pull/30620
        return load_model(
            spec=ModelSpec(
                model='google/flan-t5-small',
                model_type=ModelLoaderSpec(
                    model_type='hf-automodel',
                    classname='AutoClassModel',
                    filename='autoclass_model.py',
                    is_extension=False,
                ),
                model_load_args={'task_type': 'T5ForConditionalGeneration'},
            ),
            extensions_path=pathlib.Path(os.getcwd()),
        )


# @pytest.fixture(scope='module')
# def readability_metric() -> ReadabilityMetric:
#     metric_file = (
#         Path(__file__).parents[1] / 'voiceai' / 'nlp' / 'llm_evaluate' /
#         'custom_metrics' / 'readability_metric.py'
#     )
#     return evaluate.load(str(metric_file))


# @pytest.fixture(scope='module')
# def metrics_summary_data() -> Dict[str, Any]:
#     path = (
#         pathlib.Path(__file__).parents[0] / 'assets' / 'summaries.csv')
#     return pandas.read_csv(str(path)).to_dict(orient='list')
