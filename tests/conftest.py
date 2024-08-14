import os
import pathlib
import warnings

import pytest

from llm_evaluate.helpers.profile_utils import ConfigurationProfile
from llm_evaluate.helpers.profile_utils import load_profile
from llm_evaluate.models import load_model
from llm_evaluate.models import Model
from llm_evaluate.models import ModelLoaderSpec
from llm_evaluate.models import ModelSpec
from llm_evaluate.models.autoclass_model import AutoClassModel
from llm_evaluate.models.gcp_model import GCPModel
from llm_evaluate.models.openai_model import OpenAIModel
from llm_evaluate.models.pipeline_model import PipelineModel


@pytest.fixture(scope='module')
def model_auto_left() -> AutoClassModel:
    return AutoClassModel(
        't5-small',
        tokenizer_args={'truncation_side': 'left'},
        model_load_args={
            'max_input_tokens': 3,
            'device_map': 'cpu',
            'task_type': 'T5ForConditionalGeneration',
        },
    )


@pytest.fixture(scope='module')
def model_auto_right() -> AutoClassModel:
    return AutoClassModel(
        't5-small',
        tokenizer_args={'truncation_side': 'right'},
        model_load_args={
            'max_input_tokens': 3,
            'device_map': 'cpu',
            'task_type': 'T5ForConditionalGeneration',
        },
    )


@pytest.fixture(scope='module')
def model_pipeline_left() -> PipelineModel:
    return PipelineModel(
        't5-small',
        tokenizer_args={'truncation_side': 'left'},
        model_load_args={
            'max_new_tokens': 3,
            'device_map': 'cpu',
            'task': 'text2text-generation',
        },
    )


@pytest.fixture(scope='module')
def model_pipeline_right() -> PipelineModel:
    return PipelineModel(
        't5-small',
        tokenizer_args={'truncation_side': 'right'},
        model_load_args={
            'max_input_tokens': 3,
            'max_new_tokens': 3,
            'device_map': 'cpu',
            'task': 'text2text-generation',
        },
    )


@pytest.fixture(scope='module')
def model_gcp() -> GCPModel:
    return GCPModel('text-bison@001')


@pytest.fixture(scope='module')
def model_openai() -> OpenAIModel:
    return OpenAIModel('gpt-3.5-turbo')


@pytest.fixture(scope='module')
def default_profile() -> ConfigurationProfile:
    path = (
        pathlib.Path(__file__).parents[0] / 'assets' / 'profile.yaml'
    )
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
