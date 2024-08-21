# pylint: disable=line-too-long
import warnings
from typing import Any

import pytest

from llm_evaluate.models.autoclass_model import AutoClassModel
from llm_evaluate.models.pipeline_model import PipelineModel


@pytest.mark.parametrize(
    'model_name, text, expected',
    [
        ('model_auto_right', 'Joan is an applied scientist.', 'Joan is an applied'),
        (
            'model_auto_left', 'Joan is an applied scientist.',
            'an applied scientist.',
        ),
        (
            'model_pipeline_right', 'Joan is an applied scientist.',
            'Joan is an applied',
        ),
        (
            'model_pipeline_left', 'Joan is an applied scientist.',
            'an applied scientist.',
        ),
    ],
)
def test_truncation(
    model_name: str, text: str,
    expected: str, request: Any,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=FutureWarning)
        # https://github.com/huggingface/transformers/pull/30620
        model = request.getfixturevalue(model_name)
    tokenized = model.tokenizer(text, max_length=5)
    output = model.tokenizer.decode(tokenized['input_ids'])[0]
    assert expected == output


def test_defaults() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=FutureWarning)
        # https://github.com/huggingface/transformers/pull/30620
        for model in (
            AutoClassModel(
                't5-small',
                model_load_args={
                    'task_type': 'T5ForConditionalGeneration',
                },
            ),
            PipelineModel(
                't5-small',
                model_load_args={
                    'task': 'text2text-generation',
                },
            ),
        ):
            assert model.tokenizer._tokenizer.truncation_side == 'right'  # pylint: disable=protected-access
            assert model.tokenizer._tokenizer.model_max_length == 3000  # pylint: disable=protected-access


@pytest.mark.parametrize(
    'text',
    [
        'Joan is an applied scientist.',
        'The quick brown fox jumped over the lazy fox.',
        'Lorem ipsum dolor sit amet, consectetur adipiscing elit. '
        'Phasellus imperdiet quam vitae rhoncus accumsan.',
    ],
)
def test_tokenizer_ops(text: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=FutureWarning)
        # https://github.com/huggingface/transformers/pull/30620
        for model in (
            AutoClassModel(
                't5-small',
                model_load_args={
                    'task_type': 'T5ForConditionalGeneration',
                },
            ),
            PipelineModel(
                't5-small',
                model_load_args={
                    'task': 'text2text-generation',
                },
            ),
        ):
            tokenized = model.tokenizer.tokenize([text])
            assert tokenized['input_ids'].size(
            )[1] == model.tokenizer.get_num_tokens(text)


@pytest.mark.parametrize(
    'model_name, text, expected',
    [
        ('model_auto_right', 'Joan is an applied scientist.', 'Joan ist ein'),
        ('model_auto_left', 'Joan is an applied scientist.', 'Joan ist ein'),
        ('model_pipeline_right', 'Joan is an applied scientist.', 'Joan ist ein'),
        (
            'model_pipeline_left', 'Joan is an applied scientist.',
            'Joan ist ein',
        ),
    ],
)
def test_process(
    model_name: str, text: str, expected: str,
    request: Any,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=FutureWarning)
        # https://github.com/huggingface/transformers/pull/30620
        model = request.getfixturevalue(model_name)
    output = model.process(text, max_new_tokens=3)[0]
    assert output == expected


@pytest.mark.parametrize(
    'model_name, truncation_side, text, expected',
    [
        (
            'model_gcp',
            'left',
            'In this tutorial, you will learn how to use the PaLM API.',
            'to use the PaLM API.',
        ),
        (
            'model_gcp',
            'right',
            'In this tutorial, you will learn how to use the PaLM API.',
            'In this tutorial, you will',
        ),
        (
            'model_openai',
            'left',
            'In this tutorial, you will learn how to use the PaLM API.',
            ' the PaLM API.',
        ),
        (
            'model_openai',
            'right',
            'In this tutorial, you will learn how to use the PaLM API.',
            'In this tutorial, you',
        ),
    ],
)
def test_truncation_gcp_openai(
    model_name: str, truncation_side: str,
    text: str, expected: str, request: Any,
) -> None:
    model = request.getfixturevalue(model_name)
    output = model.truncate(
        text, max_length=5, truncation_side=truncation_side,
    )
    assert expected == output


@pytest.mark.parametrize(
    'model_name, truncation_side, max_tokens',
    [
        ('model_gcp', 'right', 8192),
        ('model_openai', 'right', 3968),
    ],
)
def test_defaults_gcp_openai(
    model_name: str, truncation_side: str,
    max_tokens: int, request: Any,
) -> None:
    model = request.getfixturevalue(model_name)
    assert model.truncation_side == truncation_side
    assert model.max_input_tokens == max_tokens
