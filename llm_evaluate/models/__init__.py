import logging
import pathlib
import warnings
from typing import Any
from typing import Dict
from typing import Optional

from ..helpers.utils import dynamic_load_class
from .model import Model
from .model import ModelLoaderSpec
from .model import ModelSpec


def load_model_catalog(data: dict[str, Any]) -> dict[str, ModelLoaderSpec]:
    logging.debug('loading model catalog...')
    specs = [
        {
            'model_type': model_name,
            **data[model_name],
        }
        for model_name in data
    ]
    if not specs:
        raise ValueError('No models available')
    return {spec['model_type']: ModelLoaderSpec(**spec) for spec in specs}


def load_model_spec(
        spec: dict[str, Any], loaders: dict[str, ModelLoaderSpec],
) -> ModelSpec:
    logging.debug('loading model spec...')
    try:
        spec = {
            **spec,
            'model_type': loaders[spec['model_type']],
        }
    except KeyError as e:
        logging.exception(
            'could not find a model loader named %s',
            spec['model_type'], exc_info=e,
        )
        raise e
    return ModelSpec(**spec)


def load_model(spec: ModelSpec, extensions_path: pathlib.Path) -> Model:
    logging.debug('loading model from spec: %s', spec)
    filepath = spec.model_type.get_load_path(extensions_path)
    filename = spec.model_type.filename.replace('.py', '')
    module_name = f'llm_evaluate.models.{filename}'
    model_cls = dynamic_load_class(
        filepath=str(filepath),
        class_name=spec.model_type.classname,
        module_name=module_name,
    )

    if not model_cls:
        raise ValueError('Model loader not found!')
    return model_cls(  # type: ignore
        model_or_path=spec.model,
        tokenizer_args=spec.tokenizer_args,
        model_load_args=spec.model_load_args,
        prompt_start_placeholder=spec.add_to_prompt_start,
        prompt_end_placeholder=spec.add_to_prompt_end,
    )
