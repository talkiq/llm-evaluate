import logging
import os
import pathlib
from dataclasses import dataclass
from typing import Any
from typing import Dict

from ..helpers.utils import camel_to_snake
from ..helpers.utils import dynamic_load_class
from .parser import Parser


@dataclass(kw_only=True)
class ParserSpec:
    name: str
    filename: str
    is_extension: bool = True

    def get_load_path(self, extensions_path: str) -> pathlib.Path:
        path = extensions_path if self.is_extension else pathlib.Path(
            __file__).parent
        path = path / self.filename
        assert os.path.exists(path), f'Parser {self.name} not found in {extensions_path}'
        return path


def load_parser_catalog(data: Dict[str, Any]) -> Dict[str, ParserSpec]:
    logging.debug('loading parser catalog...')
    specs = [
        {
            'name': parser_name,
            **data[parser_name],
        }
        for parser_name in data
    ]
    if not specs:
        raise ValueError('No parsers available')
    return {spec['name']: ParserSpec(**spec) for spec in specs}


def load_parser(spec: ParserSpec, extensions_path: pathlib.Path,
                **kwargs: Any) -> Parser:
    logging.debug('loading parser: %s with kwargs: %s from: %s',
                  spec.name, kwargs, spec.get_load_path(extensions_path=extensions_path))
    module_name = f'voiceai.nlp.llm_evaluate.parsers.{camel_to_snake(spec.name)}'
    parser = dynamic_load_class(
        filepath=spec.get_load_path(extensions_path=extensions_path),
        class_name=spec.name, module_name=module_name)(**kwargs)
    assert isinstance(parser, Parser)
    return parser
