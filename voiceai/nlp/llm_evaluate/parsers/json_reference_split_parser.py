import json
import logging
import re
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import omegaconf
import pandas

from .reference_split_parser import ReferenceSplitParser


class JsonReferenceSplitParser(ReferenceSplitParser):
    def __init__(self, keys: List[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert all(key is not None for key in keys)
        self.keys = keys
        self.none_pattern = re.compile(r':\s*None')
        self.none_replacement = ': null'

    @staticmethod
    def clean(text: str) -> str:
        return text.replace('```json', '').replace('```', '')

    def parse(
        self, prompt: str, output: Union[str, List[str]],
    ) -> Tuple[List[Optional[Union[str, List[str]]]], bool]:
        '''Only remove trailing and starting whitespace.'''
        if pandas.isna(output) or not output or not str(output).strip():
            return [None for _ in self.keys], False
        if isinstance(output, omegaconf.dictconfig.DictConfig):
            output_dict = dict(output)
        else:
            output = self.none_pattern.sub(self.none_replacement, output)
            output = self.clean(output)
            try:
                output_dict = json.loads(output.strip())
            except BaseException:  # pylint: disable=bare-except
                logging.warning('could not parse: %s', output)
                return [None for _ in self.keys], False
        output_dict = {str(key): val for key, val in output_dict.items()}
        return [output_dict.get(str(key)) for key in self.keys], True
