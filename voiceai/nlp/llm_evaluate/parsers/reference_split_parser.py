import abc
from typing import List
from typing import Tuple
from typing import Union

from .parser import Parser


class ReferenceSplitParser(Parser):
    @abc.abstractmethod
    def parse(
        self, prompt: str, output: Union[str, List[str]],
    ) -> Tuple[List[Union[str, List[str]]], bool]:
        '''Parse expected response so it can be used for evaluation.'''
