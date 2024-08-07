import abc
import logging
from typing import Any
from typing import List
from typing import Optional
from typing import Union


class Parser:
    '''Simple base class for all parsers.'''

    def __init__(self, **kwargs: Any) -> None:
        logging.debug('unused kwargs: %s', kwargs)

    @abc.abstractmethod
    def parse(
        self, prompt: str, output: Union[Optional[str], List[str]],
    ) -> Union[str, List[str]]:
        '''Parse a response into something that can be used for evaluation.'''
