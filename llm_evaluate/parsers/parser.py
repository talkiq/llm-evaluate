import abc
import logging
from typing import Any


class Parser:
    """Simple base class for all parsers."""

    def __init__(self, **kwargs: Any) -> None:
        logging.debug('unused kwargs: %s', kwargs)

    @abc.abstractmethod
    def parse(
        self, prompt: str, output: str | None | list[str],
    ) -> str | list[str]:
        """Parse a response into something that can be used for evaluation."""
