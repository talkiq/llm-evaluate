import abc

from .parser import Parser


class ReferenceSplitParser(Parser):
    @abc.abstractmethod
    def parse(
        self, prompt: str, output: str | list[str],
    ) -> tuple[list[str | list[str]], bool]:
        """Parse expected response so it can be used for evaluation."""
