import omegaconf
import pandas

from .reference_split_parser import ReferenceSplitParser


class DefaultReferenceSplitParser(ReferenceSplitParser):
    def parse(
        self, prompt: str, output: str | list[str],
    ) -> tuple[list[str | list[str]], bool]:
        """Only remove trailing and starting whitespace."""
        if isinstance(output, (list, omegaconf.listconfig.ListConfig)):
            output = [str(ref).strip() for ref in output]
        elif not isinstance(
            output, (
                list, omegaconf.listconfig.ListConfig,
            ),
        ) and (
                pandas.isna(output) or not str(output).strip()
        ):
            return [None], True
        else:
            output = str(output).strip()
        return [output], True
