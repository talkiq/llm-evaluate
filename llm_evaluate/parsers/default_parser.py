from typing import Any

import pandas

from .parser import Parser


class DefaultParser(Parser):
    def __init__(
        self,
        none_value: str | None,
        case_sensitive: bool,
        multi_y: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.case_sensitive = case_sensitive
        self.multi_y = multi_y
        if none_value:
            if not self.case_sensitive:
                self.none_value = none_value.strip().lower()
            else:
                self.none_value = none_value.strip()
        else:
            self.none_value = None

    def parse(
        self, prompt: str,
        output: str | list[str],
    ) -> str | None | list[str]:
        """Only remove trailing and starting whitespace."""
        if not isinstance(output, list):
            output = str(output).strip()
            if self.multi_y:
                output = [output]
        if (
            (isinstance(output, list) and not output)
            or (not isinstance(output, list) and pandas.isna(output))
        ):
            return [self.none_value] if self.multi_y else self.none_value

        if isinstance(output, list) and self.multi_y:
            if not self.case_sensitive:
                return [str(out).lower() for out in output]
            return output
        if isinstance(output, list) and not self.multi_y:
            assert len(output) == 1
            output = output[0]
        return str(output) if self.case_sensitive else str(output).lower()
