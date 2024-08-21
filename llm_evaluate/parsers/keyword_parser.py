from typing import Any

from .default_parser import DefaultParser


class KeywordParser(DefaultParser):
    def __init__(
        self, none_value: str | None, case_sensitive: bool, multi_y: bool,
        labels: list[str], **kwargs: Any,
    ) -> None:
        super().__init__(
            none_value=none_value, case_sensitive=case_sensitive,
            multi_y=multi_y, **kwargs,
        )
        self.labels = [lbl.strip() for lbl in labels]
        if not self.case_sensitive:
            self.labels = [lbl.lower() for lbl in labels]
        # self.punctuations = re.compile(r'\W|\s')

    def parse(
        self, prompt: str,
        output: str | list[str],
    ) -> str | None | list[str]:
        """Prase LLM response for keywords that can be used for evaluation."""
        if isinstance(output, list):
            output = [out for out in output if out]

        if not output:
            return [] if self.multi_y else self.none_value

        assert len(
            output,
        ) == 1, f'Multiclass classification not tested. Got: {output}'
        output = output[0]

        if not self.case_sensitive:
            output = output.lower()

        # response = self.punctuations.sub(' ', output)
        response = output
        if self.multi_y:
            return [label for label in self.labels if label in response]
        for label in self.labels:
            if label in response:
                return label
        return self.none_value
