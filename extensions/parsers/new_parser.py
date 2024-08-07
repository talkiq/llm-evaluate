from typing import Any

from voiceai.nlp.llm_evaluate.parsers import Parser

class NewParser(Parser):
    def parse(self, prompt: str, output: Any) -> str:
        # do any processsing here
        return output
