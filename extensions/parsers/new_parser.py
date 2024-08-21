from llm_evaluate.parsers import Parser


class NewParser(Parser):
    def parse(self, prompt: str, output: str) -> str:
        # do any processsing here
        return output
