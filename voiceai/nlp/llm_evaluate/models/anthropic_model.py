import logging
from typing import Any
from typing import List

import backoff
from anthropic import AnthropicVertex

from .model import Model
from .model import Tokenizer
from .model import TruncationSide


class AnthropicTokenizer(Tokenizer):

    def __init__(self, _model: str, **kwargs: str):
        self._tokenizer = lambda x: x
        self.max_tokens = kwargs['max_input_tokens']
        self.truncation_side = TruncationSide(
            kwargs.pop('truncation_side', TruncationSide.RIGHT.value))

    def tokenize(self, inputs: List[str], **_kwargs: Any) -> List[List[str]]:
        """Tokenize input strings to their token ids."""
        return inputs

    def decode(self, outputs: Any) -> List[str]:
        """Decode and detokenize the model output."""
        return outputs

    def get_num_tokens(self, input_: str) -> int:
        """Get the count of tokens in the input string."""
        return len(input_)

    def __call__(self, inputs: List[str], **kwds: Any) -> List[List[str]]:
        return self.tokenize(inputs)


class AnthropicModel(Model):
    def init(self) -> None:
        self.truncation_side = self.model_load_args.get(
            'truncation_side', TruncationSide.RIGHT.value)
        self.model_name = self.model_or_path
        self.max_input_tokens = self.model_load_args.get(
            'max_input_tokens', 200000)
        self.max_new_tokens = self.model_load_args.get('max_new_tokens', 4096)
        self.anthropic_model = AnthropicVertex(
            project_id='talkiq-data', region='us-central1')
        self.init_tokenizer(self.model_or_path)

    @classmethod
    def truncate(cls, query: str, max_length: int,
                 truncation_side: str) -> str:
        # TODO: Max length based on token limit, this is crude
        overflow = len(query.split()) - max_length
        if overflow > 0:
            if truncation_side == 'right':
                query = ' '.join(query.split()[:-overflow])
            else:
                query = ' '.join(query.split()[overflow:])

            logging.warning(
                'token limit reached, truncating from %s.', truncation_side)
        return query

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_time=180,
    )
    def query_model(self, query: str, max_tokens: int) -> str:
        query = self.truncate(
            query, self.max_input_tokens, self.truncation_side)
        response = self.anthropic_model.messages.create(
            max_tokens=max_tokens,
            messages=[{'role': 'user', 'content': query}],
            model=self.model_name,
        )
        return response.content[0].text

    def process(self, inputs: List[str], **model_kwargs: Any) -> List[str]:
        responses = [
            self.query_model(
                input_,
                max_tokens=model_kwargs.get(
                    'max_new_tokens', self.max_new_tokens),
            )
            for input_ in inputs
        ]
        logging.debug('prompt/response:\n>>>%s\n<<<%s',
                      inputs[0], responses[0])
        return responses

    def init_tokenizer(self, model_or_path: str, **kwargs: Any) -> None:
        self._tokenizer = AnthropicTokenizer(
            model_or_path,
            **{**kwargs, **{'max_input_tokens': self.max_input_tokens}})
