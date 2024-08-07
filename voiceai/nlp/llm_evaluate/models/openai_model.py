import logging
import os
from typing import Any
from typing import List

import backoff
import openai
import tiktoken

from .model import Model
from .model import Tokenizer
from .model import TruncationSide


class OpenAITokenizer(Tokenizer):

    def __init__(self, model: str, **kwargs: str):
        self._tokenizer = tiktoken.encoding_for_model(model)

        # https://platform.openai.com/docs/models
        if model == 'gpt-3.5-turbo':
            max_input_tokens = 4096
        else:
            max_input_tokens = 4096

        self.max_tokens = kwargs.pop('model_max_length', max_input_tokens)
        self.truncation_side = TruncationSide(
            kwargs.pop('truncation_side', TruncationSide.RIGHT.value))

    def tokenize(self, inputs: List[str], **kwargs: Any) -> List[List[str]]:
        """Tokenize input strings to their token ids."""
        return [self._tokenizer.encode(input_) for input_ in inputs]

    def decode(self, outputs: Any) -> List[str]:
        """Decode and detokenize the model output."""
        return [self._tokenizer.decode(query) for query in outputs]

    def get_num_tokens(self, input_: str) -> int:
        """Get the count of tokens in the input string."""
        tokenized = self._tokenizer.encode(input_)
        return len(tokenized)

    def __call__(self, inputs: List[str], **kwds: Any) -> List[List[str]]:
        return self.tokenize(inputs)


class OpenAIModel(Model):
    def init(self) -> None:
        self.client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        self.model_encoding = tiktoken.encoding_for_model(self.model_or_path)

        self.truncation_side = self.model_load_args.get(
            'truncation_side', TruncationSide.RIGHT.value)
        self.max_output_tokens = self.model_load_args.get(
            'max_new_tokens', 128)
        self.max_input_tokens = self.model_load_args.get('max_input_tokens',
                                                         4096 - self.max_output_tokens)
        self.temperature = self.model_load_args.get('temperature', 0.)
        self.init_tokenizer(self.model_or_path)

    def truncate(self, query: str, max_length: int,
                 truncation_side: str) -> str:
        tokenized = self.model_encoding.encode(query)
        overflow = len(tokenized) - max_length
        if overflow > 0:
            if truncation_side == 'right':
                query = self.model_encoding.decode(tokenized[:-overflow])
            else:
                query = self.model_encoding.decode(tokenized[overflow:])

            logging.warning(
                'token limit reached, truncating from %s.', truncation_side)
        return query

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_time=180,
    )
    def query_openai(self, query: str) -> str:
        query = self.truncate(
            query, self.max_input_tokens, self.truncation_side)

        messages = [{'role': 'user', 'content': query}]

        response = self.client.chat.completions.create(
            model=self.model_or_path,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )
        return response.choices[0].message.content

    def process(self, inputs: List[str], **model_kwargs: Any) -> List[str]:
        responses = [self.query_openai(input_) for input_ in inputs]
        logging.debug(
            'prompt/response:\n>>>%s\n<<<%s',
            inputs[0],
            responses[0])
        return responses

    def init_tokenizer(self, model_or_path: str, **kwargs: Any) -> None:
        self._tokenizer = OpenAITokenizer(model=model_or_path)
