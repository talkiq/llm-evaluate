import copy
import logging
from typing import Any
from typing import List

import llama_cpp

from ..helpers.constants import DEFAULT_MAX_INPUT_TOKENS
from ..helpers.constants import DEFAULT_MAX_OUTPUT_TOKENS
from .model import Model
from .model import Tokenizer
from .model import TruncationSide


class LlamaCppTokenizer(Tokenizer):
    def __init__(self, model: llama_cpp.Llama, **kwargs: Any) -> None:
        super().__init__()
        self._tokenizer = model
        self.truncation_side = TruncationSide(
            kwargs.pop('truncation_side', TruncationSide.RIGHT.value))
        self.max_tokens = kwargs.get(
            'model_max_length', DEFAULT_MAX_INPUT_TOKENS)

    def tokenize(self, inputs: List[str], **kwargs: Any) -> List[List[int]]:
        tokenized = [
            self._tokenizer.tokenize(input_.encode('utf-8'), **kwargs)
            for input_ in inputs
        ]
        if self.max_tokens:
            if self.truncation_side == TruncationSide.RIGHT:
                return [tokens[:self.max_tokens] for tokens in tokenized]
            return [tokens[-self.max_tokens:] for tokens in tokenized]
        return tokenized

    def get_num_tokens(self, input_: str) -> int:
        return len(self.tokenize([input_])[0])

    def decode(self, outputs: List[List[int]]) -> List[str]:
        return [
            self._tokenizer.detokenize(output).decode('utf-8')
            for output in outputs
        ]


class LlamaCpp(Model):
    """Source: https://github.com/abetlen/llama-cpp-python"""

    def init(self) -> None:
        self.model_load_args = {
            'n_ctx': DEFAULT_MAX_INPUT_TOKENS + DEFAULT_MAX_OUTPUT_TOKENS,
            **(self.model_load_args or {}),
        }

        _model_load_args = copy.deepcopy(self.model_load_args)
        self.model_load_args = {}
        for arg in _model_load_args:
            if arg not in {
                'n_ctx', 'n_gqa', 'n_gpu_layers', 'n_threads',
                'flash_attn',  # for Qwen2 otherwise gibberish output
            }:
                logging.warning(
                    'model load arg is not supported by LLamaCpp: %s', arg)
                continue
            self.model_load_args[arg] = _model_load_args[arg]

        self.model = llama_cpp.Llama(
            model_path=self.model_or_path,
            **self.model_load_args)
        self.init_tokenizer('', **{**self.model_load_args, **_model_load_args})

    def process(self, inputs: List[str], **model_kwargs: Any) -> List[str]:
        model_kwargs = {
            'echo': False,
            'max_tokens': DEFAULT_MAX_OUTPUT_TOKENS,
            **(model_kwargs or {}),
        }

        _model_kwargs = model_kwargs
        model_kwargs = {}
        for arg, val in _model_kwargs.items():
            if arg not in {
                'max_tokens', 'echo', 'top_k', 'top_p', 'temperature', 'repeat_penalty',
            }:
                logging.warning(
                    'model inference arg is not supported by LLamaCpp: %s', arg)
                continue
            model_kwargs[arg] = val

        responses = [
            self.model(input_, **model_kwargs)['choices'][0]['text']
            for input_ in inputs
        ]

        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            encoded = self.tokenizer.tokenize(inputs)
            actual_prompt = self.tokenizer.decode(encoded)
            logging.debug('number of input tokens: %s', len(encoded[0]))
            logging.debug('prompt/response:\n>>>%s\n<<<%s',
                          actual_prompt[0], responses[0])
        return responses

    def init_tokenizer(self, model_or_path: str, **kwargs: Any) -> None:
        self._tokenizer = LlamaCppTokenizer(self.model, **kwargs)
