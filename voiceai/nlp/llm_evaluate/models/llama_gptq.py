import logging
import os
from typing import Any
from typing import List

import auto_gptq
import transformers

from ..helpers.constants import DEFAULT_MAX_INPUT_TOKENS
from ..helpers.constants import DEFAULT_MAX_OUTPUT_TOKENS
from .autoclass_model import AutoClassTokenizer
from .model import Model


class LlamaGPTQ(Model):
    def init(self) -> None:
        self.model_load_args = {
            'max_new_tokens': DEFAULT_MAX_OUTPUT_TOKENS,
            **(self.model_load_args or {}),
        }
        self.tokenizer_args = {
            'model_max_length': DEFAULT_MAX_INPUT_TOKENS,
            **(self.tokenizer_args or {}),
        }
        self.model = auto_gptq.AutoGPTQForCausalLM.from_quantized(
            self.model_or_path, device_map=self.model_load_args.pop(
                'device_map', 'auto'),
            inject_fused_attention=self.model_load_args.pop('inject_fused_attention', True))
        self.init_tokenizer('', **self.tokenizer_args)

        self.pipeline = transformers.TextGenerationPipeline(
            model=self.model,
            tokenizer=self._tokenizer._tokenizer,  # pylint: disable=protected-access
            **self.model_load_args)

    def process(self, inputs: List[str], **model_kwargs: Any) -> List[str]:
        outputs = [self.pipeline(input_, **model_kwargs)[0]['generated_text'][len(input_):]
                   for input_ in inputs]
        logging.debug('prompt/response:\n>>>%s\n<<<%s', inputs[0], outputs[0])
        return outputs

    def init_tokenizer(self, model_or_path: str, **kwargs: Any) -> None:
        base_model_path = os.path.expanduser(kwargs.pop('base_model_path'))
        self._tokenizer = AutoClassTokenizer(base_model_path, **kwargs)
