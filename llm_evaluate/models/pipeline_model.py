import logging
from typing import Any

import transformers

from ..helpers.constants import DEFAULT_MAX_INPUT_TOKENS
from ..helpers.constants import DEFAULT_MAX_OUTPUT_TOKENS
from ..helpers.torch_utils import parse_torch_dtype
from .autoclass_model import AutoClassTokenizer
from .model import Model


class PipelineModel(Model):
    def init(self) -> None:
        self.model_load_args = {
            'torch_dtype': 'auto',
            'num_beams': 1,
            'temperature': 1.,
            'max_input_tokens': DEFAULT_MAX_INPUT_TOKENS,
            'max_new_tokens': DEFAULT_MAX_OUTPUT_TOKENS,
            **(self.model_load_args or {}),
        }
        self.model_load_args['torch_dtype'] = parse_torch_dtype(
            self.model_load_args['torch_dtype'],
        )

        self.tokenizer_args = {
            **({
                'model_max_length': self.model_load_args.pop(
                    'max_input_tokens',
                ),
            }
                if 'max_input_tokens' in self.model_load_args else {}),
            **(self.tokenizer_args or {}),
        }
        self.init_tokenizer(self.model_or_path, **self.tokenizer_args)
        task = self.model_load_args.pop('task', 'text2text-generation')
        self.pipeline = transformers.pipeline(
            model=self.model_or_path,
            tokenizer=self.tokenizer._tokenizer,  # pylint: disable=protected-access
            task=task,
            framework='pt',
            **self.model_load_args,
        )

    def process(self, inputs: list[str], **model_kwargs: Any) -> list[str]:
        outputs = self.pipeline(inputs, **model_kwargs)
        responses = [out['generated_text'] for out in outputs]
        logging.debug(
            'prompt/response:\n>>>%s\n<<<%s',
            inputs[0],
            responses[0],
        )
        return responses

    def init_tokenizer(self, model_or_path: str, **kwargs: Any) -> None:
        self._tokenizer = AutoClassTokenizer(
            model_or_path,
            **{**self.tokenizer_args, **kwargs},
        )
