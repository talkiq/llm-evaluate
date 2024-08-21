import importlib
import logging
from typing import Any

import torch
import transformers

from ..helpers.constants import DEFAULT_MAX_INPUT_TOKENS
from ..helpers.constants import DEFAULT_MAX_OUTPUT_TOKENS
from ..helpers.constants import TORCH_DEVICE
from ..helpers.torch_utils import parse_torch_dtype
from .model import Model
from .model import Tokenizer
from .model import TruncationSide


class AutoClassTokenizer(Tokenizer):
    def __init__(self, model_or_path: str, **kwargs: Any) -> None:
        super().__init__()
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_or_path, **kwargs,
        )
        self.truncation_side = TruncationSide(
            kwargs.pop('truncation_side', TruncationSide.RIGHT.value),
        )
        self.max_tokens = kwargs.pop(
            'model_max_length', DEFAULT_MAX_INPUT_TOKENS,
        )
        self.encoding_args = kwargs

    def tokenize(self, inputs: list[str], **kwargs: Any) -> torch.Tensor:
        return self._tokenizer(
            inputs, **{
                'return_tensors': 'pt',
                **self.encoding_args,
                **kwargs,
            },
        )

    def get_num_tokens(self, input_: str) -> int:
        encoded = self._tokenizer([input_], **self.encoding_args)['input_ids']
        return len(encoded[0])

    def decode(self, outputs: list[str]) -> list[str]:
        return self._tokenizer.batch_decode(  # type: ignore
            outputs, skip_special_tokens=True,
        )


class AutoClassModel(Model):
    def init(self) -> None:
        self.model_load_args = {
            'torch_dtype': 'auto',
            'max_input_tokens': DEFAULT_MAX_INPUT_TOKENS,
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
        self.class_name = self.model_load_args.pop(
            'task_type', 'AutoModelForCausalLM',
        )
        self.model = self._load_pretrained_model(
            self.class_name, self.model_or_path,
            **self.model_load_args,
        )

        if isinstance(self.model, torch.nn.Module):
            total_params = sum(p.numel() for p in self.model.parameters())
            logging.info(
                'loaded model with %sB parameters', round(
                    total_params / 1e9, 1,
                ),
            )

    @staticmethod
    def _load_pretrained_model(
        class_name: str, model_or_path: str, **load_kwargs: Any,
    ) -> torch.nn.Module:
        if TORCH_DEVICE == 'cpu':
            offload_kwards = {
                'offload_folder': 'offload',
                'offload_state_dict': True,
            }
        else:
            offload_kwards = {}

        module = importlib.import_module('.', package='transformers')
        model_class = getattr(module, class_name)
        return model_class.from_pretrained(
            model_or_path,
            **offload_kwards,
            **load_kwargs,
        )

    def process(
        self, inputs: list[str],
        **model_kwargs: Any,
    ) -> list[str]:
        padding = bool(len(inputs) > 1)
        # the default cuda:0 works for both # workers = 1 and # workers > 1,
        # because for # workers > 1, "CUDA_VISIBLE_DEVICES" re-maps device ids
        encoded = self.tokenizer.tokenize(inputs, padding=padding).to(
            TORCH_DEVICE,
        )

        self.model.train(False)
        with torch.no_grad():
            outputs = self.model.generate(
                **encoded,
                **{
                    'num_beams': 1,
                    'temperature': 1.,
                    'max_new_tokens': DEFAULT_MAX_OUTPUT_TOKENS,
                    **model_kwargs,
                },
            ).to('cpu')

        text_outputs = self.tokenizer.decode(outputs)
        responses = [
            text[0] if isinstance(text, (list, tuple)) else text
            for text in text_outputs
        ]

        # Hack for dealing with CausalLM models
        if self.class_name.endswith('CausalLM'):
            responses = [
                resp[len(inp):]
                for inp, resp in zip(inputs, responses)
            ]

        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            actual_prompt = self.tokenizer.decode(encoded['input_ids'])
            logging.debug(
                'number of input tokens: %s',
                encoded['input_ids'].size()[1],
            )
            logging.debug(
                'prompt/response:\n>>>%s\n<<<%s',
                actual_prompt[0],
                responses[0],
            )
        return responses

    def init_tokenizer(self, model_or_path: str, **kwargs: Any) -> None:
        self._tokenizer = AutoClassTokenizer(
            model_or_path,
            **{**self.tokenizer_args, **kwargs},
        )
