import abc
import enum
import json
import logging
import os
import pathlib
from dataclasses import dataclass
from typing import Any

import torch
import transformers

from ..helpers.utils import parse_kwargs


class TruncationSide(enum.Enum):
    LEFT = 'left'
    RIGHT = 'right'


@dataclass(kw_only=True)
class ModelLoaderSpec:
    model_type: str
    classname: str
    filename: str
    is_extension: bool = True

    def get_load_path(self, extensions_path: pathlib.Path) -> pathlib.Path:
        if self.is_extension:
            path = extensions_path / self.filename
        else:
            path = pathlib.Path(__file__).parent / self.filename
        assert os.path.exists(path), f'Model not found: {path}'
        return path

    def to_dict(self) -> dict[str, Any]:
        return {
            'model_type': self.model_type,
            'classname': self.classname,
            'filename': self.filename,
            'is_extension': self.is_extension,
        }


@dataclass(kw_only=True)
class ModelSpec:
    model: str
    model_type: ModelLoaderSpec
    tokenizer_args: dict[str, Any] | None = None
    model_load_args: dict[str, Any] | None = None
    model_inference_args: dict[str, Any] | None = None
    add_to_prompt_start: str | None = None
    add_to_prompt_end: str | None = None

    def __init__(
        self,
        model: str,
        model_type: ModelLoaderSpec,
        tokenizer_args: dict[str, Any] | None = None,
        model_load_args: dict[str, Any] | None = None,
        model_inference_args: dict[str, Any] | None = None,
        add_to_prompt_start: str | None = None,
        add_to_prompt_end: str | None = None,
    ) -> None:
        self.model = model
        self.model_type = model_type
        self.tokenizer_args = (
            parse_kwargs(tokenizer_args) if tokenizer_args else {}
        )
        self.model_load_args = (
            parse_kwargs(model_load_args) if model_load_args else {}
        )
        self.model_inference_args = (
            parse_kwargs(model_inference_args) if model_inference_args else {}
        )
        self.add_to_prompt_start = add_to_prompt_start
        self.add_to_prompt_end = add_to_prompt_end

    def to_dict(self) -> dict[str, str]:
        return {
            'model': self.model,
            'model_type': self.model_type.to_dict(),
            'tokenizer_args': self.tokenizer_args,
            'model_load_args': self.model_load_args,
            'model_inference_args': self.model_inference_args,
            'add_to_prompt_start': self.add_to_prompt_start,
            'add_to_prompt_end': self.add_to_prompt_end,
        }

    def __str__(self) -> str:
        return json.dumps(self.to_dict())


class Tokenizer:
    _tokenizer: Any
    truncation_side: TruncationSide
    max_tokens: int

    @abc.abstractmethod
    def tokenize(self, inputs: list[str], **kwargs: Any) -> torch.Tensor:
        """Tokenize input strings to their token ids."""

    @abc.abstractmethod
    def decode(self, outputs: Any) -> list[str]:
        """Decode and detokenize the model output."""

    @abc.abstractmethod
    def get_num_tokens(self, input_: str) -> int:
        """Get the count of tokens in the input string."""

    def __call__(self, inputs: list[str], **kwds: Any) -> torch.Tensor:
        return self.tokenize(inputs, **kwds)


class Model:
    """Base class for evaluating models."""

    # pylint: disable=too-many-instance-attributes
    _tokenizer: Tokenizer

    def __init__(
        self,
        model_or_path: str,
        prompt_start_placeholder: str | None = None,
        prompt_end_placeholder: str | None = None,
        tokenizer_args: dict[str, str] | None = None,
        model_load_args: dict[str, str] | None = None,
        model_inference_args: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.model_or_path = model_or_path
        self.placeholder_start = prompt_start_placeholder
        self.placeholder_end = prompt_end_placeholder
        self.num_placeholder_tokens: int | None = None
        self.placeholder_start_tokens: int | None = None
        self.placeholder_end_tokens: int | None = None
        self.tokenizer_args = tokenizer_args or {}
        self.model_load_args = model_load_args or {}
        self.model_inference_args = model_inference_args or {}
        self.kwargs = kwargs

        # Call the implementation specific initialization function
        self.init()

    @abc.abstractmethod
    def init(self) -> None:
        """Initialize model specific objects."""
        raise NotImplementedError('All models should extend init() method.')

    def _strip_placeholders(self, message: str) -> str:
        def has_placeholder_string(string: str, placeholder: str) -> bool:
            return string.startswith(
                placeholder,
            ) or string.endswith(placeholder)

        if self.placeholder_start and has_placeholder_string(
                message, self.placeholder_start,
        ):
            message = message.replace(self.placeholder_start, '').strip()
        if self.placeholder_end and has_placeholder_string(
                message, self.placeholder_end,
        ):
            message = message.replace(self.placeholder_end, '').strip()
        return message

    def window_with_placeholders(
        self, text: str, prompt: str | None = None,
    ) -> torch.Tensor:
        # pylint: disable=too-complex,too-many-branches,too-many-statements
        # TODO: How would this work for a non-tensor implementation?
        text = self._strip_placeholders(text)
        if prompt:
            prompt = self._strip_placeholders(prompt)
            text = f'{prompt}\n{text}'

        if not (self.placeholder_start or self.placeholder_end):
            self.num_placeholder_tokens = 0
            self.placeholder_start_tokens = 0
            self.placeholder_end_tokens = 0

        # TODO: Perf changes, makes tokenization slow due to multiple runs.
        tokenized_input = self.tokenizer.tokenize([text])
        if (
            isinstance(
                tokenized_input,
                (dict, transformers.tokenization_utils_base.BatchEncoding),
            )
            and 'input_ids' in tokenized_input
        ):
            text_tokens = tokenized_input['input_ids']
        else:
            text_tokens = tokenized_input

        if self.num_placeholder_tokens is None:
            self.placeholder_start_tokens = self.tokenizer.get_num_tokens(
                self.placeholder_start,
            ) if self.placeholder_start else 0

            self.placeholder_end_tokens = self.tokenizer.get_num_tokens(
                self.placeholder_end,
            )if self.placeholder_end else 0

            self.num_placeholder_tokens = (
                self.placeholder_start_tokens
                + self.placeholder_end_tokens
                + 2  # account for newlines in the interpolation below
            )

        # problematic with batch size > 1 because we are not catering for
        if isinstance(text_tokens, list):
            # TODO: Factor for batch size > 1
            if not self.tokenizer.max_tokens >= (
                    len(text_tokens[0]) + self.num_placeholder_tokens
            ):
                overflow = max(
                    0, (
                        len(text_tokens[0]) + self.num_placeholder_tokens
                    ) - self.tokenizer.max_tokens,
                )
                logging.debug('trucating input text by %s tokens', overflow)
                if self.tokenizer.truncation_side == TruncationSide.LEFT:
                    text_tokens[0] = text_tokens[0][overflow:]
                    # only consider full utterances assuming linebreaks as
                    # delimiter
                    tokenized = self.tokenizer.decode(text_tokens)[0]
                    tokenized = '\n'.join(tokenized.splitlines()[1:])
                elif overflow:  # right side & some placeholders
                    text_tokens[0] = text_tokens[0][
                        :(
                            -1 * self.num_placeholder_tokens
                        )
                    ]
                    # only consider full utterances assuming linebreaks as
                    # delimiter
                    tokenized = self.tokenizer.decode(text_tokens)[0]
                    tokenized = '\n'.join(tokenized.splitlines()[:-1])
                else:
                    raise ValueError('Should not reach this...')
            else:
                tokenized = self.tokenizer.decode(text_tokens)[0]
        else:
            if not self.tokenizer.max_tokens >= (
                    text_tokens.size()[1] + self.num_placeholder_tokens
            ):
                overflow = max(
                    0, (
                        text_tokens.size()[1] + self.num_placeholder_tokens
                    ) - self.tokenizer.max_tokens,
                )
                logging.debug('trucating input text by %s tokens', overflow)
                if self.tokenizer.truncation_side == TruncationSide.LEFT:
                    text_tokens = text_tokens[:, overflow:]
                    # only consider full utterances assuming linebreaks as
                    # delimiter
                    tokenized = self.tokenizer.decode(text_tokens)[0]
                    tokenized = '\n'.join(tokenized.splitlines()[1:])
                elif overflow:  # right side & some placeholders
                    text_tokens = text_tokens[:, :(-1 * overflow)]
                    # only consider full utterances assuming linebreaks as
                    # delimiter
                    tokenized = self.tokenizer.decode(text_tokens)[0]
                    tokenized = '\n'.join(tokenized.splitlines()[:-1])
                else:
                    raise ValueError('Should not reach this...')
            else:
                tokenized = self.tokenizer.decode(text_tokens)[0]
        if self.num_placeholder_tokens:
            prompt = (
                f'{self.placeholder_start}\n{tokenized}\n'
                f'{self.placeholder_end}\n'
            )
        else:
            prompt = tokenized
        logging.debug(
            'number of tokens in sample: %s',
            self.tokenizer.get_num_tokens(prompt),
        )
        return prompt

    def preprocess(
        self,
        inputs: list[str],
        prompt: str | None = None,
    ) -> list[str]:
        # logging.debug('raw inputs: %s', inputs)
        inputs = [
            self.window_with_placeholders(text=input_, prompt=prompt)
            for input_ in inputs
        ]
        # logging.debug('preprocessed inputs: %s', inputs)
        return inputs

    @abc.abstractmethod
    def process(self, inputs: list[str], **model_kwargs: Any) -> list[str]:
        """Given a list of input corpora, generate responses."""

    @abc.abstractmethod
    def init_tokenizer(self, model_or_path: str, **kwargs: Any) -> None:
        """Initialize the tokenizer."""

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer
