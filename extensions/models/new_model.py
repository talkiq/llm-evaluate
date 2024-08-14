import logging
from typing import Any

import torch
import transformers

from llm_evaluate.helpers.constants import TORCH_DEVICE
from llm_evaluate.models.autoclass_model import AutoClassModel


class SequenceClassificationAutoClassModel(AutoClassModel):
    def __init__(
        self, model_or_path: str, class_name: str | None = None,
        tokenizer_args: dict[str, str] | None = None,
        model_load_args: dict[str, str] | None = None,
        prompt_start_placeholder: str | None = None,
        prompt_end_placeholder: str | None = None,
    ) -> None:
        if class_name:
            raise ValueError(
                'Cannot specify class name for SequenceClassification models.',
            )
        super().__init__(
            model_or_path=model_or_path,
            prompt_start_placeholder=prompt_start_placeholder,
            prompt_end_placeholder=prompt_end_placeholder,
            class_name=class_name,
            tokenizer_args=tokenizer_args,
            model_load_args=model_load_args,
        )

    @staticmethod
    def _load_pretrained_model(
        class_name: str | None, model_or_path: str, **load_kwargs: Any,
    ) -> torch.nn.Module:
        return transformers.AutoModelForSequenceClassification.from_pretrained(
            model_or_path,
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
            outputs = self.model(
                **encoded,
                **model_kwargs,
            )
        labels = [
            self.model.config.id2label[idx]
            for idx in torch.argmax(
                outputs.logits.cpu(), dim=-1,
            ).numpy().tolist()
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
                labels[0],
            )
        return labels
