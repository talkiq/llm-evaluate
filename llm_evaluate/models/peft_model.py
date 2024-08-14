import os
from typing import Any

import peft

from .autoclass_model import AutoClassModel


class PeftModel(AutoClassModel):
    @staticmethod
    def _load_pretrained_model(
        class_name: str, model_or_path: str, **load_kwargs: Any,
    ) -> peft.PeftModel:
        peft_path = os.path.expanduser(load_kwargs.pop('peft_path'))

        # TODO: if loading a already quantized model file (e.g., saved by
        #  bitsandbytes), no need to provide the
        #  quantization_config (Jun 9, 2024).
        base_model = AutoClassModel._load_pretrained_model(
            class_name, model_or_path, **load_kwargs,
        )
        return peft.PeftModel.from_pretrained(
            base_model, peft_path, device_map=load_kwargs['device_map'],
        )
