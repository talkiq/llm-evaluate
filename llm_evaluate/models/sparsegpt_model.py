import os
from typing import Any

import torch

from .autoclass_model import AutoClassModel


class SparseGptModel(AutoClassModel):
    @staticmethod
    def _load_pretrained_model(
        class_name: str, model_or_path: str, **load_kwargs: Any,
    ) -> torch.nn.Module:
        sparse_path = os.path.expanduser(load_kwargs.pop('sparse_path'))
        model = AutoClassModel._load_pretrained_model(
            class_name,
            sparse_path, **load_kwargs,
        )
        return model
