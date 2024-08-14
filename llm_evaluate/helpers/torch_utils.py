import torch

from .constants import TORCH_DEVICE


def parse_torch_dtype(dtype_str: str | None) -> torch.dtype:
    if dtype_str == 'auto':
        return dtype_str

    torch_dtype = 'auto' if TORCH_DEVICE == 'cpu' else torch.float16
    if dtype_str == 'bfloat16':
        torch_dtype = torch.bfloat16
    elif dtype_str == 'float16':
        torch_dtype = torch.float16
    elif dtype_str == 'float32':
        torch_dtype = torch.float32
    return torch_dtype
