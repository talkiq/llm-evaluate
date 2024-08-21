import torch

TORCH_DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DEFAULT_RUNTIME_BENCHMARK = 'runtime-measurements'
DEFAULT_MAX_INPUT_TOKENS = 3000
DEFAULT_MAX_OUTPUT_TOKENS = 1000
