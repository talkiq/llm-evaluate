# pylint: disable=global-variable-undefined
import logging
import multiprocessing.managers
import os
import pathlib
from typing import Any

import torch

from ..models import load_model
from ..models import ModelSpec


# initialize worker processes
def init_worker(
    l_shared_cuda_devices: multiprocessing.managers.ListProxy,
    model_spec: ModelSpec, extensions_path: pathlib.Path,
) -> None:
    global model_obj
    global torch_device  # may not be necessary

    device_ids = l_shared_cuda_devices.pop()

    # may not work unless setting them before import torch
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
        str(i) for i in device_ids
    )  # re-maps torch device ids
    os.environ['WORLD_SIZE'] = str(len(device_ids))

    # torch.cuda.current_device()
    logging.info(
        'init worker pid=%s with cuda devices: %s',
        os.getpid(),
        device_ids,
    )
    # for sending tokenized input vectors (torch device_id re-mapped)
    torch_device = 'cuda:0'

    model_obj = load_model(spec=model_spec, extensions_path=extensions_path)


def worker(runner_obj: Any, batch: Any, model_kwargs: Any) -> Any:
    """Run the model on each sample and return the results."""
    torch.cuda.empty_cache()

    inputs, _ = batch

    global model_obj  # pylint: disable=global-variable-not-assigned
    global torch_device  # pylint: disable=global-variable-not-assigned

    prompt_inputs = model_obj.preprocess(
        inputs=list(inputs),
        prompt=runner_obj.benchmark_dataset_spec.prompt,
    )
    raw_outputs = model_obj.process(
        inputs=prompt_inputs,
        **model_kwargs,
    )

    input_tokens = [
        model_obj.tokenizer.get_num_tokens(
            text,
        ) for text in inputs
    ]
    output_tokens = [
        model_obj.tokenizer.get_num_tokens(text) for text in
        raw_outputs
    ]

    return prompt_inputs, raw_outputs, input_tokens, output_tokens
