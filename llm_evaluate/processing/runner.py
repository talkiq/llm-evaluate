import concurrent.futures
import logging
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

import pandas
import torch
from tqdm import tqdm

from ..datasets import EvaluationDataset
from ..datasets import load_dataset
from ..datasets.spec import DatasetTaskSpec
from ..helpers.configuration_profile import ConfigurationProfile
from ..helpers.configuration_profile import ExtensionType
from ..models import Model
from .benchmark import BenchmarkDatasetSpec
from .mp_worker import worker


class Runner:
    """Run a model and get results."""

    def __init__(
        self, batch_size: int, dataset: EvaluationDataset,
        benchmark_dataset_spec: BenchmarkDatasetSpec,
    ) -> None:
        self.batch_size = batch_size
        self.dataset = dataset
        self.benchmark_dataset_spec = benchmark_dataset_spec

    def extension_fn(
        self,
        list_handle: list[Any],
        items: list[Any],
        task_spec: DatasetTaskSpec,
    ) -> None:
        if len(items) == self.batch_size:
            if (
                task_spec.multi_y
                and self.batch_size == 1
                and not isinstance(items[0], (list, tuple))
            ):
                list_handle.append(items)
            else:
                list_handle.extend(items)
        elif len(items) < self.batch_size:
            if isinstance(items, str):
                list_handle.append(items)
            else:
                list_handle.extend(items)
        else:
            list_handle.append([items])

    def inference_loop(
        self, executor: Model | concurrent.futures.Executor,
        **model_kwargs: Any,
    ) -> Any:
        # pylint: disable=too-many-locals
        if isinstance(executor, Model):
            logging.debug('>>> model executor: %s', type(executor))
            for batch in tqdm(self.dataset.get_dataloader()):
                inputs, task_references = batch
                prompt_inputs = executor.preprocess(
                    inputs=list(inputs),
                    prompt=self.benchmark_dataset_spec.prompt,
                )
                raw_outputs = executor.process(
                    inputs=prompt_inputs,
                    **model_kwargs,
                )

                input_tokens = [
                    executor.tokenizer.get_num_tokens(text) for text in
                    inputs
                ]
                output_tokens = [
                    executor.tokenizer.get_num_tokens(text) for text
                    in raw_outputs
                ]
                yield (
                    inputs, task_references, prompt_inputs, raw_outputs,
                    input_tokens, output_tokens,
                )

        elif isinstance(executor, concurrent.futures.Executor):
            logging.debug('>>> cf executor: %s', type(executor))
            d_batch = dict(enumerate(self.dataset.get_dataloader()))

            with tqdm(total=len(d_batch)) as pbar:
                futures = {
                    executor.submit(
                        worker, self, batch, model_kwargs,
                    ): batch_idx
                    for batch_idx, batch in d_batch.items()
                }

                batch_result = {}

                for future in concurrent.futures.as_completed(futures):
                    batch_idx = futures[future]
                    batch_result[batch_idx] = future.result()
                    pbar.update(1)

            for batch_idx, result in sorted(
                batch_result.items(),
                key=lambda x: x[0],
            ):
                (
                    prompt_inputs, raw_outputs,
                    input_tokens, output_tokens,
                ) = result
                batch = d_batch[batch_idx]
                inputs, task_references = batch
                yield (
                    inputs, task_references, prompt_inputs,
                    raw_outputs, input_tokens, output_tokens,
                )

        else:
            raise TypeError(
                'Incorrect executor type. '
                'Must be: Model | concurrent.futures.Executor',
            )

    def _inference(
        self, executor: Model | concurrent.futures.Executor,
        profile: ConfigurationProfile, **model_kwargs: Any,
    ) -> pandas.DataFrame:
        # pylint: disable=too-many-locals
        collated_results = defaultdict(list)
        for out_inference in self.inference_loop(executor, **model_kwargs):
            (
                inputs, task_references, prompt_inputs, raw_outputs,
                input_tokens, output_tokens,
            ) = out_inference

            # Split the output of the model into different substrings
            # for each task
            task_split_outputs = [
                self.dataset.spec.get_reference_split_parser(
                    parser_spec=profile.parsers[
                        self.dataset.spec.reference_split_parser
                    ],
                    extensions_path=profile.extensions[
                        ExtensionType.PARSERS
                    ],
                ).parse(
                    prompt=input_, output=out,
                )
                for input_, out in zip(inputs, raw_outputs)
            ]
            output_parsing_successes = [
                x[1] for x in task_split_outputs
                for _ in self.dataset.spec.tasks.items()
            ]
            task_split_outputs = [x[0] for x in task_split_outputs]

            logging.debug('references: %s', task_references)
            logging.debug('raw outputs: %s', raw_outputs)
            logging.debug(
                'split using %s with results: %s',
                self.dataset.spec.reference_split_parser,
                task_split_outputs,
            )

            num_tasks = len(list(self.dataset.spec.tasks.keys()))
            batch_size = len(output_parsing_successes) // num_tasks
            logging.debug(
                'computed num_tasks=%s and batch_size=%s',
                num_tasks,
                batch_size,
            )
            assert (
                self.batch_size == batch_size
                or self.batch_size > batch_size
            )
            assert len(output_parsing_successes) % num_tasks == 0, (
                'Some error in batch as the references and '
                "parsing don't match"
            )

            task_spec_names = list(self.dataset.spec.tasks.keys()) * batch_size
            task_spec_specs = list(
                self.dataset.spec.tasks.values(),
            ) * batch_size

            if num_tasks > 1:
                # Create copies for each task element for the same row
                prompt_inputs = [
                    item for item in prompt_inputs for _ in range(num_tasks)
                ]
                input_tokens = [
                    item for item in input_tokens for _ in range(num_tasks)
                ]
                output_tokens = [
                    item for item in output_tokens for _ in range(num_tasks)
                ]
                raw_outputs = [
                    item for item in raw_outputs for _ in range(num_tasks)
                ]
                inputs = [item for item in inputs for _ in range(num_tasks)]

                # Unzip the references and outputs
                task_references = [
                    task for row in task_references for task in row
                ]
                task_split_outputs = [
                    task for row in task_split_outputs for task in row
                ]

            for (
                task_reference, raw_task_output, output_parsing_success,
                task_name, task_spec, num_input_tokens, num_output_tokens,
                raw_output, prompt_input, row_input,
            ) in zip(
                task_references, task_split_outputs, output_parsing_successes,
                task_spec_names, task_spec_specs, input_tokens, output_tokens,
                raw_outputs, prompt_inputs, inputs,
            ):
                if isinstance(task_reference, int):
                    task_reference = [task_reference]
                parsed_task_outputs = [
                    task_spec.get_model_output_parser(
                        parser_spec=profile.parsers[
                            self.dataset.spec.tasks[
                                task_name
                            ].model_output_parser
                        ],
                        extensions_path=profile.extensions[
                            ExtensionType.PARSERS
                        ],
                    ).parse(
                        prompt=row_input, output=raw_task_output,
                    ),
                ]
                postprocessed_task_outputs = self.dataset.task_datasets[
                    task_spec.name
                ].postprocess_outputs(parsed_task_outputs)
                postprocessed_task_references = self.dataset.task_datasets[
                    task_spec.name
                ].postprocess_references(
                    task_reference,
                )

                logging.debug(
                    'parsed outputs using %s: %s',
                    self.dataset.spec.tasks[task_name].model_output_parser,
                    postprocessed_task_outputs,
                )

                collated_results['dataset'].append(self.dataset.spec.name)
                collated_results['task'].append(task_name)

                self.extension_fn(
                    collated_results['prompt+input'],
                    [prompt_input],
                    task_spec,
                )
                self.extension_fn(
                    collated_results['reference'],
                    postprocessed_task_references,
                    task_spec,
                )
                self.extension_fn(
                    collated_results['output_raw'],
                    [raw_output],
                    task_spec,
                )
                collated_results['num_input_tokens'].append(num_input_tokens)
                collated_results['num_output_tokens'].append(num_output_tokens)
                collated_results['parsing_success'].append(
                    output_parsing_success,
                )

                if isinstance(task_reference, torch.Tensor):
                    task_reference = task_reference.cpu().numpy().tolist()

                self.extension_fn(
                    collated_results['eval_reference'],
                    task_reference,
                    task_spec,
                )
                self.extension_fn(
                    collated_results['eval_output'],
                    postprocessed_task_outputs,
                    task_spec,
                )
        return pandas.DataFrame(collated_results)

    def compute(
        self, executor: Model | concurrent.futures.Executor,
        profile: ConfigurationProfile, **model_kwargs: Any,
    ) -> Iterator[tuple[str, dict[str, float], pandas.DataFrame]]:
        data = self._inference(executor, profile, **model_kwargs)
        for task_name in self.dataset.spec.tasks:
            task_output = data[data['task'] == task_name]
            eval_reference = task_output['eval_reference'].tolist()
            eval_output = task_output['eval_output'].tolist()
            if not self.dataset.spec.tasks[task_name].multi_y and (
                eval_output and isinstance(
                    eval_output[0], list,
                )
            ):
                eval_output = [r for row in eval_output for r in row]
                assert len(eval_output) == len(eval_reference)
            computed_metrics = {
                key: round(float(val), 4) if isinstance(val, float) else val
                for key, val in self.benchmark_dataset_spec.tasks[
                    task_name
                ].compute_metrics(
                    references=eval_reference,
                    predictions=eval_output,
                    metrics_catalog=profile.metrics,
                    extension_filepath=profile.extensions[
                        ExtensionType.METRICS
                    ],
                ).items()
            }
            parsing_errors = (~task_output['parsing_success']).sum().item()
            computed_metrics = {
                **computed_metrics,
                'parsing-errors-count': parsing_errors,
                'parsing-errors-ratio': float(parsing_errors) / len(
                    task_output['parsing_success'],
                ),
            }
            yield task_name, computed_metrics, task_output


def load_benchmark_runners(
    profile: ConfigurationProfile,
    # benchmarks: Dict[str, BenchmarkSpec],
    # datasets: Dict[str, DatasetSpec],
    benchmark_names: list[str] | None = None,
    max_samples: int | None = None,
    batch_size: int = 1,
) -> Iterator[Runner]:
    benchmark_datasets = [
        dataset
        for catalog in profile.benchmarks
        for dataset in catalog.datasets.values()
        if not benchmark_names or catalog.benchmark_name in benchmark_names
    ]

    if not benchmark_datasets:
        available = ', '.join([
            catalog.benchmark_name
            for catalog in profile.benchmarks
        ])
        raise KeyError(
            'No matching benchmarks found. Please choose from '
            f'this list: {available}',
        )

    for dataset_spec in benchmark_datasets:
        logging.debug(
            'loading benchmark dataset: %s',
            dataset_spec.dataset_name,
        )
        dataset = load_dataset(
            name=dataset_spec.dataset_name,
            profile=profile,
            max_samples=max_samples,
            batch_size=batch_size,
        )

        yield Runner(
            dataset=dataset,
            benchmark_dataset_spec=dataset_spec,
            batch_size=batch_size,
        )
