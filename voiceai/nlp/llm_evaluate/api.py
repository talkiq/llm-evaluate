# pylint: disable=dangerous-default-value
import concurrent.futures
import itertools
import json
import logging
import multiprocessing
import os
import pathlib
import re
import sys
import time
from datetime import datetime
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

import numpy
import pandas
import torch
import tqdm
import yaml

from .datasets.utils import load_dataset
from .helpers.constants import DEFAULT_RUNTIME_BENCHMARK
from .helpers.gpu_monitor import GpuMonitor
from .helpers.latency_utils import make_measure_stats
from .helpers.latency_utils import make_measure_time
from .helpers.profile import ExtensionType
from .helpers.profile import Profile
from .models import load_model
from .processing.mp_worker import init_worker
from .processing.runner import load_benchmark_runners


def make_run_name(model: str, run_name: Optional[str] = None) -> str:
    if not run_name:
        run_name = model
        for char in ('~', '/', '\\', '.'):
            run_name = run_name.replace(char, '_')
        run_name = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}-{run_name}'
    run_name = re.sub(r'\W|\s', '_', run_name)
    return run_name


def set_output_verbosity(verbose: bool) -> None:
    if verbose:
        log_lvl = logging.DEBUG
    else:
        log_lvl = logging.INFO
    logging.basicConfig(stream=sys.stdout, level=log_lvl)


def _benchmark(
    profile: Profile,
    benchmarks: List[str] = [],
    run_name: Optional[str] = None,
    max_samples: Optional[int] = None,
    batch_size: int = 1,
    verbose: Optional[bool] = False,
    mp_gpu_workers: int = 1,
) -> None:
    set_output_verbosity(verbose)
    cli_args = {
        'model': profile.model.to_dict(),
        'benchmark': benchmarks,
        'run_name': run_name,
        'max_samples': max_samples,
        'batch_size': batch_size,
        'mp_gpu_workers': mp_gpu_workers,
    }
    logging.info('Args:\n%s', yaml.dump(cli_args))
    run_name = make_run_name(profile.model.model, run_name)
    logging.debug('setting run_name to %s', run_name)

    report: Dict[str, Any] = {}
    outputs = pandas.DataFrame()

    if 0 <= mp_gpu_workers <= 1:
        model_obj = load_model(
            spec=profile.model,
            extensions_path=profile.extensions[ExtensionType.MODELS])
        for runner in tqdm.tqdm(
            load_benchmark_runners(
                benchmark_names=benchmarks,
                profile=profile,
                max_samples=max_samples,
                batch_size=batch_size),
        ):
            runner_results = list(runner.compute(
                model_obj, profile, **profile.model.model_inference_args))

            report[runner.dataset.identifier] = {}
            for task_name, computed_metrics, dataset_outputs in runner_results:
                logging.info('Interim results for %s: \n%s',
                             runner.dataset.identifier,
                             yaml.dump(computed_metrics))
                report[runner.dataset.identifier][task_name] = computed_metrics
                outputs = pandas.concat([outputs, dataset_outputs])

    else:
        # Once tested, this can merge with the above if, i.e., always uses a
        # cf_executor to run benchmark, and load only one model by default

        # device_count will be set based on os.environ["CUDA_VISIBLE_DEVICES"]
        device_count = torch.cuda.device_count()
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            # assuming comma "," separated device ids
            logging.warning(
                'CUDA_VISIBLE_DEVICES is set to %s',
                os.environ['CUDA_VISIBLE_DEVICES'])
            visible_gpu_devices = [
                int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        else:
            visible_gpu_devices = list(range(device_count))

        # make sure all are valid
        assert all(torch.cuda.get_device_name(i) for i in visible_gpu_devices)
        assert 0 <= mp_gpu_workers <= len(visible_gpu_devices), (
            f'mp_gpu_workers must be <= # visible GPUs ({len(visible_gpu_devices)})')

        with multiprocessing.Manager() as manager:
            d_shared_cuda_devices = {i: [] for i in range(mp_gpu_workers)}
            # l_shared_cuda_num = manager.list(range(mp_gpu_workers))
            for i, device_id in enumerate(visible_gpu_devices):
                d_shared_cuda_devices[i % mp_gpu_workers].append(device_id)

            l_shared_cuda_devices = manager.list(
                d_shared_cuda_devices.values())
            logging.warning('l_shared_cuda_devices: %s', l_shared_cuda_devices)
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=mp_gpu_workers,
                mp_context=multiprocessing.get_context('spawn'),
                initializer=init_worker,
                initargs=(l_shared_cuda_devices, profile.model,
                          profile.extensions[ExtensionType.MODELS]),
            ) as cf_executor:

                for runner in tqdm.tqdm(
                        load_benchmark_runners(
                            profile=profile,
                            benchmark_names=benchmarks,
                            max_samples=max_samples,
                            batch_size=batch_size),
                ):
                    runner_results = list(runner.compute(
                        cf_executor, profile, **profile.model.model_inference_args))

                    report[runner.dataset.identifier] = {}
                    for task_name, computed_metrics, dataset_outputs in runner_results:
                        logging.info('Interim results for %s: \n%s',
                                     runner.dataset.identifier,
                                     yaml.dump(computed_metrics))
                        report[runner.dataset.identifier][
                            task_name] = computed_metrics
                        outputs = pandas.concat([outputs, dataset_outputs])

    logging.info('\nMetrics\n-------\n%s\n', yaml.dump(report))

    metrics_file = f'metrics-{run_name}.csv'
    logging.info('Writing metrics to "%s"', metrics_file)
    flat_report: List[Dict[str, Any]] = []
    for dataset_, metrics in sorted(report.items(), key=lambda k: k[0]):
        flat_report.append({
            'dataset': dataset_,
            'args': json.dumps(cli_args),
            **metrics,
        })
    pandas.DataFrame(flat_report).to_csv(metrics_file, index=False)

    outputs_file = f'outputs-{run_name}.csv'
    logging.info('Writing outputs to "%s"', outputs_file)
    outputs.to_csv(outputs_file, index=False)


def _get_runtime_stats(
    profile: Profile,
    batch_size: int = 1,
    verbose: Optional[bool] = False,
    gpu_monitor: Optional[bool] = True,
) -> None:
    NUM_RUNS = 2
    set_output_verbosity(verbose)

    cli_args = {
        'model': profile.model.to_dict(),
        'batch_size': batch_size,
    }
    logging.info('Args:\n%s', yaml.dump(cli_args))
    logging.info('Measuring load times...')

    def load_closure() -> Iterator[float]:
        start = time.perf_counter()
        model_ = load_model(spec=profile.model,
                            extensions_path=profile.extensions[ExtensionType.MODELS])
        yield time.perf_counter() - start
        del model_
    load_stats = make_measure_time(load_closure)(runs=2)

    benchmarks = [benchmark_dataset_spec
                  for catalog in profile.benchmarks
                  for benchmark_dataset_spec in catalog.datasets.values()
                  if catalog.benchmark_name == DEFAULT_RUNTIME_BENCHMARK]
    assert len(benchmarks) == 1, 'expecting one benchmark for runtime-stats'

    dataset = load_dataset(
        name=benchmarks[0].dataset_name,
        profile=profile,
        batch_size=batch_size,
        max_samples=None,
    )
    model_obj = load_model(spec=profile.model,
                           extensions_path=profile.extensions[ExtensionType.MODELS])

    logging.info('Warming up...')
    model_obj.process(inputs=['Hello, world!'],
                      **profile.model.model_inference_args)

    logging.info('Measuring inference times...')
    run_inputs = []
    run_outputs = []

    def inference_closure() -> Iterator[Tuple[int, int, float, float, float]]:
        for inputs, _ in tqdm.tqdm(dataset.get_dataloader()):
            prompt_inputs = model_obj.preprocess(
                inputs=list(inputs), prompt=benchmarks[0].prompt,
            )
            start = time.perf_counter()
            outputs = model_obj.process(
                inputs=prompt_inputs,
                **profile.model.model_inference_args)
            inference_time = time.perf_counter() - start
            num_input_tokens = sum(
                model_obj.tokenizer.get_num_tokens(in_)
                for in_ in prompt_inputs)
            num_output_tokens = max(sum(
                model_obj.tokenizer.get_num_tokens(out_)
                for out_ in outputs), 1)  # to avoid zero division
            time_per_token = (inference_time / num_output_tokens) * 1000
            time_per_token_norm = (inference_time / (
                num_input_tokens + num_output_tokens)) * 1000
            run_inputs.extend(prompt_inputs)
            run_outputs.extend(outputs)
            yield (num_input_tokens, num_output_tokens, time_per_token,
                   time_per_token_norm, inference_time)
    with GpuMonitor(enabled=gpu_monitor) as monitor:
        inference_stats = make_measure_stats(inference_closure)(runs=NUM_RUNS)
        gpu_stats = monitor.get_memory_stats()

    report = {
        'args': cli_args,
        'load_stats': load_stats,
        'inference_stats': dict(
            zip(('input_tokens', 'output_tokens', 'time_per_token_ms',
                 'time_per_token_normalized_ms', 'latency_s'),
                inference_stats)),
        'gpu_memory': gpu_stats,
    }
    logging.info('Report:\n%s', yaml.dump(report))

    run_name = make_run_name(profile.model.model, None)
    stats_file = f'{run_name}_runtime_stats.csv'
    logging.info('Writing stats to "%s"', stats_file)
    df = pandas.json_normalize(report)
    df.to_csv(stats_file, index=False)

    run_data = pandas.DataFrame({
        'prompt': run_inputs,
        'output': run_outputs,
    }).groupby(['prompt']).agg(outputs=('output', list)).reset_index(drop=False)

    for idx in range(NUM_RUNS):
        run_data[f'output_run_{idx}'] = run_data['outputs'].apply(
            lambda x: x[idx])  # pylint: disable=cell-var-from-loop
    run_data = run_data.drop(columns=['outputs'])
    run_data['model_args'] = json.dumps(cli_args)

    outputs_file = f'{run_name}_outputs.csv'
    logging.info('Writing inputs/outputs to %s', outputs_file)
    run_data.to_csv(outputs_file, index=False)


def _get_context_stats(
    profile: Profile,
    batch_size: int = 1,
    verbose: Optional[bool] = False,
    gpu_monitor: Optional[bool] = True,
) -> None:
    set_output_verbosity(verbose)
    cli_args = {
        'model': profile.model.to_dict(),
        'batch_size': batch_size,
    }
    logging.info('Args:\n%s', yaml.dump(cli_args))
    logging.info('Measuring memory usage per context window...')
    model_obj = load_model(spec=profile.model,
                           extensions_path=profile.extensions[ExtensionType.MODELS])
    with open(pathlib.Path(__file__).parents[0] / 'local_assets' / 'story.txt',
              encoding='utf-8') as f:
        story = (' '.join([line.strip()
                           for line in f.readlines()
                           if not line.startswith('#')
                           ])).split()

    story_generator = itertools.cycle(story)
    instruction = 'Continue the story below.'

    def inference_closure(
    ) -> Iterator[Tuple[int, int, float, float, float, Dict[str, float]]]:
        with tqdm.tqdm(range(20)) as pbar:
            for i in pbar:
                factor = (i + 1) * 10
                num_input_tokens = 10 * factor

                model_obj.init_tokenizer(model_obj.model_or_path, **{
                    **profile.model.tokenizer_args,
                    'model_max_length': num_input_tokens,
                })

                for _ in range(2):
                    with GpuMonitor(enabled=gpu_monitor) as monitor:
                        text = ' '.join(
                            [next(story_generator, 'word') for _ in range(2000)])
                        prompt_inputs = model_obj.preprocess(inputs=[text],
                                                             prompt=instruction)
                        start = time.perf_counter()
                        outputs = model_obj.process(
                            inputs=prompt_inputs,
                            **profile.model.model_inference_args,
                        )
                        total_elapsed = time.perf_counter() - start
                        memory_stats = monitor.get_memory_stats()
                    num_output_tokens = model_obj.tokenizer.get_num_tokens(
                        outputs[0])
                    time_per_token = (total_elapsed / num_output_tokens) * 1000
                    time_per_token_norm = (total_elapsed / (
                        num_input_tokens + num_output_tokens)) * 1000
                    yield (
                        num_input_tokens, num_output_tokens,
                        total_elapsed, time_per_token, time_per_token_norm, memory_stats,
                    )
                    torch.cuda.empty_cache()
                pbar.set_postfix({
                    'tokens': num_input_tokens,
                    'peak memory': memory_stats['peak']})

    summary = []
    latencies = []
    try:
        for (input_length, output_length, latency, per_token_latency,
             per_token_latency_norm, memory) in inference_closure():
            latencies.append(latency)
            summary.append({
                'input tokens': input_length,
                'generated tokens': output_length,
                'latency (s)': latency,
                'time per token (ms)': per_token_latency,
                'time per token normalized (ms)': per_token_latency_norm,
                'memory (GB)': memory['peak'],
            })
    except Exception:
        logging.exception('caught an error while processing context window '
                          'length, results are incomplete!')

    df = pandas.DataFrame(summary)
    df = df.groupby(['input tokens']).agg(
        ['min', 'max', 'mean', ('p95', lambda x: numpy.percentile(x, q=95))],
    ).sort_values(by=['input tokens'])

    logging.info('Args:\n%s', yaml.dump(cli_args))
    logging.info('Stats:\n%s', df)

    run_name = make_run_name(profile.model.model, None)
    stats_file = f'{run_name}_context_length_stats.csv'
    logging.info('Writing stats to "%s"', stats_file)
    df['args'] = json.dumps(cli_args)
    df.to_csv(stats_file, index=False)
