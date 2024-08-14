# LLM Evaluate

Tool for benchmarking and evaluating LLMs. Basically, given a model checkpoint,
the tool performs a set of predefined or custom benchmarks and reports results.

## Installation

### Install via Pip

The package can be directly installed into the python environment using `pip`:

```console
    pip install git+git://github.com/talkiq/llm-evaluate.git
```

## Quick Start

The package comes with a sample starter template to help get started with the tool quickly.

- A demo configuration with the model definiton [sample_config.yaml](./sample-config.yaml)
- A few benchmarks defined in [benchmarks.yaml](./llm_evaluate/configs/benchmarks.yaml)
- A few demo datasets defined in [datasets.yaml](./llm_evaluate/configs/datasets.yaml)

Other artifacts are defined in the [configs directory](./llm_evaluate/configs/).

To perform an accuracy benchmark on the model, we can run the following command:

```console
    llm-evaluate benchmark --benchmarks demo sample_config.yaml
```

To collect runtime telemetry for the model, we can run the following command:

```console
    llm-evaluate stats-runtime sample_config.yaml
```

## Basic concepts

- `Dataset`: A definition of a dataset that can be externally or internally stored.
- `Metric`: A measure of the model's perfermance. The tool uses [evaluate](https://github.com/huggingface/evaluate).
- `Benchmark`: A collection of datasets, metrics and parsers. A benchmark can be named and reused repeatedly.
- `Parser`: A function to interpret prompt or LLM responoses before further processing. We have 3 type of parsers:

  - `ReferenceSplitParser`: A parser that can convert multitask prompt responses into individual task references and also apply the same operation on LLM's corresponding output.
  - `TaskReferenceParser`: A parser that can further process individual task references.
  - `ModelOutputParser`: A parser that can process individual model outputs.

- `Model`: The model that can be used for evaluation. The tool is compatible with [HuggingFace Auto Class Models](https://huggingface.co/docs/transformers/v4.30.0/en/model_doc/auto), [HuggingFace Pipelines](https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/pipelines) and/or external APIs (currently, OpenAI and VertexGenAI are supported).

## Some common ops

Some common things one may have to do to run evaluation are described below.
Mostly, this boils down to adding datasets, benchmarks and parsing.

All of the steps below require a package release.

### Adding a dataset

Update [configs/default/datasets.yaml](llm_evaluate/configs/default/datasets.yaml) with the following:

- Create a new entry in the dataset list.
- Fill in the mandatory fields:

  - `name`: Name of the dataset
  - `description`: Description of the dataset
  - `column_input`: Column containing the prompts
  - `column_reference`: Column containing the reference outputs
  - `reference_split_parser`: Parser for preprocessing ground truths or raw LLM outputs in case there are multiple task responses in a single message.
  - `tasks`: A dictionary of tasks within the dataset.

    - `task_type`: Type of task - one of `generation` or `classification`.
    - `multi_y`: Task references have multiple ground truths for a single sample.
    - `case_sensitive`: If the ground truth labels are case sensitive.
    - `none_value`: Default value to give to the output label if it is empty.
    - `key`: Optional, specifies the key entry in a JSON message for this task. Used by `JsonReferenceSplitParser`.
    - `model_output_parser`: A parser to process model's output.
    - `task_reference_parser`: A parser to further process a task's raw references, if needed.

  - `metadata`: A dictionary with the following fields:

    - `version`: Version of the dataset.
    - `format_`: Currently, `csv`, `jsonl` or `json`.
    - `source`: One of `local`, `gcs` or `custom` (for external API hosted datasets)
    - `path`: Path for the dataset, for instance, path on cloud storage.

For instance, a entry for a GCS backed dataset (with the input prompt in `prompt` and desired output in `reference` columns) would look something like this:

```{yaml}
my_dataset:
    task_type: classification
    description: Example entry for a GCS backed dataset
    column_input: prompt
    column_reference: reference
    reference_split_parser: JsonReferenceSplitParser
    tasks:
        my_task:
            task_type: classification
            task_reference_parser: DefaultParser
            model_output_parser: KeywordParser
            none_value: Unknown
            key: answer
            labels:
            - A
            - B
            - C
    metadata:
        format_: csv
        source: gcs
        path: gs://path/to/dataset/file.csv
        version: 'test'
```

#### Notes for custom datasets

Adding a custom (API sourced) dataset requires some additional code changes:

- Subclass `CustomDataset` class from [llm_evaluate/datasets/custom_dataset.py](llm_evaluate/datasets/custom_dataset.py)
- Implement the required `load_all()` method.
- For more details, look at existing examples for [extensions](extensions/datasets/).

### Creating a benchmark

Update [configs/default/benchmarks.yaml](llm_evaluate/configs/default/benchmarks.yaml) with the following:

- Create a new entry in the benchmarks list.
- Fill the mandatory fields:

  - Top level key: Name of the benchmark. Can be any name without spaces.
  - A list of datasets, added as keys, metrics and parsing information that form up to build that benchmark:

    - Key name: Reference to an existing dataset. Required match the name of the dataset in `datasets.yaml`.

    - `prompt`: Optional prompt prefix to prepend to every sample during evaluation.
    - `tasks`:

        - key name: matches the name of a task the dataset
        - `metrics`: A list of named metrics from [evaluate](https://github.com/huggingface/evaluate) to be used for that task.

An example of a benchmark `my_benchmark` with a dataset `my_dataset`:

```{yaml}
my_benchmark:
    my_dataset:
        prompt: A simple prompt
        tasks:
            my_task:
                metrics:
                - rouge
                - accuracy
                - custom_new_metric
```

### Adding a parser

Adding a parser requires a code change. Basically:

- Subclass the `Parser` class [llm_evaluate/parsers/parser.py](llm_evaluate/parsers/parser.py).
- Implement the `parse()` method.
- For examples, look at [default_parser.py](llm_evaluate/parsers/default_parser.py) and [keyword_parser.py](llm_evaluate/parsers/keyword_parser.py)

### Adding a custom metric

Similar to adding a parser, adding a custom metric requires a code change. To add a new metric:

- Create a copy of the `new_metric.py` file [extensions/metrics/new_metric.py](extensions/metrics/new_metric.py).
- Complete the fields marked by `TODO` with code and documentation and rename the file with your metric name.

## Usage

There are 4 basic commands right now:

- `benchmark` - Useful for running a predefined dataset.
- `stats-runtime` - For general runtime stats for a model.
- `stats-context-window` - Fo stats on different context lengths for a model.

```{console}
    $ llm-evaluate --help
    Usage: llm-evaluate [OPTIONS] COMMAND [ARGS]...

    ╭─ Commands ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │ benchmark                                        Run a model against predefined benchmarks.                                                                          │
    │ stats-context-window                             Get stats on model's context window & memory usage.                                                                 │
    │ stats-runtime                                    Get stats on model's runtime.                                                                                       │
    ╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

### Command: benchmark

```{console}
    $ llm-evaluate benchmark --help
    Usage: llm-evaluate benchmark [OPTIONS] PROFILE

    Run a model against predefined benchmarks.

    ╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │ *    profile      TEXT  Path to YAML configuration profile for the model. [default: None] [required]                                                                 │
    ╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │ --benchmark                      TEXT     Optionally specify only a few select benchmarks eg. --benchmark internal --benchmark external                              │
    │ --run-name                       TEXT     Run name to prefix output files. [default: None]                                                                           │
    │ --max-samples                    INTEGER  Maximum number of samples for benchmarking per dataset. [default: None]                                                    │
    │ --batch-size                     INTEGER  Batch size for mini-batch evaluation. [default: 1]                                                                         │
    │ --mp-gpu-workers                 INTEGER  Number of multiprocessing workers. [default: 1]                                                                            │
    │ --verbose        --no-verbose             Dislpay debug logs. [default: no-verbose]                                                                                  │
    │ --help                                    Show this message and exit.                                                                                                │
    ╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```

Example using [flan-t5-xxl](./llm_evaluate/configs/custom/flan-t5-xxl.yaml) with the model binaries stored locally at `~/data/models/flan-t5-xxl`:

```{console}

    $ llm-evaluate benchmark llm_evaluate/configs/custom/flan-t5-xxl.yaml --benchmark internal --run-name test --max-samples 10

    Metrics
    -------
    action_items@4:
    rouge1: 0.10891483028030927
    rouge2: 0.05562770562770562
    rougeL: 0.09773440214575269
    rougeLsum: 0.10441330122832151
    call_purpose_categories_full_call_sides@6:
    accuracy: 0.5
    f1: 0.4333333333333333
    precision: 0.3857142857142857
    recall: 0.5
    call_purpose_categories_utterances@6:
    accuracy: 0.7
    f1: 0.6333333333333333
    precision: 0.6
    recall: 0.7
    chapter_summary_review@2022-03-29:
    rouge1: 0.27549660435170464
    rouge2: 0.08602447734217675
    rougeL: 0.2058981682512737
    rougeLsum: 0.20673639630484614
    sentiment_internal@2022-05-22:
    accuracy: 0.7
    f1: 0.667948717948718
    precision: 0.8125
    recall: 0.7


    INFO:root:Writing metrics to "metrics-test.csv"
    INFO:root:Writing outputs to "outputs-test.csv"
```

Outputs:

- Metrics from each of the benchmarks.
- Raw model inputs & outputs for debugging (saved as CSV files).

### Command: stats-runtime

```{console}

    $ llm-evaluate stats-runtime --help
    Usage: llm-evaluate stats-runtime [OPTIONS] PROFILE

    Get stats on model's runtime.

    ╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │ *    profile      TEXT  Path to YAML configuration profile for the model. [default: None] [required]                       │
    ╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │ --batch-size                    INTEGER  Batch size for mini-batch evaluation. [default: 1]                                │
    │ --verbose       --no-verbose             Dislpay debug logs. [default: no-verbose]                                         │
    │ --help                                   Show this message and exit.                                                       │
    ╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

Example: Run the stock [google/flan-t5-small](./llm_evaluate/configs/custom/flan-t5-small.yaml) model with customizations in the configuration profile:

```{console}

    $ llm-evaluate stats-runtime llm_evaluate/configs/custom/flan-t5-small.yaml
    INFO:root:Measuring load times...
    INFO:root:Measuring inference times...
    100%|█████████████████████████████████████████████████████████████████████████████████████████| 20/20 [04:03<00:00, 12.17s/it]
    100%|█████████████████████████████████████████████████████████████████████████████████████████| 20/20 [04:01<00:00, 12.09s/it]
    INFO:root:Report:
    args:
    batch_size: 1
    model: google/flan-t5-small
    model_inference_arg:
        max_new_tokens: 1000
        num_beams: 5
        temperature: 0.5
    model_load_arg:
        device_map: sequential
        max_input_tokens: 3000
        torch_dtype: bfloat16
    model_task_type: T5ForConditionalGeneration
    model_type: automodel
    tokenizer_arg:
        model_max_length: 3000
        truncation: longest_first
        truncation_side: right
    gpu_memory:
        mean: 9.642
        p01: 9.618
        p95: 9.65
        p99: 9.65
        peak: 9.65
        stdev: 0.103
    inference_stats:
        mean: 12.127
        p01: 0.659
        p95: 15.68
        p99: 15.8
        stdev: 3.693
    load_stats:
        mean: 6.877
        p01: 0.988
        p95: 12.285
        p99: 12.766
        stdev: 6.009

    INFO:root:Writing stats to "runtime_stats.csv"
```

### Command: stats-context-window

```{console}

    $ llm-evaluate stats-context-window --help
    Usage: llm-evaluate stats-context-window [OPTIONS] PROFILE

    Get stats on model's context window & memory usage.

    ╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │ *    profile      TEXT  Path to YAML configuration profile for the model. [default: None] [required]                       │
    ╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │ --batch-size                    INTEGER  Batch size for mini-batch evaluation. [default: 1]                                │
    │ --verbose       --no-verbose             Dislpay debug logs. [default: no-verbose]                                         │
    │ --help                                   Show this message and exit.                                                       │
    ╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

Example: Run LLama2-Chat optimized using Llama.cpp library (stored locally at `~/data/models/llama2-cpp`) [llama2-cpp.yaml](llm_evaluate/configs/custom/llama2-cpp.yaml) model with customizations in the configuration profile:

```{console}

    INFO:root:Args:
    batch_size: 1
    model: /home/harshsaini/data/models/llama2-cpp/ggml-model-q4_0.bin
    model_inference_arg:
    max_new_tokens: 512
    num_beams: 5
    temperature: 0.5
    model_load_arg:
    n_ctx: 4000
    model_task_type: null
    model_type: llama-cpp
    tokenizer_arg:
    max_tokens: 3000

    INFO:root:Stats:
                generated tokens                    latency (s)  ... time per token (ms) memory (GB)
                            min  max   mean    p95         min  ...                 p95         min     max    mean     p95
    input tokens                                                  ...
    100                        40  100   76.4  100.0   19.541414  ...          487.559597      11.537  11.537  11.537  11.537
    200                         7  129   93.8  129.0   17.363498  ...         2045.059656      11.537  11.537  11.537  11.537
    300                        38  129   89.6  129.0   21.992356  ...          555.494385      11.537  11.537  11.537  11.537
    400                        13  129   58.6  118.0   18.156507  ...         1262.216721      11.537  11.537  11.537  11.537
    500                         4  129   64.2  129.0   18.344462  ...         3820.323508      11.537  11.537  11.537  11.537
    600                        49  129   87.2  126.0   20.377329  ...          395.606101      11.537  11.537  11.537  11.537
    700                        11  129   72.2  129.0   17.617694  ...         1425.553178      11.537  11.537  11.537  11.537
    800                        54  129   93.8  127.4   20.896465  ...          370.108579      11.537  11.537  11.537  11.537
    900                        58  129  106.6  129.0   21.811075  ...          352.831850      11.537  11.537  11.537  11.537
    1000                        4  107   51.8  105.0   16.865100  ...         3805.886147      11.537  11.537  11.537  11.537
    1100                       33   82   56.0   80.0   18.623258  ...          580.942581      11.537  11.537  11.537  11.537
    1200                       44  129   85.2  128.0   20.406143  ...          457.194141      11.537  11.537  11.537  11.537
    1300                       16  129   58.8  123.8   19.152370  ...         1186.204717      11.537  11.537  11.537  11.537
    1400                        5  129   68.2  128.0   16.762951  ...         3124.485000      11.537  11.537  11.537  11.537
    1500                       36  129   83.2  129.0   19.797821  ...          517.538090      11.537  11.537  11.537  11.537
    1600                       28  129   83.2  129.0   20.244846  ...          669.765093      11.537  11.537  11.537  11.537
    1700                        7  114   58.4  106.6   17.381077  ...         2117.777818      11.537  11.537  11.537  11.537
    1800                       67  117   88.4  113.0   22.849887  ...          362.527236      11.537  11.537  11.537  11.537
    1900                       73  129  110.4  129.0   23.340018  ...          307.801323      11.537  11.537  11.537  11.537
    2000                       12  129   83.6  129.0   18.121643  ...         1401.317769      11.537  11.537  11.537  11.537

    [20 rows x 16 columns]
    INFO:root:Writing stats to "context_length_stats.csv"
```
