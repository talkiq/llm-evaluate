import typer

from .api import _benchmark
from .api import _get_context_stats
from .api import _get_runtime_stats
from .api import set_output_verbosity
from .helpers.profile_utils import load_profile

app = typer.Typer()


@app.command(
    name='benchmark',
    help='Run a model against predefined benchmarks.',
)
def run_benchmark(
    profile: str = typer.Argument(
        ..., help='Path to YAML configuration profile for the model.',
    ),
    benchmarks: list[str] = typer.Option(
        [], help='Optionally specify only a few select benchmarks '
                 'eg. --benchmark internal --benchmark external',
    ),
    run_name: str | None = typer.Option(
        None, help='Run name to prefix output files.',
    ),
    max_samples: int | None = typer.Option(
        None, help='Maximum number of samples for benchmarking per dataset.',
    ),
    batch_size: int = typer.Option(
        1, help='Batch size for mini-batch evaluation.',
    ),
    mp_gpu_workers: int = typer.Option(
        1, help='Number of multiprocessing workers.',
    ),
    verbose: bool = typer.Option(
        False, help='Dislpay debug logs.',
    ),
) -> None:
    set_output_verbosity(verbose)
    profile_ = load_profile(profile)
    _benchmark(
        profile=profile_,
        benchmarks=benchmarks,
        run_name=run_name,
        max_samples=max_samples,
        batch_size=batch_size,
        verbose=verbose,
        mp_gpu_workers=mp_gpu_workers,
    )


@app.command(name='stats-runtime', help="Get stats on model's runtime.")
def run_runtime(
    profile: str = typer.Argument(
        ..., help='Path to YAML configuration profile for the model.',
    ),
    batch_size: int = typer.Option(
        1, help='Batch size for mini-batch evaluation.',
    ),
    verbose: bool = typer.Option(
        False, help='Dislpay debug logs.',
    ),
    gpu_monitor: bool = typer.Option(
        True, help='Enables GPU monitoring',
    ),
) -> None:
    set_output_verbosity(verbose)
    profile_ = load_profile(profile)
    _get_runtime_stats(
        profile=profile_,
        batch_size=batch_size,
        verbose=verbose,
        gpu_monitor=gpu_monitor,
    )


@app.command(
    name='stats-context-window',
    help="Get stats on model's context window & memory usage.",
)
def run_context(
    profile: str = typer.Argument(
        ..., help='Path to YAML configuration profile for the model.',
    ),
    batch_size: int = typer.Option(
        1, help='Batch size for mini-batch evaluation.',
    ),
    verbose: bool = typer.Option(
        False, help='Dislpay debug logs.',
    ),
    gpu_monitor: bool = typer.Option(
        True, help='Enables GPU monitoring',
    ),
) -> None:
    set_output_verbosity(verbose)
    profile_ = load_profile(profile)
    _get_context_stats(
        profile=profile_,
        batch_size=batch_size,
        verbose=verbose,
        gpu_monitor=gpu_monitor,
    )


def main() -> None:
    app()


if __name__ == '__main__':
    app()
