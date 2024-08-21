from collections.abc import Callable
from collections.abc import Iterator

import numpy


def get_stats(elapsed: list[float]) -> dict[str, float]:
    """Calculates some basic stats."""
    arr_elapsed = numpy.array(elapsed)
    return {
        'mean': round(float(numpy.mean(arr_elapsed)), 3),
        'stdev': round(float(numpy.std(arr_elapsed)), 3),
        'p01': round(float(numpy.percentile(arr_elapsed, 1)), 3),
        'p95': round(float(numpy.percentile(arr_elapsed, 95)), 3),
        'p99': round(float(numpy.percentile(arr_elapsed, 99)), 3),
    }


def make_measure_time(
        callback: Callable[[], Iterator[float]],
) -> Callable[[int], dict[str, float]]:
    def _measure_time(runs: int = 10) -> dict[str, float]:
        elapsed = []
        for _ in range(runs):
            elapsed.extend(list(callback()))
        return get_stats(elapsed)
    return _measure_time

# TODO: Make this a bit simpler to read


def make_measure_stats(
    callback: Callable[[], Iterator[list[float]]],
) -> Callable[[int], list[dict[str, float]]]:
    def _measure_stats(runs: int = 10) -> list[dict[str, float]]:
        elapsed = []
        for _ in range(runs):
            elapsed.extend(list(callback()))
        num_measures = len(elapsed[0])
        measures: list[list[float]] = [
            [] for _ in range(num_measures)
        ]
        for elp in elapsed:
            for i in range(num_measures):
                measures[i].append(elp[i])
        return [get_stats(measure) for measure in measures]
    return _measure_stats
