import multiprocessing
import time
from typing import Any

import pynvml as nv

from .latency_utils import get_stats


class _NvmlMonitor:
    def __init__(
        self, queue: multiprocessing.SimpleQueue,
        interval: float,
    ) -> None:
        nv.nvmlInit()
        self.queue = queue
        self.stopped = False
        self.interval = interval
        self.num_gpus = nv.nvmlDeviceGetCount()
        self.run()

    def run(self) -> None:
        while not self.stopped:
            memory_used = self.measure_current_memory_used()
            self.queue.put(memory_used)
            time.sleep(self.interval)

    def measure_current_memory_used(self) -> float:
        return float(
            sum(
                nv.nvmlDeviceGetMemoryInfo(
                    nv.nvmlDeviceGetHandleByIndex(gpu),
                ).used
                for gpu in range(self.num_gpus)
            ),
        )


class GpuMonitor:
    def __init__(self, enabled: bool = True, interval: float = .1) -> None:
        super().__init__()
        self.enabled = enabled
        self.interval = interval
        self.process: multiprocessing.Process | None = None
        self.queue: multiprocessing.SimpleQueue | None = None
        self.observations: list[float] = []

    def get_memory_stats(self) -> dict[str, float]:
        if not self.enabled:
            return {}

        while not self.queue.empty():
            self.observations.append(self.queue.get() / (2**30))
        return {
            **get_stats(self.observations),
            'peak': round(max(self.observations), 3),
        }

    def __enter__(self) -> 'GpuMonitor':
        if not self.enabled:
            return self
        self.queue = multiprocessing.SimpleQueue()
        self.process = multiprocessing.Process(
            target=lambda: _NvmlMonitor(
                queue=self.queue, interval=self.interval,
            ),
        )
        self.process.start()
        return self

    def __exit__(
        self, exc_type: Any, exc_value: Any,
        traceback: Any,
    ) -> None:
        if not self.enabled:
            return
        self.process.terminate()
        self.process.join()
        # self.queue.close()  # Enable when switched to py3.11+ CI
        self.process.close()
