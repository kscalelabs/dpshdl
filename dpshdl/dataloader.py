"""Defines a class for loading data from a dataset.

This is a greatly simplified dataloader implementation. We start a pool of
worker processes which continually load data from the dataset and put it on a
queue. The main process then reads from the queue and yields batches of data
as lists. The user is responsible for performing any additional downstream
tasks like collation.
"""

import logging
import multiprocessing as mp
import os
import random
import threading
import time
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from queue import Queue
from threading import Event
from types import TracebackType
from typing import Callable, Generic, Self, TypeVar

import numpy as np

from dpshdl.dataset import Dataset, ErrorHandlingDataset
from dpshdl.testing import print_sample, run_test
from dpshdl.utils import configure_logging

logger = logging.getLogger(__name__)

T = TypeVar("T")
Tc = TypeVar("Tc")


class Timer:
    def __init__(self) -> None:
        super().__init__()

        self.start_time = 0.0
        self.elapsed_time = 0.0

    def __enter__(self) -> Self:
        self.start_time = time.time()
        return self

    def __exit__(self, _t: type[BaseException] | None, _e: BaseException | None, _tr: TracebackType | None) -> None:
        self.elapsed_time = time.time() - self.start_time


@dataclass(frozen=True)
class Stats:
    worker_id: int
    elapsed_time: float


@dataclass(frozen=True)
class DataloaderItem(Generic[T]):
    item: T | None
    exception: BaseException | None
    stats: Stats


@dataclass(frozen=True)
class CollatedDataloaderItem(Generic[T]):
    item: T | None
    exception: BaseException | None
    stats: list[Stats]


def init_random(worker_id: int = 0) -> None:
    random.seed(1337 + worker_id)
    np.random.seed(1337 + worker_id)


def dataloader_worker_init_fn(worker_id: int, num_workers: int) -> None:
    configure_logging(prefix=f"{worker_id}")
    init_random(worker_id)


def collate_worker_init_fn() -> None:
    configure_logging(prefix="col")
    init_random(-1)


def dataloader_worker(
    worker_init_fn: Callable[[int, int], None],
    dataset: Dataset[T, Tc],
    queue: "Queue[DataloaderItem[T]]",
    stop_event: Event,
    worker_id: int,
    num_workers: int,
    raise_errs: bool,
) -> None:
    worker_init_fn(worker_id, num_workers)
    dataset.worker_init(worker_id, num_workers)
    dataset_iterator = dataset.__iter__()
    while True:
        if stop_event.is_set():
            break
        try:
            with Timer() as timer:
                sample = dataset_iterator.__next__()
            queue.put(DataloaderItem(sample, None, Stats(worker_id, timer.elapsed_time)))
        except BaseException as e:
            if raise_errs:
                raise
            queue.put(DataloaderItem(None, e, Stats(worker_id, 0.0)))
            break


def collate_worker(
    worker_init_fn: Callable[[], None],
    samples_queue: "Queue[DataloaderItem[T]]",
    collated_queue: "Queue[CollatedDataloaderItem[Tc]]",
    collate_fn: Callable[[list[T]], Tc | None],
    stop_event: Event,
    batch_size: int,
    raise_errs: bool,
) -> None:
    worker_init_fn()
    samples: list[DataloaderItem[T]] = []
    while True:
        if stop_event.is_set():
            break
        sample = samples_queue.get()
        if sample.exception is not None:
            collated_queue.put(CollatedDataloaderItem(None, sample.exception, [sample.stats]))
            break
        if sample.item is None:
            raise RuntimeError("`item` should not be `None` unless there was an exception")
        samples.append(sample)
        if len(samples) == batch_size:
            try:
                if (collated := collate_fn([s.item for s in samples if s.item is not None])) is not None:
                    collated_queue.put(CollatedDataloaderItem(collated, None, [s.stats for s in samples]))
            except BaseException as e:
                if raise_errs:
                    raise
                collated_queue.put(CollatedDataloaderItem(None, e, [s.stats for s in samples]))
                break
            finally:
                samples = []


def default_num_workers(default: int) -> int:
    if hasattr(os, "sched_getaffinity"):
        try:
            return len(os.sched_getaffinity(0))
        except Exception:
            pass
    if (cpu_count := os.cpu_count()) is not None:
        return cpu_count
    return default


class Dataloader(Generic[T, Tc]):
    """Defines a dataloader for loading data from a dataset.

    Usage:

    .. code-block:: python

        with Dataloader(dataset, num_workers=4, batch_size=4) as dataloader:
            for batch in dataloader:
                # Do something with batch

    Parameters:
        dataset: The dataset to load data from.
        collate_fn: The function to use to collate the samples into a batch.
        num_workers: The number of workers to use. If 0, the dataset will be
            loaded on the main process. Otherwise, the dataset will be loaded
            on a pool of workers. If not provided, it will default to the
            number of CPUs on the system.
        batch_size: The batch size to use.
        prefetch_factor: The number of batches to pre-load from the dataset.
        ctx: The multiprocessing context to use. If not provided, the default
            context will be used.
        dataloader_worker_init_fn: The initialization function to use for
            the dataloader workers, which takes the workd ID and the number of
            workers as input.
        collate_worker_init_fn: The initialization function to use for the
            collate worker, which takes no inputs.
        item_callback: A callback function to call for each item before it is
            returned. This can be used for logging or debugging purposes.
        raise_errs: If set, raise worker errors instead of passing them to
            the error queue.
    """

    def __init__(
        self,
        dataset: Dataset[T, Tc],
        num_workers: int | None = None,
        batch_size: int = 1,
        prefetch_factor: int = 2,
        ctx: BaseContext | None = None,
        dataloader_worker_init_fn: Callable[[int, int], None] = dataloader_worker_init_fn,
        collate_worker_init_fn: Callable[[], None] = collate_worker_init_fn,
        item_callback: Callable[[CollatedDataloaderItem[Tc]], None] = lambda _: None,
        raise_errs: bool = False,
    ) -> None:
        super().__init__()

        if num_workers is None:
            num_workers = default_num_workers(0)
        if num_workers < 0:
            raise ValueError("`num_workers` must be greater than or equal to 0")
        if batch_size < 1:
            raise ValueError("`batch_size` must be greater than or equal to 1")
        if prefetch_factor < 1:
            raise ValueError("`prefetch_factor` must be greater than or equal to 1")

        self.dataset = dataset
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.ctx = mp.get_context() if ctx is None else ctx
        self.dataloader_worker_init_fn = dataloader_worker_init_fn
        self.collate_worker_init_fn = collate_worker_init_fn
        self.item_callback = item_callback
        self.raise_errs = raise_errs
        self.manager = self.ctx.Manager()

        self.processes: list[mp.Process] | None = None
        self.single_process_threads: tuple[threading.Thread, threading.Thread] | None = None
        self.samples_queue: Queue[DataloaderItem[T]] = self.manager.Queue(maxsize=batch_size * prefetch_factor)
        self.collated_queue: Queue[CollatedDataloaderItem[Tc]] = self.manager.Queue(maxsize=prefetch_factor)
        self.stop_event: Event = self.manager.Event()

    def test(
        self,
        max_samples: int = 10,
        log_interval: int | None = 1,
        print_fn: Callable[[int, Tc], None] = print_sample,
    ) -> None:
        """Defines a function for doing adhoc testing of the dataset.

        Args:
            max_samples: The maximum number of samples to test.
            log_interval: How often to log a sample. If None, don't log any
                samples.
            print_fn: The function to use for printing samples.
        """
        with self:
            run_test(
                ds=self,
                max_samples=max_samples,
                log_interval=log_interval,
                print_fn=print_fn,
            )

    def __iter__(self) -> "Dataloader[T, Tc]":
        return self

    def __next__(self) -> Tc:
        if self.processes is None and self.single_process_threads is None:
            raise RuntimeError("Dataloader is not running")
        item = self.collated_queue.get()
        self.item_callback(item)
        if item.exception:
            raise RuntimeError(f"Exception for worker ID(s) {item.worker_ids}: {item.exception}")
        if item.item is None:
            raise RuntimeError("`item` should not be `None` unless there was an exception")
        return item.item

    def __enter__(self) -> Self:
        if self.processes is None and self.single_process_threads is None:
            if self.num_workers > 0:
                self.processes = []
                for i in range(self.num_workers):
                    process = mp.Process(
                        target=dataloader_worker,
                        args=(
                            self.dataloader_worker_init_fn,
                            self.dataset,
                            self.samples_queue,
                            self.stop_event,
                            i,
                            self.num_workers,
                            self.raise_errs,
                        ),
                        daemon=True,
                        name=f"dataloader-worker-{i}",
                    )
                    process.start()
                    self.processes.append(process)
                collate_process = mp.Process(
                    target=collate_worker,
                    args=(
                        self.collate_worker_init_fn,
                        self.samples_queue,
                        self.collated_queue,
                        self.dataset.collate,
                        self.stop_event,
                        self.batch_size,
                        self.raise_errs,
                    ),
                    name="dataloader-worker-collate",
                )
                collate_process.start()
                self.processes.append(collate_process)
            else:
                dataloader_thread = threading.Thread(
                    target=dataloader_worker,
                    args=(
                        self.dataloader_worker_init_fn,
                        self.dataset,
                        self.samples_queue,
                        self.stop_event,
                        0,
                        1,
                        self.raise_errs,
                    ),
                )
                dataloader_thread.start()
                collate_thread = threading.Thread(
                    target=collate_worker,
                    args=(
                        self.collate_worker_init_fn,
                        self.samples_queue,
                        self.collated_queue,
                        self.dataset.collate,
                        self.stop_event,
                        self.batch_size,
                        self.raise_errs,
                    ),
                )
                collate_thread.start()
                self.single_process_threads = (dataloader_thread, collate_thread)
        else:
            raise RuntimeError("Dataloader is already running")
        return self

    def __exit__(self, _t: type[BaseException] | None, _e: BaseException | None, _tr: TracebackType | None) -> None:
        if self.processes is None and self.single_process_threads is None:
            raise RuntimeError("DataLoader is not running")

        # Stops the dataloader workers by setting the stop event and clearing
        # the queue. Clearing the queue is important because otherwise the
        # workers will block while trying to put items on the queue.
        self.stop_event.set()
        try:
            while not self.collated_queue.empty():
                self.collated_queue.get_nowait()
        except Exception:
            pass
        try:
            while not self.samples_queue.empty():
                self.samples_queue.get_nowait()
        except Exception:
            pass

        # Joins the dataloader workers.
        if self.processes is not None:
            for process in self.processes:
                process.terminate()
            self.processes = None
        elif self.single_process_threads is not None:
            dataloader_thread, collate_thread = self.single_process_threads
            dataloader_thread.join()
            collate_thread.join()
            self.single_process_thread = None
        else:
            raise RuntimeError("Unexpected state")


class _DummyDataset(Dataset[int, list[int]]):
    def next(self) -> int:
        return random.randint(0, 10)

    def collate(self, items: list[int]) -> list[int]:
        assert random.random() > 0.1, "A random error"
        time.sleep(random.random() * 0.1)
        return items


def test_error_handling_dataset_adhoc(test_samples: int = 1000000) -> None:
    """Tests the error handling dataset."""
    Dataloader(
        ErrorHandlingDataset(
            _DummyDataset(),
            flush_every_n_steps=test_samples,
            flush_every_n_seconds=None,
        ),
        batch_size=2,
    ).test(
        max_samples=test_samples,
        print_fn=lambda i, sample: logger.info("Sample %d: %s", i, sample),
    )


if __name__ == "__main__":
    # python -m dpshdl.dataloader
    test_error_handling_dataset_adhoc()
