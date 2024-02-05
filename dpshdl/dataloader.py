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
import threading
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from queue import Queue
from threading import Event
from types import TracebackType
from typing import Callable, Generic, Self, TypeVar

from dpshdl.dataset import Dataset
from dpshdl.testing import print_sample, run_test

logger = logging.getLogger(__name__)

T = TypeVar("T")
Tc = TypeVar("Tc")


@dataclass(frozen=True)
class DataloaderItem(Generic[T]):
    item: T | None
    exception: BaseException | None
    worker_id: int


def dataloader_worker(
    dataset: Dataset[T, Tc],
    queue: "Queue[DataloaderItem[T]]",
    stop_event: Event,
    worker_id: int,
    num_workers: int,
) -> None:
    dataset.worker_init(worker_id, num_workers)
    dataset_iterator = dataset.__iter__()
    while True:
        if stop_event.is_set():
            return
        try:
            sample = dataset_iterator.__next__()
            queue.put(DataloaderItem(sample, None, worker_id))
        except BaseException as e:
            queue.put(DataloaderItem(None, e, worker_id))
            break


def collate_worker(
    samples_queue: "Queue[DataloaderItem[T]]",
    collated_queue: "Queue[DataloaderItem[Tc]]",
    collate_fn: Callable[[list[T]], Tc],
    stop_event: Event,
    batch_size: int,
) -> None:
    samples: list[T] = []
    while True:
        if stop_event.is_set():
            return
        try:
            sample = samples_queue.get()
            if sample.exception is not None:
                collated_queue.put(DataloaderItem(None, sample.exception, sample.worker_id))
                break
            if sample.item is None:
                raise RuntimeError("`item` should not be `None` unless there was an exception")
            samples.append(sample.item)
            if len(samples) == batch_size:
                collated_queue.put(DataloaderItem(collate_fn(samples), None, -1))
                samples = []
        except BaseException as e:
            collated_queue.put(DataloaderItem(None, e, -1))
            break


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
    """

    def __init__(
        self,
        dataset: Dataset[T, Tc],
        num_workers: int | None = None,
        batch_size: int = 1,
        prefetch_factor: int = 2,
        ctx: BaseContext | None = None,
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
        self.manager = self.ctx.Manager()

        self.processes: list[mp.Process] | None = None
        self.single_process_threads: tuple[threading.Thread, threading.Thread] | None = None
        self.samples_queue: Queue[DataloaderItem[T]] = self.manager.Queue(maxsize=batch_size * prefetch_factor)
        self.collate_queue: Queue[DataloaderItem[Tc]] = self.manager.Queue(maxsize=prefetch_factor)
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
        run_test(self, max_samples, log_interval, print_fn)

    def __iter__(self) -> "Dataloader[T, Tc]":
        return self

    def __next__(self) -> Tc:
        if self.processes is None and self.single_process_threads is None:
            raise RuntimeError("Dataloader is not running")
        item = self.collate_queue.get()
        if item.exception:
            raise item.exception
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
                        args=(self.dataset, self.samples_queue, self.stop_event, i, self.num_workers),
                        daemon=True,
                        name=f"dataloader-worker-{i}",
                    )
                    process.start()
                    self.processes.append(process)
                collate_process = mp.Process(
                    target=collate_worker,
                    args=(
                        self.samples_queue,
                        self.collate_queue,
                        self.dataset.collate,
                        self.stop_event,
                        self.batch_size,
                    ),
                )
                collate_process.start()
                self.processes.append(collate_process)
            else:
                dataloader_thread = threading.Thread(
                    target=dataloader_worker,
                    args=(self.dataset, self.samples_queue, self.stop_event, 0, 1),
                )
                dataloader_thread.start()
                collate_thread = threading.Thread(
                    target=collate_worker,
                    args=(
                        self.samples_queue,
                        self.collate_queue,
                        self.dataset.collate,
                        self.stop_event,
                        self.batch_size,
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
        while not self.samples_queue.empty():
            self.samples_queue.get()
        while not self.collate_queue.empty():
            self.collate_queue.get()

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
