"""Defines an interface for loading data."""

import bdb
import logging
import random
import re
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections import Counter, deque
from dataclasses import dataclass
from queue import Queue
from typing import Callable, Deque, Generic, Iterator, Sequence, TypeVar, final

import numpy as np

from dpshdl.collate import collate
from dpshdl.numpy import worker_chunk
from dpshdl.testing import print_sample, run_test
from dpshdl.utils import TextBlock, render_text_blocks

logger = logging.getLogger(__name__)

T = TypeVar("T")  # The type of the dataset item.
Tc = TypeVar("Tc")  # The type of the collated item.
Tarrays = TypeVar("Tarrays", bound=tuple[np.ndarray, ...])


class Dataset(Iterator[T], Generic[T, Tc], ABC):
    """Defines the dataset interface.

    Datasets are analogous to a PyTorch iterable datasets that iterates
    forever. This means that there is no concept of an epoch or dataset size.
    """

    def __init__(self) -> None:
        super().__init__()

        self.worker_id = 0
        self.num_workers = 1

    def worker_init(self, worker_id: int, num_workers: int) -> None:
        """Initializes the dataset worker.

        This method is called once per worker when the dataset is used in a
        dataloader.

        Args:
            worker_id: The ID of the worker.
            num_workers: The number of workers in the worker pool.
            lock: A lock shared by all dataset workers in the dataloader.
        """
        self.worker_id = worker_id
        self.num_workers = num_workers

    @abstractmethod
    def next(self) -> T:
        """Returns the next item in the dataset.

        Returns:
            The next item in the dataset.
        """

    def collate(self, items: list[T]) -> Tc | None:
        """Collates  a list of items into a single item.

        Args:
            items: The items in a batch.

        Returns:
            The collated items.
        """
        return collate(items)

    @final
    def __iter__(self) -> "Dataset[T, Tc]":
        # Don't override this! Use `worker_init` instead.
        return self

    @final
    def __next__(self) -> T:
        # Don't override this! Use `next` instead.
        return self.next()

    def test(
        self,
        max_samples: int = 10,
        handle_errors: bool = False,
        log_interval: int | None = 1,
        print_fn: Callable[[int, T], None] = print_sample,
        batch_fn: Callable[[list[int], list[T]], None] | None = None,
        batch_size: int | None = None,
        log_batch_interval: int | None = 1,
    ) -> None:
        """Defines a function for doing adhoc testing of the dataset.

        Args:
            max_samples: The maximum number of samples to test.
            handle_errors: If set, wraps the dataset in an error handling
                wrapper that will catch and log exceptions.
            log_interval: How often to log a sample. If None, don't log any
                samples.
            print_fn: The function to use for printing samples.
            batch_fn: The function to use for printing batches.
            batch_size: Call the ``batch_fn`` on batches of this size.
            log_batch_interval: Log a batch after this many batches.
        """
        ds = (
            ErrorHandlingDataset(
                self,
                flush_every_n_steps=max_samples,
                flush_every_n_seconds=None,
            )
            if handle_errors
            else self
        )

        if batch_fn is None:

            def batch_fn(
                indices: list[int],
                samples: list[T],
                truncate: int | None = 80,
                replace_whitespace: bool = True,
            ) -> None:
                batch = ds.collate(samples)
                batch_str = str(batch)
                if replace_whitespace:
                    batch_str = re.sub(r"\s+", " ", batch_str)
                if truncate is not None and len(batch_str) > truncate:
                    batch_str = batch_str[: truncate - 3] + "..."
                logger.info("Samples %d - %d: %s", indices[0], indices[-1], batch_str)

        run_test(
            ds=ds,
            max_samples=max_samples,
            log_interval=log_interval,
            print_fn=print_fn,
            batch_fn=batch_fn,
            batch_size=batch_size,
            log_batch_interval=log_batch_interval,
        )


class TensorDataset(Dataset[Tarrays, Tarrays], Generic[Tarrays]):
    """Defines a dataset that yields samples from a tensor.

    All provided tensors should have the same shape in the ``dim`` dimension.

    Parameters:
        tensors: The tensors to sample from.
        dim: The dimension to sample from.
        stack_dim: The dimension to stack the tensors on.
    """

    def __init__(self, *tensors: np.ndarray, dim: int = 0, stack_dim: int = 0) -> None:
        super().__init__()

        self.tensors = tensors
        self.dim = dim
        self.stack_dim = stack_dim

        self._worker_tensors = list(tensors)

        # Gets the number of samples.
        self.num_samples = tensors[0].shape[dim]
        self._worker_num_samples = self.num_samples
        if not all(t.shape[dim] == self.num_samples for t in tensors):
            raise ValueError("All tensors must have the same shape in the specified dimension.")

        self.rand = np.random.RandomState(0)

    def worker_init(self, worker_id: int, num_workers: int) -> None:
        super().worker_init(worker_id, num_workers)

        self._worker_tensors = [worker_chunk(t, worker_id, num_workers) for t in self.tensors]
        self._worker_num_samples = self._worker_tensors[0].shape[self.dim]
        if not all(t.shape[self.dim] == self._worker_num_samples for t in self._worker_tensors):
            raise ValueError("All tensors must have the same shape in the specified dimension.")

    def next(self) -> Tarrays:
        index = self.rand.randint(0, self._worker_num_samples)
        return tuple(np.take(t, index, axis=0) for t in self._worker_tensors)  # type: ignore[return-value]

    def collate(self, items: list[Tarrays]) -> Tarrays:
        collated_items = (np.stack([item[i] for item in items]) for i in range(len(items[0])))
        return tuple(collated_items)  # type: ignore[return-value]


class ChunkedDataset(Dataset[T, Tc], Generic[T, Tc], ABC):
    """Defines a dataset that yields chunks of samples in a separate thread.

    This dataset is useful for implementing IO bound datasets such as
    datasets that read from disk or download data from the internet. Subclasses
    should implement the `next_chunk` method, which iterates over the next
    chunk of data. For example, a dataset that reads from a spinning disk
    might implement `next_chunk` as follows:

    .. code-block:: python

        def next_chunk(self) -> Iterator[T]:
            path = self.paths[self.current_path]
            self.current_path = (self.current_path + 1) % len(self.paths)
            with open(path, "rb", buffering=BIG_BUFFERING_SIZE) as fh:
                for line in fh:
                    yield line

    In the above example, we use a large buffer size to reduce the number of
    IO operations. Since this happens in a separate thread, we don't need to
    worry about creating a new file pointer for each chunk.

    Parameters:
        max_queue_size: The maximum number of samples to keep in the
            queue at a time.
    """

    def __init__(self, max_queue_size: int = 0) -> None:
        super().__init__()

        self._max_queue_size = max_queue_size
        self._next_chunk_thread: tuple[threading.Thread, Queue[T], Queue[Exception]] | None = None

    @abstractmethod
    def next_chunk(self) -> Iterator[T]:
        """Returns the next chunk of data.

        Returns:
            The next chunk of data.
        """

    def chunked_dataset_thread(self, next_chunk_queue: Queue[T], error_queue: Queue[Exception]) -> None:
        while True:
            try:
                for sample in self.next_chunk():
                    next_chunk_queue.put(sample)
            except (bdb.BdbQuit, KeyboardInterrupt, StopIteration):
                raise
            except Exception as e:
                error_queue.put(e)

    def next(self) -> T:
        if self._next_chunk_thread is None:
            next_chunk_queue: Queue[T] = Queue(maxsize=self._max_queue_size)
            error_queue: Queue[Exception] = Queue()
            thread = threading.Thread(
                target=self.chunked_dataset_thread,
                args=(next_chunk_queue, error_queue),
                daemon=True,
            )
            thread.start()
            self._next_chunk_thread = (thread, next_chunk_queue, error_queue)

        # Gets the next sample event and the queues.
        _, next_chunk_queue, error_queue = self._next_chunk_thread

        # If there are any errors in the error queue, raise them.
        if not error_queue.empty():
            raise error_queue.get()

        return next_chunk_queue.get()


class RoundRobinDataset(Dataset[T, Tc], Generic[T, Tc]):
    """Defines a dataset that yields samples in round robin fashion.

    Parameters:
        datasets: The datasets to sample from.
    """

    def __init__(self, datasets: Sequence[Dataset[T, Tc]], collate_fn: Callable[[list[T]], Tc | None]) -> None:
        super().__init__()

        self.datasets = datasets
        self.collate_fn = collate_fn

        self.i = 0

    def worker_init(self, worker_id: int, num_workers: int) -> None:
        super().worker_init(worker_id, num_workers)

        for dataset in self.datasets:
            dataset.worker_init(worker_id, num_workers)

    def next(self) -> T:
        next_item = self.datasets[self.i].next()
        self.i = (self.i + 1) % len(self.datasets)
        return next_item

    def collate(self, items: list[T]) -> Tc | None:
        return self.collate_fn(items)


class RandomDataset(Dataset[T, Tc], Generic[T, Tc]):
    """Defines a dataset that randomly samples from a list of datasets.

    Parameters:
        datasets: The datasets to sample from.
    """

    def __init__(
        self,
        datasets: Sequence[Dataset[T, Tc]],
        collate_fn: Callable[[list[T]], Tc | None],
        stop_on_first: bool = False,
    ) -> None:
        super().__init__()

        self.datasets = datasets
        self.collate_fn = collate_fn
        self.stop_on_first = stop_on_first

    def worker_init(self, worker_id: int, num_workers: int) -> None:
        super().worker_init(worker_id, num_workers)

        for dataset in self.datasets:
            dataset.worker_init(worker_id, num_workers)

    def next(self) -> T:
        return random.choice(self.datasets).next()

    def collate(self, items: list[T]) -> Tc | None:
        return self.collate_fn(items)


class InMemoryDataset(Dataset[T, Tc], Generic[T, Tc]):
    """Repeatedly yields from a pool of samples which are stored in-memory.

    Parameters:
        dataset: The dataset to draw samples for the pool from.
        num_samples: The maximum size of the pool.
        yield_in_order: If True, yield the samples in the order they were
            drawn from the original dataset, otherwise yield randomly.
    """

    def __init__(
        self,
        dataset: Dataset[T, Tc],
        num_samples: int,
        yield_in_order: bool = False,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.num_samples = num_samples
        self.yield_in_order = yield_in_order

        self.pool: Deque[T] = deque()

    def worker_init(self, worker_id: int, num_workers: int) -> None:
        super().worker_init(worker_id, num_workers)

        self.dataset.worker_init(worker_id, num_workers)

    def next(self) -> T:
        if len(self.pool) < self.num_samples:
            self.pool.append(self.dataset.next())
        if self.yield_in_order:
            item = self.pool[0]
            self.pool.rotate(-1)
        else:
            item = self.pool[random.randint(0, len(self.pool) - 1)]
        return item

    def collate(self, items: list[T]) -> Tc | None:
        return self.dataset.collate(items)


def get_loc(num_excs: int = 1) -> str:
    _, _, exc_tb = sys.exc_info()
    if exc_tb is None or (exc_tb := exc_tb.tb_next) is None:
        return "unknown"
    exc_strs: list[str] = []
    for _ in range(num_excs):
        exc_strs += [f"{exc_tb.tb_frame.f_code.co_filename}:{exc_tb.tb_lineno}"]
        if (exc_tb := exc_tb.tb_next) is None:
            break
    return "\n".join(exc_strs)


@dataclass(frozen=True)
class ExceptionSummary:
    title: str
    num_steps: int
    elapsed_time: float
    num_exceptions: int
    top_exception_messages: list[tuple[str, int]]
    top_exception_types: list[tuple[str, int]]
    top_exception_locations: list[tuple[str, int]]
    last_exception: Exception | None

    def __str__(self) -> str:
        blocks: list[list[TextBlock]] = []

        blocks += [
            [
                TextBlock(
                    f"{self.title} ({self.num_steps} steps, {self.elapsed_time:.2f} seconds)",
                    color="red",
                    bold=True,
                    width=60,
                    center=True,
                ),
                TextBlock("Count", color="yellow", bold=False, width=10, center=True),
                TextBlock("Percent", color="yellow", bold=False, width=10, center=True),
            ],
        ]

        def get_header(s: str) -> list[list[TextBlock]]:
            return [
                [
                    TextBlock(s, color="yellow", bold=True, width=60),
                    TextBlock("", width=10),
                    TextBlock("", width=10),
                ],
            ]

        def get_line(ks: str, v: int) -> list[list[TextBlock]]:
            line = [
                TextBlock(ks, width=60, no_sep=True),
                TextBlock(f"{v}", width=10, no_sep=True),
                TextBlock(f"{v / self.num_steps * 100:.2f}%", width=10, no_sep=True),
            ]
            return [line]

        # Logs unique exception strings.
        blocks += get_header("Exceptions")
        for k, v in self.top_exception_messages:
            blocks += get_line(k, v)

        # Logs the individual exception classes.
        blocks += get_header("Types")
        for k, v in self.top_exception_types:
            blocks += get_line(k, v)

        # Logs by line number.
        blocks += get_header("Locations")
        for k, v in self.top_exception_locations:
            blocks += get_line(k, v)

        # Logs the total number of exceptions.
        blocks += get_header("Total")
        blocks += get_line("Total", self.num_exceptions)

        return render_text_blocks(blocks)


class ExceptionSummaryWriter:
    """Defines a utility class for storing and logging exceptions.

    Parameters:
        title: The title for each summary.
        max_exceptions: The maximum number of unique exceptions to log.
    """

    def __init__(self, title: str, max_exceptions: int = 10) -> None:
        super().__init__()

        self.title = title
        self.max_exceptions = max_exceptions

        self.exceptions: Counter[str] = Counter()
        self.exception_classes: Counter[str] = Counter()
        self.exception_locs: Counter[str] = Counter()

        self.last_exception: Exception | None = None
        self.num_steps = 0
        self.start_time = time.time()
        self.step_has_error = False
        self.total_exceptions = 0

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    def next(self) -> None:
        self.num_steps += 1
        self.step_has_error = False

    def __len__(self) -> int:
        return len(self.exceptions)

    def __bool__(self) -> bool:
        return len(self.exceptions) > 0

    def add_exception(self, exc: Exception, loc: str) -> None:
        self.last_exception = exc
        self.exceptions[f"{exc.__class__.__name__}: {exc}"] += 1
        self.exception_classes[exc.__class__.__name__] += 1
        self.exception_locs[loc] += 1
        if not self.step_has_error:
            self.step_has_error = True
            self.total_exceptions += 1

    def summary(self) -> ExceptionSummary:
        return ExceptionSummary(
            title=self.title,
            num_steps=self.num_steps,
            elapsed_time=self.elapsed_time,
            num_exceptions=self.total_exceptions,
            top_exception_messages=self.exceptions.most_common(self.max_exceptions),
            top_exception_types=self.exception_classes.most_common(self.max_exceptions),
            top_exception_locations=self.exception_locs.most_common(self.max_exceptions),
            last_exception=self.last_exception,
        )

    def clear(self) -> None:
        self.exceptions.clear()
        self.exception_classes.clear()
        self.exception_locs.clear()

        self.num_steps = 0
        self.start_time = time.time()
        self.step_has_error = False
        self.total_exceptions = 0

    def __str__(self) -> str:
        return str(self.summary())


class ErrorHandlingDataset(Dataset[T, Tc]):
    """Defines a wrapper for safely handling errors in iterable datasets.

    Parameters:
        dataset: The dataset to wrap.
        sleep_backoff: The initial sleep time after an exception.
        sleep_backoff_power: The power to raise the sleep time by after
            each consecutive exception.
        maximum_exceptions: The maximum number of consecutive exceptions
            to allow before raising an error.
        backoff_after: The number of exceptions to allow before backing
            off (i.e. increasing the sleep time).
        traceback_depth: The number of stack frames to include in the
            exception traceback.
        flush_every_n_steps: Flush the exception summary every N steps.
        flush_every_n_seconds: Flush the exception summary every N seconds.
        log_exceptions_all_workers: If set, log exceptions from all workers.
        log_level: The log level to use for logging exceptions.
    """

    def __init__(
        self,
        dataset: Dataset[T, Tc],
        sleep_backoff: float = 0.1,
        sleep_backoff_power: float = 2.0,
        maximum_exceptions: int = 10,
        backoff_after: int = 5,
        traceback_depth: int = 3,
        flush_every_n_steps: int | None = None,
        flush_every_n_seconds: float | None = 60.0,
        log_exceptions_all_workers: bool = False,
        log_level: int = logging.INFO,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.sleep_backoff = sleep_backoff
        self.sleep_backoff_power = sleep_backoff_power
        self.maximum_exceptions = maximum_exceptions
        self.backoff_after = backoff_after
        self.traceback_depth = traceback_depth
        self.flush_every_n_steps = flush_every_n_steps
        self.flush_every_n_seconds = flush_every_n_seconds
        self.log_exceptions_all_workers = log_exceptions_all_workers
        self.log_exceptions = True
        self.log_level = log_level

        self.exc_summary = ExceptionSummaryWriter("Error Summary")
        self.col_exc_summary = ExceptionSummaryWriter("Collate Error Summary")

    def should_flush_summary(self) -> bool:
        if self.flush_every_n_steps is not None and self.exc_summary.num_steps >= self.flush_every_n_steps:
            return True
        if self.flush_every_n_seconds is not None and self.exc_summary.elapsed_time >= self.flush_every_n_seconds:
            return True
        return False

    def should_flush_col_summary(self) -> bool:
        if self.flush_every_n_steps is not None and self.col_exc_summary.num_steps >= self.flush_every_n_steps:
            return True
        if self.flush_every_n_seconds is not None and self.col_exc_summary.elapsed_time >= self.flush_every_n_seconds:
            return True
        return False

    def worker_init(self, worker_id: int, num_workers: int) -> None:
        super().worker_init(worker_id, num_workers)

        self.dataset.worker_init(worker_id, num_workers)
        if worker_id != 0 and not self.log_exceptions_all_workers:
            self.log_exceptions = False

    def next(self) -> T:
        num_exceptions = 0
        backoff_time = self.sleep_backoff
        self.exc_summary.next()

        if self.should_flush_summary():
            if self.log_exceptions and self.exc_summary:
                logger.warning("Exception summary:\n%s", self.exc_summary.summary())
            self.exc_summary.clear()

        while num_exceptions < self.maximum_exceptions:
            try:
                return self.dataset.next()
            except (bdb.BdbQuit, KeyboardInterrupt, StopIteration):
                raise
            except Exception as e:
                self.exc_summary.add_exception(e, get_loc(self.traceback_depth))
            num_exceptions += 1
            if num_exceptions > self.backoff_after:
                logger.error("Encountered %d exceptions, backing off for %f seconds", num_exceptions, backoff_time)
                time.sleep(backoff_time)
                backoff_time *= self.sleep_backoff_power
        raise RuntimeError(f"Reached max exceptions {self.maximum_exceptions}\n{self.exc_summary.summary()}")

    def collate(self, items: list[T]) -> Tc | None:
        self.col_exc_summary.next()

        if self.should_flush_col_summary():
            if self.log_exceptions and self.col_exc_summary:
                logger.log(self.log_level, "Exception summary:\n%s", self.col_exc_summary.summary())
            self.col_exc_summary.clear()

        try:
            return self.dataset.collate(items)
        except (bdb.BdbQuit, KeyboardInterrupt, StopIteration):
            raise
        except Exception as e:
            self.col_exc_summary.add_exception(e, get_loc(self.traceback_depth))
            return None


def test_error_handling_dataset_adhoc() -> None:
    """Tests the error handling dataset."""

    class DummyDataset(Dataset[int, list[int]]):
        def next(self) -> int:
            n = random.random()
            if n < 0.1:
                assert False
            elif n < 0.2:
                raise ValueError("ValueError")
            elif n < 0.3:
                1 / 0
            return random.randint(0, 10)

        def collate(self, items: list[int]) -> list[int]:
            return items

    DummyDataset().test(
        max_samples=100,
        handle_errors=True,
        print_fn=lambda i, sample: logger.info("Sample %d: %d", i, sample),
        batch_size=10,
    )


if __name__ == "__main__":
    # python -m dpshdl.dataset
    test_error_handling_dataset_adhoc()
