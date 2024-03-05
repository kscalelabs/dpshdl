"""Defines a utility class for pre-loading samples into device memory.

When you are training a model, you usually get a sample from the dataloader,
then need to move it into device memory. This host-to-device transfer can be
slow, so it is beneficial to pre-load the next sample into device memory while
the current sample is being processed.
"""

from queue import Queue
from threading import Event, Thread
from types import TracebackType
from typing import Callable, ContextManager, Generic, Iterable, Iterator, TypeVar

from dpshdl.testing import print_sample, run_test

T = TypeVar("T")
Tc_co = TypeVar("Tc_co", covariant=True)
Tp_co = TypeVar("Tp_co", covariant=True)


class Prefetcher(Iterable[Tp_co], Generic[Tc_co, Tp_co]):
    """Helper class for pre-loading samples into device memory."""

    def __init__(
        self,
        to_device_func: Callable[[Tc_co], Tp_co],
        dataloader: Iterator[Tc_co],
        prefetch_size: int = 2,
    ) -> None:
        super().__init__()

        self.to_device_func = to_device_func
        self.dataloader = dataloader
        self.sample_queue: Queue[Tp_co] = Queue(maxsize=prefetch_size)
        self.stop_event = Event()
        self.enqueue_thread: Thread | None = None

    def _enqueue_samples(self) -> None:
        for sample in self.dataloader:
            if self.stop_event.is_set():
                break
            self.sample_queue.put(self.to_device_func(sample))

    def __iter__(self) -> Iterator[Tp_co]:
        if self.enqueue_thread is None:
            raise RuntimeError("Prefetcher is not running.")
        return self

    def __next__(self) -> Tp_co:
        if self.enqueue_thread is None:
            raise RuntimeError("Prefetcher is not running.")
        return self.sample_queue.get()

    def __enter__(self) -> "Prefetcher[Tc_co, Tp_co]":
        if isinstance(self.dataloader, ContextManager):
            self.dataloader = self.dataloader.__enter__()
        if self.enqueue_thread is not None:
            raise RuntimeError("Prefetcher is already running.")
        self.enqueue_thread = Thread(target=self._enqueue_samples, daemon=True)
        self.enqueue_thread.start()
        return self

    def __exit__(self, _t: type[BaseException] | None, _e: BaseException | None, _tr: TracebackType | None) -> None:
        if self.enqueue_thread is None:
            raise RuntimeError("Prefetcher is not running.")
        self.stop_event.set()
        self.enqueue_thread.join()
        self.enqueue_thread = None
        if isinstance(self.dataloader, ContextManager):
            self.dataloader.__exit__(_t, _e, _tr)

    def test(
        self,
        max_samples: int = 10,
        log_interval: int | None = 1,
        print_fn: Callable[[int, Tp_co], None] = print_sample,
    ) -> None:
        """Defines a function for doing adhoc testing of the prefetcher.

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
