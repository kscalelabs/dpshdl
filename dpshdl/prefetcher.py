"""Defines a utility class for pre-loading samples into device memory.

When you are training a model, you usually get a sample from the dataloader,
then need to move it into device memory. This host-to-device transfer can be
slow, so it is beneficial to pre-load the next sample into device memory while
the current sample is being processed.
"""

from types import TracebackType
from typing import Callable, Generic, Iterable, Iterator, TypeVar

from dpshdl.dataloader import Dataloader
from dpshdl.testing import print_sample, run_test

T = TypeVar("T")
Tc_co = TypeVar("Tc_co", covariant=True)
Tp_co = TypeVar("Tp_co", covariant=True)


class Prefetcher(Iterable[Tp_co], Generic[Tc_co, Tp_co]):
    """Helper class for pre-loading samples into device memory."""

    def __init__(self, to_device_func: Callable[[Tc_co], Tp_co], dataloader: Dataloader[T, Tc_co]) -> None:
        super().__init__()

        self.to_device_func = to_device_func
        self.dataloader = dataloader
        self.next_sample: Tp_co | None = None

    def get_sample(self) -> Tp_co:
        return self.to_device_func(next(self.dataloader))

    def __iter__(self) -> Iterator[Tp_co]:
        return self

    def __next__(self) -> Tp_co:
        if self.next_sample is None:
            self.next_sample = self.get_sample()
        sample, self.next_sample = self.next_sample, self.get_sample()
        return sample

    def __enter__(self) -> "Prefetcher[Tc_co, Tp_co]":
        self.dataloader.__enter__()
        return self

    def __exit__(self, _t: type[BaseException] | None, _e: BaseException | None, _tr: TracebackType | None) -> None:
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
