"""Defines a utility class for pre-loading samples into device memory.

When you are training a model, you usually get a sample from the dataloader,
then need to move it into device memory. This host-to-device transfer can be
slow, so it is beneficial to pre-load the next sample into device memory while
the current sample is being processed.
"""

from types import TracebackType
from typing import Any, Callable, Generic, Iterable, Iterator, TypeVar

from dpshdl.dataloader import Dataloader

T = TypeVar("T")
Tc_co = TypeVar("Tc_co", covariant=True)


class Prefetcher(Iterable[Tc_co], Generic[Tc_co]):
    """Helper class for pre-loading samples into device memory."""

    def __init__(self, to_device_func: Callable[[Any], Any], dataloader: Dataloader[T, Tc_co]) -> None:
        super().__init__()

        self.to_device_func = to_device_func
        self.dataloader = dataloader
        self.next_sample: Tc_co | None = None

    def get_sample(self) -> Tc_co:
        return self.to_device_func(next(self.dataloader))

    def __iter__(self) -> Iterator[Tc_co]:
        return self

    def __next__(self) -> Tc_co:
        if self.next_sample is None:
            self.next_sample = self.get_sample()
        sample, self.next_sample = self.next_sample, self.get_sample()
        return sample

    def __enter__(self) -> "Prefetcher[Tc_co]":
        self.dataloader.__enter__()
        return self

    def __exit__(self, _t: type[BaseException] | None, _e: BaseException | None, _tr: TracebackType | None) -> None:
        self.dataloader.__exit__(_t, _e, _tr)
