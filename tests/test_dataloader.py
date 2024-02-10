"""Tests the dataloader."""

import itertools
import random
from typing import Iterator

import numpy as np
import pytest

from dpshdl.collate import collate
from dpshdl.dataloader import Dataloader
from dpshdl.dataset import ChunkedDataset, Dataset


class DummyDataset(Dataset[int, np.ndarray]):
    def start(self) -> None:
        pass

    def next(self) -> int:
        return random.randint(0, 5)

    def collate(self, items: list[int]) -> np.ndarray:
        return collate(items)


@pytest.mark.parametrize("num_workers", [0, 1, 2])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_dataloader(num_workers: int, batch_size: int) -> None:
    ds = DummyDataset()
    with Dataloader(ds, num_workers=num_workers, batch_size=batch_size) as loader:
        for sample in itertools.islice(loader, 10):
            assert sample.shape == (batch_size,)


class DummyChunkedDataset(ChunkedDataset[int, np.ndarray]):
    def __init__(self, value: int) -> None:
        super().__init__()

        self.value = value

    def next_chunk(self) -> Iterator[int]:
        for i in range(self.value):
            yield i

    def collate(self, items: list[int]) -> np.ndarray:
        return collate(items)


@pytest.mark.parametrize("num_workers", [1, 2])
def test_chunked_dataloader(num_workers: int) -> None:
    ds = DummyChunkedDataset(5)
    with Dataloader(ds, num_workers=num_workers, batch_size=2) as loader:
        assert all(t.shape == (2,) for t in itertools.islice(loader, 10))


if __name__ == "__main__":
    # python -m tests.test_dataloader
    test_dataloader(0, 1)
