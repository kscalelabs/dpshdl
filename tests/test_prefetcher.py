"""Prefetcher tests."""

import itertools
import random

import numpy as np
import pytest

from dpshdl.collate import collate
from dpshdl.dataloader import Dataloader
from dpshdl.dataset import Dataset
from dpshdl.prefetcher import Prefetcher


class DummyDataset(Dataset[int, np.ndarray]):
    def start(self) -> None:
        pass

    def next(self) -> int:
        return random.randint(0, 5)

    def collate(self, items: list[int]) -> np.ndarray:
        return collate(items)


def to_device_fn(sample: np.ndarray) -> np.ndarray:
    return sample


@pytest.mark.parametrize("num_workers", [0, 1, 2])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_dataloader(num_workers: int, batch_size: int) -> None:
    ds = DummyDataset()
    ld = Dataloader(ds, num_workers=num_workers, batch_size=batch_size)
    with Prefetcher(to_device_fn, ld) as pf:
        for sample in itertools.islice(pf, 10):
            assert sample.shape == (batch_size,)


if __name__ == "__main__":
    # python -m tests.test_prefetcher
    test_dataloader(0, 1)
