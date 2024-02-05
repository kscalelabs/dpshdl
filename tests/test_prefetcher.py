"""Prefetcher tests."""

import itertools
import random

import numpy as np
import pytest

import dpshdl as dl


class DummyDataset(dl.Dataset[int, np.ndarray]):
    def start(self) -> None:
        pass

    def next(self) -> int:
        return random.randint(0, 5)

    def collate(self, items: list[int]) -> np.ndarray:
        return dl.collate_non_null(items)


def to_device_fn(sample: np.ndarray) -> np.ndarray:
    return sample


@pytest.mark.parametrize("num_workers", [0, 1, 2])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_dataloader(num_workers: int, batch_size: int) -> None:
    ds = DummyDataset()
    ld = dl.Dataloader(ds, num_workers=num_workers, batch_size=batch_size)
    with dl.Prefetcher(to_device_fn, ld) as pf:
        for sample in itertools.islice(pf, 10):
            assert sample.shape == (batch_size,)


if __name__ == "__main__":
    # python -m tests.test_prefetcher
    test_dataloader(0, 1)
