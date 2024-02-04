"""Tests the dataloader."""

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


@pytest.mark.parametrize("num_workers", [0, 1, 2])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_dataloader(num_workers: int, batch_size: int) -> None:
    ds = DummyDataset()
    with dl.Dataloader(ds, num_workers=num_workers, batch_size=batch_size) as loader:
        for sample in itertools.islice(loader, 10):
            assert sample.shape == (batch_size,)


if __name__ == "__main__":
    # python -m tests.utils.test_dataloader
    test_dataloader(0, 1)
