# dpshdl

A framework-agnostic library for loading data.

## Installation

```bash
pip install dpshdl
```

## Usage

Datasets should override a single method, `next`, which returns a single sample.

```python
from dpshdl.dataset import Dataset
from dpshdl.dataloader import Dataloader
import numpy as np

class MyDataset(Dataset[np.ndarray, np.ndarray]):
    def next(self) -> int:
        return 1

    def collate(self, items: list[int]) -> np.ndarray:
        return np.array(items)

loader = Dataloader(MyDataset(), batch_size=2)

# Loops forever.
for sample in loader:
    assert sample.shape == (2,)
```
