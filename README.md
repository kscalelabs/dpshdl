# dpshdl

A framework-agnostic library for loading data.

## Installation

```bash
pip install dpshdl
```

## Usage

Datasets should override a single method, `next`, which returns a single sample.

```python
import dpshdl as dl
import numpy as np

class MyDataset(dl.Dataset[np.ndarray]):
    def next(self) -> np.ndarray:
        return np.random.rand(10)

loader = dl.Dataloader(MyDataset(), batch_size=2)

# Loops forever.
for sample in loader:
    print(sample)
```
