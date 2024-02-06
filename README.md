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

class MyDataset(Dataset[int, np.ndarray]):
    def next(self) -> int:
        return 1

# Loops forever.
with Dataloader(MyDataset(), batch_size=2) as loader:
    for sample in loader:
        assert sample.shape == (2,)
```

### Error Handling

You can wrap any dataset in an `ErrorHandlingDataset` to catch and log errors:

```python
from dpshdl.dataset import ErrorHandlingDataset

with Dataloader(ErrorHandlingDataset(MyDataset()), batch_size=2) as loader:
    ...
```

This wrapper will detect errors in the `next` function and log error summaries, to avoid crashing the entire program.

### Ad-hoc Testing

While developing datasets, you usually want to loop through a few samples to make sure everything is working. You can do this easily as follows:

```python
MyDataset().test(
    max_samples=100,
    handle_errors=True,  # To automatically wrap the dataset in an ErrorHandlingDataset.
    print_fn=lambda i, sample: print(f"Sample {i}: {sample}")
)
```

### Collating

This package provides a default implementation of dataset collating, which can be used as follows:

```python
from dpshdl.collate import collate

class MyDataset(Dataset[int, np.ndarray]):
    def collate(self, items: list[int]) -> np.ndarray:
        return collate(items)
```

Alternatively, you can implement your own custom collating strategy:

```python
from dpshdl.collate import collate

class MyDataset(Dataset[int, list[int]]):
    def collate(self, items: list[int]) -> list[int]:
        return items
```

There are additional arguments that can be passed to the `collate` function to automatically handle padding and batching:

```python
from dpshdl.collate import pad_all, pad_sequence
import functools
import random
import numpy as np

items = [np.random.random(random.randint(5, 10)) for _ in range(5)]  # Randomly sized arrays.
collate(items)  # Will fail because the arrays are of different sizes.
collate(items, pad=True)  # Use the default padding strategy.
collate(items, pad=functools.partial(pad_all, left_pad=True))  # Left-padding.
collate(items, pad=functools.partial(pad_sequence, dim=0, left_pad=True))  # Pads a specific dimension.
```

### Prefetching

Sometimes it is a good idea to trigger a host-to-device transfer before a batch of samples is needed, so that it can take place asynchronously while other computation is happening. This is called prefetching. This package provides a simple utility class to do this:

```python
from dpshdl.dataset import Dataset
from dpshdl.dataloader import Dataloader
from dpshdl.prefetcher import Prefetcher
import numpy as np
import torch
from torch import Tensor


class MyDataset(Dataset[int, np.ndarray]):
    def next(self) -> int:
        return 1


def to_device_func(sample: np.ndarray) -> Tensor:
    # Because this is non-blocking, the H2D transfer can take place in the
    # background while other computation is happening.
    return torch.from_numpy(sample).to("cuda", non_blocking=True)


with Prefetcher(to_device_func, Dataloader(MyDataset(), batch_size=2)) as loader:
    for sample in loader:
        assert sample.device.type == "cuda"
```
