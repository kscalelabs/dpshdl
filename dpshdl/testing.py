"""Utility functions for testing datasets and dataloaders."""

import itertools
import logging
import re
import time
from typing import Callable, Iterable, TypeVar

from dpshdl.utils import configure_logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


def print_sample(
    index: int,
    sample: T,
    truncate: int | None = 80,
    replace_whitespace: bool = True,
) -> None:
    """Prints a sample from a dataset.

    Args:
        index: The index of the sample.
        sample: The sample to print.
        truncate: If set, truncates the sample to this length.
        replace_whitespace: If set, replaces all whitespace with a single
            space.
    """
    sample_str = str(sample)
    if replace_whitespace:
        sample_str = re.sub(r"\s+", " ", sample_str)
    if truncate is not None and len(sample_str) > truncate:
        sample_str = sample_str[: truncate - 3] + "..."
    logger.info("Sample %d: %s", index, sample_str)


def run_test(
    ds: Iterable[T],
    max_samples: int = 10,
    log_interval: int | None = 1,
    print_fn: Callable[[int, T], None] = print_sample,
) -> None:
    """Defines a function for doing adhoc testing of the dataset.

    Args:
        ds: The dataset to test.
        max_samples: The maximum number of samples to test.
        handle_errors: If set, wraps the dataset in an error handling
            wrapper that will catch and log exceptions.
        log_interval: How often to log a sample. If None, don't log any
            samples.
        print_fn: The function to use for printing samples.
    """
    configure_logging()
    start_time = time.time()
    for i, sample in enumerate(itertools.islice(ds, max_samples)):
        if log_interval is not None and i % log_interval == 0:
            print_fn(i, sample)
    elapsed_time = time.time() - start_time
    samples_per_second = i / elapsed_time
    logger.info("Tested %d samples in %f seconds (%f samples per second)", i + 1, elapsed_time, samples_per_second)
