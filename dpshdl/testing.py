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


def print_batch(
    indices: list[int],
    samples: list[T],
    truncate: int | None = 80,
    replace_whitespace: bool = True,
) -> None:
    """Prints a batch of samples.

    Args:
        indices: The indices of the samples.
        samples: The samples to print.
        truncate: If set, truncates the sample to this length.
        replace_whitespace: If set, replaces all whitespace with a single
            space.
    """
    batch_str = str(samples)
    if replace_whitespace:
        batch_str = re.sub(r"\s+", " ", batch_str)
    if truncate is not None and len(batch_str) > truncate:
        batch_str = batch_str[: truncate - 3] + "..."
    logger.info("Samples %d - %d: %s", indices[0], indices[-1], batch_str)


def run_test(
    ds: Iterable[T],
    max_samples: int = 10,
    log_interval: int | None = 1,
    print_fn: Callable[[int, T], None] = print_sample,
    batch_fn: Callable[[list[int], list[T]], None] = print_batch,
    batch_size: int | None = None,
    log_batch_interval: int | None = 1,
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
        batch_fn: The function to use for printing batches.
        batch_size: Call the ``batch_fn`` on batches of this size.
        log_batch_interval: Log a batch after this many batches.
    """
    configure_logging()
    start_time = time.time()
    samples: tuple[list[int], list[T]] | None = None if batch_size is None else ([], [])
    batch_id = 0
    for i, sample in enumerate(itertools.islice(ds, max_samples)):
        if log_interval is not None and i % log_interval == 0:
            print_fn(i, sample)
        if samples is not None:
            samples[0].append(i)
            samples[1].append(sample)
            if len(samples[0]) == batch_size:
                batch_id += 1
                if batch_id == log_batch_interval:
                    batch_fn(samples[0], samples[1])
                    batch_id = 0
                samples = [], []
    elapsed_time = time.time() - start_time
    samples_per_second = i / elapsed_time
    logger.info("Tested %d samples in %f seconds (%f samples per second)", i + 1, elapsed_time, samples_per_second)
