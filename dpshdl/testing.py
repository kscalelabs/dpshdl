"""Utility functions for testing datasets and dataloaders."""

import itertools
import logging
import re
import time
from typing import Iterable

from dpshdl.utils import configure_logging

logger = logging.getLogger(__name__)


def run_test(
    ds: Iterable,
    max_samples: int = 10,
    log_interval: int | None = 1,
    truncate: int | None = 80,
    replace_whitespace: bool = True,
) -> None:
    """Defines a function for doing adhoc testing of the dataset.

    Args:
        ds: The dataset to test.
        max_samples: The maximum number of samples to test.
        handle_errors: If set, wraps the dataset in an error handling
            wrapper that will catch and log exceptions.
        log_interval: How often to log a sample. If None, don't log any
            samples.
        truncate: The maximum number of characters to show in a sample.
            If None, shows the entire sample.
        replace_whitespace: If set, replaces whitespace characters with
            spaces.
    """
    configure_logging()
    start_time = time.time()
    ws_regex = re.compile(r"\s+") if replace_whitespace else None
    for i, sample in enumerate(itertools.islice(ds, max_samples)):
        if log_interval is not None and i % log_interval == 0:
            sample_str = str(sample)
            if ws_regex is not None:
                sample_str = ws_regex.sub(" ", sample_str)
            if truncate is not None and len(sample_str) > truncate:
                sample_str = sample_str[: truncate - 3] + "..."
            logger.info("Sample %d: %s", i, sample_str)
    elapsed_time = time.time() - start_time
    samples_per_second = i / elapsed_time
    logger.info("Tested %d samples in %f seconds (%f samples per second)", i + 1, elapsed_time, samples_per_second)
