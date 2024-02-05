"""Defines the top-level dataload API."""

__version__ = "0.0.3"

from ._collate import CollateMode, collate, collate_non_null
from ._dataloader import Dataloader, DataloaderItem
from ._dataset import ChunkedDataset, Dataset, ErrorHandlingDataset, RandomDataset, RoundRobinDataset, TensorDataset
from ._experiments import FileDownloader, check_md5, check_sha256
from ._impl.mnist import MNIST
from ._numpy import one_hot, partial_flatten, worker_chunk
from ._prefetcher import Prefetcher
from .utils import (
    COLOR_INDEX,
    Color,
    ColoredFormatter,
    TextBlock,
    color_parts,
    colored,
    configure_logging,
    format_timedelta,
    outlined,
    render_text_blocks,
    show_error,
    show_info,
    show_warning,
    uncolored,
    wrapped,
)
