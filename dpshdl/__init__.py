"""Defines the top-level dataload API."""

__version__ = "0.0.1"

from .collate import CollateMode, collate, collate_non_null
from .dataloader import Dataloader, DataloaderItem
from .dataset import ChunkedDataset, Dataset, ErrorHandlingDataset, RandomDataset, RoundRobinDataset
from .experiments import FileDownloader, check_md5, check_sha256
from .impl.mnist import MNIST
from .numpy import one_hot, partial_flatten, worker_chunk
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
