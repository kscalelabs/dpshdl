"""Implements the MNIST dataset."""

import array
import gzip
import logging
import struct
from pathlib import Path
from typing import Literal

import numpy as np

from dpshdl.dataset import TensorDataset
from dpshdl.experiments import FileDownloader
from dpshdl.numpy import one_hot as one_hot_fn

logger = logging.getLogger(__name__)

MnistDtype = Literal["int8", "float32"]


class MNIST(TensorDataset[tuple[np.ndarray, np.ndarray]]):
    def __init__(
        self,
        train: bool,
        root_dir: Path,
        dtype: MnistDtype = "int8",
        one_hot: bool = False,
    ) -> None:
        self.train = train
        self.dtype = dtype
        self.one_hot = one_hot

        base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
        images_name = "train-images-idx3-ubyte.gz" if train else "t10k-images-idx3-ubyte.gz"
        labels_name = "train-labels-idx1-ubyte.gz" if train else "t10k-labels-idx1-ubyte.gz"

        images_path = FileDownloader(
            base_url + images_name,
            "mnist",
            images_name,
            root_dir=root_dir,
        ).ensure_downloaded()

        labels_path = FileDownloader(
            base_url + labels_name,
            "mnist",
            labels_name,
            root_dir=root_dir,
        ).ensure_downloaded()

        with gzip.open(labels_path, "rb") as fh:
            _ = struct.unpack(">II", fh.read(8))
            labels = np.array(array.array("B", fh.read()), dtype=np.uint8)

        with gzip.open(images_path, "rb") as fh:
            _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            images = np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

        # Process the labels and images.
        self.labels = one_hot_fn(labels, 10) if one_hot else labels
        self.images = self.as_dtype(images)

        super().__init__(self.images, self.labels)

    def as_dtype(self, images: np.ndarray) -> np.ndarray:
        if self.dtype == "int8":
            return images
        elif self.dtype == "float32":
            return images.astype(np.float32) / 255.0
        else:
            raise ValueError(f"Unknown dtype: {self.dtype}")
