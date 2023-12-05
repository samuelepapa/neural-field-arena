import json
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import h5py
import numpy as np
from absl import logging

from neural_dataset.core import ClassificationNeFDataset


def get_mean_std(path, split_name):
    if isinstance(path, str):
        path = Path(path)
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"Metadata file not found at {metadata_path}")
    with metadata_path.open() as f:
        metadata = json.load(f)

    if split_name not in metadata:
        raise ValueError(f"Split {split_name} not found in metadata.json")

    return np.array(metadata[split_name]["mean"]), np.array(metadata[split_name]["std"])


def get_param_keys(path: Union[str, Path]) -> List[str]:
    first_h5py_file = list(Path(path).glob("*.hdf5"))[0]
    with h5py.File(first_h5py_file, "r") as f:
        # get the param keys from binary
        param_keys = [x[0] for x in json.loads(f["param_config"][0])]

    return param_keys


def get_param_structure(path: Union[str, Path]) -> List[str]:
    first_h5py_file = list(Path(path).glob("*.hdf5"))[0]
    with h5py.File(first_h5py_file, "r") as f:
        # get the param keys from binary
        param_structure = json.loads(f["param_config"][0])

    return param_structure


class ClassificationNeuralMNIST(ClassificationNeFDataset):
    splits = {
        "train": (0, 55000),
        "val": (55000, 60000),
        "test": (60000, 70000),
    }

    def __init__(
        self, path: Union[str, Path], split: Literal["train", "val", "test"] = "train", **kwargs
    ):
        if split not in self.splits:
            raise ValueError(f"Split {split} not supported, must be one of `train`, `val`, `test`")

        start_idx, end_idx = self.splits[split]

        super().__init__(path, start_idx, end_idx, **kwargs)


class ClassificationNeuralCIFAR10(ClassificationNeFDataset):
    splits = {
        "train": (0, 45000),
        "val": (45000, 50000),
        "test": (50000, 60000),
    }

    def __init__(
        self,
        path: Union[str, Path],
        split: Literal["train", "val", "test"] = "train",
        **kwargs,
    ):
        if split not in self.splits:
            raise ValueError(f"Split {split} not supported, must be one of `train`, `val`, `test`")

        start_idx, end_idx = self.splits[split]

        super().__init__(path, start_idx, end_idx, **kwargs)


class ClassificationNeuralShapeNet(ClassificationNeFDataset):
    splits = {
        "train": (0, 27840),
        "val": (27840, 31320),
        "test": (31320, 34800),
    }

    def __init__(
        self,
        path: Union[str, Path],
        split: Literal["train", "val", "test"] = "train",
        **kwargs,
    ):
        if split not in self.splits:
            raise ValueError(f"Split {split} not supported, must be one of `train`, `val`, `test`")

        start_idx, end_idx = self.splits[split]

        super().__init__(path, start_idx, end_idx, **kwargs)


class ClassificationNeuralMicroImageNet(ClassificationNeFDataset):
    splits = {
        "train": (0, 45000),
        "val": (45000, 50000),
        "test": (50000, 60000),
    }

    def __init__(
        self,
        path: Union[str, Path],
        split: Literal["train", "val", "test"] = "train",
        **kwargs,
    ):
        if split not in self.splits:
            raise ValueError(f"Split {split} not supported, must be one of `train`, `val`, `test`")

        start_idx, end_idx = self.splits[split]

        super().__init__(path, start_idx, end_idx, split_type="exact", **kwargs)
