from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch

from neural_dataset.utils import get_param_structure

PARAM_KEY = "params"


def get_dataset_size(hdf5_files: Sequence[Union[str, Path]]) -> int:
    """Compute the total number of elements in a list of hdf5 files.

    Args:
        hdf5_files (Sequence[Union[str, Path]]): The list of hdf5 files.

    Returns:
        int: The total number of elements in the hdf5 files.
    """
    num_elements = 0
    for file in hdf5_files:
        with h5py.File(file, "r") as f:
            num_elements += f[PARAM_KEY].shape[0]

    return num_elements


def load_data_from_hdf5(
    hdf5_files: Sequence[Union[str, Path]],
    data_keys: List[str] = None,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Load data from a list of hdf5 files.

    Args:
        hdf5_files (Sequence[Union[str, Path]]): The list of hdf5 files.
        data_keys (List[str], optional): The list of keys to load from the hdf5 files. Defaults to None.
        start_idx (int, optional): The starting index of the data to load. Defaults to 0.
        end_idx (Optional[int], optional): The ending index of the data to load. Defaults to None.

    Returns:
        Dict[str, np.ndarray]: A dictionary of the loaded data.
    """
    if data_keys is None:
        data_keys = [PARAM_KEY]
    if end_idx is None:
        end_idx = get_dataset_size(hdf5_files)
    data = {}
    idx = 0
    dataset_size = end_idx - start_idx
    for file in hdf5_files:
        elements_in_file, elements_to_load = -1, -1
        with h5py.File(file, "r") as f:
            for key in data_keys:
                assert key in f.keys(), f"Key {key} not found in {file}"
                if key not in data:
                    target_shape = (end_idx - start_idx, *f[key].shape[1:])
                    data[key] = np.zeros(target_shape, dtype=f[key].dtype)
                if elements_in_file == -1:
                    elements_in_file = f[key].shape[0]
                    file_start_idx = max(0, start_idx - idx)
                    file_end_idx = min(elements_in_file, end_idx - idx)
                    elements_to_load = max(0, file_end_idx - file_start_idx)
                else:
                    assert (
                        elements_in_file == f[key].shape[0]
                    ), f"All keys must have the same number of elements, violated in {file}."
                if elements_to_load == 0:
                    break
                data_idx = idx - start_idx
                data[key][data_idx : data_idx + elements_to_load] = f[key][
                    file_start_idx:file_end_idx
                ]
        idx += elements_in_file
        if idx >= end_idx:
            break
    if idx < end_idx:
        raise RuntimeError(
            f"Could not load enough data, requested {end_idx - start_idx} elements but only loaded {idx - start_idx} elements."
        )
    return data


class PreloadedNeFDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: Union[str, Path],
        start_idx: Union[int, float] = 0.0,
        end_idx: Union[int, float] = 1.0,
        split_type: Literal["fractional", "exact", None] = None,
        data_prefix: str = "",
        data_keys: List[str] = None,
        transform: Optional[Union[Callable, Dict[str, Callable]]] = None,
    ):
        if isinstance(path, str):
            path = Path(path)
        assert path.exists(), f"Path {path.absolute()} does not exist"
        assert path.is_dir(), f"Path {path.absolute()} is not a directory"

        file_pattern = f"{data_prefix}*.hdf5"
        paths = list(path.glob(f"{data_prefix}*.hdf5"))

        if len(paths) == 0:
            raise ValueError(
                f"No files found at `{path.absolute()}` with pattern `{file_pattern}`"
            )

        # Determine data split
        if split_type is None:
            if isinstance(start_idx, float) and isinstance(end_idx, float):
                split_type = "fractional"
            elif isinstance(start_idx, int) and isinstance(end_idx, int):
                split_type = "exact"
            else:
                raise ValueError(
                    f"Split type could not be inferred from start_idx={start_idx} and end_idx={end_idx}"
                )
        if split_type == "fractional":
            num_elements = get_dataset_size(paths)
            start_idx = int(start_idx * num_elements)
            end_idx = int(end_idx * num_elements)
        elif split_type == "exact":
            start_idx = int(start_idx)
            end_idx = int(end_idx)

        self.data = load_data_from_hdf5(paths, data_keys, start_idx, end_idx)
        self.data_keys = data_keys if data_keys is not None else list(self.data.keys())
        self.path = path
        self.param_structure = get_param_structure(path)
        self.transform = transform

    def __len__(self):
        return self.data[self.data_keys[0]].shape[0]

    def __getitem__(self, idx):
        batch_item = {key: self.data[key][idx] for key in self.data_keys}
        if self.transform is not None:
            batch_item = self.transform(batch_item)
        return batch_item


class ClassificationNeFDataset(PreloadedNeFDataset):
    def __init__(
        self,
        path: Union[str, Path],
        start_idx: Union[int, float] = 0.0,
        end_idx: Union[int, float] = 1.0,
        split_type: Literal["fractional", "exact", None] = None,
        data_prefix: str = "",
        transform: Optional[Union[Callable, Dict[str, Callable]]] = None,
    ):
        super().__init__(
            path,
            start_idx,
            end_idx,
            split_type,
            data_prefix,
            data_keys=["params", "labels"],
            transform=transform,
        )


class MNISTNeFDataset(PreloadedNeFDataset):
    def __init__(
        self, path: Union[str, Path], split: Literal["train", "val", "test"] = "train", **kwargs
    ):
        if split == "train":
            start_idx = 0
            end_idx = 55000
        elif split == "val":
            start_idx = 55000
            end_idx = 60000
        elif split == "test":
            start_idx = 60000
            end_idx = 70000
        else:
            raise ValueError(f"Split {split} not supported, must be one of `train`, `val`, `test`")

        super().__init__(path, start_idx, end_idx, split_type="exact", **kwargs)


class CIFAR10NeFDataset(PreloadedNeFDataset):
    def __init__(
        self,
        path: Union[str, Path],
        split: Literal["train", "val", "test"] = "train",
        **kwargs,
    ):
        if split == "train":
            start_idx = 0
            end_idx = 45000
        elif split == "val":
            start_idx = 45000
            end_idx = 50000
        elif split == "test":
            start_idx = 50000
            end_idx = 60000
        else:
            raise ValueError(f"Split {split} not supported, must be one of `train`, `val`, `test`")

        super().__init__(path, start_idx, end_idx, split_type="exact", **kwargs)
