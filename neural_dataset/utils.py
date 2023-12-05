import json
import os
import re
from glob import glob
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
import torch.utils.data as data

from neural_dataset import transform


def numpy_collate(batch: Union[np.ndarray, Sequence[Any], Any]):
    """Collate function for numpy arrays.

    This function acts as replacement to the standard PyTorch-tensor collate function in PyTorch DataLoader.

    Args:
        batch: Batch of data. Can be a numpy array, a list of numpy arrays, or nested lists of numpy arrays.

    Returns:
        Batch of data as (potential list or tuple of) numpy array(s).
    """
    # dict collate

    if isinstance(batch, np.ndarray):
        return batch
    elif isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    elif isinstance(batch[0], dict):
        return {key: numpy_collate([d[key] for d in batch]) for key in batch[0]}
    else:
        return np.array(batch)


def torch_collate(batch: Union[np.ndarray, Sequence[Any], Any]):
    """Collate function for numpy arrays.

    This function acts as replacement to the standard PyTorch-tensor collate function in PyTorch DataLoader.

    Args:
        batch: Batch of data. Can be a numpy array, a list of numpy arrays, or nested lists of numpy arrays.

    Returns:
        Batch of data as (potential list or tuple of) numpy array(s).
    """
    # dict collate

    if isinstance(batch, np.ndarray):
        return torch.tensor(batch)
    elif isinstance(batch[0], np.ndarray):
        return torch.tensor(np.stack(batch))
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    elif isinstance(batch[0], dict):
        return {key: numpy_collate([d[key] for d in batch]) for key in batch[0]}
    else:
        return torch.tensor(batch)


def get_param_structure(path: str):
    # find the first parameter hdf5 file
    hdf5_path = glob(os.path.join(path, "*.hdf5"))[0]
    with h5py.File(hdf5_path, "r") as hdf5_file:
        param_structure = json.loads(hdf5_file["param_config"][0])
        return param_structure


def get_param_keys(path):
    return [key for key, shape in get_param_structure(path=path)]


def splits_to_names(split_sizes: Union[List[int], List[float]]) -> List[str]:
    start_idx = 0
    split_names = []
    for split_size in split_sizes:
        end_idx = start_idx + split_size
        split_names.append(f"{start_idx}_{end_idx}")
        start_idx = end_idx
    return split_names


def is_range_in_file(start_idx, end_idx, file_start_idx, file_end_idx):
    return start_idx < file_end_idx and end_idx > file_start_idx


def create_path_start_end_list(path_start_end_idxs, start_idx, end_idx):
    used_files = []

    for path, file_start_end in path_start_end_idxs:
        file_start_idx, file_end_idx = file_start_end
        if is_range_in_file(start_idx, end_idx, file_start_idx, file_end_idx):
            file_start_idx, file_end_idx = file_start_end

            used_files.append(
                (
                    path,
                    max(start_idx - file_start_idx, 0),
                    min(end_idx - file_start_idx, file_end_idx - file_start_idx),
                )
            )

    return used_files


def start_end_idx_from_path(path: str) -> Tuple[int, int]:
    """Get start and end index from path.

    Args:
        path: Path from which to extract the start and end index.

    Returns:
        Tuple with start and end index.
    """
    start_idx = int(Path(path).stem.split("_")[1].split("-")[0])
    end_idx = int(Path(path).stem.split("_")[1].split("-")[1])
    return start_idx, end_idx


def path_from_name_idxs(name: str, start_idx: int, end_idx: int) -> str:
    return f"{name}_{start_idx}-{end_idx}.hdf5"
