from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.data as data
from ml_collections import ConfigDict

import neural_dataset.nef_pipe as nef_pipe
from neural_dataset.augmentations import Augmentation
from neural_dataset.utils import numpy_collate


def make_dataloader(
    path: str,
    batch_size: int,
    seed: int = 42,
    shuffle: bool = False,
    split_start: Union[float, int] = 0.0,
    split_end: Union[float, int] = 1.0,
    split_type: str = "fractional",  # or "exact"
    collate_fn: Callable = numpy_collate,
    num_workers: int = 0,
    transform: Optional[Union[Callable, Dict[str, Callable]]] = None,
    shard_over_files: bool = True,
    pipe_class: Callable = nef_pipe.PreloadedNeFDataset,
    preload: bool = True,
    drop_last: bool = True,
    persistent_workers: bool = True,
):
    assert preload, "Only preloaded data loaders are supported at the moment."
    data_pipe = pipe_class(
        path,
        start_idx=split_start,
        end_idx=split_end,
        split_type=split_type,
        transform=transform,
    )
    loader = data.DataLoader(
        data_pipe,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=persistent_workers and num_workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )
    return loader


def build_nef_data_loader_group(
    data_config: ConfigDict,
    collate_fn: Callable = numpy_collate,
    transform: Optional[Union[Callable, Dict[str, Callable]]] = None,
):
    """Creates data loaders for a set of datasets.

    Args:
        data_config: ConfigDict with the following possible keys:
            path: Path to the data.
            shuffle_slice (optional): Whether to shuffle the slices of the data during training.
            preload (optional): Whether to preload the data.
            split (optional): Split of the data to use. If None (default), all data is used.
            data_prefix (optional): Prefix of the data files.
        collate_fn: Collate function to use for the data loaders.
        transform: Transform to use for the datasets.

    Returns:
        List of data loaders.
    """
    assert transform is None or isinstance(
        transform, (Augmentation, list, tuple)
    ), f"Transform must be a list of transforms is instead {type(transform)}"

    data_split = data_config.get("split", [None])

    num_loaders = len(data_split) if data_split is not None else 1

    # when only one transform is provided, assume that it is the same for all loaders
    if isinstance(transform, Augmentation):
        transform = [transform] + [None for _ in range(num_loaders - 1)]
    elif transform is None:
        transform = [None] * num_loaders
    assert (
        len(transform) == num_loaders
    ), f"there must be a number of transforms ({len(transform)}) equal to the number of loaders ({num_loaders})."

    all_loaders = []

    start_idx = 0
    for loader_idx, split in enumerate(data_split):
        if split is None:
            end_idx = None
        else:
            end_idx = start_idx + split

        loader = build_nef_data_loader(
            data_config,
            start_idx=start_idx,
            end_idx=end_idx,
            is_train=loader_idx == 0,
            collate_fn=collate_fn,
            transform=transform[loader_idx] if transform is not None else None,
        )
        all_loaders.append(loader)
        start_idx = end_idx

    return all_loaders


def build_nef_data_loader(
    data_config: ConfigDict,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    is_train: bool = True,
    collate_fn: Callable = numpy_collate,
    transform: Optional[Callable] = None,
):
    """Creates data loaders for a set of datasets.

    Args:
        data_config: ConfigDict with the following possible keys:
            path: Path to the data.
            shuffle_slice (optional): Whether to shuffle the slices of the data during training.
            preload (optional): Whether to preload the data.
            data_pipe_class (optional): The class of the data pipe to use.
            batch_size (optional): Batch size to use in the data loaders.
            num_workers (optional): Number of workers for each dataset.
            persistent_workers (optional): Whether to use persistent workers.
            seed (optional): Seed to initialize the workers and shuffling with.
        modes: Modes for which data loaders are created. Each mode corresponds to a file prefix.
            If None (default), all files in the data path are used for a single mode.

    Returns:
        List of data loaders.
    """
    path = data_config.get("path", "")
    num_workers = data_config.get("num_workers", 0)
    pipe_class = getattr(nef_pipe, data_config.get("data_pipe_class", "PreloadedNeFDataset"))
    data_pipe = pipe_class(
        path,
        start_idx=start_idx,
        end_idx=end_idx,
        split_type="exact",
        transform=transform,
    )
    loader = data.DataLoader(
        data_pipe,
        batch_size=data_config.get("batch_size", 128),
        shuffle=is_train,
        drop_last=is_train,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and data_config.get("persistent_workers", is_train),
        generator=torch.Generator().manual_seed(data_config.get("seed", 42)),
    )
    return loader
