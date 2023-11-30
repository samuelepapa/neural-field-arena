from pathlib import Path
from typing import List, Literal, Tuple, Union

from neural_dataset.core import ClassificationNeFDataset


class ClassificationNeuralMNIST(ClassificationNeFDataset):
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


class ClassificationNeuralCIFAR10(ClassificationNeFDataset):
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


class ClassificationNeuralShapeNet(ClassificationNeFDataset):
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


class ClassificationNeuralMicroImageNet(ClassificationNeFDataset):
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
