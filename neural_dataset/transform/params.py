from copy import deepcopy
from typing import Any, List, Literal, Tuple, Union

import numpy as np

from neural_dataset.transform.core import (
    Transform,
    JointParameterTransformation,
    TensorTransformation,
    _replace,
)

Number = Union[int, float, np.ndarray]

ParametersList = List[np.ndarray]
ParameterVector = np.ndarray

try:
    import jax
    import jax.numpy as jnp

    Number = Union[Number, jnp.ndarray]

    ParametersList = Union[ParametersList, List[jnp.ndarray]]
    ParameterVector = Union[ParameterVector, jnp.ndarray]

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
try:
    import torch

    # add torch.Tensor to Number typing
    Number = Union[Number, torch.Tensor]

    ParametersList = Union[ParametersList, List[torch.Tensor]]
    ParameterVector = Union[ParameterVector, torch.Tensor]

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

ParameterAny = Union[ParametersList, ParameterVector]


def param_vector_to_list(
    param: np.ndarray, param_structure: List[Tuple[str, Tuple[int]]]
) -> List[np.ndarray]:
    """Converts a parameter vector into a list of parameters.

    Args:
        param: Parameter vector.
        param_structure: Structure of the parameter list.

    Returns:
        List of parameters.
    """
    param_list = []
    start_idx = 0
    for param_name, param_shape in param_structure:
        end_idx = start_idx + np.prod(param_shape)
        param_list.append(param[start_idx:end_idx].reshape(param_shape))
        start_idx = end_idx
    return param_list


def param_vector_to_list_relativistic(
    param: np.ndarray, param_structure: List[Tuple[str, Tuple[int], Tuple[int]]]
) -> List[np.ndarray]:
    """Converts a parameter vector into a list of parameters.

    Args:
        param: Parameter vector.
        param_structure: Structure of the parameter list.

    Returns:
        List of parameters.
    """
    param_list = []
    for param_name, param_shape, (start_idx, end_idx) in param_structure:
        param_list.append(param[start_idx:end_idx].reshape(param_shape))

    return param_list


def param_list_to_vector(
    param_list: List[np.ndarray],
) -> np.ndarray:
    """Converts a list of parameters into a vector.

    Args:
        param_list: Parameter list.

    Returns:
        Vector of parameters.
    """
    param = np.concatenate([p.flatten() for p in param_list]).flatten()

    return param


def index_sorting_layers_MFN(param_name, num_layers):
    # bias before kernel, ordered based on layer number
    if param_name.startswith("output_linear."):
        index = 4 * num_layers - 2
    else:
        index = 2 * int(param_name.split(".")[0].split("_")[-1])

    if param_name.startswith("linears_"):
        index += num_layers * 2

    if param_name.endswith(".bias"):
        return index
    elif param_name.endswith(".kernel"):
        return index + 1
    else:
        raise ValueError(f"param_name (`{param_name}`) must end with either `.bias` or `.kernel`.")


def get_param_normalization_stats(
    nef_dataset: Any,
    norm_type: Literal["per_layer", "per_parameter", "global"],
) -> Tuple[np.ndarray, np.ndarray]:
    """Get normalization stats for a NeF dataset.

    Args:
        nef_dataset (PreloadedNeFDataset): The NeF dataset.
        norm_type (Literal["per_layer", "per_parameter", "global"]): The type of normalization to use.

    Raises:
        ValueError: If the normalization type is not supported.
        AssertionError: If the dataset does not have the required attributes.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The mean and standard deviation of the parameters.
    """
    assert hasattr(nef_dataset, "data"), "nef_dataset must have a `data` attribute"
    assert (
        "params" in nef_dataset.data
    ), "Parameters could not be found in nef_dataset.data with key `params`."
    params = nef_dataset.data["params"]
    if norm_type == "global":
        mean = np.mean(params)
        std = np.std(params)
    elif norm_type == "per_parameter":
        mean = np.mean(params, axis=0)
        std = np.std(params, axis=0)
    elif norm_type == "per_layer":
        assert hasattr(
            nef_dataset, "param_structure"
        ), "nef_dataset must have a `param_structure` attribute to calculate per-layer normalization stats."
        assert nef_dataset.param_structure is not None, "nef_dataset.param_structure is None. "
        param_structure = nef_dataset.param_structure
        mean = np.zeros_like(params[0])
        std = np.zeros_like(params[0])
        start_idx = 0
        for param_name, param_shape in param_structure:
            end_idx = start_idx + np.prod(param_shape)
            param = params[:, start_idx:end_idx]
            # Mean and std shared across parameters in same layer.
            mean[start_idx:end_idx] = np.mean(param)
            std[start_idx:end_idx] = np.std(param)
            start_idx = end_idx
    else:
        raise ValueError(f"Normalization type `{norm_type}` not supported.")
    return mean, std


class ParametersToListMFN(Transform):
    """Converts a parameter vector into a list of parameters.

    Args:
        param_structure: Structure of the parameter list. For example, the one saved
            along with the NeF dataset.
    """

    def __init__(self, param_structure: List[Tuple[str, Tuple[int]]]):
        super().__init__()
        assert (
            len(param_structure) % 4 == 0
        ), f"There should be an even number of layers, but there are {len(param_structure)}"

        self.original_param_structure = param_structure

        num_layers = len(param_structure) // 4
        self.param_structure_idxs = [
            index_sorting_layers_MFN(x[0], num_layers) for x in param_structure
        ]
        self.param_structure = []
        for original_idx, param_structure_idx in enumerate(self.param_structure_idxs):
            prev_lengths = [np.prod(x[1]) for x in param_structure[:original_idx]]
            start_idx = sum(prev_lengths)
            end_idx = start_idx + np.prod(param_structure[original_idx][1])
            new_local_info = (
                param_structure[param_structure_idx][0],
                param_structure[param_structure_idx][1],
                (start_idx, end_idx),
            )
            self.param_structure.append(new_local_info)

        self.param_keys = [x[0] for x in self.param_structure]

    def transform(self, x):
        return param_vector_to_list_relativistic(x, self.param_structure)

    def __call__(self, x: dict):
        params = x["params"]
        return _replace(x, params=self.transform(params))

    def __repr__(self):
        return f"ParametersToListMFN(param_structure={self.param_structure})"


def index_sorting_layers_SIREN(param_name, num_layers):
    # bias before kernel, ordered based on layer number
    if param_name.startswith("output_linear."):
        index = 2 * (num_layers - 1)
    else:
        index = 2 * int(param_name.split(".")[0].split("_")[-1])

    if param_name.endswith(".bias"):
        return index
    elif param_name.endswith(".kernel"):
        return index + 1
    else:
        raise ValueError(f"param_name (`{param_name}`) must end with either `.bias` or `.kernel`.")


class ParametersToListSIREN(ParametersToListMFN):
    """Converts a parameter vector into a list of parameters.

    Args:
        param_structure: Structure of the parameter list. For example, the one saved
            along with the NeF dataset.
    """

    def __init__(self, param_structure: List[Tuple[str, Tuple[int]]]):
        super().__init__(param_structure=param_structure)
        assert (
            len(param_structure) % 2 == 0
        ), f"There should be an even number of layers, but there are {len(param_structure)}"

        num_layers = len(param_structure) // 2
        param_structure_idxs = [
            index_sorting_layers_SIREN(x[0], num_layers) for x in param_structure
        ]
        self.param_structure = []
        for param_structure_idx in param_structure_idxs:
            self.param_structure.append(param_structure[param_structure_idx])

        self.param_keys = [x[0] for x in self.param_structure]

    def __repr__(self):
        return f"ParametersToListSIREN(param_structure={self.param_structure})"


class ListToParameters(TensorTransformation):
    """Converts a parameter list into a parameter vector."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pytorch_transform(self, params: ParametersList) -> ParameterVector:
        return torch.cat([x.flatten() for x in params], dim=0).flatten()

    def numpy_transform(self, params: ParametersList) -> ParameterVector:
        return np.concatenate([x.flatten() for x in params], axis=0).flatten()

    def jax_transform(self, params: ParametersList) -> ParameterVector:
        return jnp.concatenate([x.flatten() for x in params], axis=0).flatten()

    def __call__(self, x: dict) -> dict:
        params = deepcopy(x["params"])
        return _replace(x, params=self.transform(params))

    def __repr__(self):
        return (
            f"ListToParameters(platform={self.platform}, seed={self.seed}, device={self.device})"
        )


class Normalize(Transform):
    """Normalizes the input to have zero mean and unit variance."""

    def __init__(self, mean: Number, std: Number, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.mean = mean
        self.std = std

    def __call__(self, x: dict, rng=None):
        params = deepcopy(x["params"])
        params = (params - self.mean) / (self.std + self.epsilon)
        return _replace(x, params=params), rng

    def __repr__(self) -> str:
        return f"Normalize(mean={self.mean}, std={self.std}, epsilon={self.epsilon})"


class UnNormalize(Transform):
    """Un-Normalizes the input to have zero mean and unit variance."""

    def __init__(self, mean: Number, std: Number, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.mean = mean
        self.std = std

    def __call__(self, x: dict, rng=None):
        params = deepcopy(x["params"])
        params = (params * (self.std + self.epsilon)) + self.mean
        return _replace(x, params=params), rng

    def __repr__(self) -> str:
        return f"UnNormalize(mean={self.mean}, std={self.std}, epsilon={self.epsilon})"


class ToTensor(Transform):
    """Converts a numpy array to a torch tensor."""

    def __init__(self):
        super().__init__()

    def __call__(self, x: dict, rng):
        params = deepcopy(x["params"])
        return _replace(x, params=torch.from_numpy(params)), rng

    def __repr__(self) -> str:
        return "ToTensor()"
