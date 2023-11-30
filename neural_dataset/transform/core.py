from abc import ABC, abstractmethod
from copy import copy, deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

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


class Transform(ABC):
    """Abstract base class for transformations.

    transformations are thought of as being applied to each sample in the dataset.
    """

    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError


class Compose(Transform):
    """Class for composing transformations.

    Args:
        transformations: List of transformations to compose.
    """

    def __init__(self, transformations: List[Transform]):
        self.transformations = transformations

    def __call__(self, x, rng: Optional[Any] = None):
        for transformation in self.transformations:
            x, rng = transformation(x, rng=rng)
        return x

    def __add__(self, composed_aug):
        return Compose(self.transformations + composed_aug.transformations)

    def __repr__(self) -> str:
        s = "Compose(\n"
        for aug in self.transformations:
            s += f"{aug}\n"
        s += ")"

        return s


class Identity(Transform):
    """Identity transformation."""

    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return x

    def __repr__(self) -> str:
        return "Identity()"


class TensorTransformation(Transform, ABC):
    """Base transformation class for operations on tensors. Supports numpy, jax, and pytorch.

    Args:
        platform: Platform to use. Can be "numpy", "jax", "pytorch", or "auto".
            The latter will automatically detect the platform based on the input.
        seed: Seed for the random number generator. If None, the default global
            random number generator will be used. If an integer, a new random number
            generator will be created with the given seed. If a random number generator
            is given, it will be used directly.
    """

    def __init__(
        self,
        platform: str = "numpy",
        seed: Optional[Any] = None,
        device: Any = None,
        init_example: dict = None,
    ):
        self.platform = platform
        self.seed = seed
        self.device = device

        if hasattr(self, "transform"):
            # If transform is already defined, then we will ignore the platform and seed
            pass
        elif platform == "numpy":
            self._setup_numpy()
        elif platform == "jax":
            self._setup_jax()
        elif platform == "pytorch" or platform == "torch":
            self._setup_pytorch()
        elif platform == "auto":
            self._setup_auto()
        else:
            raise NotImplementedError(f"Platform {platform} not implemented")

        if init_example is not None:
            self.init_transform(init_example)

    def _setup_numpy(self):
        self._set_transform(self.numpy_transform)
        self._set_init_transform(self.numpy_init_transform)

    def _setup_jax(self):
        self._set_transform(self.jax_transform)
        self._set_init_transform(self.jax_init_transform)

    def _setup_pytorch(self):
        self._set_transform(self.pytorch_transform)
        self._set_init_transform(self.pytorch_init_transform)

    def _setup_auto(self):
        self._set_transform(self.auto_transform)
        self._set_init_transform(None)

    def _set_transform(self, transform):
        self.transform = transform

    @abstractmethod
    def numpy_transform(self, x, rng: Optional[Any] = None):
        raise NotImplementedError

    @abstractmethod
    def jax_transform(self, x, rng: Optional[Any] = None):
        raise NotImplementedError

    @abstractmethod
    def pytorch_transform(self, x, rng: Optional[Any] = None):
        raise NotImplementedError

    def _set_init_transform(self, init_transform):
        self.init_transform = init_transform

    def numpy_init_transform(self, x, rng: Optional[Any] = None):
        return x, rng

    def jax_init_transform(self, x, rng: Optional[Any] = None):
        return x, rng

    def pytorch_init_transform(self, x, rng: Optional[Any] = None):
        return x, rng

    def auto_transform(self, x: dict, rng: Optional[Any] = None):
        params = x.params
        if isinstance(params, np.ndarray):
            self._setup_numpy()
        elif JAX_AVAILABLE and isinstance(x, jnp.ndarray):
            self._setup_jax()
        elif TORCH_AVAILABLE and isinstance(x, torch.Tensor):
            self._setup_pytorch()
        else:
            raise NotImplementedError(
                f"Auto transform failed to find a suitable transformation for type {type(x)}. Platforms available: numpy{', jax' if JAX_AVAILABLE else ', (unable to import jax)'}{', pytorch' if TORCH_AVAILABLE else ', (unable to import pytorch)'}."
            )
        return self.transform(x, rng=rng)

    def __call__(self, x: dict, rng: Optional[Any] = None):
        y = deepcopy(x)
        return self.transform(y, rng=rng)

    def __repr__(self):
        return (
            f"TensorTransformation(platform={self.platform}, seed={self.seed}, device={self.device})"
        )

    def __str__(self):
        return repr(self)


def _replace(data, **kwargs):
    new_dict = deepcopy(data)
    for key, value in kwargs.items():
        new_dict[key] = value

    for key in data.keys():
        if key not in kwargs.keys():
            new_dict[key] = data[key]

    return new_dict


class JointParameterTransformation(TensorTransformation, ABC):
    """Base transformation class for operations on joint parameters, i.e. all parameters being
    augmented together. This is useful when there is dependency between different parameters.
    Supports numpy, jax, and pytorch.

    Args:
        platform: Platform to use. Can be "numpy", "jax", "pytorch", or "auto".
            The latter will automatically detect the platform based on the input.
        seed: Seed for the random number generator. If None, the default global
            random number generator will be used. If an integer, a new random number
            generator will be created with the given seed. If a random number generator
            is given, it will be used directly.
        param_keys: List of parameter keys. May not be optional for some transformations.
    """

    def __init__(
        self,
        platform: str = "numpy",
        seed: Optional[Any] = None,
        param_keys: Optional[List[str]] = None,
        device: Any = None,
    ):
        super().__init__(platform=platform, seed=seed, device=device)
        self.param_keys = param_keys

    def __call__(self, x: dict, rng: Optional[Any] = None):
        """Perform transformation.

        Args:
            x: Input to augment. If single parameter, the whole parameter will be
                augmented. If a tuple, the first element is assumed to be the parameters,
                and second or later ones tensors not to be augmented. If parameters are
                a list, ensure that it is passed within a list, i.e. x=[params,...] with
                params = [param1, param2, ...].

        Returns:
            Augmented input.
        """
        params = deepcopy(x["params"])
        params, rng = self.transform(params, rng=rng)
        return _replace(x, params=params), rng

    def __repr__(self):
        return f"JointParameterTransformation(platform={self.platform})"


class IndividualParameterTransformation(TensorTransformation, ABC):
    """Base transformation class for operations on individual parameters, i.e. each parameter being
    augmented independently of each other. Supports numpy, jax, and pytorch.

    Args:
        platform: Platform to use. Can be "numpy", "jax", "pytorch", or "auto".
            The latter will automatically detect the platform based on the input.
        seed: Seed for the random number generator. If None, the default global
            random number generator will be used. If an integer, a new random number
            generator will be created with the given seed. If a random number generator
            is given, it will be used directly.
        param_keys: List of parameter keys. If None, all parameters will be augmented.
        exclude_params: List of parameter keys to exclude from transformation.
    """

    def __init__(
        self,
        param_keys: List[str],
        platform: str = "numpy",
        seed: Optional[Any] = None,
        selected_param_idxs: Optional[List[str]] = None,
        device: Any = None,
    ):
        super().__init__(platform=platform, seed=seed, device=device)
        self.param_keys = param_keys
        self.selected_param_idxs = selected_param_idxs

    @abstractmethod
    def numpy_transform(self, x, param_key: Optional[str] = None):
        raise NotImplementedError

    @abstractmethod
    def jax_transform(self, x, param_key: Optional[str] = None):
        raise NotImplementedError

    @abstractmethod
    def pytorch_transform(self, x, param_key: Optional[str] = None):
        raise NotImplementedError

    def __call__(self, x: dict, rng: Optional[Any] = None):
        """Perform transformation.

        Args:
            x: Input to augment. If single parameter, the whole parameter will be
                augmented. If a tuple, the first element is assumed to be the parameters,
                and second or later ones tensors not to be augmented. If parameters are
                a list, ensure that it is passed within a list, i.e. x=[params,...] with
                params = [param1, param2, ...]. The parameters should be in the same order
                as the param_keys.

        Returns:
            Augmented input.
        """
        params = deepcopy(x["params"])

        params, rng = self.init_transform(params, rng=rng)

        if self.selected_param_idxs is None:
            self.selected_param_idxs = list(range(len(params)))

        for i in self.selected_param_idxs:
            params[i], rng = self.transform(params[i], param_key=self.param_keys[i], rng=rng)

        return _replace(x, params=params), rng

    def __repr__(self):
        return f"ParameterTransformation(platform={self.platform})"
