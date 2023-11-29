from copy import deepcopy
from typing import List, Optional, Tuple, Union

import numpy as np

from neural_dataset.augmentations.core import (
    Augmentation,
    IndividualParameterAugmentation,
    JointParameterAugmentation,
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


class RandomTranslateMFN(JointParameterAugmentation):
    def __init__(
        self,
        *args,
        min_translation: Number = 0.0,
        max_translation: Number = 0.1,
        **kwargs,
    ):
        """
        MFN is expected to have the following sorting of the parameters:
        [bias_filter_0, kernel_filter_0,
        ...,
        bias_filter_n, kernel_filter_n,
        bias_linear_0, kernel_linear_,
        ...,
        bias_linear_n, kernel_linear_n]
        """
        super().__init__(*args, **kwargs)
        self.min_translation = min_translation
        self.max_translation = max_translation

        if isinstance(self.min_translation, (float, int)):
            self.translation_shape = (1,)
        else:
            self.translation_shape = tuple(self.min_translation.shape)

        # must be a multiple of 4, as there should be the same number of filters
        # as the number of linears, and each layer has a bias and a kernel
        assert (
            len(self.param_keys) % 4 == 0
        ), f"Number of parameters must be a multiple of 4, but got {len(self.param_keys)}"
        num_filters = len(self.param_keys) // 4

        self.biases_indices = list(range(0, 2 * num_filters, 2))
        self.kernels_indices = list(range(1, 2 * num_filters, 2))

    def numpy_transform(self, params: ParametersList, rng: np.random.Generator):
        if not isinstance(params, list):
            raise ValueError("Input must be a list of parameters")

        stack_axis = 0

        biases = np.stack(
            [x for i, x in enumerate(params) if i in self.biases_indices], axis=stack_axis
        )
        kernels = np.stack(
            [x for i, x in enumerate(params) if i in self.kernels_indices], axis=stack_axis
        )

        translation_vector = rng.uniform(
            self.min_translation, self.max_translation, size=(kernels.shape[-2],)
        )

        bias_translation_vector = np.matmul(translation_vector, kernels)

        new_biases = biases + bias_translation_vector

        for i, j in enumerate(self.biases_indices):
            params[j] = new_biases[i]

        return params, rng

    def jax_transform(self, params: ParametersList, rng: jax.random.PRNGKey):
        if not isinstance(params, list):
            raise ValueError("Input must be a list of parameters")

        stack_axis = 0

        biases = jnp.stack(
            [x for i, x in enumerate(params) if i in self.biases_indices], axis=stack_axis
        )
        kernels = jnp.stack(
            [x for i, x in enumerate(params) if i in self.kernels_indices], axis=stack_axis
        )

        new_rng, translation_rng = jax.random.split(rng)
        translation_vector = jax.random.uniform(
            key=translation_rng,
            shape=(kernels.shape[-2],),
            minval=self.min_translation,
            maxval=self.max_translation,
        )

        bias_translation_vector = jnp.matmul(translation_vector, kernels)

        new_biases = biases + bias_translation_vector

        for i, j in enumerate(self.biases_indices):
            params[j] = new_biases[i]

        return params, new_rng

    def pytorch_transform(self, params: ParametersList, rng: torch.Generator):
        if not isinstance(params, list):
            raise ValueError("Input must be a list of parameters")

        stack_dim = 0

        biases = torch.stack(
            [x for i, x in enumerate(params) if i in self.biases_indices], dim=stack_dim
        )
        kernels = torch.stack(
            [x for i, x in enumerate(params) if i in self.kernels_indices], dim=stack_dim
        )

        batched_translation_vector = (
            torch.rand(
                (kernels.shape[-2],),
                generator=rng,
                device=self.device,
                dtype=biases.dtype,
            )
            * (self.max_translation - self.min_translation)
            + self.min_translation
        )

        bias_translation_vector = torch.matmul(batched_translation_vector, kernels)

        new_biases = biases + bias_translation_vector

        for i, j in enumerate(self.biases_indices):
            params[j] = new_biases[i]

        return params, rng


class RandomTranslateSIREN(JointParameterAugmentation):
    def __init__(
        self,
        *args,
        min_translation: Number = 0.0,
        max_translation: Number = 0.1,
        **kwargs,
    ):
        """
        MFN is expected to have the following sorting of the parameters:
        [bias_filter_0, kernel_filter_0,
        ...,
        bias_filter_n, kernel_filter_n,
        bias_linear_0, kernel_linear_,
        ...,
        bias_linear_n, kernel_linear_n]
        """
        super().__init__(*args, **kwargs)
        self.min_translation = min_translation
        self.max_translation = max_translation

        if isinstance(self.min_translation, (float, int)):
            self.translation_shape = (1,)
        else:
            self.translation_shape = tuple(self.min_translation.shape)

        self.bias_index = 0
        self.kernel_index = 1

    def numpy_transform(self, params: ParametersList, rng: np.random.Generator):
        if not isinstance(params, list):
            raise ValueError("Input must be a list of parameters")

        bias = params[self.bias_index]
        kernel = params[self.kernel_index]

        translation_vector = rng.uniform(
            self.min_translation, self.max_translation, size=(kernel.shape[-2],)
        )

        bias_translation_vector = np.matmul(translation_vector, kernel)

        new_bias = bias + bias_translation_vector

        params[self.bias_index] = new_bias

        return params, rng

    def jax_transform(self, params: ParametersList, rng: jax.random.PRNGKey):
        if not isinstance(params, list):
            raise ValueError("Input must be a list of parameters")

        bias = params[self.bias_index]
        kernel = params[self.kernel_index]

        new_rng, translation_rng = jax.random.split(rng)
        translation_vector = jax.random.uniform(
            key=translation_rng,
            shape=(kernel.shape[-2],),
            minval=self.min_translation,
            maxval=self.max_translation,
        )

        bias_translation_vector = jnp.matmul(translation_vector, kernel)

        new_bias = bias + bias_translation_vector

        params[self.bias_index] = new_bias

        return params, new_rng

    def pytorch_transform(self, params: ParametersList, rng: torch.Generator):
        if not isinstance(params, list):
            raise ValueError("Input must be a list of parameters")

        bias = params[self.bias_index]
        kernel = params[self.kernel_index]

        batched_translation_vector = (
            torch.rand(
                (kernel.shape[-2],),
                generator=rng,
                device=self.device,
                dtype=bias.dtype,
            )
            * (self.max_translation - self.min_translation)
            + self.min_translation
        )

        bias_translation_vector = torch.matmul(batched_translation_vector, kernel)

        new_bias = bias + bias_translation_vector

        params[self.bias_index] = new_bias

        return params, rng


class RandomRotate(IndividualParameterAugmentation):
    def __init__(
        self,
        *args,
        min_angle: float = 0.0,
        max_angle: float = 3.14,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.min_angle = min_angle
        self.max_angle = max_angle

        assert (
            self.min_angle <= self.max_angle
        ), f"min_angle must be smaller than max_angle, but got min_angle={self.min_angle} and max_angle={self.max_angle}"

    def numpy_init_transform(self, params, rng: np.random.Generator):
        angle = rng.uniform(self.min_angle, self.max_angle)
        self.rotation_matrix = np.empty((2, 2), dtype=np.float32)
        self.rotation_matrix[0, 0] = np.cos(angle)
        self.rotation_matrix[1, 0] = np.sin(angle)
        self.rotation_matrix[0, 1] = -self.rotation_matrix[1, 0]
        self.rotation_matrix[1, 1] = self.rotation_matrix[0, 0]

        return params, rng

    def jax_init_transform(self, params, rng):
        new_rng, rng = jax.random.split(rng)
        angle = jax.random.uniform(rng) * (self.max_angle - self.min_angle) + self.min_angle
        self.rotation_matrix = jnp.array(
            [
                [jnp.cos(angle), jnp.sin(angle)],
                [-jnp.sin(angle), jnp.cos(angle)],
            ],
            dtype=jnp.float32,
        )

        return params, new_rng

    def pytorch_init_transform(self, params, rng):
        angle = (
            torch.rand((1,), generator=rng, device=self.device) * (self.max_angle - self.min_angle)
            + self.min_angle
        )
        self.rotation_matrix = torch.tensor(
            [
                [torch.cos(angle), torch.sin(angle)],
                [-torch.sin(angle), torch.cos(angle)],
            ],
            device=self.device,
            dtype=torch.float32,
        )

        return params, rng

    def numpy_transform(self, x, param_key: Optional[str] = None, rng: np.random.Generator = None):
        return np.matmul(self.rotation_matrix, x), rng

    def jax_transform(self, x, param_key: Optional[str] = None, rng: jax.random.PRNGKey = None):
        return jnp.matmul(self.rotation_matrix, x), rng

    def pytorch_transform(self, x, param_key: Optional[str] = None, rng: torch.Generator = None):
        return torch.matmul(self.rotation_matrix, x), rng

    def __repr__(self):
        return f"RandomRotate(platform={self.platform}, min_angle={self.min_angle}, max_angle={self.max_angle}{', exclude_params=' + str(self.exclude_params) if self.exclude_params else ''})"


class RandomScale(IndividualParameterAugmentation):
    def __init__(self, *args, min_scale: float = 0.5, max_scale: float = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_scale = min_scale
        self.max_scale = max_scale

        if isinstance(self.min_scale, float):
            self.scale_shape = (1,)
        else:
            self.scale_shape = tuple(self.min_scale.shape)

        assert (
            self.min_scale <= self.max_scale
        ), f"min_scale must be smaller than max_scale, but got min_scale={self.min_scale} and max_scale={self.max_scale}"

    def numpy_init_transform(self, params: ParametersList, rng: np.random.Generator):
        self.scale = rng.uniform(self.min_scale, self.max_scale, size=self.scale_shape)
        return params, rng

    def jax_init_transform(self, params: ParametersList, rng: jax.random.PRNGKey):
        new_rng, rng = jax.random.split(rng)
        self.scale = (
            jax.random.uniform(rng, self.scale_shape) * (self.max_scale - self.min_scale)
            + self.min_scale
        )

        return params, new_rng

    def pytorch_init_transform(self, params: ParametersList, rng: torch.Generator):
        self.scale = (
            torch.rand(self.scale_shape, generator=rng, device=self.device)
            * (self.max_scale - self.min_scale)
            + self.min_scale
        )

        return params, rng

    def numpy_transform(self, x, param_key: Optional[str] = None, rng: np.random.Generator = None):
        return x * self.scale, rng

    def jax_transform(self, x, param_key: Optional[str] = None, rng: jax.random.PRNGKey = None):
        return x * self.scale, rng

    def pytorch_transform(self, x, param_key: Optional[str] = None, rng: torch.Generator = None):
        return x * self.scale, rng

    def __repr__(self):
        return f"RandomScale(platform={self.platform}, min_scale={self.min_scale}, max_scale={self.max_scale}{', exclude_params=' + str(self.exclude_params) if self.exclude_params else ''})"
