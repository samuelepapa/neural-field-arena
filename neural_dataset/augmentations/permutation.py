from copy import copy
from typing import List, Tuple, Union

import numpy as np

from neural_dataset.augmentations.core import JointParameterAugmentation

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


class RandomMLPWeightPermutation(JointParameterAugmentation):
    """Randomly permutes the weights of the network without changing the network structure and
    outputs.

    Args:
        platform: Platform to use. Can be "numpy", "jax", "pytorch", or "auto".
            The latter will automatically detect the platform based on the input.
        seed: Seed for the random number generator. If None, the default global
            random number generator will be used. If an integer, a new random number
            generator will be created with the given seed. If a random number generator
            is given, it will be used directly.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias_idxs = list(range(0, len(self.param_keys), 2))
        self.kernel_idxs = list(range(1, len(self.param_keys), 2))

    def pytorch_init_transform(self, x, rng):
        return self.init_transform(x, rng)

    def jax_init_transform(self, x, rng):
        return self.init_transform(x, rng)

    def numpy_init_transform(self, x, rng):
        return self.init_transform(x, rng)

    def pre_transform(self, x, rng, transform_func):
        assert isinstance(x, (list)), "Input must be a list of parameters"
        # x = copy(x)  # Copy list to avoid modifying original
        for i, (layer_idx1, layer_idx2) in enumerate(
            zip(self.kernel_idxs[:-1], self.kernel_idxs[1:])
        ):
            bias_idx1 = self.bias_idxs[i]

            output_dim_permute = [x[layer_idx1], x[bias_idx1]]
            input_dim_permute = [x[layer_idx2]]
            assert (
                output_dim_permute[0].shape[-1] == output_dim_permute[1].shape[-1]
            ), f"Output dimensions must match: {[op.shape for op in output_dim_permute]}"
            assert (
                output_dim_permute[0].shape[-1] == input_dim_permute[0].shape[-2]
            ), "Input and output dimensions must match"
            new_layers, rng = transform_func(
                output_dim_permute=output_dim_permute, input_dim_permute=input_dim_permute, rng=rng
            )
            x[layer_idx1] = new_layers[0]
            x[bias_idx1] = new_layers[1]
            x[layer_idx2] = new_layers[2]

        return x, rng

    def _set_transform(self, transform):
        self.transform = lambda x, rng: self.pre_transform(x, rng, transform)

    def numpy_transform(self, output_dim_permute, input_dim_permute, rng):
        # TODO: Consider supporting different permutation per batch element
        perm = rng.permutation(output_dim_permute[0].shape[-1])
        return [op[..., perm] for op in output_dim_permute] + [
            ip[..., perm, :] for ip in input_dim_permute
        ], rng

    def jax_transform(self, output_dim_permute, input_dim_permute, rng):
        new_rng, perm_rng = jax.random.split(rng)
        perm = jax.random.permutation(perm_rng, output_dim_permute[0].shape[-1])
        return [op[..., perm] for op in output_dim_permute] + [
            ip[..., perm, :] for ip in input_dim_permute
        ], new_rng

    def pytorch_transform(self, output_dim_permute, input_dim_permute, rng):
        perm = torch.randperm(output_dim_permute[0].shape[-1])
        return [op[..., perm] for op in output_dim_permute] + [
            ip[..., perm, :] for ip in input_dim_permute
        ], rng


class RandomFourierNetWeightPermutation(JointParameterAugmentation):
    """Randomly permutes the weights of the network without changing the network structure and
    outputs.

    Args:
        platform: Platform to use. Can be "numpy", "jax", "pytorch", or "auto".
            The latter will automatically detect the platform based on the input.
        seed: Seed for the random number generator. If None, the default global
            random number generator will be used. If an integer, a new random number
            generator will be created with the given seed. If a random number generator
            is given, it will be used directly.
    """

    def pre_transform(self, x, rng, transform_func):
        """
        Structure of x is always:
        [0] - filters.0.bias
        [1] - filters.0.kernel
        [2] - filters.1.bias
        [3] - filters.1.kernel
        ...
        [2 * num_filters] - linears.0.bias
        [2 * num_filters + 1] - linears.0.kernel
        [2 * num_filters + 2] - linears.1.bias
        [2 * num_filters + 3] - linears.1.kernel
        ...
        [4 * num_filters - 2] - output_linear.bias
        [4 * num_filters - 1] - output_linear.kernel
        where 4 * num_filters = len(x)
        """
        assert isinstance(x, (list)), "Input must be a list of parameters"
        assert (
            len(x) % 4 == 0
        ), f"Input must be a valid list of parameters (length should be a multiple of 4), but has length {len(x)}"

        num_filters = len(x) // 4

        idxs_filter_kernels = [2 * i + 1 for i in range(num_filters)]
        idxs_filter_biases = [2 * i for i in range(num_filters)]
        idxs_linears_kernels = [2 * num_filters + 2 * i + 1 for i in range(num_filters)]
        idxs_linears_biases = [2 * num_filters + 2 * i for i in range(num_filters)]

        perm, rng = transform_func(x[idxs_filter_kernels[0]].shape[-1], rng)

        x[idxs_filter_kernels[0]] = self.permute_rows(x[idxs_filter_kernels[0]], perm)
        x[idxs_filter_biases[0]] = self.permute_rows_biases(x[idxs_filter_biases[0]], perm)
        x[idxs_linears_kernels[0]] = self.permute_cols(x[idxs_linears_kernels[0]], perm)

        for i in range(num_filters - 1):
            perm, rng = transform_func(x[idxs_filter_kernels[0]].shape[-1], rng)
            # perm = torch.randperm(x[idxs_filter_kernels[0]].shape[-1])
            x[idxs_linears_kernels[i + 1]] = self.permute_cols(
                x[idxs_linears_kernels[i + 1]], perm
            )
            x[idxs_linears_kernels[i]] = self.permute_rows(x[idxs_linears_kernels[i]], perm)
            x[idxs_linears_biases[i]] = self.permute_rows_biases(x[idxs_linears_biases[i]], perm)
            x[idxs_filter_kernels[i + 1]] = self.permute_rows(x[idxs_filter_kernels[i + 1]], perm)
            x[idxs_filter_biases[i + 1]] = self.permute_rows_biases(
                x[idxs_filter_biases[i + 1]], perm
            )

        return x, rng

    def _set_transform(self, transform):
        self.transform = lambda x, rng: self.pre_transform(x, rng, transform)

    def numpy_transform(self, x, rng: np.random.Generator):
        return rng.permutation(x), rng

    def pytorch_transform(self, x, rng):
        return torch.randperm(x, generator=rng), rng

    def jax_transform(self, x, rng):
        new_rng, perm_rng = jax.random.split(rng)
        return jax.random.permutation(perm_rng, x), new_rng

    def permute_rows(self, x, perm):
        return x[..., perm]

    def permute_cols(self, x, perm):
        return x[perm]

    def permute_rows_biases(self, x, perm):
        return x[perm]
