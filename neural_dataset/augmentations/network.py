from typing import Any, Dict, List, Optional, Tuple, Union

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


class RandomQuantileWeightDropout(JointParameterAugmentation):
    """Randomly masks out weights of the network below a certain quantile. The quantile is
    uniformly sampled between min_quantile and max_quantile.

    Args:
        platform: Platform to use. Can be "numpy", "jax", "pytorch", or "auto".
            The latter will automatically detect the platform based on the input.
        seed: Seed for the random number generator. If None, the default global
            random number generator will be used. If an integer, a new random number
            generator will be created with the given seed. If a random number generator
            is given, it will be used directly.
        param_keys: List of parameter keys. Optional, not needed for this augmentation.
        min_quantile: Minimum quantile to use when sampling to quantile to use to mask out weights.
        max_quantile: Maximum quantile to use when sampling to quantile to use to mask out weights.
    """

    def __init__(
        self,
        *args,
        min_quantile: float = 0.0,
        max_quantile: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.min_quantile = min_quantile
        self.max_quantile = max_quantile
        assert 0 <= self.min_quantile <= 1, "min_quantile must be between 0 and 1"
        assert 0 <= self.max_quantile <= 1, "max_quantile must be between 0 and 1"
        assert (
            self.min_quantile <= self.max_quantile
        ), "min_quantile must be smaller than max_quantile"

    def numpy_transform(self, x: ParametersList, rng: Optional[np.random.Generator] = None):
        comb_tensor = np.concatenate([t.reshape(self.batch_size, -1) for t in x], axis=1)
        quantile = self.rng.uniform(self.min_quantile, self.max_quantile, size=(self.batch_size,))
        threshold = np.quantile(np.abs(comb_tensor), quantile, axis=1)
        return [t * (np.abs(t) >= threshold) for t in x], rng

    def jax_transform(self, x: ParametersList, rng: Optional[jax.random.PRNGKey] = None):
        comb_tensor = jnp.concatenate([t.reshape(self.batch_size, -1) for t in x], axis=1)
        new_rng, quant_rng = jax.random.split(rng)
        quantile = jax.random.uniform(
            key=quant_rng,
            shape=(self.batch_size,),
            minval=self.min_quantile,
            maxval=self.max_quantile,
        )[0]
        threshold = jnp.quantile(jnp.abs(comb_tensor), quantile, axis=1)
        return [t * (jnp.abs(t) >= threshold) for t in x], new_rng

    def pytorch_transform(self, x: ParametersList, rng: Optional[torch.Generator] = None):
        comb_tensor = torch.cat([t.reshape(self.batch_size, -1) for t in x], dim=1)

        quantile = (
            torch.rand((self.batch_size,), generator=rng) * (self.max_quantile - self.min_quantile)
            + self.min_quantile
        )
        threshold = torch.quantile(torch.abs(comb_tensor), quantile, dim=1)

        return [t * (torch.abs(t) >= threshold) for t in x], rng


class RandomDropout(IndividualParameterAugmentation):
    """Randomly sets parameters to zero.

    Args:
        platform: Platform to use. Can be "numpy", "jax", "pytorch", or "auto".
            The latter will automatically detect the platform based on the input.
        seed: Seed for the random number generator. If None, the default global
            random number generator will be used. If an integer, a new random number
            generator will be created with the given seed. If a random number generator
            is given, it will be used directly.
        param_keys: List of parameter keys. If None, all parameters will be augmented.
        exclude_params: List of parameter keys to exclude from augmentation.
        p: Probability of setting a parameter to zero. Can be a float or a dictionary
            mapping parameter keys to probabilities.
    """

    def __init__(self, *args, p: Union[float, Dict[str, float]] = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        if isinstance(self.p, dict):
            assert self.param_keys is not None, "param_keys must be specified if p is a dict"
            assert len(self.p) + len(self.exclude_params) == len(
                self.param_keys
            ), f"Number of parameters ({len(self.p)}) must match number of param keys ({len(self.param_keys)}) minus number of excluded parameters ({len(self.exclude_params)})"
            assert all(
                [k in self.param_keys for k in self.p]
            ), "param_keys must contain all keys in p"
            assert all(
                [0 <= prob <= 1 for prob in self.p.values()]
            ), "probabilities must be between 0 and 1"
        else:
            assert isinstance(self.p, float), "p must be a float"
            assert 0 <= self.p <= 1, "p must be between 0 and 1"

    def _get_p(self, param_key: Optional[str] = None):
        if isinstance(self.p, dict):
            assert param_key is not None, "param_key must be specified if p is a dict"
            assert param_key in self.p, f"param_key {param_key} not found in p"
            return self.p[param_key]
        else:
            return self.p

    def numpy_transform(
        self, x, param_key: Optional[str] = None, rng: Optional[np.random.Generator] = None
    ):
        p = self._get_p(param_key)
        return x * (rng.random(x.shape) > p), rng

    def jax_transform(
        self, x, param_key: Optional[str] = None, rng: Optional[jax.random.PRNGKey] = None
    ):
        p = self._get_p(param_key)
        new_rng, rng = jax.random.split(rng)
        return x * (jax.random.uniform(rng, x.shape) > p), new_rng

    def pytorch_transform(
        self, x, param_key: Optional[str] = None, rng: Optional[torch.Generator] = None
    ):
        p = self._get_p(param_key)
        return x * (torch.rand(x.shape, generator=rng) > p), rng

    def __repr__(self):
        return f"RandomDropout(platform={self.platform}, p={self.p}{', exclude_params=' + str(self.exclude_params) if self.exclude_params else ''})"


class RandomGaussianNoise(IndividualParameterAugmentation):
    """Adds Gaussian noise to parameters.

    Args:
        platform: Platform to use. Can be "numpy", "jax", "pytorch", or "auto".
            The latter will automatically detect the platform based on the input.
        seed: Seed for the random number generator. If None, the default global
            random number generator will be used. If an integer, a new random number
            generator will be created with the given seed. If a random number generator
            is given, it will be used directly.
        param_keys: List of parameter keys. If None, all parameters will be augmented.
        exclude_params: List of parameter keys to exclude from augmentation.
        sigma: Standard deviation of the Gaussian noise. Can be a float or a dictionary
            mapping parameter keys to standard deviations.
    """

    def __init__(self, *args, sigma: Union[Dict[str, Any], float] = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma = sigma

    def _get_sigma(self, param_key: Optional[str] = None):
        if isinstance(self.sigma, dict):
            assert param_key is not None, "param_key must be specified if sigma is a dict"
            assert param_key in self.sigma, f"param_key {param_key} not found in sigma"
            return self.sigma[param_key]
        else:
            return self.sigma

    def numpy_transform(
        self, x, param_key: Optional[str] = None, rng: Optional[np.random.Generator] = None
    ):
        sigma = self._get_sigma(param_key)
        return x + rng.normal(0, sigma, size=x.shape), rng

    def jax_transform(
        self, x, param_key: Optional[str] = None, rng: Optional[jax.random.PRNGKey] = None
    ):
        sigma = self._get_sigma(param_key)
        new_rng, rng = jax.random.split(rng)
        return x + jax.random.normal(rng, x.shape) * sigma, new_rng

    def pytorch_transform(
        self, x, param_key: Optional[str] = None, rng: Optional[torch.Generator] = None
    ):
        sigma = self._get_sigma(param_key)
        return x + torch.normal(0, sigma, size=x.shape, generator=rng, device=self.device), rng

    def __repr__(self):
        return f"GaussianNoise(platform={self.platform}, sigma={self.sigma}{', exclude_params=' + str(self.exclude_params) if self.exclude_params else ''})"
