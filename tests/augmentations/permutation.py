import numpy as np
from absl.testing import absltest

from neural_dataset.augmentations import (
    RandomFourierNetWeightPermutation,
    RandomMLPWeightPermutation,
)
from tests.augmentations.utils import (
    allclose_multi_platform,
    create_MFN,
    create_SIREN,
    test_augmentation,
)

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class AugmentationPermutationTest(absltest.TestCase):
    def setUp(self):
        num_layers = 4
        hidden_dim = 5
        input_dim = 2
        output_dim = 3
        # create the parameters for the networks used in the tests
        self.params_MFN, self.param_keys_MFN = create_MFN(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            input_dim=input_dim,
            output_dim=output_dim,
        )
        self.params_SIREN, self.param_keys_SIREN = create_SIREN(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            input_dim=input_dim,
            output_dim=output_dim,
        )

    def _test_permutation_SIREN(self, params, param_keys, platform):
        random_permute = RandomMLPWeightPermutation(
            platform=platform, seed=0, param_keys=param_keys
        )
        new_datapoint, datapoint, rng = test_augmentation(
            self, random_permute, params, param_keys, platform=platform, num_iter=2, log=False
        )

        self.assertFalse(
            allclose_multi_platform(
                datapoint["params"][0], new_datapoint["params"][0], platform=platform
            )
        )

        for i in range(len(datapoint["params"])):
            self.assertEqual(datapoint["params"][i].shape, new_datapoint["params"][i].shape)

    def _test_permutation_MFN(self, params, param_keys, platform):
        random_permute = RandomFourierNetWeightPermutation(
            platform=platform, seed=0, param_keys=param_keys
        )
        new_datapoint, datapoint, rng = test_augmentation(
            self, random_permute, params, param_keys, platform=platform, num_iter=2, log=False
        )

        self.assertFalse(
            allclose_multi_platform(
                datapoint["params"][0], new_datapoint["params"][0], platform=platform
            )
        )

        for i in range(len(datapoint["params"])):
            self.assertEqual(datapoint["params"][i].shape, new_datapoint["params"][i].shape)

    def test_MFN(self):
        self._test_permutation_MFN(self.params_MFN, self.param_keys_MFN, platform="numpy")
        if TORCH_AVAILABLE:
            self._test_permutation_MFN(self.params_MFN, self.param_keys_MFN, platform="pytorch")
        if JAX_AVAILABLE:
            self._test_permutation_MFN(self.params_MFN, self.param_keys_MFN, platform="jax")

    def test_SIREN(self):
        self._test_permutation_SIREN(self.params_SIREN, self.param_keys_SIREN, platform="numpy")
        if TORCH_AVAILABLE:
            self._test_permutation_SIREN(
                self.params_SIREN, self.param_keys_SIREN, platform="pytorch"
            )
        if JAX_AVAILABLE:
            self._test_permutation_SIREN(self.params_SIREN, self.param_keys_SIREN, platform="jax")
