import numpy as np
from absl.testing import absltest

from neural_dataset.augmentations import RandomDropout
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


class AugmentationNetworkTest(absltest.TestCase):
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

    def _test_dropout(self, params, keys, platform):
        p = 0.2

        aug = RandomDropout(platform=platform, p=p, param_keys=keys)

        new_datapoint, datapoint, rng = test_augmentation(
            self, aug, params, keys, platform=platform, num_iter=1, log=False
        )

        # check that around p of the parameters are zero
        average_zero = 0.0
        sum_items = 0.0
        for i in range(2000):
            new_datapoint, datapoint, rng = test_augmentation(
                self, aug, params, keys, platform=platform, num_iter=1, log=False, rng=rng
            )
            local_average = 0.0
            local_sum_items = 0.0
            for param in new_datapoint["params"]:
                local_average += float((param <= 1e-7).sum())
                local_sum_items += float(np.prod(param.shape))

            average_zero += local_average / local_sum_items
            sum_items += 1.0

        average_zero /= sum_items

        self.assertAlmostEqual(
            average_zero,
            p,
            delta=1e-2,
            msg=f"Expected {p} of the parameters to be zero, but got {average_zero}",
        )

    def test_MFN(self):
        self._test_dropout(self.params_MFN, self.param_keys_MFN, platform="numpy")
        if TORCH_AVAILABLE:
            self._test_dropout(self.params_MFN, self.param_keys_MFN, platform="pytorch")
        if JAX_AVAILABLE:
            self._test_dropout(self.params_MFN, self.param_keys_MFN, platform="jax")

    def test_SIREN(self):
        self._test_dropout(self.params_SIREN, self.param_keys_SIREN, platform="numpy")
        if TORCH_AVAILABLE:
            self._test_dropout(self.params_SIREN, self.param_keys_SIREN, platform="pytorch")
        if JAX_AVAILABLE:
            self._test_dropout(self.params_SIREN, self.param_keys_SIREN, platform="jax")
