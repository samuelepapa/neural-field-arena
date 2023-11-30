from absl.testing import absltest

from neural_dataset.transform import (
    RandomRotate,
    RandomScale,
    RandomTranslateMFN,
    RandomTranslateSIREN,
)
from tests.transformations.utils import (
    allclose_multi_platform,
    create_MFN,
    create_SIREN,
    test_transformation,
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


class TransformationGeometricTest(absltest.TestCase):
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

    def _test_translation_SIREN(self, params, param_keys, platform):
        random_translate = RandomTranslateSIREN(
            platform=platform, param_keys=param_keys, min_translation=0, max_translation=0.5
        )

        new_datapoint, datapoint, rng = test_transformation(
            self, random_translate, params, param_keys, platform=platform, num_iter=1, log=False
        )

        self.assertFalse(
            allclose_multi_platform(
                datapoint["params"][0], new_datapoint["params"][0], platform=platform
            )
        )

    def _test_translation(self, params, param_keys, platform):
        random_translate = RandomTranslateMFN(
            platform=platform, param_keys=param_keys, min_translation=0, max_translation=0.5
        )

        new_datapoint, datapoint, rng = test_transformation(
            self, random_translate, params, param_keys, platform=platform, num_iter=1, log=False
        )

        self.assertFalse(
            allclose_multi_platform(
                datapoint["params"][0], new_datapoint["params"][0], platform=platform
            )
        )

    def _test_rotation(self, params, param_keys, selected_param_idxs, platform):
        random_rotation = RandomRotate(
            platform=platform,
            param_keys=param_keys,
            min_angle=0,
            max_angle=3.14,
            selected_param_idxs=selected_param_idxs,
        )
        new_datapoint, datapoint, _ = test_transformation(
            self, random_rotation, params, param_keys, platform=platform, num_iter=1, log=False
        )
        first_datapoint, _, rng = test_transformation(
            self, random_rotation, params, param_keys, platform=platform, num_iter=1, log=False
        )
        second_datapoint, _, rng = test_transformation(
            self,
            random_rotation,
            params,
            param_keys,
            platform=platform,
            num_iter=1,
            log=False,
            rng=rng,
        )

        for i in selected_param_idxs:
            self.assertFalse(
                allclose_multi_platform(
                    datapoint["params"][i], new_datapoint["params"][i], platform=platform
                )
            )
            self.assertTrue(
                allclose_multi_platform(
                    first_datapoint["params"][i], new_datapoint["params"][i], platform=platform
                )
            )
            self.assertFalse(
                allclose_multi_platform(
                    first_datapoint["params"][i], second_datapoint["params"][i], platform=platform
                )
            )

        for i in range(len(param_keys)):
            if i not in selected_param_idxs:
                self.assertTrue(
                    allclose_multi_platform(
                        datapoint["params"][i], new_datapoint["params"][i], platform=platform
                    )
                )

    def _test_scale(self, params, param_keys, selected_param_idxs, platform):
        random_translate = RandomScale(
            platform=platform,
            param_keys=param_keys,
            min_scale=0.5,
            max_scale=2,
            selected_param_idxs=selected_param_idxs,
        )

        new_datapoint, datapoint, rng = test_transformation(
            self, random_translate, params, param_keys, platform=platform, num_iter=1, log=False
        )

        for i in selected_param_idxs:
            self.assertFalse(
                allclose_multi_platform(
                    datapoint["params"][i], new_datapoint["params"][i], platform=platform
                )
            )

        for i in range(len(param_keys)):
            if i not in selected_param_idxs:
                self.assertTrue(
                    allclose_multi_platform(
                        datapoint["params"][i], new_datapoint["params"][i], platform=platform
                    )
                )

    def test_MFN(self):
        self._test_translation(self.params_MFN, self.param_keys_MFN, platform="numpy")
        self._test_rotation(self.params_MFN, self.param_keys_MFN, [1, 3, 5], platform="numpy")
        self._test_scale(self.params_MFN, self.param_keys_MFN, [0, 2, 3, 4], platform="numpy")
        if TORCH_AVAILABLE:
            self._test_translation(self.params_MFN, self.param_keys_MFN, platform="pytorch")
            self._test_rotation(
                self.params_MFN, self.param_keys_MFN, [1, 3, 5], platform="pytorch"
            )
            self._test_scale(
                self.params_MFN, self.param_keys_MFN, [0, 2, 3, 4], platform="pytorch"
            )
        if JAX_AVAILABLE:
            self._test_translation(self.params_MFN, self.param_keys_MFN, platform="jax")
            self._test_rotation(self.params_MFN, self.param_keys_MFN, [1, 3, 5], platform="jax")
            self._test_scale(self.params_MFN, self.param_keys_MFN, [0, 2, 3, 4], platform="jax")

    def test_SIREN(self):
        self._test_translation_SIREN(self.params_SIREN, self.param_keys_SIREN, platform="numpy")
        self._test_rotation(
            self.params_SIREN,
            self.param_keys_SIREN,
            [
                1,
            ],
            platform="numpy",
        )
        self._test_scale(self.params_SIREN, self.param_keys_SIREN, [0, 3, 4], platform="numpy")
        if TORCH_AVAILABLE:
            self._test_translation_SIREN(
                self.params_SIREN, self.param_keys_SIREN, platform="pytorch"
            )
            self._test_rotation(
                self.params_SIREN,
                self.param_keys_SIREN,
                [
                    1,
                ],
                platform="pytorch",
            )
            self._test_scale(
                self.params_SIREN, self.param_keys_SIREN, [0, 3, 4], platform="pytorch"
            )
        if JAX_AVAILABLE:
            self._test_translation_SIREN(self.params_SIREN, self.param_keys_SIREN, platform="jax")
            self._test_rotation(
                self.params_SIREN,
                self.param_keys_SIREN,
                [
                    1,
                ],
                platform="jax",
            )
            self._test_scale(self.params_SIREN, self.param_keys_SIREN, [0, 3, 4], platform="jax")
