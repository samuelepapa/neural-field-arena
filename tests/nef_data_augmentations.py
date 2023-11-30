import os

import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest

from neural_dataset.transform import (
    ParametersToList,
    RandomDropout,
    RandomGaussianNoise,
    RandomMLPWeightPermutation,
    RandomQuantileWeightDropout,
    RandomRotate,
    RandomScale,
)


class TestParametersToList(absltest.TestCase):
    def test_simple_network(self):
        network = [
            ("fc1.weight", (3, 32)),
            ("fc1.bias", (32,)),
            ("fc2.weight", (32, 64)),
            ("fc2.bias", (64,)),
            ("fc3.weight", (64, 10)),
            ("fc3.bias", (10,)),
        ]
        self.transformation = ParametersToList(param_structure=network)
        inp = np.random.normal(size=(sum([np.prod(p[1]) for p in network]),))
        out = self.transformation(inp)
        self.assertTrue(isinstance(out, (list, tuple)))
        self.assertEqual(len(out), len(network))
        for i, (_, shape) in enumerate(network):
            self.assertEqual(out[i].shape, shape)
            self.assertTrue(np.all(inp[: np.prod(shape)] == out[i].reshape(-1)))
            inp = inp[np.prod(shape) :]


class TestRandomQuantileWeightDropout(absltest.TestCase):
    def test_numpy(self):
        self.quantile_dropout = RandomQuantileWeightDropout(
            platform="numpy", seed=42, min_quantile=0.0, max_quantile=0.2
        )
        inp = np.random.normal(size=(100, 100))
        out = self.quantile_dropout(inp)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual(out.shape, (100, 100))
        eq = out == inp
        zeros = out == 0
        num_zeros = zeros.sum()
        self.assertEqual(eq.sum() + num_zeros, np.prod(out.shape))
        self.assertLessEqual(num_zeros / np.prod(out.shape), 0.2)
        self.assertGreaterEqual(num_zeros / np.prod(out.shape), 0.0)
        self.assertLessEqual(np.abs(inp[zeros]).max(), np.abs(inp[eq]).min())
        # Check that dropout changes seed when called
        out2 = self.quantile_dropout(inp)
        self.assertFalse(np.all(out == out2))

    def test_torch(self):
        self.quantile_dropout = RandomQuantileWeightDropout(
            platform="torch", seed=42, min_quantile=0.15, max_quantile=0.2
        )
        inp = torch.randn(100, 100)
        out = self.quantile_dropout(inp)
        self.assertTrue(isinstance(out, torch.Tensor))
        self.assertEqual(out.shape, (100, 100))
        eq = out == inp
        zeros = out == 0
        num_zeros = zeros.sum()
        self.assertEqual(eq.sum() + num_zeros, np.prod(out.shape))
        self.assertLessEqual(num_zeros / np.prod(out.shape), 0.2)
        self.assertGreaterEqual(num_zeros / np.prod(out.shape), 0.15)
        self.assertLessEqual(inp[zeros].abs().max(), inp[eq].abs().min())
        # Check that dropout changes seed when called
        out2 = self.quantile_dropout(inp)
        self.assertFalse(torch.all(out == out2))

    def test_jax(self):
        self.quantile_dropout = RandomQuantileWeightDropout(
            platform="jax", seed=42, min_quantile=0.1, max_quantile=0.2
        )
        inp = jax.random.normal(jax.random.PRNGKey(123), shape=(100, 100))
        out = self.quantile_dropout(inp)
        self.assertTrue(isinstance(out, jax.Array))
        self.assertEqual(out.shape, (100, 100))
        eq = out == inp
        zeros = out == 0
        num_zeros = zeros.sum()
        self.assertEqual(eq.sum() + num_zeros, np.prod(out.shape))
        self.assertLessEqual(num_zeros / np.prod(out.shape), 0.2)
        self.assertGreaterEqual(num_zeros / np.prod(out.shape), 0.1)
        self.assertLessEqual(jnp.abs(inp[zeros]).max(), jnp.abs(inp[eq]).min())
        # Check that dropout changes seed when called
        out2 = self.quantile_dropout(inp)
        self.assertFalse(jnp.all(out == out2))

    def test_auto(self):
        self.quantile_dropout = RandomQuantileWeightDropout(
            platform="auto", seed=42, max_quantile=0.2
        )
        inp = torch.randn(100, 100)
        out = self.quantile_dropout(inp)
        self.assertTrue(isinstance(out, torch.Tensor))
        self.assertEqual(out.shape, (100, 100))

    def test_weight_list(self):
        self.quantile_dropout = RandomQuantileWeightDropout(
            platform="numpy", seed=42, max_quantile=0.2
        )
        inp = [
            np.random.normal(size=(np.random.randint(32, 128), np.random.randint(32, 128)))
            for _ in range(8)
        ]
        out = self.quantile_dropout([inp])[0]
        self.assertTrue(isinstance(out, (list, tuple)))
        self.assertEqual(len(out), len(inp))
        for _, (inp_i, out_i) in enumerate(zip(inp, out)):
            self.assertEqual(inp_i.shape, out_i.shape)
            eq = out_i == inp_i
            zeros = out_i == 0
            num_zeros = zeros.sum()
            self.assertEqual(eq.sum() + num_zeros, np.prod(out_i.shape))
            self.assertLess(num_zeros / np.prod(out_i.shape), 0.2)
            self.assertLessEqual(np.abs(inp_i[zeros]).max(), np.abs(inp_i[eq]).min())


class TestRandomDropout(absltest.TestCase):
    def test_numpy(self):
        self.dropout = RandomDropout(platform="numpy", seed=42, p=0.5)
        inp = np.random.normal(size=(100, 100))
        out = self.dropout(inp)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual(out.shape, (100, 100))
        eq = out == inp
        zeros = out == 0
        num_zeros = zeros.sum()
        self.assertEqual(eq.sum() + num_zeros, np.prod(out.shape))
        self.assertAlmostEqual(num_zeros / np.prod(out.shape), 0.5, delta=0.1)
        # Check that dropout changes seed when called
        out2 = self.dropout(inp)
        self.assertFalse(np.all(out == out2))

    def test_torch(self):
        self.dropout = RandomDropout(platform="torch", seed=42, p=0.5)
        inp = torch.randn(100, 100)
        out = self.dropout(inp)
        self.assertTrue(isinstance(out, torch.Tensor))
        self.assertEqual(out.shape, (100, 100))
        eq = out == inp
        zeros = out == 0
        num_zeros = zeros.sum()
        self.assertEqual(eq.sum() + num_zeros, np.prod(out.shape))
        self.assertAlmostEqual(num_zeros / np.prod(out.shape), 0.5, delta=0.1)
        # Check that dropout changes seed when called
        out2 = self.dropout(inp)
        self.assertFalse(torch.all(out == out2))

    def test_jax(self):
        self.dropout = RandomDropout(platform="jax", seed=42, p=0.5)
        inp = jax.random.normal(jax.random.PRNGKey(123), shape=(100, 100))
        out = self.dropout(inp)
        self.assertTrue(isinstance(out, jax.Array))
        self.assertEqual(out.shape, (100, 100))
        eq = out == inp
        zeros = out == 0
        num_zeros = zeros.sum()
        self.assertEqual(eq.sum() + num_zeros, np.prod(out.shape))
        self.assertAlmostEqual(num_zeros / np.prod(out.shape), 0.5, delta=0.1)
        # Check that dropout changes seed when called
        out2 = self.dropout(inp)
        self.assertFalse(jnp.all(out == out2))

    def test_auto(self):
        self.dropout = RandomDropout(platform="auto", seed=42, p=0.5)
        inp = np.random.normal(size=(100, 100))
        out = self.dropout(inp)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual(out.shape, (100, 100))

    def test_random_p(self):
        for _ in range(5):
            p = np.random.uniform(0.1, 0.9)
            self.dropout = RandomDropout(platform="numpy", seed=42, p=p)
            inp = np.random.normal(size=(100, 100))
            out = self.dropout(inp)
            self.assertTrue(isinstance(out, np.ndarray))
            self.assertEqual(out.shape, (100, 100))
            eq = out == inp
            zeros = out == 0
            num_zeros = zeros.sum()
            self.assertEqual(eq.sum() + num_zeros, np.prod(out.shape))
            self.assertAlmostEqual(num_zeros / np.prod(out.shape), p, delta=0.1)

    def test_param_list(self):
        param_keys = [
            "conv1.weight",
            "conv1.bias",
            "conv2.weight",
            "conv2.bias",
            "fc1.weight",
            "fc1.bias",
        ]
        exclude_params = ["conv1.bias", "conv2.bias"]
        self.dropout = RandomDropout(
            platform="numpy", seed=42, p=0.5, param_keys=param_keys, exclude_params=exclude_params
        )
        inp = [
            np.random.normal(size=(np.random.randint(32, 128), np.random.randint(32, 128)))
            for _ in range(len(param_keys))
        ]
        out = self.dropout([inp])[0]
        self.assertTrue(isinstance(out, (list, tuple)))
        self.assertEqual(len(out), len(param_keys))
        for i, (inp_i, out_i) in enumerate(zip(inp, out)):
            if param_keys[i] in exclude_params:
                self.assertTrue(np.all(inp_i == out_i))
            else:
                self.assertFalse(np.all(inp_i == out_i))
                num_zeros = (out_i == 0).sum()
                self.assertAlmostEqual(num_zeros / np.prod(out_i.shape), 0.5, delta=0.1)

    def test_param_p(self):
        param_keys = [
            "conv1.weight",
            "conv1.bias",
            "conv2.weight",
            "conv2.bias",
            "fc1.weight",
            "fc1.bias",
        ]
        exclude_params = ["conv1.bias", "conv2.bias"]
        p = {"conv1.weight": 0.2, "conv2.weight": 0.4, "fc1.weight": 0.6, "fc1.bias": 0.8}
        self.dropout = RandomDropout(
            platform="numpy", seed=42, p=p, param_keys=param_keys, exclude_params=exclude_params
        )
        inp = [
            np.random.normal(size=(np.random.randint(32, 128), np.random.randint(32, 128)))
            for _ in range(len(param_keys))
        ]
        out = self.dropout([inp])[0]
        self.assertTrue(isinstance(out, (list, tuple)))
        self.assertEqual(len(out), len(param_keys))
        for i, (key, inp_i, out_i) in enumerate(zip(param_keys, inp, out)):
            if param_keys[i] in exclude_params:
                self.assertTrue(np.all(inp_i == out_i))
            else:
                self.assertFalse(np.all(inp_i == out_i))
                num_zeros = (out_i == 0).sum()
                self.assertAlmostEqual(num_zeros / np.prod(out_i.shape), p[key], delta=0.1)


class TestGaussianNoise(absltest.TestCase):
    def test_numpy(self):
        self.gaussian_noise = RandomGaussianNoise(platform="numpy", seed=42, sigma=1.0)
        inp = np.zeros((100, 100))
        out = self.gaussian_noise(inp)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual(out.shape, (100, 100))
        self.assertFalse(np.all(inp == out))
        self.assertAlmostEqual(np.std(out), 1.0, delta=0.1)

    def test_torch(self):
        self.gaussian_noise = RandomGaussianNoise(platform="torch", seed=42, sigma=1.0)
        inp = torch.zeros((100, 100))
        out = self.gaussian_noise(inp)
        self.assertTrue(isinstance(out, torch.Tensor))
        self.assertEqual(out.shape, (100, 100))
        self.assertFalse(torch.all(inp == out))
        self.assertAlmostEqual(torch.std(out), 1.0, delta=0.1)

    def test_jax(self):
        self.gaussian_noise = RandomGaussianNoise(platform="jax", seed=42, sigma=1.0)
        inp = jnp.zeros((100, 100))
        out = self.gaussian_noise(inp)
        self.assertTrue(isinstance(out, jax.Array))
        self.assertEqual(out.shape, (100, 100))
        self.assertFalse(jnp.all(inp == out))
        self.assertAlmostEqual(jnp.std(out), 1.0, delta=0.1)

    def test_auto(self):
        self.gaussian_noise = RandomGaussianNoise(platform="auto", seed=42, sigma=1.0)
        inp = jnp.zeros((100, 100))
        out = self.gaussian_noise(inp)
        self.assertTrue(isinstance(out, jax.Array))
        self.assertEqual(out.shape, (100, 100))
        self.assertFalse(np.all(inp == out))
        self.assertAlmostEqual(np.std(out), 1.0, delta=0.1)

    def test_random_sigma(self):
        for _ in range(5):
            sigma = np.random.uniform(0.5, 2.0)
            self.gaussian_noise = RandomGaussianNoise(platform="numpy", seed=42, sigma=sigma)
            inp = np.zeros((100, 100))
            out = self.gaussian_noise(inp)
            self.assertTrue(isinstance(out, np.ndarray))
            self.assertEqual(out.shape, (100, 100))
            self.assertFalse(np.all(inp == out))
            self.assertAlmostEqual(np.std(out), sigma, delta=0.1)


class TestRandomRotate(absltest.TestCase):
    def test_numpy(self):
        self.rotation = RandomRotate(
            platform="numpy", seed=42, min_angle=-2 * np.pi, max_angle=2 * np.pi
        )
        inp = np.ones((100, 2))
        out = self.rotation(inp)
        self.assertTrue(isinstance(out, np.ndarray))
        self.assertEqual(out.shape, (100, 2))
        self.assertFalse(np.all(inp == out))

    def test_torch(self):
        self.rotation = RandomRotate(
            platform="torch", seed=42, min_angle=-2 * np.pi, max_angle=2 * np.pi
        )
        inp = torch.ones((100, 2))
        out = self.rotation(inp)
        self.assertTrue(isinstance(out, torch.Tensor))
        self.assertEqual(out.shape, (100, 2))
        self.assertFalse(torch.all(inp == out))

    def test_jax(self):
        self.rotation = RandomRotate(
            platform="jax", seed=42, min_angle=-2 * np.pi, max_angle=2 * np.pi
        )
        inp = jnp.ones((100, 2))
        out = self.rotation(inp)
        self.assertTrue(isinstance(out, jax.Array))
        self.assertEqual(out.shape, (100, 2))
        self.assertFalse(jnp.all(inp == out))

    def test_auto(self):
        self.rotation = RandomRotate(
            platform="auto", seed=42, min_angle=-2 * np.pi, max_angle=2 * np.pi
        )
        inp = jnp.ones((100, 2))
        out = self.rotation(inp)
        self.assertTrue(isinstance(out, jax.Array))
        self.assertEqual(out.shape, (100, 2))
        self.assertFalse(np.all(inp == out))


class TestRandomMLPWeightPermutation(absltest.TestCase):
    def test_numpy(self):
        param_keys = [
            "params.Dense_0.bias",
            "params.Dense_0.kernel",
            "params.Dense_1.bias",
            "params.Dense_1.kernel",
            "params.Dense_2.bias",
            "params.Dense_2.kernel",
        ]
        layer_sizes = [2, 16, 32, 10]
        params = []
        for key in param_keys:
            layer_idx = int(key.split("_")[1].split(".")[0])
            if "bias" in key:
                params.append(np.random.normal(size=(layer_sizes[layer_idx + 1],)))
            else:
                params.append(
                    np.random.normal(size=(layer_sizes[layer_idx], layer_sizes[layer_idx + 1]))
                )
        self.weight_permutation = RandomMLPWeightPermutation(
            platform="numpy", seed=42, param_keys=param_keys
        )
        out = self.weight_permutation([params])
        self.assertTrue(isinstance(out, (list, tuple)))
        self.assertEqual(len(out), 1)
        self.assertTrue(isinstance(out[0], (list, tuple)))
        self.assertEqual(len(out[0]), len(param_keys))
        for i, (inp_i, out_i) in enumerate(zip(params, out[0])):
            self.assertEqual(inp_i.shape, out_i.shape)
            if param_keys[i] != "params.Dense_2.bias":
                self.assertFalse(np.all(inp_i == out_i))
            self.assertAlmostEqual(np.std(out_i), np.std(inp_i), delta=0.001)
            self.assertAlmostEqual(np.mean(out_i), np.mean(inp_i), delta=0.001)

    def test_torch(self):
        param_keys = [
            "params.Dense_0.bias",
            "params.Dense_0.kernel",
            "params.Dense_1.bias",
            "params.Dense_1.kernel",
            "params.Dense_2.bias",
            "params.Dense_2.kernel",
        ]
        layer_sizes = [2, 16, 32, 10]
        params = []
        for key in param_keys:
            layer_idx = int(key.split("_")[1].split(".")[0])
            if "bias" in key:
                params.append(torch.randn(layer_sizes[layer_idx + 1]))
            else:
                params.append(torch.randn(layer_sizes[layer_idx], layer_sizes[layer_idx + 1]))
        self.weight_permutation = RandomMLPWeightPermutation(
            platform="torch", seed=42, param_keys=param_keys
        )
        out = self.weight_permutation([params])
        self.assertTrue(isinstance(out, (list, tuple)))
        self.assertEqual(len(out), 1)
        self.assertTrue(isinstance(out[0], (list, tuple)))
        self.assertEqual(len(out[0]), len(param_keys))
        for i, (inp_i, out_i) in enumerate(zip(params, out[0])):
            self.assertEqual(inp_i.shape, out_i.shape)
            if param_keys[i] != "params.Dense_2.bias":
                self.assertFalse(torch.all(inp_i == out_i))
            self.assertAlmostEqual(torch.std(out_i), torch.std(inp_i), delta=0.001)
            self.assertAlmostEqual(torch.mean(out_i), torch.mean(inp_i), delta=0.001)

    def test_jax(self):
        param_keys = [
            "params.Dense_0.bias",
            "params.Dense_0.kernel",
            "params.Dense_1.bias",
            "params.Dense_1.kernel",
            "params.Dense_2.bias",
            "params.Dense_2.kernel",
        ]
        layer_sizes = [2, 16, 32, 10]
        params = []
        for key in param_keys:
            layer_idx = int(key.split("_")[1].split(".")[0])
            if "bias" in key:
                params.append(
                    jax.random.normal(
                        jax.random.PRNGKey(123),
                        shape=(
                            1,
                            layer_sizes[layer_idx + 1],
                        ),
                    )
                )
            else:
                params.append(
                    jax.random.normal(
                        jax.random.PRNGKey(123),
                        shape=(1, layer_sizes[layer_idx], layer_sizes[layer_idx + 1]),
                    )
                )
        self.weight_permutation = RandomMLPWeightPermutation(
            platform="jax", seed=42, param_keys=param_keys
        )
        out = self.weight_permutation([params])
        self.assertTrue(isinstance(out, (list, tuple)))
        self.assertEqual(len(out), 1)
        self.assertTrue(isinstance(out[0], (list, tuple)))
        self.assertEqual(len(out[0]), len(param_keys))
        for i, (inp_i, out_i) in enumerate(zip(params, out[0])):
            self.assertEqual(inp_i.shape, out_i.shape)
            if param_keys[i] != "params.Dense_2.bias":
                self.assertFalse(jnp.all(inp_i == out_i))
            self.assertAlmostEqual(jnp.std(out_i), jnp.std(inp_i), delta=0.001)
            self.assertAlmostEqual(jnp.mean(out_i), jnp.mean(inp_i), delta=0.001)
