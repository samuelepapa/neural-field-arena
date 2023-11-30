import time

import numpy as np
from absl import logging
from absl.testing import absltest

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


def create_MFN(num_layers=4, hidden_dim=5, input_dim=2, output_dim=3, identity=False):
    params = []
    param_keys = []

    def tensor_generator(*args):
        if identity:
            if len(args) == 1:
                return np.zeros(*args)
            elif len(args) == 2:
                return np.eye(*args)
            else:
                return np.ones(args)
        else:
            return np.random.rand(*args)

    for cur_layer in range(num_layers):
        param_keys.append(f"filter_{cur_layer}.bias")
        params.append(tensor_generator(hidden_dim))
        param_keys.append(f"filter_{cur_layer}.kernel")
        params.append(tensor_generator(input_dim, hidden_dim))

    for cur_layer in range(num_layers - 1):
        param_keys.append(f"linear_{cur_layer}.bias")
        params.append(tensor_generator(hidden_dim))
        param_keys.append(f"linear_{cur_layer}.kernel")
        params.append(tensor_generator(hidden_dim, hidden_dim))

    param_keys.append("output_layer.bias")
    params.append(tensor_generator(output_dim))
    param_keys.append("output_layer.kernel")
    params.append(tensor_generator(hidden_dim, output_dim))

    return params, param_keys


def create_SIREN(num_layers=4, hidden_dim=5, input_dim=2, output_dim=3):
    params = []
    param_keys = []

    param_keys.append("linear_0.bias")
    params.append(np.random.rand(hidden_dim))
    param_keys.append("linear_0.kernel")
    params.append(np.random.rand(input_dim, hidden_dim))

    for cur_layer in range(1, num_layers - 1):
        param_keys.append(f"linear_{cur_layer}.bias")
        params.append(np.random.rand(hidden_dim))
        param_keys.append(f"linear_{cur_layer}.kernel")
        params.append(np.random.rand(hidden_dim, hidden_dim))

    param_keys.append("output_layer.bias")
    params.append(np.random.rand(output_dim))
    param_keys.append("output_layer.kernel")
    params.append(np.random.rand(hidden_dim, output_dim))

    return params, param_keys


def allclose_multi_platform(array_a, array_b, platform="pytorch"):
    if platform == "pytorch":
        return torch.allclose(array_a, array_b)
    elif platform == "jax":
        return jnp.allclose(array_a, array_b)
    elif platform == "numpy":
        return np.allclose(array_a, array_b)
    else:
        raise ValueError(f"Unknown platform {platform}")


def test_transformation(
    testcase: absltest.TestCase,
    transformation,
    params,
    param_keys,
    platform="pytorch",
    num_iter=20000,
    log=True,
    rng=None,
):
    # convert params to torch tensors if platform is pytorch
    if platform == "pytorch":
        params = [torch.tensor(x, dtype=torch.float32) for x in params]
        if rng is None:
            rng = torch.Generator().manual_seed(42)
    elif platform == "jax":
        params = [jnp.array(x, dtype=jnp.float32) for x in params]
        if rng is None:
            rng = jax.random.PRNGKey(42)
    elif platform == "numpy":
        params = [np.array(x, dtype=np.float32) for x in params]
        if rng is None:
            rng = np.random.default_rng(42)

    # setup the transform function
    if platform == "jax":
        aug_func = jax.jit(transformation)
    else:
        aug_func = transformation

    datapoint = {"params": params}

    start_time = time.time()
    for i in range(num_iter):
        new_datapoint, rng = aug_func(datapoint, rng)
        if i == 0:
            for j, (param, new_param) in enumerate(
                zip(datapoint["params"], new_datapoint["params"])
            ):
                testcase.assertEqual(param.shape, new_param.shape)

    if log:
        logging.info(f"Time taken: {time.time() - start_time:.3f}")

    return new_datapoint, datapoint, rng
