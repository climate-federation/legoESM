"""Batched Equinox layers and network shapes."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.wave.networks.mlp import LayerNorm, Linear, MLP, SourceTermMLP


def test_linear_matches_batched_matmul():
    layer = Linear(5, 3, key=jax.random.key(0))
    x = jnp.arange(30, dtype=jnp.float32).reshape(6, 5) / 30.0
    expected = x @ layer.weight.T + layer.bias
    assert np.allclose(np.asarray(layer(x)), np.asarray(expected))


def test_layernorm_matches_manual_formula():
    layer = LayerNorm(4, eps=1e-5)
    x = jnp.asarray([[1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0]])
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    expected = (x - mean) / jnp.sqrt(var + 1e-5) * layer.weight + layer.bias
    assert np.allclose(np.asarray(layer(x)), np.asarray(expected))


def test_mlp_shapes_and_depth_order():
    model = MLP(6, 2, depth=2, hidden_dim=4, use_layernorm=True, key=jax.random.key(0))
    assert len(model.layers) == 5
    out = model(jnp.ones((7, 6)))
    assert out.shape == (7, 2)


def test_mlp_rejects_wrong_input_width():
    model = MLP(6, 2, depth=1, hidden_dim=4, key=jax.random.key(0))
    with pytest.raises(ValueError, match="Expected last dim"):
        model(jnp.ones((7, 5)))


def _manual_norm(v, gain, bias, eps):
    mean = v.mean(axis=-1, keepdims=True)
    var = ((v - mean) ** 2).mean(axis=-1, keepdims=True)
    return (v - mean) / jnp.sqrt(var + eps) * gain + bias


def test_transport_mlp_layer_order_and_activation():
    """Pin linear -> norm -> relu -> final linear for the transport MLP.

    The manual reference uses non-commuting matrices and negative biases, so
    a swapped layer, a missing relu, or a relu after the final linear fails.
    """
    import equinox as eqx

    eps = 1e-5
    model = MLP(2, 2, depth=1, hidden_dim=2, use_layernorm=True,
                layernorm_eps=eps, key=jax.random.key(0))
    w0 = jnp.asarray([[1.0, 0.5], [-0.5, 1.0]])
    w1 = jnp.asarray([[1.0, 2.0], [0.0, 1.0]])
    b0 = jnp.asarray([-0.25, 0.5])
    b1 = jnp.asarray([0.3, -0.4])
    gain = jnp.asarray([1.5, 0.5])
    shift = jnp.asarray([0.1, -0.2])
    model = eqx.tree_at(lambda m: m.layers[0].weight, model, w0)
    model = eqx.tree_at(lambda m: m.layers[0].bias, model, b0)
    model = eqx.tree_at(lambda m: m.layers[1].weight, model, gain)
    model = eqx.tree_at(lambda m: m.layers[1].bias, model, shift)
    model = eqx.tree_at(lambda m: m.layers[2].weight, model, w1)
    model = eqx.tree_at(lambda m: m.layers[2].bias, model, b1)

    x = jnp.asarray([[1.0, -2.0], [0.5, 3.0]])
    h = jax.nn.relu(_manual_norm(x @ w0.T + b0, gain, shift, eps))
    expected = h @ w1.T + b1
    np.testing.assert_allclose(
        np.asarray(model(x)), np.asarray(expected), rtol=1e-5, atol=1e-6)


def test_mlp_without_layernorm_relu_follows_each_hidden_linear():
    import equinox as eqx

    model = MLP(2, 2, depth=1, hidden_dim=2, use_layernorm=False,
                key=jax.random.key(0))
    w0 = jnp.asarray([[1.0, 0.5], [-0.5, 1.0]])
    w1 = jnp.asarray([[1.0, 2.0], [0.0, 1.0]])
    b0 = jnp.asarray([-0.25, 0.5])
    b1 = jnp.asarray([0.3, -0.4])
    model = eqx.tree_at(lambda m: m.layers[0].weight, model, w0)
    model = eqx.tree_at(lambda m: m.layers[0].bias, model, b0)
    model = eqx.tree_at(lambda m: m.layers[1].weight, model, w1)
    model = eqx.tree_at(lambda m: m.layers[1].bias, model, b1)

    x = jnp.asarray([[1.0, -2.0], [0.5, 3.0]])
    expected = jax.nn.relu(x @ w0.T + b0) @ w1.T + b1
    np.testing.assert_allclose(
        np.asarray(model(x)), np.asarray(expected), rtol=1e-5, atol=1e-6)


def test_source_term_mlp_shapes():
    model = SourceTermMLP(12, 3, hidden_0=4, hidden_1=4, hidden_2=4,
                          key=jax.random.key(0))
    assert model(jnp.ones((2, 12))).shape == (2, 3)


def test_source_term_mlp_layer_order_and_activation():
    """Pin the four-linears-with-relu-after-each-norm order.

    The source MLP is [linear, norm, relu] x3 then a final linear; the
    manual reference below uses non-commuting matrices and negative biases,
    so a swapped linear layer or a missing relu changes the result.
    """
    import equinox as eqx

    eps = 1e-5
    model = SourceTermMLP(2, 2, hidden_0=2, hidden_1=2, hidden_2=2,
                          layernorm_eps=eps, key=jax.random.key(0))
    w0 = jnp.asarray([[1.0, 0.5], [-0.5, 1.0]])
    w1 = jnp.asarray([[1.0, 2.0], [0.0, 1.0]])
    w2 = jnp.asarray([[1.0, 0.0], [1.0, 1.0]])
    w3 = jnp.asarray([[1.0, -1.0], [0.5, 2.0]])
    model = eqx.tree_at(lambda m: m.linear0.weight, model, w0)
    model = eqx.tree_at(lambda m: m.linear1.weight, model, w1)
    model = eqx.tree_at(lambda m: m.linear2.weight, model, w2)
    model = eqx.tree_at(lambda m: m.linear3.weight, model, w3)
    for name in ("linear0", "linear1", "linear2", "linear3"):
        model = eqx.tree_at(
            lambda m, n=name: getattr(m, n).bias, model,
            jnp.asarray([-0.25, 0.5]))
    for name in ("norm0", "norm1", "norm2"):
        model = eqx.tree_at(
            lambda m, n=name: getattr(m, n).weight, model,
            jnp.asarray([1.5, 0.5]))
        model = eqx.tree_at(
            lambda m, n=name: getattr(m, n).bias, model,
            jnp.asarray([0.1, -0.2]))

    x = jnp.asarray([[1.0, -2.0], [0.5, 3.0]])

    def norm(v, g, b):
        mean = v.mean(axis=-1, keepdims=True)
        var = ((v - mean) ** 2).mean(axis=-1, keepdims=True)
        return (v - mean) / jnp.sqrt(var + eps) * g + b

    def linear(v, w):
        return v @ w.T + jnp.asarray([-0.25, 0.5])

    h = jax.nn.relu(norm(linear(x, w0), jnp.asarray([1.5, 0.5]),
                         jnp.asarray([0.1, -0.2])))
    h = jax.nn.relu(norm(linear(h, w1), jnp.asarray([1.5, 0.5]),
                         jnp.asarray([0.1, -0.2])))
    h = jax.nn.relu(norm(linear(h, w2), jnp.asarray([1.5, 0.5]),
                         jnp.asarray([0.1, -0.2])))
    expected = linear(h, w3)
    np.testing.assert_allclose(
        np.asarray(model(x)), np.asarray(expected), rtol=1e-5, atol=1e-6)
