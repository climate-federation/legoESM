"""Batched MLP matching the NeuralSWM Flax-NNX network.

Equinox's ``nn.Linear``/``nn.LayerNorm`` are unbatched: they take a
single vector and expect ``equinox.filter_vmap`` for a batch.  The wave
network evaluates hundreds of thousands of columns per step, so this
module stores the same weights (``(out, in)`` kernels and ``(out,)``
biases) but applies them to a leading batch axis directly.

The source builds ``[Linear, LayerNorm, Linear, LayerNorm, Linear]`` for
``depth=2`` and applies ``relu`` after every LayerNorm, with no
activation after the final linear layer.  The port keeps that order, the
source's LayerNorm epsilon, and Flax's lecun-normal kernel initialisation
(only used before weights are loaded).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

# Architecture of the released NeuralSWM weights.  These are shape defaults
# for constructing a fresh model; loading a checkpoint replaces every leaf.
_RELEASED_LAYERNORM_EPS = 1e-5
_RELEASED_MLP_HIDDEN = 256
_RELEASED_SOURCE_HIDDEN = (128, 256, 256)


class Linear(eqx.Module):
    """``y = x @ W.T + b`` for ``x`` of shape ``(..., in)``."""

    weight: jax.Array  # (out, in)
    bias: jax.Array    # (out,)

    def __init__(self, in_features: int, out_features: int, *, key: jax.Array):
        # Flax nnx.Linear default kernel_init: truncated lecun_normal with
        # fan_in on the input axis, and zero bias.
        init = jax.nn.initializers.lecun_normal(in_axis=-1, out_axis=-2)
        self.weight = init(key, (out_features, in_features), jnp.float32)
        self.bias = jnp.zeros((out_features,), dtype=self.weight.dtype)

    def __call__(self, x: jax.Array) -> jax.Array:
        return x @ self.weight.T + self.bias


class LayerNorm(eqx.Module):
    """Layer normalisation over the last axis, batched."""

    weight: jax.Array
    bias: jax.Array
    eps: float = eqx.field(static=True)

    def __init__(self, features: int, *, eps: float = _RELEASED_LAYERNORM_EPS):
        self.weight = jnp.ones((features,), dtype=jnp.float32)
        self.bias = jnp.zeros((features,), dtype=jnp.float32)
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        return (x - mean) / jnp.sqrt(var + self.eps) * self.weight + self.bias


class MLP(eqx.Module):
    """Generic MLP with the source's activation placement."""

    layers: tuple
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    use_layernorm: bool = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        depth: int = 2,
        hidden_dim: int = _RELEASED_MLP_HIDDEN,
        use_layernorm: bool = True,
        layernorm_eps: float = _RELEASED_LAYERNORM_EPS,
        key: jax.Array,
    ):
        if depth < 0:
            raise ValueError("depth must be >= 0")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.use_layernorm = use_layernorm

        keys = jax.random.split(key, 2 * depth + 2)
        layers = []
        cursor = 0
        if depth == 0:
            layers.append(Linear(in_channels, out_channels, key=keys[cursor]))
        else:
            layers.append(Linear(in_channels, hidden_dim, key=keys[cursor]))
            cursor += 1
            if use_layernorm:
                layers.append(LayerNorm(hidden_dim, eps=layernorm_eps))
            for _ in range(depth - 1):
                layers.append(Linear(hidden_dim, hidden_dim, key=keys[cursor]))
                cursor += 1
                if use_layernorm:
                    layers.append(LayerNorm(hidden_dim, eps=layernorm_eps))
            layers.append(Linear(hidden_dim, out_channels, key=keys[cursor]))
        self.layers = tuple(layers)

    def __call__(self, x: jax.Array) -> jax.Array:
        if x.shape[-1] != self.in_channels:
            raise ValueError(f"Expected last dim == {self.in_channels}, got {x.shape[-1]}")

        orig_shape = x.shape[:-1]
        x = jnp.reshape(x, (-1, self.in_channels))

        if self.depth == 0:
            y = self.layers[0](x)
            return jnp.reshape(y, (*orig_shape, self.out_channels))

        for layer in self.layers[:-1]:
            x = layer(x)
            if isinstance(layer, LayerNorm):
                x = jax.nn.relu(x)
            elif not self.use_layernorm:
                x = jax.nn.relu(x)

        y = self.layers[-1](x)
        return jnp.reshape(y, (*orig_shape, self.out_channels))


class SourceTermMLP(eqx.Module):
    """The four-linear source-term network of the released weights."""

    linear0: Linear
    norm0: LayerNorm
    linear1: Linear
    norm1: LayerNorm
    linear2: Linear
    norm2: LayerNorm
    linear3: Linear

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        hidden_0: int = _RELEASED_SOURCE_HIDDEN[0],
        hidden_1: int = _RELEASED_SOURCE_HIDDEN[1],
        hidden_2: int = _RELEASED_SOURCE_HIDDEN[2],
        layernorm_eps: float = _RELEASED_LAYERNORM_EPS,
        key: jax.Array,
    ):
        keys = jax.random.split(key, 4)
        self.linear0 = Linear(in_channels, hidden_0, key=keys[0])
        self.norm0 = LayerNorm(hidden_0, eps=layernorm_eps)
        self.linear1 = Linear(hidden_0, hidden_1, key=keys[1])
        self.norm1 = LayerNorm(hidden_1, eps=layernorm_eps)
        self.linear2 = Linear(hidden_1, hidden_2, key=keys[2])
        self.norm2 = LayerNorm(hidden_2, eps=layernorm_eps)
        self.linear3 = Linear(hidden_2, out_channels, key=keys[3])

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jax.nn.relu(self.norm0(self.linear0(x)))
        x = jax.nn.relu(self.norm1(self.linear1(x)))
        x = jax.nn.relu(self.norm2(self.linear2(x)))
        return self.linear3(x)


__all__ = ["LayerNorm", "Linear", "MLP", "SourceTermMLP"]
