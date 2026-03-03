"""Learned spectral convolution for SFNO.

Applies a learnable linear operator in spherical harmonic space via
complex matrix multiplication. This replaces the classical spectral
convolution of Fourier Neural Operators with one that respects the
geometry of the sphere.

The operation is:
    out[k, o] = sum_i W[k, o, i] * in[k, i]

where k indexes SH coefficients, o indexes output channels, and i
indexes input channels. W is complex-valued.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import equinox as eqx


class SpectralConv(eqx.Module):
    """Learned spectral convolution in spherical harmonic space.

    Stores complex weights of shape (n_sh, out_channels, in_channels)
    split into real and imaginary parts for compatibility with JAX
    pytree operations.

    Parameters
    ----------
    n_sh : int
        Number of spherical harmonic coefficients.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    key : jax.random.PRNGKey
        Random key for initialization.
    """
    weight_real: jnp.ndarray  # (n_sh, out_channels, in_channels)
    weight_imag: jnp.ndarray  # (n_sh, out_channels, in_channels)
    n_sh: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)

    def __init__(
        self,
        n_sh: int,
        in_channels: int,
        out_channels: int,
        *,
        key: jax.Array,
    ):
        self.n_sh = n_sh
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Xavier-like initialization scaled by spectral dimension
        scale = 1.0 / math.sqrt(n_sh * in_channels)
        key_r, key_i = jax.random.split(key)

        self.weight_real = scale * jax.random.normal(
            key_r, (n_sh, out_channels, in_channels)
        )
        self.weight_imag = scale * jax.random.normal(
            key_i, (n_sh, out_channels, in_channels)
        )

    def __call__(self, coeffs: jnp.ndarray) -> jnp.ndarray:
        """Apply spectral convolution.

        Parameters
        ----------
        coeffs : complex array, shape (n_sh, in_channels)
            Input spectral coefficients.

        Returns
        -------
        complex array, shape (n_sh, out_channels)
            Output spectral coefficients.
        """
        W = self.weight_real + 1j * self.weight_imag  # (n_sh, out, in)
        # Einstein summation: for each SH coefficient k,
        # out[k, o] = sum_i W[k, o, i] * coeffs[k, i]
        return jnp.einsum("koi,ki->ko", W, coeffs)
