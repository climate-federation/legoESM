"""Single SFNO processor block.

Each block applies:
    1. Layer normalization
    2. SH analysis (grid → spectral)
    3. Learned spectral convolution
    4. SH synthesis (spectral → grid)
    5. GELU activation
    6. Pointwise MLP
    7. Skip connection

The SH transforms reuse legoESM's existing ``sh_analysis_3d`` and
``sh_synthesis_3d`` from ``grids/gaussian.py``, treating the channel/
embed dimension identically to the level axis.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

from legoesm.grids.gaussian import (
    GaussianGrid,
    sh_analysis_3d,
    sh_synthesis_3d,
)
from legoesm.ml.spectral_conv import SpectralConv


class SFNOBlock(eqx.Module):
    """Single SFNO processor block.

    Parameters
    ----------
    grid : GaussianGrid
        Gaussian grid (used for SH transforms; not stored as a
        learnable parameter).
    embed_dim : int
        Embedding dimension (channel count).
    mlp_expansion : int
        Expansion factor for the pointwise MLP hidden dimension.
    key : jax.random.PRNGKey
        Random key for initialization.
    """
    spectral_conv: SpectralConv
    ln: eqx.nn.LayerNorm
    mlp_linear1: eqx.nn.Linear
    mlp_linear2: eqx.nn.Linear
    embed_dim: int = eqx.field(static=True)

    def __init__(
        self,
        grid: GaussianGrid,
        embed_dim: int,
        mlp_expansion: int = 4,
        *,
        key: jax.Array,
    ):
        self.embed_dim = embed_dim
        k1, k2, k3 = jax.random.split(key, 3)

        self.spectral_conv = SpectralConv(
            n_sh=grid.n_sh,
            in_channels=embed_dim,
            out_channels=embed_dim,
            key=k1,
        )

        self.ln = eqx.nn.LayerNorm(embed_dim)

        mlp_hidden = embed_dim * mlp_expansion
        self.mlp_linear1 = eqx.nn.Linear(embed_dim, mlp_hidden, key=k2)
        self.mlp_linear2 = eqx.nn.Linear(mlp_hidden, embed_dim, key=k3)

    def __call__(
        self,
        x: jnp.ndarray,
        grid: GaussianGrid,
    ) -> jnp.ndarray:
        """Forward pass through one SFNO block.

        Parameters
        ----------
        x : array, shape (n_lat, n_lon, embed_dim)
            Input field on the Gaussian grid.
        grid : GaussianGrid
            Grid for SH transforms.

        Returns
        -------
        array, shape (n_lat, n_lon, embed_dim)
            Output field with skip connection.
        """
        residual = x

        # 1. Layer normalization (over embed_dim axis)
        x = jax.vmap(jax.vmap(self.ln))(x)  # (n_lat, n_lon, embed_dim)

        # 2. SH analysis: grid → spectral
        # sh_analysis_3d expects (n_lat, n_lon, nlev) and returns (n_sh, nlev)
        # Cast to float64 for SHT accuracy (matching existing spectral model pattern)
        x_f64 = x.astype(jnp.float64)
        coeffs = sh_analysis_3d(grid, x_f64)  # (n_sh, embed_dim) complex128

        # 3. Learned spectral convolution
        coeffs = self.spectral_conv(coeffs)    # (n_sh, embed_dim) complex128

        # 4. SH synthesis: spectral → grid
        x = sh_synthesis_3d(grid, coeffs)      # (n_lat, n_lon, embed_dim) float64
        x = x.astype(jnp.float32)

        # 5. GELU activation
        x = jax.nn.gelu(x)

        # 6. Pointwise MLP
        x = jax.vmap(jax.vmap(self.mlp_linear1))(x)  # (n_lat, n_lon, mlp_hidden)
        x = jax.nn.gelu(x)
        x = jax.vmap(jax.vmap(self.mlp_linear2))(x)  # (n_lat, n_lon, embed_dim)

        # 7. Skip connection
        return x + residual
