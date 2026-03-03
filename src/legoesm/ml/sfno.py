"""Spherical Fourier Neural Operator (SFNO).

Full SFNO model: encoder → N processor blocks → decoder, with an
optional big residual (skip) connection for residual prediction.

The architecture follows NVIDIA/AI2's FourCastNet v2 / ACE design:
1. Pointwise encoder lifts input channels to embed_dim
2. N SFNOBlocks apply spectral convolution + pointwise MLP
3. Pointwise decoder projects back to output channels
4. Big skip: output = input + decoder(processor(encoder(input)))

References
----------
- Bonev et al. (2023). Spherical Fourier Neural Operators: Learning
  Stable Dynamics on the Sphere. ICML.
- Watt-Meyer et al. (2023). ACE: A fast, skillful learned global
  atmospheric model for climate prediction. arXiv:2310.02074.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import equinox as eqx

from legoesm.grids.gaussian import GaussianGrid
from legoesm.ml.sfno_block import SFNOBlock


class SFNOConfig(NamedTuple):
    """Configuration for the SFNO model.

    Attributes
    ----------
    in_channels : int
        Number of input channels (state variables packed into last axis).
    out_channels : int
        Number of output channels (predicted state variables).
    embed_dim : int
        Internal embedding dimension for the processor.
    n_blocks : int
        Number of SFNO processor blocks.
    mlp_expansion : int
        Expansion factor for pointwise MLPs within blocks.
    residual_prediction : bool
        If True, predict residuals: output = input[:out_channels] + decoder(...).
    """
    in_channels: int = 4
    out_channels: int = 4
    embed_dim: int = 256
    n_blocks: int = 8
    mlp_expansion: int = 4
    residual_prediction: bool = True


class SFNO(eqx.Module):
    """Spherical Fourier Neural Operator.

    Parameters
    ----------
    config : SFNOConfig
        Model configuration.
    grid : GaussianGrid
        Gaussian grid (used for SH transforms within blocks).
    key : jax.random.PRNGKey
        Random key for initialization.
    """
    encoder: eqx.nn.Linear
    blocks: list[SFNOBlock]
    decoder: eqx.nn.Linear
    config: SFNOConfig = eqx.field(static=True)

    def __init__(
        self,
        config: SFNOConfig,
        grid: GaussianGrid,
        *,
        key: jax.Array,
    ):
        self.config = config
        k_enc, k_dec, k_blocks = jax.random.split(key, 3)

        # Pointwise encoder: in_channels → embed_dim
        self.encoder = eqx.nn.Linear(
            config.in_channels, config.embed_dim, key=k_enc
        )

        # Processor: N SFNO blocks
        block_keys = jax.random.split(k_blocks, config.n_blocks)
        self.blocks = [
            SFNOBlock(
                grid=grid,
                embed_dim=config.embed_dim,
                mlp_expansion=config.mlp_expansion,
                key=block_keys[i],
            )
            for i in range(config.n_blocks)
        ]

        # Pointwise decoder: embed_dim → out_channels
        self.decoder = eqx.nn.Linear(
            config.embed_dim, config.out_channels, key=k_dec
        )

    def __call__(
        self,
        x: jnp.ndarray,
        grid: GaussianGrid,
    ) -> jnp.ndarray:
        """Forward pass through the full SFNO.

        Parameters
        ----------
        x : array, shape (n_lat, n_lon, in_channels)
            Input field on the Gaussian grid.
        grid : GaussianGrid
            Grid for SH transforms.

        Returns
        -------
        array, shape (n_lat, n_lon, out_channels)
            Predicted output field.
        """
        # Save input for big residual skip
        x_in = x

        # Encoder: (n_lat, n_lon, in_channels) → (n_lat, n_lon, embed_dim)
        x = jax.vmap(jax.vmap(self.encoder))(x)

        # Processor: N SFNO blocks
        for block in self.blocks:
            x = block(x, grid)

        # Decoder: (n_lat, n_lon, embed_dim) → (n_lat, n_lon, out_channels)
        x = jax.vmap(jax.vmap(self.decoder))(x)

        # Big residual skip (ACE-style)
        if self.config.residual_prediction:
            n_out = self.config.out_channels
            x = x + x_in[..., :n_out]

        return x
