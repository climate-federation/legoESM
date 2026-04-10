"""Flexible vertical soil grid (Task 8A).

Provides configurable soil layer spacing with geometric (power-of-2)
default and custom override. All operations are JAX-compatible.

References
----------
- CLM5 Technical Note, Section 2.2: Soil Layer Structure
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class SoilGridConfig(NamedTuple):
    """Configuration for the vertical soil grid.

    Default: 8 layers with geometric (power-of-2) spacing, ~6.4 m total.
    """
    n_layers: int = 8
    dz_top: float = 0.025          # thickness of first layer [m]
    growth_factor: float = 2.0     # each layer is growth_factor x previous
    total_depth: float = 0.0       # if >0, auto-compute dz_top to fill depth


class SoilGrid(NamedTuple):
    """Vertical soil grid arrays."""
    dz: jnp.ndarray            # (n_layers,) layer thicknesses [m]
    z_node: jnp.ndarray        # (n_layers,) depth of layer midpoints [m]
    z_interface: jnp.ndarray   # (n_layers+1,) depth of layer interfaces [m]
    dz_interface: jnp.ndarray  # (n_layers-1,) distance between adjacent midpoints [m]
    n_layers: int


def make_soil_grid(config: SoilGridConfig = SoilGridConfig()) -> SoilGrid:
    """Build a SoilGrid from configuration.

    Parameters
    ----------
    config : SoilGridConfig

    Returns
    -------
    SoilGrid
    """
    n = config.n_layers

    if config.total_depth > 0:
        # Auto-compute dz_top so layers fill the target depth
        # Sum of geometric series: S = dz_top * (r^n - 1) / (r - 1) for r != 1
        r = config.growth_factor
        if abs(r - 1.0) < 1e-10:
            dz_top = config.total_depth / n
        else:
            dz_top = config.total_depth * (r - 1.0) / (r ** n - 1.0)
    else:
        dz_top = config.dz_top

    r = config.growth_factor
    # Layer thicknesses: dz_top * r^k for k = 0, ..., n-1
    dz = dz_top * r ** jnp.arange(n, dtype=float)

    # Interface depths (cumulative sum with 0 at top)
    z_interface = jnp.concatenate([jnp.zeros(1), jnp.cumsum(dz)])

    # Node depths (midpoints)
    z_node = 0.5 * (z_interface[:-1] + z_interface[1:])

    # Distance between adjacent midpoints
    dz_interface = z_node[1:] - z_node[:-1]

    return SoilGrid(
        dz=dz,
        z_node=z_node,
        z_interface=z_interface,
        dz_interface=dz_interface,
        n_layers=n,
    )


def make_soil_grid_custom(dz_array: tuple | list) -> SoilGrid:
    """Build a SoilGrid from explicit layer thicknesses.

    Parameters
    ----------
    dz_array : sequence of float
        Layer thicknesses from top to bottom [m].

    Returns
    -------
    SoilGrid
    """
    dz = jnp.array(dz_array, dtype=float)
    n = len(dz_array)
    z_interface = jnp.concatenate([jnp.zeros(1), jnp.cumsum(dz)])
    z_node = 0.5 * (z_interface[:-1] + z_interface[1:])
    dz_interface = z_node[1:] - z_node[:-1]
    return SoilGrid(dz=dz, z_node=z_node, z_interface=z_interface,
                    dz_interface=dz_interface, n_layers=n)
