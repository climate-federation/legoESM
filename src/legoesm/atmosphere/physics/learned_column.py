"""Learned column physics parameterization (Rasp et al. 2018).

Provides a per-column neural network that replaces traditional subgrid
physics (radiation, convection, turbulence, boundary layer) with a
learned MLP operating independently on each atmospheric column.

The column MLP takes local state (T, u, v, q, p_s) and predicts
temperature tendencies.  No horizontal coupling — the spectral PE
dynamical core handles all resolved transport.

Two coupling interfaces:

- ``make_column_physics_fn``: returns a ``physics_fn`` compatible with
  ``SpectralPrimitiveEquationModel.step(physics_fn=...)`` and
  ``spectral_pe_tendencies(..., physics_tendency=...)``.
- ``build_column_physics``: factory that creates a ``NeuralPhysics``
  model from grid/level configuration.

References
----------
- Rasp, S., Pritchard, M. S., & Gentine, P. (2018). Deep learning to
  represent subgrid processes in climate models. PNAS, 115(39),
  9684-9689.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.atmosphere.dynamics.spectral_pe import (
    SpectralHydrostaticState,
    spectral_pe_to_grid,
)
from legoesm.grids.gaussian import GaussianGrid, sh_analysis_3d
from legoesm.training.neural_physics import NeuralPhysics, _pack_column_features


def build_column_physics(
    nlev: int,
    hidden_dim: int = 256,
    n_layers: int = 4,
    residual_scale: float = 0.01,
    *,
    key: jax.Array,
) -> NeuralPhysics:
    """Create a column MLP physics model.

    Parameters
    ----------
    nlev : int
        Number of vertical levels.
    hidden_dim : int
        Width of hidden layers.
    n_layers : int
        Number of hidden layers.
    residual_scale : float
        Output scaling for stable initialization (untrained network
        produces near-zero tendencies).
    key : jax.Array
        PRNG key for weight initialization.

    Returns
    -------
    NeuralPhysics
        Equinox module with ``(n_input,) -> (n_output,)`` per column.
    """
    return NeuralPhysics(
        nlev=nlev,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        key=key,
        residual_scale=residual_scale,
    )


def make_column_physics_fn(
    neural_physics: NeuralPhysics,
    grid: GaussianGrid,
):
    """Create a spectral PE physics_fn from a column MLP.

    Returns a function with the interface expected by
    ``SpectralPrimitiveEquationModel.step(physics_fn=...)``::

        physics_fn(state, grid, sigma_coord)
            -> SpectralHydrostaticState  (tendencies)

    The column MLP operates per-column via ``jax.vmap``:

    1. Spectral → grid (SH synthesis)
    2. Flatten to columns (n_lat*n_lon, nlev)
    3. Normalize inputs (T/300, u/30, v/30, q*1e3, p_s/1e5)
    4. Forward MLP × residual_scale
    5. Extract dT/dt tendency
    6. Grid → spectral (SH analysis)

    Column physics only affects temperature — momentum (vor, div)
    and surface pressure (lnps) tendencies are zero.

    Parameters
    ----------
    neural_physics : NeuralPhysics
        Column MLP model (eqx.Module).
    grid : GaussianGrid
        Gaussian grid for spectral transforms.

    Returns
    -------
    callable
        Physics function for the spectral PE dycore.
    """
    nlev = neural_physics.nlev

    def physics_fn(state, grid_, sigma_coord):
        # Spectral → grid-space fields
        fields = spectral_pe_to_grid(state, grid_, sigma_coord)
        T = fields['T']          # (n_lat, n_lon, nlev)
        u = fields['u']
        v = fields['v']
        p_s = fields['p_s']      # (n_lat, n_lon)

        n_lat, n_lon = T.shape[:2]

        # Flatten to columns
        T_col = T.reshape(-1, nlev)
        u_col = u.reshape(-1, nlev)
        v_col = v.reshape(-1, nlev)
        q_col = jnp.zeros_like(T_col)       # dry spectral PE
        p_s_col = p_s.reshape(-1)
        solar_col = jnp.full_like(p_s_col, 1361.0)

        # Pack features + vmap forward (normalization built into _pack)
        features = jax.vmap(_pack_column_features)(
            T_col, u_col, v_col, q_col, p_s_col, solar_col,
        )
        y = jax.vmap(neural_physics)(features)  # (ncol, n_output)

        # Extract dT/dt (first nlev outputs)
        dT_dt = y[:, :nlev].reshape(n_lat, n_lon, nlev)

        # Convert to spectral temperature tendency
        dT_hat = sh_analysis_3d(grid_, dT_dt.astype(jnp.float64))

        # Column physics: only T tendency; zero for vor, div, lnps
        zero_3d = jnp.zeros_like(state.vor_hat.data)
        zero_2d = jnp.zeros_like(state.lnps_hat.data)

        return SpectralHydrostaticState(
            vor_hat=state.vor_hat.replace(data=zero_3d),
            div_hat=state.div_hat.replace(data=zero_3d),
            T_hat=state.T_hat.replace(data=dT_hat),
            lnps_hat=state.lnps_hat.replace(data=zero_2d),
            phis_hat=state.phis_hat.replace(data=zero_2d),
        )

    return physics_fn
