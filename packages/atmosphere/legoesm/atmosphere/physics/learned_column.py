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

from legoesm import constants
from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
    SpectralHydrostaticState,
    spectral_pe_to_grid,
)
from legoesm.grids.gaussian import GaussianGrid, sh_analysis_3d
from legoesm.atmosphere.physics.neural_physics import NeuralPhysics, pack_column_features
from legoesm.atmosphere.physics.radiation.solar import cos_zenith_angle
from legoesm.atmosphere.physics._shared import zero_like_tracers

# Machine-checked scheme contract (see tests/test_physics_contracts.py). Learned
# per-column mapping -> no hard conservation guarantee.
__physics_contract__ = {
    "summary": (
        "Learned per-column MLP physics (Rasp et al. 2018) for the spectral "
        "primitive-equation dycore: predicts a temperature tendency from the "
        "local column state, with no horizontal coupling."
    ),
    "inputs": {
        "T": "K", "u": "m/s", "v": "m/s", "q_v": "kg/kg",
        "p_s": "Pa", "solar": "W/m^2",
    },
    "outputs": {
        "dT_dt": (
            "K/s (spectral T_hat tendency; vor/div/lnps and tracer "
            "tendencies identically 0)"
        ),
    },
    "sign_convention": (
        "Learned mapping: no enforced sign or conservation. Only a "
        "temperature tendency is produced; momentum (vor/div), surface-"
        "pressure (lnps) and tracer tendencies are set to zero, so those "
        "reservoirs are untouched rather than conservatively redistributed."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": "Rasp, Pritchard & Gentine (2018), PNAS 115(39), 9684-9689",
    "idealized_test": (
        "tests/unit/test_learned_column.py; untrained network "
        "(residual_scale=0.01) -> near-zero dT/dt; vor/div/lnps tendencies "
        "exactly zero; per-column vmap without horizontal coupling"
    ),
}

# Default neural-column architecture width + residual output scale (structural).
_DEFAULT_HIDDEN_DIM = 256
_DEFAULT_RESIDUAL_SCALE = 0.01
# The NeuralPhysics rate head shares one residual_scale calibrated for
# TEMPERATURE tendencies (K/s).  Moisture tendencies live ~3 orders lower
# (SFNO per-channel scales: dT 1e-4 K/s vs dq 1e-7 kg/kg/s), so the q_v head
# is multiplied by this factor before entering the tracer tendency.  Without
# it an UNTRAINED head emits O(1e-3 kg/kg/s) and q_v explodes within a 6 h
# rollout (probe 2026-07-03: q_v -> 4e2 by step 6, NaN by step 36 — the
# column_nn retrain epoch-0 NaN).  1e-4 caps the head at
# residual_scale*tendency_cap*1e-4 = 5e-6 kg/kg/s: ~5-50x typical physical
# moisture tendencies, the same relative headroom the T head has.
_Q_HEAD_TENDENCY_FACTOR = 1.0e-4



def build_column_physics(
    nlev: int,
    hidden_dim: int = _DEFAULT_HIDDEN_DIM,
    n_layers: int = 4,
    residual_scale: float = _DEFAULT_RESIDUAL_SCALE,
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

    def physics_fn(state, grid_, sigma_coord, forcing=None):
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
        # Pull q_v from state.tracers when present (PR1's spectral PE
        # tracers); fall back to zeros for the legacy dry pipeline.
        # Without this the column MLP sees dry inputs even when ERA5
        # humidity is loaded into the IC.
        if state.tracers is not None and "q_v" in state.tracers:
            _qv_raw = state.tracers["q_v"]
            _qv_data = _qv_raw.data if hasattr(_qv_raw, "data") else _qv_raw
            q_col = _qv_data.reshape(-1, nlev).astype(T_col.dtype)
        else:
            q_col = jnp.zeros_like(T_col)
        p_s_col = p_s.reshape(-1)

        # --- surface forcing features (the prescribed-SST / AMIP pathway) ---
        # forcing dict (all traced; see ``spectral_amip_rollout``):
        #   "T_sfc"          (ncol,) prescribed surface T over OCEAN, NaN over
        #                    land → replaced by the lowest-level air T proxy
        #   "sic"            (ncol,) sea-ice fraction in [0, 1]
        #   "day_of_year"    scalar (seasonal insolation)
        #   "seconds_of_day" scalar (diurnal phase)
        # Without forcing (idealized / legacy tests): T_sfc proxy = lowest
        # model level, sic = 0, solar = S_0 (the historical constant input).
        t_lowest = T_col[:, -1]
        if forcing is not None:
            # Sanitize the inactive branch BEFORE the select: jnp.where
            # propagates NaN cotangents from the untaken branch in reverse
            # mode (codex HIGH) — land-NaN T_sfc would poison the training
            # gradient through the proxy path.
            t_raw = jnp.nan_to_num(forcing["T_sfc"], nan=0.0)
            t_sfc_col = jnp.where(
                jnp.isfinite(forcing["T_sfc"]), t_raw, t_lowest,
            )
            sic_col = jnp.clip(
                jnp.nan_to_num(forcing["sic"], nan=0.0), 0.0, 1.0,
            )
            mu0 = cos_zenith_angle(
                jnp.broadcast_to(grid_.lat[:, None], (n_lat, n_lon)).reshape(-1),
                jnp.broadcast_to(grid_.lon[None, :], (n_lat, n_lon)).reshape(-1),
                forcing["day_of_year"], forcing["seconds_of_day"] / 3600.0,
            )
            solar_col = constants.S_0 * jnp.maximum(mu0, 0.0)
        else:
            t_sfc_col = t_lowest
            sic_col = jnp.zeros_like(p_s_col)
            solar_col = jnp.full_like(p_s_col, constants.S_0)

        # Pack features + vmap forward (normalization built into _pack)
        features = jax.vmap(pack_column_features)(
            T_col, u_col, v_col, q_col, p_s_col, solar_col,
            t_sfc_col, sic_col,
        )
        y = jax.vmap(neural_physics)(features)  # (ncol, n_output)

        # Extract dT/dt (first nlev outputs)
        dT_dt = y[:, :nlev].reshape(n_lat, n_lon, nlev)

        # Convert to spectral temperature tendency
        dT_hat = sh_analysis_3d(grid_, dT_dt.astype(jnp.float64))

        # Column physics: T tendency (spectral) + q_v tendency (grid-space
        # tracer path, matching ``unpack_pe_output``'s tendency convention);
        # zero for vor, div, lnps.  Mirror the input state's remaining tracer
        # pytree as zeros so the orchestrator and dycore RHS see a consistent
        # tendency structure (matches the radiation / GWD bridges).
        zero_3d = jnp.zeros_like(state.vor_hat.data)
        zero_2d = jnp.zeros_like(state.lnps_hat.data)

        tracers_out = zero_like_tracers(state.tracers)
        if tracers_out is not None and "q_v" in tracers_out:
            # dq_v/dt from the moisture head (outputs nlev:2*nlev).  Before
            # this the head was silently discarded — the column NN had NO
            # moisture physics (no condensation sink / evaporation source).
            # Rescaled to moisture magnitudes (_Q_HEAD_TENDENCY_FACTOR).
            dq_v_dt = (
                y[:, nlev:2 * nlev] * _Q_HEAD_TENDENCY_FACTOR
            ).reshape(n_lat, n_lon, nlev)
            template = state.tracers["q_v"]
            if hasattr(template, "data") and hasattr(template, "replace"):
                tracers_out["q_v"] = template.replace(
                    data=dq_v_dt.astype(template.data.dtype),
                )
            else:
                tracers_out["q_v"] = dq_v_dt.astype(template.dtype)

        return SpectralHydrostaticState(
            vor_hat=state.vor_hat.replace(data=zero_3d),
            div_hat=state.div_hat.replace(data=zero_3d),
            T_hat=state.T_hat.replace(data=dT_hat),
            lnps_hat=state.lnps_hat.replace(data=zero_2d),
            phis_hat=state.phis_hat.replace(data=zero_2d),
            tracers=tracers_out,
        )

    return physics_fn
