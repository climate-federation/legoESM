"""Apply the geothermal bottom heat-flux BC to an ocean state per timestep.

Thin grid-agnostic wrapper around
:func:`legoesm.ocean.physics.geothermal.geothermal_bottom_heating_tendency`.
The physics core lives in ``physics/geothermal.py`` (no state dependency); this
helper layers the explicit-Euler temperature update onto the live state and
rebuilds the ``T`` field, mirroring ``ocean.coupler.tidal_mixing_apply`` /
``sss_apply``.  Works for any ocean state whose ``T`` has shape ``(..., nlev)``
(lat-lon / tripole C-grid ``(ny, nx, nlev)`` and MPAS / cube ``(nCells, nlev)``).

The update is an explicit forward step ``T <- T + dt * dT/dt|_geo``.  This is
unconditionally stable for any realistic ``dt`` because the per-step warming of
the bottom cell is ``flux_wm2*dt/(rho_0 c_sw h_bottom)`` ~ 1e-7 K at
dt=600 s — there is no diffusive coupling between levels, so no CFL constraint.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from legoesm.ocean.physics.geothermal import (
    GeothermalConfig,
    geothermal_bottom_heating_tendency,
)


def apply_geothermal_step(
    state,
    *,
    dz_live,
    wet_cell,
    dt: float,
    config: GeothermalConfig | None = None,
    flux_wm2=None,
    rho_0: float | None = None,
    c_sw: float | None = None,
):
    """Apply one explicit timestep of geothermal bottom heating to ``state.T``.

    Parameters
    ----------
    state : ocean state with a ``T`` :class:`~legoesm.core.field.Field`
        ``state.T.data`` has shape ``(..., nlev)``.
    dz_live, wet_cell : array ``(..., nlev)``
        Live layer thickness [m] and wet-cell mask, aligned with ``state.T``.
    dt : float
        Time step [s].
    config : GeothermalConfig, optional
        Provides the constant ``flux_wm2`` fall-back + ``h_min_m``.  When
        ``config.enabled`` is False this is a no-op (state returned unchanged).
    flux_wm2 : float or array ``(...)``, optional
        Overrides ``config.flux_wm2`` with a spatially-varying seafloor flux
        field [W/m^2] (e.g. the Goutorbe 2011 map).
    rho_0, c_sw : float, optional
        Reference density / specific heat; default to ``legoesm.constants``.

    Returns
    -------
    new_state with the bottom-cell-warmed ``T`` (or the input ``state`` when
    geothermal heating is disabled).
    """
    cfg = config if config is not None else GeothermalConfig()
    if not cfg.enabled:
        return state

    from legoesm import constants
    rho0 = constants.rho_ocean if rho_0 is None else rho_0
    cp = constants.c_sw if c_sw is None else c_sw
    flux = cfg.flux_wm2 if flux_wm2 is None else flux_wm2

    # Geothermal heat flux is warming-only; reject a CONCRETE negative scalar
    # flux loudly rather than silently cooling the abyss. Only Python/NumPy
    # scalars are checked -- never call float() on a JAX tracer (that would
    # break a jitted caller); a spatially-varying field is trusted as data.
    if isinstance(flux, (int, float, np.floating, np.integer)) and float(flux) < 0.0:
        raise ValueError(
            f"geothermal flux_wm2 must be >= 0 (warming-only); got {float(flux)}")

    dTdt = geothermal_bottom_heating_tendency(
        dz_live, wet_cell, flux_wm2=flux, rho_0=rho0, c_sw=cp,
        h_min_m=cfg.h_min_m)
    T0 = jnp.asarray(state.T.data)
    # Preserve T's ambient dtype (no silent float32 -> float64 upcast on Metal /
    # finite-volume runs) and ALL its field metadata (long_name, staggering)
    # via Field.replace rather than a partial Field(...) reconstruction.
    T_new = (T0 + dt * dTdt).astype(T0.dtype)
    return state._replace(T=state.T.replace(data=T_new))
