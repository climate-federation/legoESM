"""Apply Dai-Trenberth river runoff to the ocean state per timestep.

Routes the gridded runoff field [kg/m²/s, positive INTO ocean] —
typically built once at setup via
:func:`legoesm.ocean.forcing.dai_trenberth.project_runoff_to_grid` —
into the OMIP-2 virtual-salt + free-surface forcing.

Effects per step:

* Free surface ``η`` rises by ``R / ρ_0 · dt`` (incoming FW raises
  sea level).
* Top-layer salinity dilutes by the virtual-salt convention:

      dS_top/dt = − S_top · R / (ρ_0 · dz_0)

  Equivalent to ``virtual_salt_flux(R, S_top, dz_0, ρ_0)`` but
  applied as a discrete step ``S_top_new = S_top + dt · dS_dt``.
  Land cells are untouched.

Lat-lon C-grid only.  MPAS callers use
:func:`apply_runoff_step_mpas` (1-D state arrays).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.core.field import Field


def apply_runoff_step(
    state,
    *,
    R_kg_m2_s: np.ndarray | jnp.ndarray,
    z_coord,
    dt: float,
    rho_0: float | None = None,
) -> object:
    """Apply one timestep of Dai-Trenberth river runoff to a lat-lon
    C-grid ocean state.

    Parameters
    ----------
    state : LatLonCGridOceanState
        Must expose ``S``, ``eta``, ``land_mask`` Fields.
    R_kg_m2_s : array ``(n_lat, n_lon)``
        Runoff freshwater flux [kg/m²/s, positive INTO ocean].
        Typically the output of
        :func:`legoesm.ocean.forcing.dai_trenberth.project_runoff_to_grid`.
    z_coord : OceanZStarCoordinate / OceanPartialCellCoordinate
        Surface layer thickness read from ``z_coord.dz_ref[0]``.
    dt : float
        Time step [s].
    rho_0 : float, optional
        Reference seawater density [kg/m³].  Default
        ``constants.rho_ocean``.

    Returns
    -------
    new_state
    """
    if rho_0 is None:
        rho_0 = float(constants.rho_ocean)
    R = np.asarray(R_kg_m2_s, dtype=np.float64)

    dz_0 = float(np.asarray(z_coord.dz_ref)[0])
    mask = np.asarray(state.land_mask.data, dtype=np.float64)

    # Free-surface rise: dη/dt = R / ρ_0.
    eta_arr = np.asarray(state.eta.data, dtype=np.float64).copy()
    deta = R / rho_0 * dt * mask
    eta_new = eta_arr + deta

    # Top-layer salinity dilution via virtual-salt convention.
    S_arr = np.asarray(state.S.data, dtype=np.float64)
    S_top = S_arr[..., 0]
    # dS/dt = -S_top * R / (ρ_0 · dz_0)
    dS_dt = -S_top * R / (rho_0 * max(dz_0, 1.0e-6))
    S_new = S_arr.copy()
    S_new[..., 0] = S_top + dt * dS_dt * mask

    return state._replace(
        S=Field(
            jnp.asarray(S_new),
            name=state.S.name,
            dims=state.S.dims,
            units=state.S.units,
        ),
        eta=Field(
            jnp.asarray(eta_new),
            name=state.eta.name,
            dims=state.eta.dims,
            units=state.eta.units,
        ),
    )


def apply_runoff_step_mpas(
    state,
    *,
    R_kg_m2_s: np.ndarray | jnp.ndarray,
    z_coord,
    dt: float,
    rho_0: float | None = None,
) -> object:
    """MPAS counterpart of :func:`apply_runoff_step`.

    Expects ``state.S.data`` shape ``(nCells, nlev)``,
    ``state.eta.data`` and ``state.land_mask.data`` shape
    ``(nCells,)``.  Same virtual-salt + η-rise convention.
    """
    if rho_0 is None:
        rho_0 = float(constants.rho_ocean)
    R = np.asarray(R_kg_m2_s, dtype=np.float64)
    dz_0 = float(np.asarray(z_coord.dz_ref)[0])
    mask = np.asarray(state.land_mask.data, dtype=np.float64)

    eta_arr = np.asarray(state.eta.data, dtype=np.float64).copy()
    eta_new = eta_arr + R / rho_0 * dt * mask

    S_arr = np.asarray(state.S.data, dtype=np.float64)
    S_top = S_arr[..., 0]
    dS_dt = -S_top * R / (rho_0 * max(dz_0, 1.0e-6))
    S_new = S_arr.copy()
    S_new[..., 0] = S_top + dt * dS_dt * mask

    return state._replace(
        S=Field(
            jnp.asarray(S_new),
            name=state.S.name,
            dims=state.S.dims,
            units=state.S.units,
        ),
        eta=Field(
            jnp.asarray(eta_new),
            name=state.eta.name,
            dims=state.eta.dims,
            units=state.eta.units,
        ),
    )
