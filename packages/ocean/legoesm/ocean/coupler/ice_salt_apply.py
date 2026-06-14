"""Apply a PRESCRIBED sea-ice -> ocean salt-mass flux to the ocean state.

legoESM has no interactive sea-ice model, so the brine-rejection /
melt salt exchange that an ice model would supply is read from a NEMO
ORCA1 icemod diagnostic (``sfxice`` [kg/m^2/s], + INTO ocean) and
injected here once per timestep as a top-layer salinity SOURCE -- the
"sea-ice freshwater/salt export" that counters the Arctic river
freshening (a no-ice ocean accumulates runoff with no ice-driven
export and the Arctic SSS collapses).

Convention (NEMO ``src/OCE/TRA/trasbc.F90``)
--------------------------------------------
``trasbc.F90`` builds the salt content trend (line ~137)::

    sbc_tsc(ji,jj,jp_sal) = r1_rho0 * sfx(ji,jj)        ! r1_rho0 = 1/rho0

then adds it to the salinity tendency divided by the top-cell live
thickness (lines ~152-153)::

    pts(...,jp_sal,Krhs) += sbc_tsc(...,jp_sal) / e3t(ji,jj,1,Kmm)

i.e. ``dS/dt|salt = sfx / (rho0 * e3t1)``.  NEMO salinity is g/kg (PSU)
while ``sfx`` is a salt MASS flux [kg/m^2/s], so the PSU form carries a
factor 1000 (kg/kg -> g/kg)::

    dS/dt|salt = 1000 * sfxice / (rho0 * h_top)   [PSU/s]

applied to the TOP cell only.  POSITIVE ``sfxice`` (brine rejection on
freeze) SALTENS the ocean.

This module is a thin host-loop (NumPy) wrapper -- the same per-step,
state-rebuild pattern as ``sss_apply`` / ``runoff_apply`` /
``geothermal_apply``.  The numerics are NOT re-derived: the top-layer
PSU tendency is the shared canonical
:func:`legoesm.ocean.freshwater.salt_flux_salinity_tendency`
(``dS/dt = salt_flux * 1e3 / (rho_0 * dz_0)`` with the thin-cell
guard), the same closure the in-core PE step uses for a coupler-driven
``surface_forcing.salt_flux``.

Grid-agnostic: works for the lat-lon C-grid state
(``state.S`` shape ``(n_lat, n_lon, nlev)``) AND the MPAS state
(``(nCells, nlev)``) because both index the surface layer as
``S[..., 0]`` and carry a ``(...,)`` ``land_mask``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.ocean.freshwater import salt_flux_salinity_tendency


def apply_ice_salt_flux_step(
    state,
    *,
    salt_flux_kg_m2_s: np.ndarray | jnp.ndarray | None,
    h_top: np.ndarray | jnp.ndarray,
    dt: float,
    rho_0: float | None = None,
) -> object:
    """Apply one timestep of a prescribed sea-ice salt-mass flux.

    Updates the surface-layer salinity via the discrete step

        S_top_new = S_top_old + dt * dS/dt|salt
        dS/dt|salt = 1000 * salt_flux / (rho_0 * max(h_top, eps))

    (NEMO ``trasbc.F90``: ``dS/dt = sfx/(rho0*e3t1)`` with the PSU
    factor 1000 because ``sfx`` is a salt MASS flux).  Land cells
    (``land_mask = 0``) and thin/dry top cells (``h_top <= 1mm``,
    handled inside :func:`salt_flux_salinity_tendency`) are left
    untouched.  Higher layers are never modified.

    Grid-agnostic over the lat-lon C-grid (``state.S`` shape
    ``(n_lat, n_lon, nlev)``) and the MPAS state (``(nCells, nlev)``).

    Parameters
    ----------
    state : ocean state
        Must expose ``S`` (``Field``, ``(..., nlev)``) and
        ``land_mask`` (``Field``, ``(...,)``).  ``S`` is updated via
        ``Field.replace(data=...)`` so name/dims/units/dtype are
        preserved.
    salt_flux_kg_m2_s : array ``(...,)`` or None
        Ice -> ocean salt-mass flux [kg(salt)/m^2/s, POSITIVE = salt
        INTO ocean = brine rejection].  ``None`` -> no-op (the original
        state is returned unchanged), so the caller can gate the apply
        on a flag without a branch at the call site.
    h_top : array ``(...,)``
        LIVE top-cell thickness [m] (e.g. ``compute_layer_thickness(
        eta, H_bathy, z_coord)[..., 0]``), matching the thickness the
        tracer update integrates mass against.
    dt : float
        Time step [s].
    rho_0 : float, optional
        Reference seawater density [kg/m^3].  Default
        ``constants.rho_ocean``.

    Returns
    -------
    new_state
        Same type/metadata as ``state`` with the top salinity layer
        updated.  Bit-identical to the input when ``salt_flux`` is
        ``None``.
    """
    if salt_flux_kg_m2_s is None:
        return state
    if rho_0 is None:
        rho_0 = float(constants.rho_ocean)

    # Preserve the state's salinity dtype across the host update (do NOT
    # silently widen f32 -> f64 under x64): read S in its own dtype and
    # carry it back through Field.replace.
    S_arr = np.asarray(state.S.data)
    S_dtype = S_arr.dtype
    S_top = S_arr[..., 0]

    salt_flux = np.asarray(salt_flux_kg_m2_s, dtype=np.float64)
    h_top_np = np.asarray(h_top, dtype=np.float64)
    mask = np.asarray(state.land_mask.data, dtype=np.float64)

    # Shared canonical real-salt-mass closure (NOT re-derived here):
    #   dS/dt = salt_flux * 1e3 / (rho_0 * dz_0)   [PSU/s]
    # with the thin-cell guard (dz_0 < 1mm -> 0).  Use NumPy in/out so
    # this host helper stays a pure CPU update like the sibling
    # applicators; the closure is plain arithmetic + jnp.where, which
    # evaluates eagerly on the NumPy arrays.
    dS_dt = np.asarray(
        salt_flux_salinity_tendency(salt_flux, h_top_np, float(rho_0)),
        dtype=np.float64,
    )

    S_new = S_arr.copy()
    # Mask to ocean cells: brine/melt salt only enters wet columns.
    S_new[..., 0] = (S_top + dt * dS_dt * mask).astype(S_dtype)

    return state._replace(S=state.S.replace(data=jnp.asarray(S_new, dtype=S_dtype)))
