"""Shared surface-forcing flux→tendency kernels (#518 item 3).

The four surface-forcing schemes (prescribed, external, bulk_formulas,
flux_feedback) each convert a surface flux to a top-layer tendency with the
same wet-cell guarded reciprocals
``1/(rho_0 * dz)`` (wind stress, freshwater virtual salt) and
``1/(rho_0 * c_sw * dz)`` (heat).  This was re-coded inline in every scheme,
with the **reference-density source split** that #518 flags:
prescribed/external/bulk_formulas read the model constants
``eos.rho_0``/``eos.c_sw`` while flux_feedback reads its Veros-faithful
config-pinned ``cfg.rho_0``/``cfg.c_sw``.

`surface_tendency_factors` makes ``rho_0``/``c_sw`` **explicit arguments** —
one wet-cell + reference-density source of truth per call — so the eos-vs-cfg
choice is visible at the call site (a deliberate per-scheme decision, not a
hidden inconsistency) and a future unification is a one-line change.
"""

from __future__ import annotations

import jax.numpy as jnp

# Volume floor on the surface-layer thickness in the flux→tendency reciprocal.
# Keeps the divide finite on land (where ``is_ocean`` already zeroes the
# result); uniform across all four schemes.
_DZ_FLOOR_M = 1.0e-10


def surface_tendency_factors(is_ocean, dz_0, rho_0, c_sw, *, dz_floor=_DZ_FLOOR_M):
    """Wet-cell flux→tendency reciprocals ``(inv_rho_dz, inv_rho_csw_dz)``.

    ``inv_rho_dz = 1/(rho_0 * max(dz_0, dz_floor))`` on ocean cells, 0 on land —
    multiply by wind stress ``tau`` [N/m^2] for ``du/dt`` [m/s^2], or by the
    salt/freshwater flux for the kinematic salinity tendency.
    ``inv_rho_csw_dz = 1/(rho_0 * c_sw * max(dz_0, dz_floor))`` — multiply by the
    net heat flux ``Q`` [W/m^2] for ``dT/dt`` [K/s].

    Parameters
    ----------
    is_ocean : bool array — wet-cell mask (the scheme's
        ``dz_0 > min_wet_cell_thickness_m``); the reciprocals are 0 where False.
    dz_0 : array — surface-layer thickness [m].
    rho_0 : reference density [kg/m^3] — the scheme's chosen source
        (``eos.rho_0`` for prescribed/external/bulk; ``cfg.rho_0`` for the
        Veros-faithful flux_feedback path).
    c_sw : seawater specific heat [J/(kg K)] — same per-scheme source as rho_0.
    dz_floor : float — thickness floor in the reciprocal (default 1e-10 m).
    """
    dz_safe = jnp.maximum(dz_0, dz_floor)
    inv_rho_dz = jnp.where(is_ocean, 1.0 / (rho_0 * dz_safe), 0.0)
    inv_rho_csw_dz = jnp.where(is_ocean, 1.0 / (rho_0 * c_sw * dz_safe), 0.0)
    return inv_rho_dz, inv_rho_csw_dz
