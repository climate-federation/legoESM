"""Geothermal bottom heat-flux boundary condition (NEMO ``ln_trabbc``).

The solid Earth conducts heat into the ocean across the seafloor.  NEMO applies
this as a bottom boundary condition on the temperature equation (Emile-Geay &
Madec 2009): a prescribed geothermal heat flux ``Q_geo`` [W/m^2] heats the
deepest wet cell of each column,

    dT/dt |_geo = Q_geo / (rho_0 * c_sw * h_bottom)      [K/s],

applied ONLY in the bottom wet level (no-flux through the seabed below, and the
flux is purely a source — no compensating sink anywhere, because the heat enters
the ocean from outside).  The default magnitude is the global-mean continental
value (~86 mW/m^2); NEMO optionally reads the spatially-varying Goutorbe et al.
(2011) field instead, which the caller supplies via ``flux_wm2`` as a per-column
field.

This is small (~0.09 W/m^2 vs ~100s W/m^2 of surface fluxes) and acts on the
abyss, so its signature is multidecadal abyssal warming + a weak strengthening
of the deep overturning, NOT a surface-SST signal on spin-up timescales.  It is
included for structural faithfulness to NEMO ORCA1 (which runs ``ln_trabbc`` with
``geothermal_heat_flux.nc``), not to move a short-run SST score.

References
----------
Emile-Geay, J., & Madec, G. (2009). Geothermal heating, diapycnal mixing and
the abyssal circulation. *Ocean Science*, 5(2), 203-217.

Goutorbe, B., Poort, J., Lucazeau, F., & Raillard, S. (2011). Global heat flow
trends resolved from multiple geological and geophysical proxies.
*Geophys. J. Int.*, 187(3), 1405-1419.
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm import constants

# --- geothermal heat flux (Goutorbe et al. 2011; NEMO rn_geoflx_cst) ---
# Global-mean seafloor geothermal heat flux.  NEMO's namelist default for the
# constant option (nn_geoflx=1) is rn_geoflx_cst = 86.4 mW/m^2.
_GEOTHERMAL_FLUX_MEAN_WM2 = 0.0864      # [W/m^2]
# Below this live-thickness a level is treated as dry (rock) for bottom-cell
# detection and as a safe heating denominator.
_DRY_THICKNESS_M = 1.0e-3               # [m]

__physics_contract__ = {
    "summary": (
        "Geothermal bottom heat-flux boundary condition (NEMO ln_trabbc, "
        "Emile-Geay & Madec 2009). A prescribed seafloor heat flux Q_geo "
        "[W/m^2] warms ONLY the deepest wet cell of each column at rate "
        "Q_geo/(rho_0 c_sw h_bottom). Default ~86 mW/m^2 (constant); an "
        "optional per-column field (Goutorbe et al. 2011) may be supplied."
    ),
    "inputs": {
        "dz_live": "m", "wet_cell": "1", "flux_wm2": "W/m^2",
        "rho_0": "kg/m^3", "c_sw": "J/(kg K)",
    },
    "outputs": {"dT_dt": "degC/s"},
    "sign_convention": (
        "flux_wm2 >= 0 is heat INTO the ocean from the solid Earth; the "
        "returned dT/dt >= 0 and is non-zero ONLY in the deepest wet level of "
        "each column (one-hot bottom cell), zero everywhere else and on land."
    ),
    # Energy conservation in the BOUNDARY-FLUX sense (same as
    # shortwave_penetration): 100% of the prescribed seafloor flux is deposited
    # in the wet column with NO leak below the seabed and no spurious interior
    # source/sink -- the column heat budget gains exactly flux_wm2 [W/m^2],
    # nothing more, nothing less. This is the input==deposited identity the
    # acceptance test checks. (It is an external source crossing the lower
    # boundary, like a surface flux crosses the upper one; it does NOT make the
    # interior column heat constant -- that is not the claim.)
    "conserves": ["energy"],
    "differentiable": True,
    "reference": "Emile-Geay & Madec 2009, Ocean Sci. 5:203; NEMO ln_trabbc",
    "idealized_test": (
        "single column, uniform flux: the deposited column heat "
        "rho_0*c_sw*dz_bottom*dT == flux_wm2*dt (only the bottom cell warms; "
        "dry columns and land receive zero)."
    ),
}

__param_spec__ = {
    "GeothermalConfig": {
        "scheme_key": "ocean.bbc.geothermal",
        "excluded": {
            # A measured geophysical heat flux, not a model calibration knob;
            # the spatial field (Goutorbe 2011) is data supplied by the caller.
            "flux_wm2": "fixed geophysical constant (measured seafloor heat flux)",
            "h_min_m": "numerics: dry-cell / safe-denominator floor",
        },
        "params": {},
    },
}


class GeothermalConfig(NamedTuple):
    """Geothermal bottom heat-flux boundary condition configuration.

    Disabled by default (``enabled=False``) so existing runs are bit-exact.
    ``flux_wm2`` is the constant fall-back magnitude; a caller wanting the
    spatially-varying field passes it directly to the apply helper.
    """
    enabled: bool = False
    flux_wm2: float = _GEOTHERMAL_FLUX_MEAN_WM2   # constant flux [W/m^2]
    h_min_m: float = _DRY_THICKNESS_M             # dry-cell threshold [m]


def _bottom_cell_onehot(wet_cell: jnp.ndarray, dtype) -> jnp.ndarray:
    """One-hot mask (..., nlev) of the SINGLE deepest wet level per column.

    The deepest wet cell is the wet level with NO wet cell anywhere below it.
    Using the count of wet cells strictly below each level (not just the
    adjacent neighbour) makes this robust to an interior dry gap
    (``wet=[1,0,1]`` -> only the level-2 cell, never two deposits): exactly one
    cell per column is selected.  Columns with no wet cell yield an all-zero
    mask (land receives no geothermal heat)."""
    wet = (jnp.asarray(wet_cell) > 0.5).astype(dtype)
    # Number of wet cells STRICTLY below each level (reverse-cumsum minus self).
    wet_below_count = jnp.cumsum(wet[..., ::-1], axis=-1)[..., ::-1] - wet
    return wet * (wet_below_count < 0.5).astype(dtype)


def geothermal_bottom_heating_tendency(
    dz_live: jnp.ndarray,
    wet_cell: jnp.ndarray,
    *,
    flux_wm2,
    rho_0: float = constants.rho_ocean,
    c_sw: float = constants.c_sw,
    h_min_m: float = _DRY_THICKNESS_M,
) -> jnp.ndarray:
    """Temperature tendency [K/s] from the geothermal bottom heat flux.

    Non-zero ONLY in the deepest wet cell of each column.

    Parameters
    ----------
    dz_live : array ``(..., nlev)``
        Live (partial-cell, z*-scaled) layer thickness [m]; dry cells carry 0.
    wet_cell : array ``(..., nlev)``
        Wet-cell mask in {0, 1} (1 = ocean).
    flux_wm2 : float or array ``(...)``
        Seafloor geothermal heat flux [W/m^2].  Scalar (constant) or a
        per-column field broadcasting against the leading dims of ``dz_live``.
    rho_0, c_sw : float
        Reference seawater density [kg/m^3] and specific heat [J/(kg K)].
    h_min_m : float
        Safe-denominator floor [m].

    Returns
    -------
    array ``(..., nlev)``
        dT/dt [K/s], heating the bottom wet cell only.
    """
    # Respect the ambient float precision (x64 when JAX_ENABLE_X64=1, else
    # float32 on Metal/GPU finite-volume runs) instead of forcing float64.
    dz = jnp.asarray(dz_live)
    dtype = jnp.result_type(dz.dtype, jnp.float32)
    dz = dz.astype(dtype)
    is_bottom = _bottom_cell_onehot(wet_cell, dtype)     # (..., nlev)
    h_safe = jnp.where(dz > h_min_m, dz, h_min_m)
    flux = jnp.asarray(flux_wm2).astype(dtype)
    if flux.ndim == dz.ndim - 1:                         # per-column field (...,)
        flux = flux[..., None]
    elif flux.ndim not in (0, dz.ndim):
        raise ValueError(
            f"geothermal flux_wm2 ndim {flux.ndim} incompatible with dz_live "
            f"ndim {dz.ndim}: pass a scalar or a per-column field shape "
            f"{dz.shape[:-1]}")
    return flux * is_bottom / (rho_0 * c_sw * h_safe)


def nemo_tra_bbc_rate(
    qgh_wm2: float,
    e3t_0: jnp.ndarray,
    stretch_kmm: jnp.ndarray,
    wet_cell: jnp.ndarray,
    *,
    rho0: float,
    rcp: float,
) -> jnp.ndarray:
    """NEMO ``tra_bbc`` temperature Krhs increment [K/s], ``nn_geoflx = 1``.

    eosbn2.F90 ``rho0_rcp = rho0*rcp``, ``r1_rho0_rcp = 1/rho0_rcp``;
    trabbc.F90 ``qgh_trd0 = r1_rho0_rcp * rn_geoflx_cst`` (the constant is
    used as W/m2, no 1e-3); then on the bottom wet level ``mbkt`` only::

        Krhs += qgh_trd0 / ( e3t_0(mbkt) * (1 + r3t(Kmm)*tmask(mbkt)) )

    ``stretch_kmm`` is ``1 + r3t(Kmm)`` per column; ``tmask(mbkt) = 1``.
    Unlike :func:`geothermal_bottom_heating_tendency` this keeps NEMO's
    association (reciprocal first, live thickness as one divisor).
    """
    from legoesm.core.source_rounding import nemo_source_round as b

    e3t = jnp.asarray(e3t_0)
    dtype = e3t.dtype
    r1_rho0_rcp = 1.0 / (rho0 * rcp)                 # python fp64 = Fortran
    qgh_trd0 = jnp.asarray(r1_rho0_rcp * qgh_wm2, dtype=dtype)
    bottom = _bottom_cell_onehot(wet_cell, dtype) > 0.5
    denom = b(e3t * jnp.asarray(stretch_kmm, dtype=dtype)[..., None])
    safe = jnp.where(bottom, denom, jnp.asarray(1.0, dtype=dtype))
    return jnp.where(bottom, b(qgh_trd0 / safe), jnp.asarray(0.0, dtype=dtype))
