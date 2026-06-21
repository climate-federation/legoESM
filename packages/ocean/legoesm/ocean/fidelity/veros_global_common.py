"""Shared builders for the VEROS-faithful global recipes (4°, 1°, flexible).

The three ``veros_global_{4deg,1deg,flexible}_recipe`` modules carried
near-identical glue for three concerns that this module now owns once:

1. **Initial-state seeding** (:func:`build_veros_global_state`) — masked
   file T/S, rest velocity, rigid-lid ``eta ≡ 0`` and the TKE/EKE carry
   fields.  The three recipes differed ONLY in which ``TKEConfig``/
   ``EKEConfig`` constant they read ``tke_background``/``e_min`` from and in
   whether they spelled the vertical size as a module constant (``NZ``) or
   ``z_coord.n_levels`` — both equal for every built ``z_coord`` here, so
   the merge is byte-identical (verified against the recipe tests).

2. **GM/Redi + EKE isopycnal-on delta** (:func:`gm_redi_eke_isopycnal_on`) —
   the 1° and flexible recipes applied a byte-identical pair of ``_replace``
   deltas to the 4° base configs (flip ``isopycnal_diffusion=True`` and the
   ``S_max``/``taper_width_frac``/``K_iso_steep`` GM/Redi values).

3. **T-cell area column weights** (:func:`veros_area_t_generic`) — the
   ``dxt·dyt·cost`` [m²] per-latitude weight; the recipes differed only in
   how ``dxt``/``dyt`` were supplied (fixed degrees vs a stretched ``dyt``
   row), so the arithmetic is shared and each recipe passes its own deltas.

These are pure harness/recipe glue (no model numerics, no autodiff): they
re-assemble shared canonical blocks into a recipe state, per
``docs/ocean_fidelity/oracle_recipe_strategy.md``.  The per-grid
``veros_mit_tau_shift`` (cyclic-roll vs zero-ghost x-shift — genuinely
different) and ghost-axis builders (uniform vs Vinokur-stretched y) stay
in their recipes: they are NOT byte-identical and a merge would change a
recipe's wind forcing / grid metric.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from legoesm.core.field import Field
from legoesm.ocean.constants_config import VEROS_CONSTANTS_CONFIG
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanState

__all__ = [
    "build_veros_global_state",
    "gm_redi_eke_isopycnal_on",
    "veros_area_t_generic",
]


def veros_area_t_generic(
    yt_deg: np.ndarray,
    dxt_deg: np.ndarray,
    dyt_deg: np.ndarray,
    r_earth: float = VEROS_CONSTANTS_CONFIG.R_earth,
) -> np.ndarray:
    """Veros T-cell area column weights ``dxt·dyt·cost`` [m²] (per-latitude
    row; broadcast over x).

    ``degtom = R_earth·π/180`` converts the zonal/meridional cell widths
    from degrees to metres.  ``dxt_deg``/``dyt_deg`` may be scalars (uniform
    grids) or per-row arrays (stretched grids); ``yt_deg`` is the T-cell
    centre latitude in degrees.
    """
    degtom = r_earth * np.pi / 180.0
    return (
        (np.asarray(dxt_deg) * degtom)
        * (np.asarray(dyt_deg) * degtom)
        * np.cos(np.deg2rad(np.asarray(yt_deg)))
    )


def gm_redi_eke_isopycnal_on(base_gm_redi, base_eke):
    """Apply the global_1deg/global_flexible GM/Redi + EKE deltas to the
    global_4deg base configs.

    Both setups (``global_1deg.py``, ``global_flexible.py``) re-use the 4°
    parameter blocks with EXACTLY this pair of deltas, so the two recipes
    shared a byte-identical ``_replace`` chain:

    * EKE: ``isopycnal_diffusion=True`` (the flip back vs global_4deg's
      ``False`` settings default; like ACC) ⇒ ``K_iso = K_gm`` (the Redi
      tracer diffusivity follows the prognostic GM coefficient).
    * GM/Redi: ``K_iso_0=1000``, ``K_iso_steep=50``,
      ``iso_dslope=iso_slopec=0.005`` ⇒ ``S_max=5e-3``,
      ``taper_width_frac=1.0`` (mapping ``S_max=iso_slopec``,
      ``frac=iso_dslope/iso_slopec``).

    Returns ``(gm_redi_config, eke_config)`` with the EKE nested inside the
    GM/Redi config (so callers can keep both public names).
    """
    eke = base_eke._replace(
        isopycnal_diffusion=True,     # *** the flip back vs global_4deg ***
    )
    gm_redi = base_gm_redi._replace(
        S_max=5.0e-3,                 # Veros iso_slopec   (4deg: 1e-3)
        taper_width_frac=1.0,         # iso_dslope/iso_slopec (4deg: 4.0)
        K_iso_steep=50.0,             # Veros K_iso_steep  (4deg: 1000)
        eke=eke,
    )
    return gm_redi, eke


def build_veros_global_state(
    grid,
    z_coord,
    land_mask: np.ndarray,
    H_bathy: np.ndarray,
    tke_config,
    eke_config,
    T_init: np.ndarray | None = None,
    S_init: np.ndarray | None = None,
) -> LatLonCGridOceanState:
    """Initial state for a VEROS-faithful global recipe.

    File T/S (already bridged to legoESM order/shape) masked by the active
    cells, rest velocity, rigid-lid ``eta ≡ 0`` and the TKE/EKE carry
    fields.

    Seeding note (Veros parity): Veros zero-initialises ``vs.tke``/``vs.eke``
    (no setup seed).  legoESM seeds tke at ``tke_config.tke_background`` and
    eke at ``eke_config.e_min`` — the model floors both there anyway on the
    first step, so this is the same effective start.  ``eke_diss`` and the
    TKE-advection AB2 history ``dtke`` seed at zero (Veros's zero-initialised
    ``eke_diss_iw`` / ``dtke[taum1]``), keeping the ``lax.scan`` carry pytree
    constant.

    Parameters
    ----------
    grid, z_coord :
        The recipe's horizontal grid and vertical coordinate.
    land_mask, H_bathy :
        Bridged Veros bathymetry (``(n_lat, n_lon)`` incl. wall rows).
    tke_config, eke_config :
        The recipe's TKE/EKE configs — only ``tke_background`` / ``e_min``
        are read here.
    T_init, S_init :
        Optional bridged file T/S; masked by the active cells when given.
    """
    nz = z_coord.n_levels
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord,
        S_uniform=35.0, H_max=float(z_coord.H_max),
        land_mask_override=jnp.asarray(land_mask),
        H_bathy_override=jnp.asarray(H_bathy),
    )
    is_active = jnp.asarray(z_coord.is_active, dtype=state.T.data.dtype)
    if T_init is not None:
        state = state._replace(
            T=state.T.replace(data=jnp.asarray(T_init) * is_active))
    if S_init is not None:
        state = state._replace(
            S=state.S.replace(data=jnp.asarray(S_init) * is_active))

    lm = state.land_mask.data
    dtype = state.T.data.dtype
    wet3 = (lm[:, :, jnp.newaxis] > 0.5) * jnp.ones((1, 1, nz - 1), dtype=dtype)

    # Prognostic TKE + its superbee-advection AB2 history.
    tke0 = tke_config.tke_background * wet3
    state = state._replace(
        tke=Field(data=tke0, name="tke", dims=("lat", "lon", "level"),
                  units="m^2/s^2"),
        dtke=Field(data=jnp.zeros_like(tke0), name="dtke",
                   dims=("lat", "lon", "level"), units="m^2/s^3"),
    )
    # Prognostic 3-D EKE + the eke_diss carry (TKE source_eke_diss reads it).
    eke0 = eke_config.e_min * wet3
    state = state._replace(
        eke=Field(data=eke0, name="eke", dims=("lat", "lon", "level"),
                  units="m^2/s^2"),
        eke_diss=Field(data=jnp.zeros_like(eke0), name="eke_diss",
                       dims=("lat", "lon", "level"), units="m^2/s^3"),
    )
    return state
