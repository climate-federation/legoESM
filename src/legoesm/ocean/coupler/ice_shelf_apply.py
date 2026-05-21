"""Apply Holland-Jenkins ice-shelf basal melt to the ocean state.

Routes the per-cell basal-melt computation from
:func:`legoesm.ocean.physics.ice_shelf.compute_basal_melt` into the
ocean's surface forcing:

* Freshwater INTO ocean column at cavity cells (positive η + virtual-
  salt dilution of the top layer).
* Heat EXTRACTED from the top-layer T (cavity draws latent + sensible
  heat from the ocean).

Lat-lon C-grid + MPAS counterparts.  The driver supplies a static
``ice_shelf_mask`` (1 = cavity cell, 0 = no shelf) + ``ice_draft_m``
field providing the local ice-base depth that converts to the
hydrostatic pressure inside the basal-melt parameterisation.  The
basal-melt scheme reads the cavity ambient (T, S) from the top
ocean layer at each cavity cell — a simplification valid for the
mixed-layer-style cavity heat budget.  For multi-layer cavities the
caller should supply an explicit cavity-cell layer index instead.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.core.field import Field
from legoesm.ocean.physics.ice_shelf import (
    IceShelfConfig,
    compute_basal_melt,
    ice_base_pressure_dbar,
)


def apply_ice_shelf_basal_step(
    state,
    *,
    ice_shelf_mask: np.ndarray | jnp.ndarray,
    ice_draft_m: np.ndarray | jnp.ndarray,
    z_coord,
    dt: float,
    config: IceShelfConfig,
    rho_0: float | None = None,
    c_p: float | None = None,
) -> tuple[object, dict]:
    """Apply one timestep of ice-shelf basal melt to a lat-lon C-grid state.

    Reads ambient ``(T_top, S_top)`` from the surface layer at cavity
    cells.  Computes the basal-melt rate ``ṁ`` via the configured
    scheme.  Effects per step:

    * Freshwater into ocean: ``F_FW = ρ_ice · ṁ`` [kg/m²/s] adds to
      ``η`` (``dη = F_FW · dt / ρ_0``) and dilutes top-layer S via
      virtual-salt.
    * Heat extracted from ocean: ``Q = ρ_w · c_w · γ_T · (T_a − T_b)``
      cools the top-layer T by ``dT = −Q · dt / (ρ_0 · c_p · dz_0)``.

    Parameters
    ----------
    state : LatLonCGridOceanState
    ice_shelf_mask : array ``(n_lat, n_lon)`` of {0, 1}
        1 = ice-shelf cavity cell.  Land + open-water cells = 0.
    ice_draft_m : array ``(n_lat, n_lon)``
        Depth of the ice-shelf base [m, positive down].  Used for the
        in-situ freezing point ``T_f(S, p)``.
    z_coord : OceanZStarCoordinate / OceanPartialCellCoordinate
    dt : float
        Time step [s].
    config : :class:`IceShelfConfig`
        Must have ``enabled=True``; ``enabled=False`` short-circuits.
    rho_0, c_p : float, optional
        Reference seawater density [kg/m³] + heat capacity
        [J/(kg·K)].  Default ``constants.rho_ocean`` / ``constants.c_sw``.

    Returns
    -------
    new_state : same type as input
    diagnostics : dict
        ``"m_dot_m_s"``, ``"freshwater_kg_m2_s"``,
        ``"heat_extracted_W_m2"`` — per-cell fields for diagnostics.
    """
    if not config.enabled:
        zero = jnp.zeros_like(state.land_mask.data)
        return state, {
            "m_dot_m_s": zero,
            "freshwater_kg_m2_s": zero,
            "heat_extracted_W_m2": zero,
        }
    if rho_0 is None:
        rho_0 = float(constants.rho_ocean)
    if c_p is None:
        c_p = float(constants.c_sw)

    mask = np.asarray(ice_shelf_mask, dtype=np.float64)
    draft = np.asarray(ice_draft_m, dtype=np.float64)
    land_mask = np.asarray(state.land_mask.data, dtype=np.float64)
    active = mask * land_mask    # only wet cavity cells participate

    S_arr = np.asarray(state.S.data, dtype=np.float64)
    T_arr = np.asarray(state.T.data, dtype=np.float64)
    eta_arr = np.asarray(state.eta.data, dtype=np.float64).copy()

    T_top = T_arr[..., 0]
    S_top = S_arr[..., 0]
    p_ice = ice_base_pressure_dbar(jnp.asarray(draft))

    melt = compute_basal_melt(
        jnp.asarray(T_top),
        jnp.asarray(S_top),
        p_ice,
        config=config,
    )

    m_dot = np.asarray(melt.m_dot_m_s, dtype=np.float64)
    fw = np.asarray(melt.freshwater_to_ocean, dtype=np.float64)
    Q = np.asarray(melt.heat_extracted_from_ocean, dtype=np.float64)

    # Mask: only active cavity cells contribute.
    m_dot = m_dot * active
    fw = fw * active
    Q = Q * active

    # Free-surface rise from melt FW.
    eta_new = eta_arr + fw / rho_0 * dt

    # Top-layer salinity dilution: dS = -S_top · F_FW · dt / (ρ_0 · dz_0)
    dz_0 = float(np.asarray(z_coord.dz_ref)[0])
    S_new = S_arr.copy()
    S_new[..., 0] = S_top + (
        -S_top * fw / (rho_0 * max(dz_0, 1.0e-6)) * dt
    )

    # Top-layer temperature cooling: dT = -Q · dt / (ρ_0 · c_p · dz_0).
    T_new = T_arr.copy()
    T_new[..., 0] = T_top - Q * dt / (rho_0 * c_p * max(dz_0, 1.0e-6))

    new_state = state._replace(
        S=Field(jnp.asarray(S_new), name=state.S.name,
                dims=state.S.dims, units=state.S.units),
        T=Field(jnp.asarray(T_new), name=state.T.name,
                dims=state.T.dims, units=state.T.units),
        eta=Field(jnp.asarray(eta_new), name=state.eta.name,
                  dims=state.eta.dims, units=state.eta.units),
    )
    diagnostics = {
        "m_dot_m_s": m_dot,
        "freshwater_kg_m2_s": fw,
        "heat_extracted_W_m2": Q,
    }
    return new_state, diagnostics


def apply_ice_shelf_basal_step_mpas(
    state,
    *,
    ice_shelf_mask: np.ndarray | jnp.ndarray,
    ice_draft_m: np.ndarray | jnp.ndarray,
    z_coord,
    dt: float,
    config: IceShelfConfig,
    rho_0: float | None = None,
    c_p: float | None = None,
) -> tuple[object, dict]:
    """MPAS counterpart of :func:`apply_ice_shelf_basal_step`.

    Expects ``state.S.data`` and ``state.T.data`` shape
    ``(nCells, nlev)``; ``state.eta.data`` and
    ``state.land_mask.data`` shape ``(nCells,)``.  ``ice_shelf_mask``
    + ``ice_draft_m`` are 1-D ``(nCells,)``.  Same three-equation /
    linear basal-melt convention as the lat-lon path.
    """
    if not config.enabled:
        zero = jnp.zeros_like(state.land_mask.data)
        return state, {
            "m_dot_m_s": zero,
            "freshwater_kg_m2_s": zero,
            "heat_extracted_W_m2": zero,
        }
    if rho_0 is None:
        rho_0 = float(constants.rho_ocean)
    if c_p is None:
        c_p = float(constants.c_sw)

    mask = np.asarray(ice_shelf_mask, dtype=np.float64)
    draft = np.asarray(ice_draft_m, dtype=np.float64)
    land_mask = np.asarray(state.land_mask.data, dtype=np.float64)
    S_arr = np.asarray(state.S.data, dtype=np.float64)
    T_arr = np.asarray(state.T.data, dtype=np.float64)
    eta_arr = np.asarray(state.eta.data, dtype=np.float64).copy()
    # Validate every 1-D cell-axis input matches the MPAS contract:
    # ``S/T`` are ``(nCells, nlev)``; ``eta``, ``land_mask``,
    # ``ice_shelf_mask``, ``ice_draft_m`` are ``(nCells,)``.
    if mask.ndim != 1:
        raise ValueError(
            "apply_ice_shelf_basal_step_mpas: ice_shelf_mask must be 1-D, "
            f"got shape {mask.shape}"
        )
    nC = mask.shape[0]
    if (
        draft.shape != mask.shape
        or land_mask.shape != mask.shape
        or eta_arr.shape != mask.shape
    ):
        raise ValueError(
            "apply_ice_shelf_basal_step_mpas: cell-axis shapes mismatch — "
            f"mask {mask.shape}, draft {draft.shape}, "
            f"land_mask {land_mask.shape}, eta {eta_arr.shape}"
        )
    if S_arr.ndim != 2 or T_arr.shape != S_arr.shape or S_arr.shape[0] != nC:
        raise ValueError(
            "apply_ice_shelf_basal_step_mpas: state.S/T must be "
            f"(nCells={nC}, nlev), got S {S_arr.shape}, T {T_arr.shape}"
        )
    active = mask * land_mask

    T_top = T_arr[..., 0]
    S_top = S_arr[..., 0]
    p_ice = ice_base_pressure_dbar(jnp.asarray(draft))

    melt = compute_basal_melt(
        jnp.asarray(T_top),
        jnp.asarray(S_top),
        p_ice,
        config=config,
    )
    m_dot = np.asarray(melt.m_dot_m_s, dtype=np.float64)
    fw = np.asarray(melt.freshwater_to_ocean, dtype=np.float64)
    Q = np.asarray(melt.heat_extracted_from_ocean, dtype=np.float64)

    m_dot = m_dot * active
    fw = fw * active
    Q = Q * active

    eta_new = eta_arr + fw / rho_0 * dt

    dz_0 = float(np.asarray(z_coord.dz_ref)[0])
    S_new = S_arr.copy()
    S_new[..., 0] = S_top + (
        -S_top * fw / (rho_0 * max(dz_0, 1.0e-6)) * dt
    )

    T_new = T_arr.copy()
    T_new[..., 0] = T_top - Q * dt / (rho_0 * c_p * max(dz_0, 1.0e-6))

    new_state = state._replace(
        S=Field(jnp.asarray(S_new), name=state.S.name,
                dims=state.S.dims, units=state.S.units),
        T=Field(jnp.asarray(T_new), name=state.T.name,
                dims=state.T.dims, units=state.T.units),
        eta=Field(jnp.asarray(eta_new), name=state.eta.name,
                  dims=state.eta.dims, units=state.eta.units),
    )
    diagnostics = {
        "m_dot_m_s": m_dot,
        "freshwater_kg_m2_s": fw,
        "heat_extracted_W_m2": Q,
    }
    return new_state, diagnostics
