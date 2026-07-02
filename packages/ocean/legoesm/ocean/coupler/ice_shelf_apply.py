"""Apply Holland-Jenkins ice-shelf basal melt to the ocean state.

Routes the per-cell basal-melt computation from
:func:`legoesm.ocean.physics.ice_shelf.compute_basal_melt` into the
ocean column at cavity cells.

Effects per step (at the cavity cell containing the ice base):

* Cavity ambient ``(T, S)`` is read from the layer that contains the
  ice-base depth (``ice_draft_m``).  Antarctic shelves with 200 –
  1500 m drafts therefore see realistic Circumpolar Deep Water
  ambient, not the surface mixed-layer temperature.
* Heat extracted from that draft layer cools its in-situ T.
* Virtual-salt dilution of that draft layer freshens its in-situ S.
* Free surface ``η`` rises by ``F_FW · dt / ρ_0`` (barotropic mass
  response — net freshwater addition raises sea level).

Lat-lon C-grid + MPAS counterparts.  Driver supplies a static
``ice_shelf_mask`` (1 = cavity cell, 0 = no shelf) + ``ice_draft_m``
field giving the local ice-base depth that converts to the
hydrostatic pressure inside the basal-melt parameterisation AND
picks the cavity-ambient layer index.
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


def _ice_base_layer_index(
    draft_m: np.ndarray, dz_ref: np.ndarray,
) -> np.ndarray:
    """Per-cell layer index ``k`` containing the ice base.

    Uses reference layer thicknesses ``dz_ref`` to build interface
    depths ``z_iface = [0, dz_0, dz_0+dz_1, ...]``.  The cavity layer
    index is the largest ``k`` such that ``z_iface[k] <= draft_m``,
    clipped to ``[0, nlev-1]`` so deep drafts collapse to the
    deepest layer and shallow / zero drafts collapse to the
    surface layer.

    Parameters
    ----------
    draft_m : ndarray ``(...)``
        Ice-base depth [m, positive down].
    dz_ref : ndarray ``(nlev,)``
        Reference layer thickness, top → bottom.

    Returns
    -------
    k : ndarray ``(...)`` int64
    """
    if dz_ref.ndim != 1:
        raise ValueError(
            f"_ice_base_layer_index: dz_ref must be 1-D, got "
            f"shape {dz_ref.shape}"
        )
    z_iface = np.concatenate([[0.0], np.cumsum(dz_ref)])
    # ``searchsorted(side='right')`` gives the count of interfaces
    # strictly <= draft; subtract 1 for the cell-centred layer index.
    k = np.searchsorted(z_iface, draft_m, side="right") - 1
    return np.clip(k, 0, dz_ref.size - 1).astype(np.int64)


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
            "ice_base_layer_idx": jnp.zeros_like(
                state.land_mask.data, dtype=jnp.int32,
            ),
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

    nlev = S_arr.shape[-1]
    dz_ref = np.asarray(z_coord.dz_ref, dtype=np.float64)[:nlev]
    # Per-cell layer index containing the ice base.
    k_draft = _ice_base_layer_index(draft, dz_ref)      # (...,)
    k_exp = k_draft[..., None]                          # (..., 1)
    # Cavity ambient: read T, S at the draft layer rather than the
    # surface (the latter would yield mixed-layer values, not the
    # deep-water mass actually in contact with the ice shelf base).
    T_amb = np.take_along_axis(T_arr, k_exp, axis=-1)[..., 0]
    S_amb = np.take_along_axis(S_arr, k_exp, axis=-1)[..., 0]
    p_ice = ice_base_pressure_dbar(jnp.asarray(draft))

    melt = compute_basal_melt(
        jnp.asarray(T_amb),
        jnp.asarray(S_amb),
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

    # Free-surface rise from melt FW (barotropic mass response).
    eta_new = eta_arr + fw / rho_0 * dt

    # Distribute the FW dilution + heat cooling INTO the draft layer
    # only.  Vectorised per-layer scatter: ``lev_one_hot[..., k]`` is
    # 1 where the layer matches the per-cell draft index, 0 else.
    lev_idx = np.arange(nlev, dtype=np.int64)            # (nlev,)
    one_hot = (lev_idx == k_exp).astype(np.float64)      # (..., nlev)
    dz_per_layer = dz_ref                                # (nlev,)
    # Per-layer thickness at the draft layer of each cell (zeros
    # elsewhere thanks to the one-hot mask).
    dz_draft = np.maximum(
        np.take_along_axis(
            np.broadcast_to(dz_per_layer, S_arr.shape),
            k_exp, axis=-1,
        )[..., 0],
        1.0e-6,
    )
    # Salinity dilution at draft layer: dS_k = -S_amb · F_FW · dt /
    # (ρ_0 · dz_k).  Multiplied by the one-hot mask so other layers
    # are unaffected.
    dS_at_k = -S_amb * fw / (rho_0 * dz_draft) * dt
    dT_at_k = -Q * dt / (rho_0 * c_p * dz_draft)
    S_new = S_arr + one_hot * dS_at_k[..., None]
    T_new = T_arr + one_hot * dT_at_k[..., None]

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
        "ice_base_layer_idx": k_draft.astype(np.int32),
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
            "ice_base_layer_idx": jnp.zeros_like(
                state.land_mask.data, dtype=jnp.int32,
            ),
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

    nlev = S_arr.shape[-1]
    dz_ref = np.asarray(z_coord.dz_ref, dtype=np.float64)[:nlev]
    k_draft = _ice_base_layer_index(draft, dz_ref)        # (nCells,)
    k_exp = k_draft[..., None]                            # (nCells, 1)
    T_amb = np.take_along_axis(T_arr, k_exp, axis=-1)[..., 0]
    S_amb = np.take_along_axis(S_arr, k_exp, axis=-1)[..., 0]
    p_ice = ice_base_pressure_dbar(jnp.asarray(draft))

    melt = compute_basal_melt(
        jnp.asarray(T_amb),
        jnp.asarray(S_amb),
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

    lev_idx = np.arange(nlev, dtype=np.int64)
    one_hot = (lev_idx == k_exp).astype(np.float64)        # (nCells, nlev)
    dz_draft = np.maximum(
        np.take_along_axis(
            np.broadcast_to(dz_ref, S_arr.shape),
            k_exp, axis=-1,
        )[..., 0],
        1.0e-6,
    )
    dS_at_k = -S_amb * fw / (rho_0 * dz_draft) * dt
    dT_at_k = -Q * dt / (rho_0 * c_p * dz_draft)
    S_new = S_arr + one_hot * dS_at_k[..., None]
    T_new = T_arr + one_hot * dT_at_k[..., None]

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
        "ice_base_layer_idx": k_draft.astype(np.int32),
    }
    return new_state, diagnostics


def apply_isf_prescribed_melt_step(
    state,
    *,
    fwf_kg_m2_s,
    zmin_m,
    zmax_m,
    dz_live,
    wet_cell,
    dt: float,
    rho_0: float | None = None,
    c_sw: float | None = None,
    L_fus: float | None = None,
    config: IceShelfConfig | None = None,
):
    """One explicit step of the NEMO ISF 'spe' prescribed melt (isfparmlt).

    Thin grid-agnostic wrapper around
    :func:`legoesm.ocean.physics.ice_shelf.isf_prescribed_melt_tendencies`
    (the ORCA1 ``cn_isfpar_mlt='spe'`` parametrised-cavity deposit over
    the ``[zmin, zmax]`` band), mirroring ``apply_geothermal_step``:
    ``T/S <- T/S + dt * tendency``, ``eta <- eta + dt * eta_dot``.
    Works for any state whose ``T`` has shape ``(..., nlev)``.

    Budget note: melt genuinely freshens the ocean (total virtual salt
    decreases) and adds volume through ``eta``; a run with the global
    ``fix_salt``/``fix_volume`` conservation fixers on will cancel the
    global-mean of this real flux — leave them off (the OMIP CORE-II
    recipe does) or account for it in the freshwater-budget correction.

    DEVIATIONS from NEMO (documented, tracked as follow-up): (a) NEMO
    injects the melt VOLUME as a 3-D divergence source over the TBL
    (isf_hdiv_mlt) — here the salinity effect is the standard
    virtual-salt approximation of that volume route and the barotropic
    part is a 2-D ``eta`` source; (b) NEMO's nn_fwb=1 freshwater-budget
    correction includes −fwfisf_par — our surface-channel normalization
    does not see this post-step flux (residual global freshening
    ~4e-5 PSU/yr at the Depoorter total).

    ``fwf_kg_m2_s`` is the melt freshwater INTO the ocean (>= 0, NEMO
    ``sornfisf``); ``zmin_m``/``zmax_m`` the injection band [m, +down].
    """
    from legoesm.ocean.physics.ice_shelf import isf_prescribed_melt_tendencies

    rho0 = constants.rho_ocean if rho_0 is None else rho_0
    # NEMO isf constants (codex r3 #2): rcp (TEOS-10 heat capacity) and
    # the ISF-SPECIFIC latent heat rLfusisf = 0.334e6 from isf_oce.F90
    # (deliberately not phycst rLfus).
    cp = float(constants.c_p_seawater) if c_sw is None else c_sw
    lf = float(constants.L_fus_isf_nemo) if L_fus is None else L_fus
    cfg = IceShelfConfig() if config is None else config

    dT_dt, dS_dt, eta_dot = isf_prescribed_melt_tendencies(
        state.S.data, dz_live, wet_cell,
        fwf_kg_m2_s, zmin_m, zmax_m,
        rho_0=rho0, c_sw=cp, L_fus=lf, config=cfg,
    )
    T0 = jnp.asarray(state.T.data)
    S0 = jnp.asarray(state.S.data)
    eta0 = jnp.asarray(state.eta.data)
    return state._replace(
        T=state.T.replace(data=(T0 + dt * dT_dt).astype(T0.dtype)),
        S=state.S.replace(data=(S0 + dt * dS_dt).astype(S0.dtype)),
        eta=state.eta.replace(data=(eta0 + dt * eta_dot).astype(eta0.dtype)),
    )
