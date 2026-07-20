"""MPAS Voronoi-mesh adapters for vertical mixing physics.

Mirrors the lat-lon ``integration.py`` factories but adapts to the MPAS
C-grid where:
- T, S, rho are at cells (nCells, nlev)
- u is at edges (nEdges, nlev) — edge-normal scalar

For tracers (T, S), KPP applies directly cell-by-cell.

For momentum (u), KPP needs (u_east, v_north) at cells to compute the
shear-driven Richardson number.  We reconstruct cell-centered velocity
via TRiSK (Perot 2000 reconstruction), pass it to KPP, take the returned
viscosity field A_v(nCells, nlev), interpolate to edges, and apply
``vertical_diffusion_variable_K`` to the edge-normal u directly.

This avoids reconstructing back from (du/dt_east, dv/dt_north) to the
edge-normal tendency, which would be more expensive and introduce
reconstruction errors.
"""
from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from legoesm import constants
from legoesm.core.state import MPASOceanState
from legoesm.ocean.eos import (
    compute_ocean_rho,
    rho_0 as _RHO_0,
    c_sw as _C_SW,
)
from legoesm.ocean.init_mpas import reconstruct_cell_velocity
from legoesm.ocean.physics.mixing import (
    vertical_diffusion_variable_K,
    flux_divergence_zero_flux,
)
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
from legoesm.ocean.physics.vertical_mixing.tke import tke_vertical_mixing
from legoesm.ocean.physics.vertical_mixing._shared import surface_buoyancy_flux
from legoesm.ocean.vertical import (
    OceanPartialCellCoordinate,
    compute_layer_thickness,
    compute_ocean_jacobian,
)

__physics_contract__ = {
    "summary": (
        "MPAS Voronoi-mesh adapters wiring KPP and (diagnostic) Gaspar/Burchard "
        "TKE vertical mixing onto the C-grid: tracers at cells, edge-normal "
        "momentum via TRiSK cell-velocity reconstruction (single shared "
        "reconstruction); produce cell (K_v, A_v) profiles for the implicit "
        "solver and, for KPP, the edge-normal momentum + cell tracer tendencies."
    ),
    "inputs": {
        "state.T": "degC", "state.S": "psu", "state.u": "m/s (edge-normal)",
        "surface_forcing.tau_x": "N/m^2", "surface_forcing.q_net": "W/m^2",
    },
    "outputs": {
        "du_dt_edge": "m/s^2", "dT_dt": "degC/s", "dS_dt": "psu/s",
        "A_v_cells": "m^2/s", "K_v_cells": "m^2/s",
    },
    "sign_convention": (
        "K_v, A_v >= 0; KPP diffusivities diagnosed at cells then interpolated "
        "to edges; edge-normal momentum diffusion is flux-form with a zero-flux "
        "seafloor (partial-cell masking); z positive up. A coefficient producer "
        "+ flux-form applier — the budget closes in the diffusion solver; an "
        "unknown scheme raises ValueError."
    ),
    # make_kpp_physics_mpas applies flux-form edge-momentum + cell-tracer
    # tendencies (zero-flux seafloor, surface fluxes separate) = conservative
    # redistribution of column-integrated heat (energy), salt and momentum; the
    # sibling make_kpp_profiles_mpas is a coefficient-only producer (applies
    # nothing, so conserves/violates nothing).
    "conserves": ["energy", "salt", "momentum"],
    "differentiable": True,
    "reference": (
        "Large, McWilliams & Doney (1994) KPP + Gaspar et al. (1990) / "
        "Burchard (2002) TKE on the MPAS/TRiSK C-grid (Perot 2000 "
        "reconstruction; Ringler et al. 2013, Ocean Modelling 69)"
    ),
    "idealized_test": (
        "tests/ocean/unit/test_vmix_mpas_integration.py — cell-reconstructed "
        "KPP viscosity applied to edge-normal u matches the lat-lon path; "
        "tests/ocean/unit/test_mpas_tke.py — the MPAS TKE profiles match a "
        "direct grid-agnostic tke_vertical_mixing call; unsupported schemes "
        "raise ValueError."
    ),
}


# Placeholder salinity for dry cells so the EOS stays well-defined [PSU].
_EOS_SAFE_SALINITY_PSU = 35.0

# Diagnostic (Mode-B) quasi-steady TKE on MPAS: a long pseudo-timestep drives
# the backward-Euler TKE solve toward local equilibrium in a few sub-iterations
# (mirrors the lat-lon ``k_profiles`` Mode-B default used when no prognostic TKE
# field is carried on the state; ``TKEConfig.prognostic=False`` path).  Fixed
# solver settings, NOT tunables.
_TKE_DIAGNOSTIC_DT_S = 86400.0   # [s] 1-day pseudo-step → quasi-steady K
_TKE_DIAGNOSTIC_N_ITER = 3       # backward-Euler sub-iterations (K within ~few %)


def _mpas_surface_buoyancy_flux(q_net, fw, salt, T_3d, S_3d, eos_fn=None):
    """MPAS surface buoyancy flux ``B_f`` [m^2/s^3] (>0 destabilising) plus the
    kinematic surface heat/salt fluxes for the KPP boundary-layer closure.

    ``eos_fn`` (``None`` ⇒ Wright) sets the surface α/β: a non-Wright EOS
    (e.g. ``nemo_seos``) is used consistently with the interior ρ/N²/Ri.

    Single source for the MPAS-KPP surface forcing block (#518 item 1): both
    ``make_kpp_physics_mpas`` and ``make_kpp_profiles_mpas`` computed this from
    byte-identical inline code.  MPAS-specific salt convention (do NOT fold into
    the lat-lon ``integration.py`` / ``k_profiles._surface_buoyancy_flux``
    variants — they differ deliberately):

    * Freshwater (virtual salt) feeds BOTH the surface buoyancy AND the KPP
      non-local salinity flux ``Q_sfc_S``.
    * Real brine salt-mass flux feeds the surface BUOYANCY ONLY — it is
      deliberately NOT added to ``Q_sfc_S``.  The net real-salt injection is the
      explicit mass-exact floored-h_k source in
      ``mpas_ocean_baroclinic_tendencies``; adding salt to the non-local term
      would inject a second real-salt contribution that is not mass-conservative
      on partial cells (the non-local tendency is built on the full reference
      grid then masked to active levels, so its actual-thickness column integral
      is nonzero when the boundary layer reaches a shallow partial seafloor).

    Returns ``(B_f, Q_sfc_T, Q_sfc_S)``; each is ``None`` when its forcing
    channel is absent (the KPP caller treats ``None`` Q_sfc_S as "diagnose the
    non-local flux from the gradient").
    """
    # Grid-agnostic kernel (#518 item 1).  MPAS convention: the real salt-mass
    # flux feeds the surface buoyancy ONLY (real_salt_in_qs=False); the floored
    # non-local Q_sfc_S carries the freshwater term only.  Pass the MPAS module
    # constants (== canonical defaults) as the constant source.
    return surface_buoyancy_flux(
        q_net, fw, salt,
        T_3d[..., 0], S_3d[..., 0],
        g=constants.g, rho_0=_RHO_0, c_sw=_C_SW,
        real_salt_in_qs=False,
        eos_fn=eos_fn,
    )


def _vertical_diffusion_edge_partial(
    field: jnp.ndarray,
    h_e: jnp.ndarray,
    K_half: jnp.ndarray,
) -> jnp.ndarray:
    """Vertical diffusion with actual per-edge layer thicknesses.

    Same algorithm as ``vertical_diffusion_variable_K`` but uses
    explicit per-level layer thicknesses ``h_e`` (from min-rule on
    partial cells) instead of ``dz_ref * J``.  This correctly handles:

    - Thin partial bottom cells: tendency uses actual 5m thickness,
      not the 128m reference thickness.
    - Sub-seafloor levels: h_e = 0 there, so those levels get zero
      tendency (division by ``max(h_e, eps)`` produces a finite but
      irrelevant value that is zeroed by ``edge_mask_3d`` in the PE
      module).
    - Full cells: h_e = dz_ref * J, so this is bit-exact with the
      original operator on z-star (non-partial) coordinates.

    Parameters
    ----------
    field : array, shape (nEdges, nlev)
        Edge-normal velocity.
    h_e : array, shape (nEdges, nlev)
        Actual layer thickness at edges [m] (min-rule, from
        ``min_cell_to_edge(compute_layer_thickness(...))``).
        Zero at sub-seafloor levels.
    K_half : array, shape (nEdges, nlev-1)
        Diffusivity at interior interfaces [m^2/s].
        Pre-masked: zero at sub-seafloor interfaces.

    Returns
    -------
    array, shape (nEdges, nlev) : Vertical diffusion tendency [m/s^2].
    """
    nlev = field.shape[-1]
    if nlev < 2:
        return jnp.zeros_like(field)

    dtype = field.dtype
    h_e = h_e.astype(dtype)
    K_half = K_half.astype(dtype)

    # Guard against zero thickness (sub-seafloor levels).  The
    # resulting tendency is large but irrelevant — it gets multiplied
    # by edge_mask_3d=0 in the PE module.  Using a LARGE floor (1m
    # instead of 1e-10) prevents extreme values from affecting
    # numerical precision elsewhere.
    h_safe = jnp.maximum(h_e, 1.0)

    # Distance between cell centers: 0.5 * (h[k] + h[k+1]).
    # At the seafloor boundary (h[bot]=5m, h[bot+1]=0): dz_half =
    # 0.5*(5+1) = 3m (with the 1m floor).  But K_half is zero there
    # (masked), so the flux is zero regardless.  The only interface
    # where dz_half matters is between two active cells.
    dz_half = 0.5 * (h_safe[..., :-1] + h_safe[..., 1:])  # (nEdges, nlev-1)

    # Diffusive flux at interfaces: K * d(field)/dz.
    df_dz = (field[..., :-1] - field[..., 1:]) / dz_half
    flux = K_half * df_dz  # (nEdges, nlev-1)

    # Tendency at full levels: d(flux)/dz with zero-flux BCs (shared kernel,
    # #518 item 4).  ``h_safe`` carries this site's max(h_e, 1.0) partial-cell
    # floor (NOT the z* where(dz>0,dz,1) floor — different on partial cells).
    return flux_divergence_zero_flux(flux, h_safe)


def _reconstruct_mpas_cell_fields(state: MPASOceanState, mesh, z_coord,
                                  eos_fn=None):
    """Shared MPAS cell-field prep for the vertical-mixing bridges.

    Single-sources the edge→cell velocity reconstruction + land/partial-cell
    conditioning that BOTH the KPP (``_run_mpas_kpp``) and the diagnostic-TKE
    (``make_tke_profiles_mpas``) bridges need, so there is exactly ONE
    reconstruction on MPAS (CLAUDE.md "no duplicate numerics"; the KPP path
    stays byte-identical — this is a pure extraction):

    * TRiSK/Perot cell-centred (u_east, v_north) from edge-normal ``u`` — the
      shear both closures diagnose (Richardson number for KPP, ∂u/∂z shear
      production for TKE);
    * land-safe Jacobian ``J`` (land → 1.0) and in-situ density ``rho`` (land →
      ``rho_0``) via the model-selected ``eos_fn`` (``None`` ⇒ Wright default);
    * land-zeroing + the partial-cell sub-seafloor T/S/u/v fill (deepest active
      value) so neither closure sees a spurious T=0/u=0 discontinuity.

    Returns ``(u_east_w, v_north_w, T_w, S_w, rho, J, mask)`` — the land-safe
    Jacobian ``J`` is returned because the KPP physics path reuses it for the
    cell→edge Jacobian average.
    """
    T_3d = state.T.data       # (nCells, nlev)
    S_3d = state.S.data
    u_edge = state.u.data     # (nEdges, nlev)
    eta = state.eta.data
    H_bathy = state.H_bathy.data
    mask = state.land_mask.data  # (nCells,) — 1=ocean, 0=land

    # Cell-centred (u_east, v_north) from edge-normal u via TRiSK/Perot — the
    # shear driver for both closures.  (Sub-seafloor momentum leak is closed by
    # the fill + masking below, so real velocities are safe to pass.)
    u_east_raw, v_north_raw = reconstruct_cell_velocity(u_edge, mesh)

    # Density at cells.  The closures divide by the Jacobian internally; it is 0
    # on land (H_bathy=0) → NaN, so replace land J with 1.0 and land density
    # with rho_0 (those cells are masked out downstream).
    J_real = compute_ocean_jacobian(eta, H_bathy, z_coord)
    J = jnp.where(mask > 0.5, J_real, 1.0)
    rho_real = compute_ocean_rho(state, z_coord, J_real, eos_fn=eos_fn)
    rho = jnp.where(mask[:, None] > 0.5, rho_real, _RHO_0)

    # Land-zero inputs; on partial cells fill sub-seafloor levels with the
    # deepest active value so the closure sees no spurious T=0/u=0 discontinuity.
    m3 = mask[:, None]
    u_east_w = jnp.where(m3 > 0.5, u_east_raw, 0.0)
    v_north_w = jnp.where(m3 > 0.5, v_north_raw, 0.0)
    T_w = jnp.where(m3 > 0.5, T_3d, 0.0)
    S_w = jnp.where(m3 > 0.5, S_3d, _EOS_SAFE_SALINITY_PSU)  # safe S for EOS
    if hasattr(z_coord, 'is_active'):
        _active = z_coord.is_active  # (nCells, nlev) bool
        _bot_lev = z_coord.bottom_level  # (nCells,) int
        _bot_lev_safe = jnp.clip(_bot_lev, 0, T_3d.shape[1] - 1)
        _row_idx = jnp.arange(T_w.shape[0])
        _T_bot = T_w[_row_idx, _bot_lev_safe]
        _S_bot = S_w[_row_idx, _bot_lev_safe]
        _u_bot = u_east_w[_row_idx, _bot_lev_safe]
        _v_bot = v_north_w[_row_idx, _bot_lev_safe]
        T_w = jnp.where(_active, T_w, _T_bot[:, None])
        S_w = jnp.where(_active, S_w, _S_bot[:, None])
        u_east_w = jnp.where(_active, u_east_w, _u_bot[:, None])
        v_north_w = jnp.where(_active, v_north_w, _v_bot[:, None])

    return u_east_w, v_north_w, T_w, S_w, rho, J, mask


def _run_mpas_kpp(state: MPASOceanState, mesh, z_coord, surface_forcing, cfg,
                  eos_fn=None):
    """Prepare MPAS-KPP inputs and run ``kpp_vertical_mixing`` (#518 item 2).

    ``eos_fn`` (``None`` ⇒ Wright default) is the model-selected EOS callable
    used for the KPP density / Richardson / buoyancy diagnostics — threaded so
    a non-Wright EOS (e.g. ``nemo_seos`` for DINO) drives the mixing decision
    consistently with the baroclinic dycore, not silently via Wright.

    ``make_kpp_physics_mpas`` and ``make_kpp_profiles_mpas`` shared this entire
    input-preparation block verbatim: the TRiSK/Perot cell-velocity
    reconstruction, land-safe Jacobian + density, land-zeroing, and the
    partial-cell sub-seafloor fill are now factored into
    :func:`_reconstruct_mpas_cell_fields` (shared with the TKE bridge); this
    function adds only the KPP-specific surface-forcing buoyancy flux and the
    KPP call.

    Returns ``(kpp_out, J)``.  ``J`` (the land-safe Jacobian) is returned because
    the physics path reuses it for the cell→edge Jacobian average; the profiles
    path uses only ``kpp_out`` (``J`` is then dead and pruned).
    """
    # Shared edge→cell reconstruction + land/partial-cell conditioning.
    # (``mask`` is returned for the TKE bridge; KPP masks downstream from
    # ``state.land_mask`` directly, so it is unused here.)
    u_east_w, v_north_w, T_w, S_w, rho, J, _ = _reconstruct_mpas_cell_fields(
        state, mesh, z_coord, eos_fn=eos_fn)
    T_3d = state.T.data       # (nCells, nlev) — RAW surface T/S for buoyancy flux
    S_3d = state.S.data
    eta = state.eta.data

    # Surface forcing channels (None → KPP uses interior proxies).
    tau_x = getattr(surface_forcing, "tau_x", None) if surface_forcing else None
    tau_y = getattr(surface_forcing, "tau_y", None) if surface_forcing else None
    q_net = getattr(surface_forcing, "q_net", None) if surface_forcing else None
    fw = getattr(surface_forcing, "freshwater", None) if surface_forcing else None
    salt = getattr(surface_forcing, "salt_flux", None) if surface_forcing else None

    # Under-ice velocity-scale attenuation (KPPConfig.eice; NEMO nn_eice) —
    # the SAME static-config gate as the lat-lon KPP paths
    # (integration.py / k_profiles.py) so MPAS honours eice through the shared
    # grid-agnostic ``_kpp_ice_attenuation`` instead of silently no-oping it
    # (dispatch discipline).  eice=0 -> ice_frac=None -> bit-identical to the
    # pre-eice MPAS-KPP path.  ``ice_concentration`` (nCells,) rides on the
    # full surface_forcing the model threads to the KPP factory (verified:
    # ocean_model_mpas.py passes the WHOLE forcing, not a tau/q strip).
    _kpp_eice = int(getattr(cfg, "eice", 0))
    if _kpp_eice not in (0, 1, 3):
        raise ValueError(
            f"Unknown KPPConfig.eice={_kpp_eice!r}; expected 0, 1 or 3.")
    ice_frac = (getattr(surface_forcing, "ice_concentration", None)
                if (_kpp_eice != 0 and surface_forcing is not None)
                else None)
    if _kpp_eice != 0 and ice_frac is None:
        # Fail fast: eice was EXPLICITLY requested but no ice field is present
        # (surface_forcing None, or ice_concentration None).  Silently running
        # the un-attenuated closure is the dispatch footgun CLAUDE.md forbids,
        # and it would regress the old MPAS-KPP hard reject for direct-API
        # callers (MPASOceanModel / make_kpp_*_mpas) that bypass
        # run_omip_core2.main()'s ice-source guard (codex MED).  ice_frac is a
        # static None here (pytree structure), so this branch is trace-time.
        raise ValueError(
            f"KPPConfig.eice={_kpp_eice} (MPAS under-ice attenuation) requires "
            "surface_forcing.ice_concentration, but none was supplied. Provide "
            "sea-ice concentration (prognostic or prescribed SIC) or set eice=0.")

    # Surface buoyancy + kinematic T/S fluxes (shared MPAS helper).
    B_f, Q_sfc_T, Q_sfc_S = _mpas_surface_buoyancy_flux(
        q_net, fw, salt, T_3d, S_3d, eos_fn=eos_fn)

    kpp_out = kpp_vertical_mixing(
        u_east_w, v_north_w, T_w, S_w,
        rho, eta, z_coord, J, cfg,
        tau_x=tau_x, tau_y=tau_y, B_f=B_f,
        Q_sfc_T=Q_sfc_T, Q_sfc_S=Q_sfc_S,
        eos_fn=eos_fn, ice_frac=ice_frac,
    )
    return kpp_out, J


def make_kpp_physics_mpas(config: VerticalMixingConfig, eos_fn=None) -> Callable:
    """Build KPP physics_fn for MPAS Voronoi mesh.

    Parameters
    ----------
    config : VerticalMixingConfig
        Must have ``scheme="kpp"`` and ``kpp`` sub-config populated.

    Returns
    -------
    Callable
        ``physics_fn(state, mesh, z_coord, surface_forcing=None)``
        returning (du_dt_edge, dT_dt_cell, dS_dt_cell, A_v_cell, K_v_cell).
        The model assembles these into MPASOceanTendencies.
    """
    if getattr(config, "iwm", None) is not None and config.iwm.enabled:
        raise NotImplementedError(
            "VerticalMixingConfig.iwm.enabled=True is not wired on the MPAS "
            "vertical-mixing bridge yet (lat-lon / tripole only) — reject "
            "rather than silently drop the wave-driven mixing.")
    cfg = config.kpp
    # NOTE: KPPConfig.eice (under-ice velocity-scale attenuation) IS wired on
    # the MPAS KPP bridge — ``_run_mpas_kpp`` reads surface_forcing.
    # ice_concentration under the shared eice gate and threads it to
    # ``kpp_vertical_mixing``.  eice is validated (0/1/3) there per call.

    def physics_fn(
        state: MPASOceanState,
        mesh,
        z_coord,
        surface_forcing=None,
    ):
        # Shared MPAS-KPP input prep + KPP call (#518: factored helper).
        kpp_out, J = _run_mpas_kpp(
            state, mesh, z_coord, surface_forcing, cfg, eos_fn=eos_fn)
        # State accessors reused by the post-KPP edge-diffusion code below.
        T_3d = state.T.data
        eta = state.eta.data
        H_bathy = state.H_bathy.data
        u_edge = state.u.data
        mask = state.land_mask.data

        # Mask tracer tendencies — zero on land AND shallow cells.
        # KPP is designed for open-ocean boundary layers (50-500m deep).
        # In cells with fewer than 5 active levels (shallow shelves,
        # isolated water bodies like the Caspian), KPP's diffusion is
        # physically inappropriate and creates O(1 K/s) tendencies in
        # thin cells that blow up the model (Finding 10: lat=48°N,
        # lon=49°E, H=50m).  Mask these cells entirely.
        _min_kpp_levels = 5
        if hasattr(z_coord, 'bottom_level'):
            _n_active = z_coord.bottom_level + 1  # (nCells,)
            _kpp_mask = (_n_active >= _min_kpp_levels).astype(T_3d.dtype)
        else:
            _kpp_mask = jnp.ones(mask.shape, dtype=T_3d.dtype)
        _kpp_mask_3d = (_kpp_mask * mask)[:, None]  # (nCells, 1)
        # Use per-level active mask on partial-cell coords to zero tracer
        # tendencies at sub-seafloor levels.  Without this, KPP computes
        # diffusion through all nlev levels (using incorrect dz_ref
        # thicknesses below the seafloor) and the resulting tendencies
        # corrupt sub-seafloor T/S values before the model's active_3d
        # guard can restore them.
        if hasattr(z_coord, 'is_active'):
            _active_3d = z_coord.is_active.astype(T_3d.dtype)  # (nCells, nlev)
            _kpp_mask_full = _kpp_mask_3d * _active_3d
        else:
            _kpp_mask_full = _kpp_mask_3d

        # --- Partial-cell conservation rescale (tracer tendencies) ---
        # ``kpp_vertical_mixing`` builds dT/dt, dS/dt as flux-form vertical
        # divergences divided by the REFERENCE-grid thickness ``dz_ref * J``
        # (kpp.py ``dz_actual``; both the local diffusion AND the non-local
        # counter-gradient term use it).  The dycore, however, advances heat
        # / salt content weighted by the LIVE partial-cell thickness
        # ``h_k = compute_layer_thickness(eta, H_bathy, z_coord)``
        # (ocean_model_mpas.py: ``T_new = T + dt*dT_dt`` with the budget
        # measured as ``sum(dT_dt * h_k * area)``).  On a partial bottom cell
        # ``dz_ref*J > h_partial*J = h_k``, so the LIVE-thickness column
        # integral of a purely REDISTRIBUTIVE (interior-mixing) tendency is
        # NONZERO — a spurious heat/salt source/sink on partial/live cells.
        #
        # Convention: z positive up; the vertical flux is down-gradient
        # (``F = -K dT/dz``) with zero flux at the surface AND at the seafloor
        # (the sub-seafloor T/S fill above makes the seafloor-interface
        # gradient — hence its flux — exactly zero).  Rescaling by
        # ``dz_used / h_k`` turns ``dT/dt = D / dz_used`` into ``D / h_k``
        # (``D`` = interface-flux divergence, thickness-independent), so
        # ``sum_k h_k * dT/dt = sum_k D = F_surface - F_seafloor = 0`` is
        # conserved to machine precision.  On full cells (and on any
        # non-partial z*/z-level coord) ``dz_used == h_k`` exactly, so
        # ``thickness_rescale == 1`` and this is a byte-exact no-op — mirroring
        # the live-thickness edge-momentum path (``_vertical_diffusion_edge_partial``)
        # applied below.
        if isinstance(z_coord, OceanPartialCellCoordinate):
            h_live = compute_layer_thickness(eta, H_bathy, z_coord)  # (nCells, nlev)
            dz_used = z_coord.dz_ref * J[:, jnp.newaxis]  # what KPP divided by
            thickness_rescale = dz_used / jnp.maximum(h_live, 1.0e-10)
            dT_dt = kpp_out.dT_dt * thickness_rescale
            dS_dt = kpp_out.dS_dt * thickness_rescale
        else:
            dT_dt = kpp_out.dT_dt
            dS_dt = kpp_out.dS_dt
        dT_dt = jnp.where(_kpp_mask_full > 0.5, dT_dt, 0.0)
        dS_dt = jnp.where(_kpp_mask_full > 0.5, dS_dt, 0.0)

        # A_v is at half-levels (nCells, nlev-1).  Mask land cells.
        # Cap A_v to CFL-safe maximum based on the thinner of the two
        # adjacent layers: A_v_max = 0.25 * min(dz_k, dz_k+1)^2 / dt.
        # Without this cap, KPP can produce O(1-10) m²/s in deep
        # boundary layers, violating explicit-diffusion stability on
        # thin stretched-grid surface cells (dz ~ 20m, dt ~ 300s).
        # Use 0.25 (not 0.5) for safety margin.  The CFL timestep is a
        # KPP config field (default 300 s); set it to the ocean dynamics
        # dt so the cap matches the real explicit-diffusion stability
        # limit instead of a buried constant (slopbuster Pass 10).
        _dt_phys = cfg.cfl_cap_dt_s  # physics timestep [s] for CFL cap
        _dz = z_coord.dz_ref  # (nlev,)
        _dz_min_half = jnp.minimum(_dz[:-1], _dz[1:])  # (nlev-1,)
        _Av_max = 0.25 * _dz_min_half**2 / _dt_phys  # (nlev-1,)
        m3_half = mask[:, None]  # broadcast to (nCells, nlev-1)
        A_v_raw = jnp.where(m3_half > 0.5, kpp_out.A_v, 0.0)
        A_v_cells = jnp.minimum(A_v_raw, _Av_max[None, :])

        # Interpolate A_v from cells to edges and apply vertical diffusion
        # to edge-normal u directly.  Also apply the shallow-cell mask so
        # edges adjacent to shallow cells don't get KPP momentum diffusion.
        c1 = mesh.cellsOnEdge[0]
        c2 = mesh.cellsOnEdge[1]
        # Zero A_v at edges where EITHER cell is too shallow for KPP.
        _edge_kpp_mask = (_kpp_mask[c1] * _kpp_mask[c2])[:, None]
        A_v_cells_masked = A_v_cells * _kpp_mask[:, None]
        A_v_edge = 0.5 * (A_v_cells_masked[c1] + A_v_cells_masked[c2])  # (nEdges, nlev-1)
        A_v_edge = A_v_edge * _edge_kpp_mask

        # On partial-cell coordinates, zero A_v at half-level interfaces
        # below the shallower neighbor's seafloor.  Without this, the
        # diffusion operator sees nonzero A_v at inactive interfaces and
        # computes spurious fluxes through the seafloor boundary.
        if hasattr(z_coord, 'bottom_level'):
            from legoesm.ocean.dynamics.mpas_partial_cell_helpers import (
                compute_max_level_edge_bot,
                min_cell_to_edge,
            )
            bot_e = compute_max_level_edge_bot(z_coord.bottom_level, mesh)
            # Half-level k sits between full levels k and k+1.
            # Active interfaces: k = 0, ..., maxLevelEdgeBot - 1.
            # (maxLevelEdgeBot is the deepest FULL level that is active
            # on BOTH cells; the interface below it is the seafloor.)
            nlev_half = A_v_edge.shape[1]
            k_idx = jnp.arange(nlev_half, dtype=bot_e.dtype)
            active_half = (k_idx[None, :] < bot_e[:, None]).astype(A_v_edge.dtype)
            A_v_edge = A_v_edge * active_half

        # --- Edge-side vertical diffusion ---
        # On partial-cell coordinates, ``vertical_diffusion_variable_K``
        # uses ``dz_ref * J`` for layer thicknesses, which is WRONG at
        # the partial bottom cell (can be 25x too thick) and at sub-
        # seafloor levels.  This causes the diffusion tendency at the
        # thin bottom cell to be underestimated by the same factor,
        # creating a depth-integrated momentum imbalance that acts as
        # spurious bottom drag and drives the SSH blow-up observed in
        # ETOPO runs.
        #
        # Fix: compute actual per-edge layer thicknesses using the same
        # min-rule as the PE module (``min_cell_to_edge``).  This gives
        # h_e[k] = min(h_k[c1,k], h_k[c2,k]) — zero below the
        # shallower cell's seafloor, and correct partial thickness at
        # the bottom level.  Use these directly in the second-order
        # diffusion stencil (same algorithm as
        # ``vertical_diffusion_variable_K`` but with per-level dz).
        if isinstance(z_coord, OceanPartialCellCoordinate):
            h_k = compute_layer_thickness(eta, H_bathy, z_coord)
            h_e = min_cell_to_edge(h_k, mesh)  # (nEdges, nlev)
            # Per-edge CFL cap using ACTUAL thicknesses (not dz_ref).
            # A_v_max = 0.25 * min(h_e[k], h_e[k+1])^2 / dt at each
            # interface.  h_e=0 at sub-seafloor gives A_v_max=0 there
            # (redundant with active_half mask but defense-in-depth).
            _h_min_half_e = jnp.minimum(
                jnp.maximum(h_e[:, :-1], 1e-10),
                jnp.maximum(h_e[:, 1:], 1e-10),
            )  # (nEdges, nlev-1)
            _Av_max_edge = 0.25 * _h_min_half_e**2 / _dt_phys
            A_v_edge = jnp.minimum(A_v_edge, _Av_max_edge)
            du_dt_edge = _vertical_diffusion_edge_partial(
                u_edge, h_e, A_v_edge,
            )
        else:
            J_edge = 0.5 * (J[c1] + J[c2])
            du_dt_edge = vertical_diffusion_variable_K(
                u_edge, z_coord, J_edge, A_v_edge,
            )

        return du_dt_edge, dT_dt, dS_dt

    return physics_fn


def make_kpp_profiles_mpas(config: VerticalMixingConfig, eos_fn=None) -> Callable:
    """Build KPP profile-only function for MPAS implicit vertical mixing.

    Returns ``(A_v_cells, K_v_cells)`` at half-levels (nCells, nlev-1)
    — the raw KPP-produced viscosity and diffusivity profiles — WITHOUT
    computing or returning any tendencies.  The caller
    (``MPASOceanModel.step``) passes these profiles to the backward-Euler
    implicit solver together with background and convective K.

    The existing ``make_kpp_physics_mpas`` is preserved for backward
    compatibility with ``implicit_vertical_mixing=False``.

    Parameters
    ----------
    config : VerticalMixingConfig
        Must have ``scheme="kpp"`` and ``kpp`` sub-config populated.

    Returns
    -------
    Callable
        ``profiles_fn(state, mesh, z_coord, surface_forcing=None)``
        returning ``(A_v_cells, K_v_cells)`` both shape (nCells, nlev-1).
    """
    if getattr(config, "iwm", None) is not None and config.iwm.enabled:
        raise NotImplementedError(
            "VerticalMixingConfig.iwm.enabled=True is not wired on the MPAS "
            "vertical-mixing bridge yet (lat-lon / tripole only) — reject "
            "rather than silently drop the wave-driven mixing.")
    cfg = config.kpp
    # NOTE: KPPConfig.eice (under-ice velocity-scale attenuation) IS wired on
    # the MPAS KPP bridge — ``_run_mpas_kpp`` reads surface_forcing.
    # ice_concentration under the shared eice gate and threads it to
    # ``kpp_vertical_mixing``.  eice is validated (0/1/3) there per call.

    def profiles_fn(
        state: MPASOceanState,
        mesh,
        z_coord,
        surface_forcing=None,
    ):
        # Shared MPAS-KPP input prep + KPP call (#518: factored helper).
        # ``J`` is unused on the profiles path (only A_v/K_v are returned).
        kpp_out, _ = _run_mpas_kpp(
            state, mesh, z_coord, surface_forcing, cfg, eos_fn=eos_fn)
        T_3d = state.T.data
        mask = state.land_mask.data

        # Shallow-cell mask: KPP is physically inappropriate for cells
        # with fewer than 5 active levels.
        _min_kpp_levels = 5
        if hasattr(z_coord, 'bottom_level'):
            _n_active = z_coord.bottom_level + 1
            _kpp_mask = (_n_active >= _min_kpp_levels).astype(T_3d.dtype)
        else:
            _kpp_mask = jnp.ones(mask.shape, dtype=T_3d.dtype)

        # A_v and K_v at half-levels (nCells, nlev-1).
        # Mask land cells and shallow cells.
        _kpp_mask_2d = _kpp_mask * mask  # (nCells,)
        m_half = _kpp_mask_2d[:, None]   # broadcast to (nCells, nlev-1)
        A_v_cells = jnp.where(m_half > 0.5, kpp_out.A_v, 0.0)
        K_v_cells = jnp.where(m_half > 0.5, kpp_out.K_v, 0.0)

        # Sub-seafloor interface masking: zero K at interfaces below the
        # deepest active full level.
        if hasattr(z_coord, 'bottom_level'):
            nlev_half = A_v_cells.shape[1]
            k_idx = jnp.arange(nlev_half, dtype=z_coord.bottom_level.dtype)
            bot_c = z_coord.bottom_level  # (nCells,)
            active_half_c = (k_idx[None, :] < bot_c[:, None]).astype(A_v_cells.dtype)
            A_v_cells = A_v_cells * active_half_c
            K_v_cells = K_v_cells * active_half_c

        return A_v_cells, K_v_cells

    return profiles_fn


def make_tke_profiles_mpas(config: VerticalMixingConfig, eos_fn=None) -> Callable:
    """Build a DIAGNOSTIC prognostic-TKE profile function for MPAS implicit vmix.

    Wires the grid-agnostic Gaspar (1990) / Burchard (2002) TKE closure
    (:func:`legoesm.ocean.physics.vertical_mixing.tke.tke_vertical_mixing`) onto
    the MPAS Voronoi C-grid, returning ``(A_v_cells, K_v_cells)`` at half-levels
    (nCells, nlev-1) — the raw TKE-derived viscosity ``K_M`` and tracer
    diffusivity ``K_H`` for the backward-Euler implicit solver, EXACTLY like
    :func:`make_kpp_profiles_mpas`.  TKE requires ``implicit_vertical_mixing=True``
    (rejected in ``make_mpas_ocean_physics``); there is no explicit-tendency TKE
    path on MPAS.

    Reuse (no re-derivation):

    * :func:`_reconstruct_mpas_cell_fields` — the SAME edge→cell (u_east,
      v_north) TRiSK/Perot reconstruction + land-safe J/ρ + partial-cell fill
      that KPP uses (single reconstruction on MPAS);
    * ``tke_vertical_mixing`` — the whole closure (shear/buoyancy production,
      dissipation, Bougeault-Lacarrère mixing lengths); NO new closure here;
    * the KPP MPAS CFL cap ``A_v_max = 0.25·min(dz)²/cfl_cap_dt_s``
      (``TKEConfig.cfl_cap_dt_s``) on both K_M and K_H (defense-in-depth: the
      profiles feed the unconditionally-stable implicit solver, but a
      convecting column can drive ``l_k`` — and K — large before ``kappaM_max``
      binds).

    DIAGNOSTIC (Mode B) only: the closure is seeded at ``tke_background`` and
    sub-iterated to quasi-steady each call (``tke_old=None``); NO prognostic TKE
    field is carried on ``MPASOceanState``.  This mirrors the lat-lon
    ``k_profiles`` Mode-B DEFAULT (``TKEConfig.prognostic=False``).  The
    PROGNOSTIC carry (``prognostic=True``) is NOT wired on MPAS — it needs a
    seeded ``MPASOceanState.tke`` field stable across the production ``lax.scan``
    (the None→Field seed the lat-lon ``seed_scan_carry`` performs).  Reject it
    (and the other options whose extra inputs this bridge does not plumb) LOUDLY
    at factory-build time rather than silently running a different closure
    (dispatch discipline; ``vm_scheme``/cfg are static → jit-safe).

    Parameters
    ----------
    config : VerticalMixingConfig
        Must have ``scheme="tke"`` and ``tke`` sub-config populated.

    Returns
    -------
    Callable
        ``profiles_fn(state, mesh, z_coord, surface_forcing=None)`` returning
        ``(A_v_cells, K_v_cells)`` both shape (nCells, nlev-1), >= 0, finite.
    """
    cfg = config.tke

    # --- Reject options whose extra inputs the MPAS diagnostic bridge does not
    #     plumb (dispatch discipline: fail loud, never silently run a different
    #     closure).  All are static config values ⇒ raising here is jit-safe. ---
    if bool(getattr(cfg, "prognostic", False)):
        raise NotImplementedError(
            "vertical_mixing.tke.prognostic=True is not wired on the MPAS "
            "ocean yet: a prognostic TKE carry needs a seeded MPASOceanState.tke "
            "field kept pytree-stable across the production lax.scan (the "
            "None->Field seed the lat-lon seed_scan_carry performs). MPAS runs "
            "the DIAGNOSTIC quasi-steady TKE (prognostic=False), matching the "
            "lat-lon Mode-B default. Set vertical_mixing.tke.prognostic=False, "
            "or run prognostic TKE on the lat-lon C-grid.")
    if getattr(cfg, "n2_mode", "insitu") != "insitu":
        raise NotImplementedError(
            f"vertical_mixing.tke.n2_mode={getattr(cfg, 'n2_mode', 'insitu')!r} "
            "is not wired on the MPAS ocean (the adiabatic/signed-N^2 path needs "
            "the cell-centre hydrostatic pressure that this bridge does not "
            "compute). MPAS supports n2_mode='insitu'.")
    if int(getattr(cfg, "eice", 0)) != 0:
        raise NotImplementedError(
            f"vertical_mixing.tke.eice={int(getattr(cfg, 'eice', 0))} (under-ice "
            "attenuation of the lc/etau TKE sources) is not wired on the MPAS "
            "vertical-mixing bridge yet: this bridge receives no ice "
            "concentration (mpas_physics passes tau/q only). Set eice=0 on "
            "MPAS, or run the tke closure on the lat-lon/tripole C-grid where "
            "surface_forcing.ice_concentration is threaded.")
    if bool(getattr(cfg, "veros_dz_slots", False)):
        raise NotImplementedError(
            "vertical_mixing.tke.veros_dz_slots=True is not wired on the MPAS "
            "ocean (the Veros metric slots need dz_ref/jacobian/dz_surface "
            "geometry this bridge does not plumb). Set veros_dz_slots=False.")
    if getattr(cfg, "buoyancy_timing", "pre_mixing") != "pre_mixing":
        raise NotImplementedError(
            "vertical_mixing.tke.buoyancy_timing="
            f"{getattr(cfg, 'buoyancy_timing', 'pre_mixing')!r} is not wired on "
            "the MPAS ocean (post_mixing_veros needs the prognostic model-step "
            "ordering). MPAS supports buoyancy_timing='pre_mixing'.")
    if bool(getattr(cfg, "lc", False)) or getattr(cfg, "etau_mode", "none") != "none":
        raise NotImplementedError(
            "vertical_mixing.tke NEMO surface terms (lc / etau_mode) are not "
            "wired on the MPAS ocean (they need the etau latitude profile this "
            "bridge does not plumb). Set lc=False and etau_mode='none'.")
    if getattr(cfg, "advection_scheme", "none") != "none":
        raise NotImplementedError(
            "vertical_mixing.tke.advection_scheme="
            f"{getattr(cfg, 'advection_scheme', 'none')!r} requires the "
            "prognostic TKE carry (advecting a diagnostic TKE is meaningless) "
            "and is not wired on MPAS. Set advection_scheme='none'.")
    if bool(getattr(cfg, "source_eke_diss", False)):
        raise NotImplementedError(
            "vertical_mixing.tke.source_eke_diss=True requires the prognostic "
            "TKE carry (the EKE-dissipation recycling source) and is not wired "
            "on MPAS. Set source_eke_diss=False.")
    if getattr(config, "iwm", None) is not None and config.iwm.enabled:
        raise NotImplementedError(
            "VerticalMixingConfig.iwm.enabled=True is not wired on the MPAS "
            "vertical-mixing bridge yet (lat-lon / tripole only) — reject "
            "rather than silently drop the wave-driven mixing.")

    def profiles_fn(
        state: MPASOceanState,
        mesh,
        z_coord,
        surface_forcing=None,
    ):
        # Shared edge→cell reconstruction + land/partial-cell conditioning
        # (the SAME helper KPP uses — one reconstruction on MPAS).
        u_east_w, v_north_w, T_w, S_w, rho, J, mask = (
            _reconstruct_mpas_cell_fields(state, mesh, z_coord, eos_fn=eos_fn)
        )

        # Cell-centre spacing dz_half = dz_half_ref · J (nCells, nlev-1), the
        # centre-to-centre distance the closure differentiates over — matches
        # the lat-lon k_profiles broadcast (dz_half_ref · J).
        dz_half = z_coord.dz_half_ref * J[:, jnp.newaxis]  # (nCells, nlev-1)
        # Interior interface reference heights (nlev-1) for the Bryan-Lewis
        # kappaH floor (only read on the opt-in Prandtl path; harmless on the
        # default 'unit' path).  z_half_ref is negative-down; drop surface+bottom.
        z_interface = z_coord.z_half_ref[1:-1]

        # Surface wind stress → TKE surface flux (|tau|/rho_0)^{3/2}; None ⇒
        # unforced (the closure zeroes the surface flux).
        tau_x = (getattr(surface_forcing, "tau_x", None)
                 if surface_forcing is not None else None)
        tau_y = (getattr(surface_forcing, "tau_y", None)
                 if surface_forcing is not None else None)

        # Veros tke_mxl_choice=1 distance-to-boundary cap (mirrors the lat-lon
        # k_profiles branch): the buoyancy mixing length may not exceed the
        # distance to surface/seafloor.  Built from the STATIC reference geometry
        # + per-column ocean depth.  choice=2 (default) is bounded by the
        # MITgcm/OPA recursion and needs no cap.
        boundary_cap = None
        if getattr(cfg, "tke_mxl_choice", 2) == 1:
            from legoesm.ocean.physics.vertical_mixing.tke import (
                veros_mxl_choice1_boundary_cap,
            )
            boundary_cap = veros_mxl_choice1_boundary_cap(
                z_interface, z_coord.dz_half_ref, state.H_bathy.data,
            )

        # --- The grid-agnostic TKE closure (DIAGNOSTIC Mode B) ---
        # A_v = K_M (momentum viscosity), K_v = K_H (tracer diffusivity), both
        # at interior interfaces (nCells, nlev-1).  insitu N^2 needs only
        # rho + dz_half (no p_cell/dz_ref/jacobian), so nothing else is passed.
        tke_out = tke_vertical_mixing(
            u_east_w, v_north_w, T_w, S_w, rho, dz_half,
            tke_old=None,
            tau_x_surface=tau_x, tau_y_surface=tau_y,
            dt=_TKE_DIAGNOSTIC_DT_S, cfg=cfg,
            rho_0=_RHO_0, g=constants.g,
            n_iterations=_TKE_DIAGNOSTIC_N_ITER,
            z_interface=z_interface,
            boundary_cap=boundary_cap,
        )
        A_v_cells = tke_out.K_M   # (nCells, nlev-1) momentum viscosity >= 0
        K_v_cells = tke_out.K_H   # (nCells, nlev-1) tracer diffusivity >= 0

        # Mask land cells (whole column) — no shallow-cell exclusion: TKE is a
        # local closure and the CFL cap below bounds K in thin cells, so KPP's
        # <5-level open-ocean heuristic does not apply here.
        m_half = mask[:, None]   # broadcast to (nCells, nlev-1)
        A_v_cells = jnp.where(m_half > 0.5, A_v_cells, 0.0)
        K_v_cells = jnp.where(m_half > 0.5, K_v_cells, 0.0)

        # CFL cap (SAME as KPP MPAS): A_v_max = 0.25·min(dz_k, dz_k+1)^2 /
        # cfl_cap_dt_s.  Applied to BOTH K_M and K_H (the explicit-diffusion
        # stability bound is the same for momentum and tracer viscosity).  Only
        # reduces K (>= 0 preserved); 0.25 for safety margin.
        _dt_phys = cfg.cfl_cap_dt_s
        _dz = z_coord.dz_ref            # (nlev,)
        _dz_min_half = jnp.minimum(_dz[:-1], _dz[1:])   # (nlev-1,)
        _Av_max = 0.25 * _dz_min_half ** 2 / _dt_phys   # (nlev-1,)
        A_v_cells = jnp.minimum(A_v_cells, _Av_max[None, :])
        K_v_cells = jnp.minimum(K_v_cells, _Av_max[None, :])

        # Sub-seafloor interface masking on partial cells: zero K at interfaces
        # below the deepest active full level (mirrors make_kpp_profiles_mpas).
        if hasattr(z_coord, 'bottom_level'):
            nlev_half = A_v_cells.shape[1]
            k_idx = jnp.arange(nlev_half, dtype=z_coord.bottom_level.dtype)
            bot_c = z_coord.bottom_level  # (nCells,)
            active_half_c = (k_idx[None, :] < bot_c[:, None]).astype(
                A_v_cells.dtype)
            A_v_cells = A_v_cells * active_half_c
            K_v_cells = K_v_cells * active_half_c

        return A_v_cells, K_v_cells

    return profiles_fn
