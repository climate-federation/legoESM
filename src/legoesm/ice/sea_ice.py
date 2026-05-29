"""Sea ice model: thermodynamics, dynamics, and multi-category ice.

Modes (controlled by ``SeaIceConfig``):
- **Slab** (``dynamics="none"``, ``n_categories=1``): Original thermodynamic
  slab with diagnostic free-drift velocity. Fully backward compatible.
- **Free drift** (``dynamics="free_drift"``): Heuristic linear-combination
  velocity (not a force-balance solver) with optional tracer advection.
- **EVP** (``dynamics="evp"``): Elastic-Viscous-Plastic rheology with
  subcycled momentum solver (Hunke & Dukowicz 1997).
- **mEVP** (``dynamics="mevp"``): Modified-EVP pseudo-time relaxation
  (Bouillon 2013 / Kimmritz 2015). Converges to the implicit VP
  solution without the EVP elastic CFL constraint.

Multi-category ice (``n_categories > 1``) uses a simplified category
transfer scheme to redistribute ice across thickness bins.  This is
*not* a full Lipscomb (2001) linear remapping.  Snow depth is not
tracked.

Thermodynamics:
    Surface energy balance determines T_ice.
    Conductive flux through ice: F_cond = k_ice * (T_freeze - T_ice) / (h + h_min)
    Growth/melt: dh/dt = (F_cond - F_ocean) / (rho_ice * L_f)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio_ice
from legoesm.coupler.bulk_flux import simple_bulk_fluxes, compute_most_fluxes
from legoesm.coupler.coupling_fields import AtmToSurface, TileResponse
from legoesm.ice.dynamics import evp_solver, mevp_solver, free_drift_velocity
from legoesm.ice.transport import advect_ice_tracers
from legoesm.ice.itd import (
    aggregate_state,
    linear_remap,
    lipscomb_2001_remap,
)
from legoesm.ice.snow import (
    accumulate_snowfall,
    combined_conductive_flux,
    consume_from_snow_then_ice,
    consume_sublimation_from_snow_then_ice,
    snow_ice_flooding,
)
from legoesm.ice.brine import update_salinity_and_salt_flux, PSU_TO_KG_PER_KG
from legoesm.ice.ridging import apply_ridging
from legoesm.ice.shortwave import compute_ice_sw
from legoesm.ice.ponds import step_ponds
from legoesm.coupler.surface_energy import surface_radiation_fluxes
from legoesm.ice.config import SeaIceConfig
from legoesm.ice.state import (
    SeaIceState,
    DynamicSeaIceState,
    dynamic_to_slab,
)
from legoesm.surface_albedo import ice_albedo as compute_ice_albedo


def _uses_new_physics(config: SeaIceConfig) -> bool:
    """True when any Tier-1/Tier-2 new physics gate is enabled."""
    return (
        config.snow.enabled
        or config.brine.enabled
        or config.ridging.enabled
        or config.ponds.enabled
        or config.shortwave_scheme != "constant"
        or config.itd_remap != "simple"
    )


# ==============================================================================
# Main entry point
# ==============================================================================

def step_sea_ice(
    state,
    forcing: AtmToSurface,
    ocean_sst: jnp.ndarray,
    ocean_u: jnp.ndarray,
    ocean_v: jnp.ndarray,
    config: SeaIceConfig,
    U_min: float,
    dt: float,
    grid=None,
):
    """Step the sea ice model forward by dt seconds.

    Dispatches to slab or dynamic model based on config.

    Parameters
    ----------
    state : SeaIceState or DynamicSeaIceState
        Current ice state.
    forcing : AtmToSurface
        Atmospheric forcing fields.
    ocean_sst : jnp.ndarray
        Ocean SST [K], shape (6, n, n).
    ocean_u, ocean_v : jnp.ndarray
        Ocean surface currents [m/s], shape (6, n, n).
    config : SeaIceConfig
        Sea ice model parameters.
    U_min : float
        Minimum wind speed floor [m/s].
    dt : float
        Time step [s].
    grid : CubedSphereGrid, optional
        Required when ``dynamics != "none"`` or ``transport != "none"``.

    Returns
    -------
    new_state : SeaIceState or DynamicSeaIceState
    response : TileResponse
    """
    # Validate: dynamics literal + grid requirement
    if config.dynamics not in ("none", "free_drift", "evp", "mevp"):
        raise ValueError(
            f"Unknown sea-ice dynamics scheme: {config.dynamics!r}. "
            "Expected one of: 'none', 'free_drift', 'evp', 'mevp'."
        )
    if config.dynamics in ("evp", "mevp") and grid is None:
        raise ValueError(
            f"dynamics={config.dynamics!r} requires a grid argument. "
            "Pass grid=<CubedSphereGrid> to step_sea_ice()."
        )
    if config.transport == "advect" and grid is None:
        raise ValueError(
            "transport='advect' requires a grid argument. "
            "Pass grid=<CubedSphereGrid> to step_sea_ice()."
        )

    if config.dynamics == "none" and config.n_categories == 1 and not _uses_new_physics(config):
        # Original slab path — fully backward compatible
        if isinstance(state, DynamicSeaIceState):
            state = dynamic_to_slab(state)
        return _step_slab(state, forcing, ocean_sst, ocean_u, ocean_v,
                          config, U_min, dt)
    elif _uses_new_physics(config):
        # Extended physics path: snow / brine / ridging / ponds /
        # delta-Eddington / Lipscomb 2001 remap.  Requires
        # DynamicSeaIceState carrier so the new state fields are
        # available.  Slab inputs are lifted; multi-category configs
        # must arrive with a state that already carries the category
        # axis — silently expanding a 3-D slab to 4-D would lose
        # the per-category distribution.
        if isinstance(state, SeaIceState):
            if config.n_categories > 1:
                raise ValueError(
                    f"step_sea_ice: new-physics path with n_categories="
                    f"{config.n_categories} requires a DynamicSeaIceState "
                    "with shape (..., n_categories).  Build one via "
                    "init_dynamic_ice_state(shape, n_categories=...) and "
                    "distribute_to_categories before calling step_sea_ice."
                )
            from legoesm.ice.state import slab_to_dynamic
            state = slab_to_dynamic(state)
        # Validate dynamic-state shape against the configured number
        # of categories so a stale 3-D state cannot silently run a
        # multi-category config (which would skip ITD remap / ridging).
        h_dyn_shape = state.h_ice.data.shape
        if config.n_categories > 1:
            if h_dyn_shape[-1] != config.n_categories or len(h_dyn_shape) < 4:
                raise ValueError(
                    f"step_sea_ice: DynamicSeaIceState.h_ice has shape "
                    f"{h_dyn_shape}, which is inconsistent with config."
                    f"n_categories={config.n_categories}.  Rebuild the "
                    "state with init_dynamic_ice_state(shape, "
                    "n_categories=...) and distribute_to_categories."
                )
        else:
            # n_categories == 1: refuse a trailing category axis so the
            # multi-cat branch in _step_dynamic_v2 (which keys off
            # ``h.ndim > 3``) is not accidentally activated.
            if len(h_dyn_shape) > 3:
                raise ValueError(
                    f"step_sea_ice: DynamicSeaIceState.h_ice has shape "
                    f"{h_dyn_shape}, which carries a trailing category "
                    f"axis, but config.n_categories=1.  Drop the trailing "
                    "axis or raise n_categories to match."
                )
        return _step_dynamic_v2(state, forcing, ocean_sst, ocean_u, ocean_v,
                                config, U_min, dt, grid)
    else:
        return _step_dynamic(state, forcing, ocean_sst, ocean_u, ocean_v,
                             config, U_min, dt, grid)


# ==============================================================================
# Shared bulk-flux dispatch
# ==============================================================================

def _bulk_flux_dispatch(
    T_ice: jnp.ndarray,
    forcing: AtmToSurface,
    config: SeaIceConfig,
    U_min: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ``(tau_x, tau_y, shflx, lhflx)`` from configured scheme.

    The slab path duplicates this dispatch inline; the multi-category
    thermodynamics path used to skip it and silently use
    ``simple_bulk_fluxes`` regardless of ``config.bulk_scheme``.  This
    helper centralises the choice so both paths and the diagnostic
    ``_build_response`` produce consistent values.
    """
    valid_schemes = ("constant", "most", "coare3", "large_yeager")
    if config.bulk_scheme not in valid_schemes:
        raise ValueError(
            f"Unknown sea-ice bulk_scheme {config.bulk_scheme!r}; "
            f"expected one of {valid_schemes}."
        )
    wind_speed = jnp.sqrt(
        forcing.u_lowest ** 2 + forcing.v_lowest ** 2 + U_min ** 2
    )
    rho = forcing.rho_lowest
    q_sfc = saturation_mixing_ratio_ice(T_ice, forcing.p_surface)
    if config.bulk_scheme in ("most", "coare3", "large_yeager"):
        tau_x, tau_y, shflx, lhflx, _ = compute_most_fluxes(
            forcing.u_lowest, forcing.v_lowest,
            forcing.T_lowest, forcing.q_lowest,
            T_ice, q_sfc, rho,
            z_ref=config.z_ref,
            z0_init=config.z0_ice,
            scheme=config.bulk_scheme,
            n_iter=config.bulk_n_iter,
            L_latent=constants.L_s,
        )
    else:
        tau_x, tau_y, shflx, lhflx = simple_bulk_fluxes(
            forcing.u_lowest, forcing.v_lowest,
            forcing.T_lowest, forcing.q_lowest,
            T_ice, q_sfc, rho, wind_speed,
            config.Cd_ice, config.Ch_ice,
            L_latent=constants.L_s,
        )
    return tau_x, tau_y, shflx, lhflx


# ==============================================================================
# Slab thermodynamics (original implementation)
# ==============================================================================

def _step_slab(
    state: SeaIceState,
    forcing: AtmToSurface,
    ocean_sst: jnp.ndarray,
    ocean_u: jnp.ndarray,
    ocean_v: jnp.ndarray,
    config: SeaIceConfig,
    U_min: float,
    dt: float,
) -> tuple[SeaIceState, TileResponse]:
    """Original slab sea ice model (backward compatible)."""
    h = state.h_ice.data
    T_ice = state.T_ice.data
    conc = state.concentration.data

    # ---------- Surface fluxes (configured scheme via shared dispatch) ----------
    tau_x, tau_y, shflx, lhflx = _bulk_flux_dispatch(T_ice, forcing, config, U_min)

    # ---------- Thermodynamics (delegate to shared routine) ----------
    h_new, T_ice_new, conc_new = _thermo_single(
        h, T_ice, conc, forcing, ocean_sst, config, U_min, dt,
        shflx=shflx, lhflx=lhflx,
    )

    # ---------- Ice velocity (free drift, diagnostic) ----------
    # Use the shared Zubov-style drag balance instead of the dimensionally
    # broken slab formula (iter-86): the previous inline expression
    # ``drag_ocean*U_w + (drag_atm*rho_air/rho_ice)*U_a`` produced ice
    # drift two orders of magnitude smaller than the physical Nansen /
    # Zubov estimate.
    u_ice, v_ice = free_drift_velocity(
        ocean_u, ocean_v,
        forcing.u_lowest, forcing.v_lowest,
        drag_ocean=config.drag_ocean,
        drag_atm=config.drag_atm,
        rho_air=config.rho_air_ref,
        rho_ocean=config.rho_ocean_ref,
    )

    new_state = SeaIceState(
        h_ice=state.h_ice.replace(data=h_new),
        T_ice=state.T_ice.replace(data=T_ice_new),
        concentration=state.concentration.replace(data=conc_new),
    )

    # ---------- Build response ----------
    # ``jnp.full`` is one ``Broadcast`` HLO op vs the
    # ``broadcast_to(jnp.array(scalar), ...)`` form.  Same micro-fix
    # as the loop-18/20 lake / coupler / land tile responses.
    _h_dtype = h.dtype
    if config.temp_dependent_albedo:
        alpha_ice = compute_ice_albedo(T_ice_new, config.ice_albedo)
    else:
        alpha_ice = jnp.full(h.shape, config.albedo_ice, dtype=_h_dtype)

    # Direct ``ε σ T⁴ + (1-ε)·lw_down`` instead of the full
    # ``surface_radiation_fluxes`` call (which discards ``sw_net``
    # / ``lw_net``) — same direct-expression rewrite as the
    # ``ocean_tile_response`` and ``two_layer_lake`` fixes.
    lw_up_new = (
        config.emissivity_ice * constants.sigma_sb * T_ice_new ** 4
        + (1.0 - config.emissivity_ice) * forcing.lw_down
    )
    q_sfc_new = saturation_mixing_ratio_ice(T_ice_new, forcing.p_surface)

    # Sea-ice → ocean freshwater flux: the ice mass that exchanged
    # with the ocean (basal/surface melt and open-water freezing),
    # NOT counting sublimation (already in the atmospheric lhflx
    # channel).  Positive = freshwater INTO ocean (melt > freeze);
    # negative = freshwater extracted to form ice.
    #
    # NOTE on bookkeeping (codex iter-25 follow-up): a ``d(h·conc)/dt``
    # cell-mean formulation was tried but the existing ``_thermo_single``
    # bookkeeping evolves ``h`` and ``conc`` semi-independently:
    # melt-retreat shrinks ``conc`` *and* ``h`` simultaneously
    # (double-counting volume loss); lead-freeze adds area but does
    # not update ``h``, so ``h·Δconc`` over-counts the new ice by
    # ``h/h_new_ice`` (≈20× for thick existing ice).  Until the slab
    # path adopts a strict CICE V=h·A state-variable convention
    # (deferred iter-22 #1), use the per-ice-area thickness rate which
    # at least integrates correctly when ``conc`` is constant.  Codex
    # iter-25 stop-time review flagged the dV-based fix as introducing
    # sublimation-mass leakage into the ocean.
    dh_dt_total = (h_new - h) / dt
    dh_dt_sublim = jnp.where(
        h > config.h_ice_min,
        -lhflx / (config.rho_ice * constants.L_s),
        0.0,
    )
    freshwater_to_ocean = -config.rho_ice * (dh_dt_total - dh_dt_sublim)

    # Heat extracted from the ocean by this tile.  Two contributions:
    #   1) basal melt/growth: F_ocean drawn from warm ocean to melt
    #      ice base (positive when SST > T_freeze_ocean).
    #   2) open-water freezing: latent heat L_f · rho_ice · dh_open
    #      removed from the ocean to form new ice.
    # The signs work out so both are positive when ocean LOSES energy
    # to the ice tile.  Ocean tile receives this back as a sink in
    # its surface heat budget.  Audit F8.
    ice_mask_init = h > config.h_ice_min
    F_ocean = jnp.where(
        ice_mask_init,
        config.ocean_heat_transfer_coeff * jnp.maximum(
            ocean_sst - config.T_freeze_ocean, 0.0,
        ),
        0.0,
    )
    # On previously-open-water cells, all of h_new is freshly frozen
    # ice at base.  L_f · rho_ice · h_new / dt is the heat extracted
    # from the ocean per unit area.
    dh_dt_freeze_open = jnp.where(~ice_mask_init, h_new / dt, 0.0)
    open_freeze_flux = config.rho_ice * config.L_f * dh_dt_freeze_open
    ocean_heat_extraction = F_ocean + open_freeze_flux

    # Sea-ice → ocean back-reaction stress (Newton's third law).
    # The ocean→ice drag tau_oi accelerates the ice; the ice exerts
    # −tau_oi on the ocean column.  Compute the proper Cauchy drag
    # using the ocean-ice drag coefficient (config.drag_ocean = C_oi)
    # and the relative velocity, weighted by ice concentration so
    # ice-free cells contribute no stress.  Audit F9.
    du_oi = ocean_u - u_ice
    dv_oi = ocean_v - v_ice
    speed_oi = jnp.sqrt(du_oi ** 2 + dv_oi ** 2 + 1e-10)
    tau_oi_x = config.rho_ocean_ref * config.drag_ocean * speed_oi * du_oi
    tau_oi_y = config.rho_ocean_ref * config.drag_ocean * speed_oi * dv_oi
    ocean_stress_x = -tau_oi_x * conc
    ocean_stress_y = -tau_oi_y * conc

    response = TileResponse(
        T_surface=T_ice_new,
        albedo=alpha_ice,
        emissivity=jnp.full(h.shape, config.emissivity_ice, dtype=_h_dtype),
        z0=jnp.full(h.shape, config.z0_ice, dtype=_h_dtype),
        q_surface=q_sfc_new,
        shflx=shflx,
        lhflx=lhflx,
        tau_x=tau_x,
        tau_y=tau_y,
        lw_up=lw_up_new,
        u_ocean_sfc=u_ice,
        v_ocean_sfc=v_ice,
        co2_flux=jnp.zeros_like(h),
        freshwater_flux=freshwater_to_ocean,
        ocean_heat_extraction=ocean_heat_extraction,
        ocean_stress_x=ocean_stress_x,
        ocean_stress_y=ocean_stress_y,
        # Sea-ice surface moisture exchange is sublimation/deposition
        # (L_s).  lhflx already used L_s in the bulk-flux call, so
        # surface_mass_flux = lhflx / L_s recovers the correct mass.
        surface_mass_flux=lhflx / constants.L_s,
        # Slab path has no bulk-ice salinity tracking — salt channel
        # is inert.  The brine-aware path uses the dynamic state and
        # populates ``salt_flux`` in ``_build_response``.
        salt_flux=jnp.zeros_like(h),
    )

    return new_state, response


# ==============================================================================
# Dynamic sea ice (EVP + optional multi-category)
# ==============================================================================

def _step_dynamic(
    state,
    forcing: AtmToSurface,
    ocean_sst: jnp.ndarray,
    ocean_u: jnp.ndarray,
    ocean_v: jnp.ndarray,
    config: SeaIceConfig,
    U_min: float,
    dt: float,
    grid=None,
):
    """Dynamic sea ice step: dynamics → transport → thermodynamics → ITD remap.

    Parameters
    ----------
    state : DynamicSeaIceState
    grid : CubedSphereGrid, required for dynamics/transport
    """
    h = state.h_ice.data
    T_ice = state.T_ice.data
    conc = state.concentration.data
    u_ice = state.u_ice.data
    v_ice = state.v_ice.data
    s11 = state.sigma_11.data
    s22 = state.sigma_22.data
    s12 = state.sigma_12.data

    # For multi-category: aggregate for coupler response and dynamics
    if h.ndim > 3:
        h_agg, T_agg, conc_agg = aggregate_state(h, T_ice, conc)
    else:
        h_agg, T_agg, conc_agg = h, T_ice, conc

    # ---- 1. Dynamics ----
    if config.dynamics == "evp" and grid is not None:
        u_ice, v_ice, s11, s22, s12 = evp_solver(
            u_ice, v_ice, s11, s22, s12,
            h_agg, conc_agg,
            forcing.u_lowest, forcing.v_lowest,
            ocean_u, ocean_v,
            grid, dt,
            N_evp=config.N_evp,
            e_yield=config.e_yield,
            P_star=config.P_star,
            C_strength=config.C_strength,
            T_evp=config.T_evp,
            Delta_min=config.Delta_min,
            rho_ice=config.rho_ice,
            rho_air=config.rho_air_ref,
            rho_ocean=config.rho_ocean_ref,
            C_ai=config.drag_atm,
            C_oi=config.drag_ocean,
            differentiable=config.differentiable_dynamics,
        )
    elif config.dynamics == "mevp" and grid is not None:
        u_ice, v_ice, s11, s22, s12 = mevp_solver(
            u_ice, v_ice, s11, s22, s12,
            h_agg, conc_agg,
            forcing.u_lowest, forcing.v_lowest,
            ocean_u, ocean_v,
            grid, dt,
            N_mevp=config.N_mevp,
            e_yield=config.e_yield,
            P_star=config.P_star,
            C_strength=config.C_strength,
            alpha_mevp=config.alpha_mevp,
            beta_mevp=config.beta_mevp,
            Delta_min=config.Delta_min,
            rho_ice=config.rho_ice,
            rho_air=config.rho_air_ref,
            rho_ocean=config.rho_ocean_ref,
            C_ai=config.drag_atm,
            C_oi=config.drag_ocean,
            differentiable=config.differentiable_dynamics,
        )
    elif config.dynamics == "free_drift":
        u_ice, v_ice = free_drift_velocity(
            ocean_u, ocean_v,
            forcing.u_lowest, forcing.v_lowest,
            drag_ocean=config.drag_ocean,
            drag_atm=config.drag_atm,
            rho_air=config.rho_air_ref,
            rho_ocean=config.rho_ocean_ref,
        )

    # ---- 2. Transport ----
    # ``advect_ice_tracers`` handles both 2D ``(6, n, n)`` and multi-category
    # 3D ``(6, n, n, n_cat)`` shapes natively, with a single 4D vector halo
    # exchange across all quantities and categories — the prior vmap-over-
    # categories pattern issued ``n_cat × 3`` halo exchanges per timestep.
    if config.transport == "advect" and grid is not None:
        h, conc, T_ice = advect_ice_tracers(
            h, conc, T_ice, u_ice, v_ice, grid, dt,
            n_subcycles=config.transport_subcycles,
        )

    # Snapshot of aggregated ice thickness AFTER transport but BEFORE
    # thermodynamics.  This is the reference for the ice → ocean
    # freshwater / heat exchange — only the thermodynamic ΔV is an
    # ocean exchange; horizontal transport conserves ice mass and
    # should not show up as melt/freezing in the coupler response.
    # Codex iter-3 finding #2.
    if h.ndim > 3:
        h_agg_post_transport, _, _ = aggregate_state(h, T_ice, conc)
    else:
        h_agg_post_transport = h

    # ---- 3. Thermodynamics (per category or single) ----
    if h.ndim > 3:
        # Multi-category: apply thermodynamics per category via vmap.
        # ``jax.vmap`` accepts negative ``in_axes`` / ``out_axes`` and
        # vmaps over the trailing category axis directly — skipping the
        # six ``jnp.moveaxis`` round-trips used by the prior pattern.
        # JAX still produces one batched kernel for ``_thermo_single``,
        # so the savings are layout/intermediate eliminations rather
        # than fewer kernel launches; the diff is one less buffer copy
        # per category-vmap invocation under XLA fusion.
        n_cat = h.shape[-1]
        h_old = h
        conc_old = conc

        # Open-water fraction at the aggregate level — the total lead
        # area available for refreezing this step.  Per-category
        # ``(1 - conc_k)`` sums to ``n_cat - sum_k conc_k`` and badly
        # over-counts when more than one category is occupied.  We pass
        # this aggregated value into ``_thermo_single`` so the lead
        # freeze is driven by the true lead area.  Codex iter-3 #3.
        open_water_agg = jnp.clip(1.0 - jnp.sum(conc_old, axis=-1), 0.0, 1.0)

        # Compute bulk fluxes per category with the configured scheme
        # BEFORE thermodynamics, so the dynamic multi-category path uses
        # the same bulk-flux closure that the slab path and diagnostic
        # ``_build_response`` use.  Previously ``_thermo_single`` was
        # called without ``shflx``/``lhflx``, silently falling back to
        # ``simple_bulk_fluxes`` regardless of ``config.bulk_scheme`` —
        # state and diagnostics could disagree.  (Codex finding #7.)
        # ``enable_lead_freeze`` is per-category: True only for cat 0
        # so the open-water → new-ice deposit happens once, into the
        # thinnest bin.  Cats 1+ still do their own basal / surface
        # melt-growth from F_cond / F_ocean / sublim.
        def _thermo_cat(h_k, T_k, conc_k, enable_lead_freeze_k):
            _, _, shflx_k, lhflx_k = _bulk_flux_dispatch(
                T_k, forcing, config, U_min,
            )
            return _thermo_single(
                h_k, T_k, conc_k,
                forcing, ocean_sst, config, U_min, dt,
                shflx=shflx_k, lhflx=lhflx_k,
                open_water_fraction=open_water_agg,
                enable_lead_freeze=bool(enable_lead_freeze_k),
            )

        # cat-0 deposits the lead freeze; cats 1+ skip it.  Implemented
        # by building separate forward calls (cat 0 vs the others) so
        # the static bool can flow through ``_thermo_single``.
        def _thermo_cat_with_lead(args):
            h_k, T_k, c_k = args
            return _thermo_cat(h_k, T_k, c_k, True)

        def _thermo_cat_no_lead(args):
            h_k, T_k, c_k = args
            return _thermo_cat(h_k, T_k, c_k, False)

        # Split: process cat 0 with lead-freeze, then vmap cats 1+
        # without.  ``jax.lax.map``-style splits keep the static bool
        # as a Python-level constant inside each branch.
        h_0, T_0, c_0 = _thermo_cat_with_lead(
            (h[..., 0], T_ice[..., 0], conc[..., 0]),
        )
        if n_cat > 1:
            h_rest, T_rest, c_rest = jax.vmap(
                _thermo_cat_no_lead, in_axes=0, out_axes=0,
            )((
                jnp.moveaxis(h[..., 1:], -1, 0),
                jnp.moveaxis(T_ice[..., 1:], -1, 0),
                jnp.moveaxis(conc[..., 1:], -1, 0),
            ))
            h = jnp.concatenate(
                [h_0[..., None], jnp.moveaxis(h_rest, 0, -1)], axis=-1,
            )
            T_ice = jnp.concatenate(
                [T_0[..., None], jnp.moveaxis(T_rest, 0, -1)], axis=-1,
            )
            conc = jnp.concatenate(
                [c_0[..., None], jnp.moveaxis(c_rest, 0, -1)], axis=-1,
            )
        else:
            h = h_0[..., None]
            T_ice = T_0[..., None]
            conc = c_0[..., None]

        # Open-water ice growth should only be deposited into category 0
        # (thinnest). Zero out new-ice growth in empty higher categories
        # to prevent spurious ice creation in all empty categories.
        was_empty = h_old <= 0.0  # (..., n_cat) True where category had no ice
        cat_mask = jnp.arange(n_cat) > 0  # False for cat 0, True for cats 1+
        suppress = was_empty & cat_mask  # suppress growth in empty non-zero cats
        h = jnp.where(suppress, h_old, h)
        conc = jnp.where(suppress, conc_old, conc)

        # ---- 4. ITD remap (including temperature for enthalpy conservation) ----
        h, conc, T_ice = linear_remap(h_old, conc_old, h, conc, n_cat, T_new=T_ice)
    else:
        # Single-category dynamic-path thermo.  Use the same bulk-flux
        # dispatch (MOST / COARE / Large-Yeager / simple_bulk) as the
        # slab path + multi-cat path + diagnostic ``_build_response``
        # so state and diagnostics use the same closure.  Earlier this
        # branch fell through to ``simple_bulk_fluxes`` regardless of
        # ``config.bulk_scheme``.  Codex iter-39 #1.
        _, _, shflx_sc, lhflx_sc = _bulk_flux_dispatch(
            T_ice, forcing, config, U_min,
        )
        h, T_ice, conc = _thermo_single(
            h, T_ice, conc, forcing, ocean_sst, config, U_min, dt,
            shflx=shflx_sc, lhflx=lhflx_sc,
        )

    # Re-aggregate for coupler response
    if h.ndim > 3:
        h_agg, T_agg, conc_agg = aggregate_state(h, T_ice, conc)
    else:
        h_agg, T_agg, conc_agg = h, T_ice, conc

    # ---- Build response ----
    # Use the post-transport snapshot as the reference for FW / heat
    # so transport (which conserves ice mass globally) doesn't appear
    # as melt/freezing in the ocean coupler response.
    response = _build_response(
        h_agg, T_agg, conc_agg, u_ice, v_ice,
        forcing, config, U_min,
        h_old=h_agg_post_transport,
        ocean_sst=ocean_sst,
        ocean_u=ocean_u,
        ocean_v=ocean_v,
        dt=dt,
    )

    # Legacy path: snow / brine / pond fields are pass-through (zero-
    # initialised in the input state when ``_uses_new_physics`` is False).
    new_state = DynamicSeaIceState(
        h_ice=state.h_ice.replace(data=h),
        T_ice=state.T_ice.replace(data=T_ice),
        concentration=state.concentration.replace(data=conc),
        u_ice=state.u_ice.replace(data=u_ice),
        v_ice=state.v_ice.replace(data=v_ice),
        sigma_11=state.sigma_11.replace(data=s11),
        sigma_22=state.sigma_22.replace(data=s22),
        sigma_12=state.sigma_12.replace(data=s12),
        h_snow=state.h_snow,
        S_ice=state.S_ice,
        pond_area=state.pond_area,
        pond_depth=state.pond_depth,
    )

    return new_state, response


# ==============================================================================
# Thermodynamics for a single thickness class
# ==============================================================================

def _thermo_single(
    h: jnp.ndarray,
    T_ice: jnp.ndarray,
    conc: jnp.ndarray,
    forcing: AtmToSurface,
    ocean_sst: jnp.ndarray,
    config: SeaIceConfig,
    U_min: float,
    dt: float,
    shflx: jnp.ndarray | None = None,
    lhflx: jnp.ndarray | None = None,
    open_water_fraction: jnp.ndarray | None = None,
    enable_lead_freeze: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Thermodynamic update for a single thickness class.

    Parameters
    ----------
    shflx, lhflx : jnp.ndarray or None
        Pre-computed sensible and latent heat fluxes.  When *None*
        (the default), ``simple_bulk_fluxes`` is called internally.
        Pass pre-computed values when the caller uses a different
        bulk-flux scheme (e.g. MOST).
    open_water_fraction : jnp.ndarray or None
        Aggregated ``(1 - sum_k conc_k)`` from the multi-category
        dispatch.  When supplied, replaces the per-category
        ``(1 - conc)`` in the concentration-growth term so the lead
        freeze is driven by *total* lead area rather than per-category
        lead area (which sums to > 1 across categories).  Default None
        preserves the slab/single-category behaviour.
    enable_lead_freeze : bool
        When False, suppress the open-water freezing (volume and
        concentration) contribution.  Multi-category callers set this
        True for category 0 only so the lead freeze isn't double-
        counted across categories.  Codex iter-3 finding #3.
    """
    ice_mask = h > 0.0
    h_eff = jnp.maximum(h, config.h_ice_min)

    # Bulk fluxes — use caller-supplied values when available.
    if shflx is None or lhflx is None:
        wind_speed = jnp.sqrt(
            forcing.u_lowest ** 2 + forcing.v_lowest ** 2 + U_min ** 2
        )
        rho = forcing.rho_lowest
        q_sfc = saturation_mixing_ratio_ice(T_ice, forcing.p_surface)
        _, _, shflx, lhflx = simple_bulk_fluxes(
            forcing.u_lowest, forcing.v_lowest,
            forcing.T_lowest, forcing.q_lowest,
            T_ice, q_sfc, rho, wind_speed,
            config.Cd_ice, config.Ch_ice,
            L_latent=constants.L_s,  # sublimation over ice
        )

    # Albedo: use ice albedo over ice, ocean albedo over open water
    if config.temp_dependent_albedo:
        alpha_ice = compute_ice_albedo(T_ice, config.ice_albedo)
    else:
        alpha_ice = jnp.full(h.shape, config.albedo_ice, dtype=h.dtype)
    alpha = jnp.where(ice_mask, alpha_ice, config.albedo_ocean)

    # Radiation
    sw_net, lw_net, lw_up = surface_radiation_fluxes(
        forcing.sw_down, forcing.lw_down, T_ice, alpha,
        config.emissivity_ice,
    )
    Q_sfc = sw_net + lw_net - shflx - lhflx

    # Conductive flux
    F_cond = jnp.where(
        ice_mask,
        config.k_ice * (config.T_freeze_ocean - T_ice) / h_eff,
        0.0,
    )

    # Temperature: F_cond = k*(T_base - T_sfc)/h is heat arriving at
    # the surface from the warm ice base, so it ADDS to the surface budget.
    skin_cap = config.rho_ice * config.c_ice * h_eff * 0.5
    dT_dt = (Q_sfc + F_cond) / skin_cap
    T_trial = T_ice + dt * dT_dt
    T_new = jnp.where(
        ice_mask,
        jnp.clip(T_trial, config.T_ice_min, config.T_freeze_ocean),
        jnp.full(T_ice.shape, config.T_freeze_ocean, dtype=T_ice.dtype),
    )

    # Surface melt: if T_trial exceeds freezing, the excess enthalpy melts
    # ice from the top instead of being discarded by the temperature clamp.
    excess_energy = skin_cap * jnp.maximum(
        T_trial - config.T_freeze_ocean, 0.0
    ) / dt  # [W/m²]
    dh_dt_surface_melt = -excess_energy / (config.rho_ice * config.L_f)

    # Growth/melt — turbulent ocean heat transfer (not conductive scaling)
    F_ocean = config.ocean_heat_transfer_coeff * jnp.maximum(
        ocean_sst - config.T_freeze_ocean, 0.0,
    )
    dh_dt_basal = (F_cond - F_ocean) / (config.rho_ice * config.L_f)

    # Sublimation mass loss: lhflx > 0 means moisture leaves the
    # surface into the atmosphere via L_s, so the equivalent ice mass
    # is removed from the column.  When lhflx < 0 (deposition), mass
    # is added.  Without this term the surface energy budget closes
    # but the ice mass budget is open: thin polar ice would grow
    # endlessly under sublimation, biased high by ~tens of cm/year.
    # (Coupler-conservation audit F7.)
    dh_dt_sublim = jnp.where(
        ice_mask,
        -lhflx / (config.rho_ice * constants.L_s),
        0.0,
    )

    # Combine surface and basal melt/growth + sublimation for existing ice
    dh_dt_ice = dh_dt_basal + dh_dt_surface_melt + dh_dt_sublim

    freeze_flux_open = jnp.maximum(-Q_sfc, 0.0)
    dh_dt_open_raw = freeze_flux_open / (config.rho_ice * config.L_f)
    # Multi-category: only deposit lead-freeze in category 0.  Cats
    # with ``enable_lead_freeze=False`` see ``dh_dt_open = 0`` so the
    # same open-water freeze does not fire per-category.
    dh_dt_open = dh_dt_open_raw if enable_lead_freeze else jnp.zeros_like(dh_dt_open_raw)
    dh_dt = jnp.where(ice_mask, dh_dt_ice, dh_dt_open)
    h_new = jnp.maximum(h + dt * dh_dt, 0.0)

    # Concentration evolution (CICE / Icepack ``add_new_ice`` convention).
    #
    # Growth: areal concentration only increases from NEW-ICE FORMATION
    # in OPEN-WATER portions of the cell.  The driver is ``dh_dt_open``
    # (the lead-freezing rate from a destabilizing surface flux), NOT
    # ``dh_dt_ice`` (vertical growth of existing floes by basal /
    # surface / sublimation processes).  This holds whether the cell
    # is fully open water (ice_mask=False) or partially ice-covered
    # (ice_mask=True with A<1).  In the partial-cover case
    # ``(1 − A) > 0`` represents the lead fraction that can refreeze;
    # the prior formulation suppressed this entire pathway by gating on
    # ``~ice_mask``, which under-grew concentration on every partial-
    # cover cell with positive surface freezing flux.  The earlier
    # ``max(dh_dt, 0)`` formulation was wrong in the opposite direction:
    # it let basal vertical growth spread floes laterally.
    #
    # Melt: concentration decreases as floes shrink in area while their
    # thickness stays roughly constant — ``dh_dt · A / h_eff`` (sign
    # carries through, dh_dt < 0 in melt).
    # ``lead_area`` is the lead area available for refreezing.  In
    # single-category mode this is ``(1 - conc)`` of the local cell;
    # in multi-category mode the caller supplies the aggregated
    # ``(1 - sum_k conc_k)`` to avoid summing more than 1 across cats.
    lead_area = (
        (1.0 - conc) if open_water_fraction is None else open_water_fraction
    )
    dconc_growth = dh_dt_open * lead_area / config.h_new_ice
    dconc_melt = jnp.minimum(dh_dt, 0.0) * conc / h_eff
    conc_new = jnp.clip(conc + dt * (dconc_growth + dconc_melt), 0.0, 1.0)

    return h_new, T_new, conc_new


# ==============================================================================
# Build TileResponse
# ==============================================================================

def _build_response(
    h: jnp.ndarray,
    T_ice: jnp.ndarray,
    conc: jnp.ndarray,
    u_ice: jnp.ndarray,
    v_ice: jnp.ndarray,
    forcing: AtmToSurface,
    config: SeaIceConfig,
    U_min: float,
    *,
    h_old: jnp.ndarray | None = None,
    ocean_sst: jnp.ndarray | None = None,
    ocean_u: jnp.ndarray | None = None,
    ocean_v: jnp.ndarray | None = None,
    dt: float | None = None,
) -> TileResponse:
    """Build coupler response from aggregated ice fields.

    The optional ``h_old``/``ocean_*``/``dt`` arguments enable the
    dynamic multi-category path to compute the freshwater flux, ocean
    heat extraction, and ocean stress feedbacks — the slab path
    computes these inline.  Without them the response carries zero
    placeholders (legacy behaviour) which silently breaks ice→ocean
    feedback.  See codex finding #8.
    """
    q_sfc = saturation_mixing_ratio_ice(T_ice, forcing.p_surface)

    if config.temp_dependent_albedo:
        alpha_ice = compute_ice_albedo(T_ice, config.ice_albedo)
    else:
        alpha_ice = jnp.full(h.shape, config.albedo_ice, dtype=h.dtype)

    lw_up = (
        config.emissivity_ice * constants.sigma_sb * T_ice ** 4
        + (1.0 - config.emissivity_ice) * forcing.lw_down
    )

    # Recompute surface fluxes from aggregated state via shared dispatch
    tau_x, tau_y, shflx, lhflx = _bulk_flux_dispatch(
        T_ice, forcing, config, U_min,
    )

    # Ice → ocean feedbacks.  When the dynamic path threads
    # ``h_old`` + ``ocean_*`` + ``dt`` we compute the per-ice-area
    # thickness-rate FW flux that the slab path produces inline;
    # otherwise expose zero placeholders.  See the matching slab-
    # path note on the deferred d(h·conc)/dt formulation.
    if h_old is not None and dt is not None:
        dh_dt_total = (h - h_old) / dt
        ice_mask_init = h_old > config.h_ice_min
        dh_dt_sublim = jnp.where(
            ice_mask_init,
            -lhflx / (config.rho_ice * constants.L_s),
            0.0,
        )
        freshwater_flux = -config.rho_ice * (dh_dt_total - dh_dt_sublim)
    else:
        freshwater_flux = jnp.zeros_like(h)

    if (
        h_old is not None and ocean_sst is not None and dt is not None
    ):
        ice_mask_init = h_old > config.h_ice_min
        F_ocean = jnp.where(
            ice_mask_init,
            config.ocean_heat_transfer_coeff
            * jnp.maximum(ocean_sst - config.T_freeze_ocean, 0.0),
            0.0,
        )
        dh_dt_freeze_open = jnp.where(~ice_mask_init, h / dt, 0.0)
        open_freeze_flux = config.rho_ice * config.L_f * dh_dt_freeze_open
        ocean_heat_extraction = F_ocean + open_freeze_flux
    else:
        ocean_heat_extraction = jnp.zeros_like(h)

    if ocean_u is not None and ocean_v is not None:
        du_oi = ocean_u - u_ice
        dv_oi = ocean_v - v_ice
        speed_oi = jnp.sqrt(du_oi ** 2 + dv_oi ** 2 + 1e-10)
        tau_oi_x = config.rho_ocean_ref * config.drag_ocean * speed_oi * du_oi
        tau_oi_y = config.rho_ocean_ref * config.drag_ocean * speed_oi * dv_oi
        ocean_stress_x = -tau_oi_x * conc
        ocean_stress_y = -tau_oi_y * conc
    else:
        ocean_stress_x = jnp.zeros_like(h)
        ocean_stress_y = jnp.zeros_like(h)

    return TileResponse(
        T_surface=T_ice,
        albedo=alpha_ice,
        emissivity=jnp.full(h.shape, config.emissivity_ice, dtype=h.dtype),
        z0=jnp.full(h.shape, config.z0_ice, dtype=h.dtype),
        q_surface=q_sfc,
        shflx=shflx,
        lhflx=lhflx,
        tau_x=tau_x,
        tau_y=tau_y,
        lw_up=lw_up,
        u_ocean_sfc=u_ice,
        v_ocean_sfc=v_ice,
        co2_flux=jnp.zeros_like(h),
        freshwater_flux=freshwater_flux,
        ocean_heat_extraction=ocean_heat_extraction,
        ocean_stress_x=ocean_stress_x,
        ocean_stress_y=ocean_stress_y,
        # Sublimation mass flux from the aggregated ice surface.
        surface_mass_flux=lhflx / constants.L_s,
        # Salt-flux placeholder — populated by the brine-aware path
        # in ``_step_dynamic`` when ``config.brine.enabled``.  The
        # zero default keeps the freshwater convention for legacy
        # configs.
        salt_flux=jnp.zeros_like(h),
    )


# ==============================================================================
# Extended-physics dynamic sea ice step (Tier 1 + Tier 2)
# ==============================================================================

def _closing_rate_from_velocity(
    u_ice: jnp.ndarray,
    v_ice: jnp.ndarray,
    grid,
    cap: float,
) -> jnp.ndarray:
    """Net convergence rate ``max(0, −div(u_ice))`` from the velocity field.

    Returns zero everywhere when the grid does not have a usable
    horizontal divergence operator (lat-lon / MPAS routes).  This
    is a conservative fallback — ridging simply does not fire
    until the relevant grid-specific operator is plumbed in.
    """
    try:
        from legoesm.ice.rheology import strain_rates
        eps_11, eps_22, _eps_12 = strain_rates(u_ice, v_ice, grid)
        div = eps_11 + eps_22
        return jnp.clip(-div, 0.0, cap)
    except Exception:
        return jnp.zeros_like(u_ice)


def _thermo_v2(
    h: jnp.ndarray,
    T_ice: jnp.ndarray,
    conc: jnp.ndarray,
    h_snow: jnp.ndarray,
    S_ice: jnp.ndarray,
    pond_area: jnp.ndarray,
    pond_depth: jnp.ndarray,
    forcing: AtmToSurface,
    ocean_sst: jnp.ndarray,
    config: SeaIceConfig,
    U_min: float,
    dt: float,
    *,
    open_water_fraction: jnp.ndarray | None = None,
    enable_lead_freeze: bool = True,
) -> dict:
    """Extended thermodynamics: snow + brine + SW scheme + ponds aware.

    Returns a dictionary of post-step per-category fields plus
    diagnostics needed by the caller to build the coupler response
    (delta volumes for the salt budget, SW penetration, etc.).
    """
    ice_mask = h > 0.0
    h_eff = jnp.maximum(h, config.h_ice_min)

    # Bulk fluxes via shared dispatch (consistent across paths).
    tau_x, tau_y, shflx, lhflx = _bulk_flux_dispatch(T_ice, forcing, config, U_min)

    # 1. Snow accumulation from precipitation (positive on ice cells).
    # ``config.snow.enabled`` is a static Python bool — safe to branch
    # on under jit; the ``has_precipitation`` array flag is folded
    # into ``accumulate_snowfall`` via array masking so JIT/scan can
    # trace the branch without lowering a tracer-boolean conversion.
    if config.snow.enabled:
        precip_snow_gated = forcing.precip_snow * jnp.asarray(
            forcing.has_precipitation, dtype=forcing.precip_snow.dtype,
        )
        h_snow, snow_to_ocean_open = accumulate_snowfall(
            h_snow, precip_snow_gated, ice_mask, dt,
            rho_snow=config.snow.rho_snow,
        )
    else:
        snow_to_ocean_open = jnp.zeros_like(h)

    # 2. Shortwave: alpha + absorbed + penetrated.
    sw_result = compute_ice_sw(
        forcing.sw_down, T_ice, h, h_snow,
        pond_area, pond_depth,
        scheme=config.shortwave_scheme,
        albedo_const=config.albedo_ice,
    )
    alpha = sw_result.albedo_eff
    sw_absorbed = sw_result.sw_absorbed_surface
    sw_penetrated = sw_result.sw_penetrated

    # Longwave net.
    lw_up = (
        config.emissivity_ice * constants.sigma_sb * T_ice ** 4
        + (1.0 - config.emissivity_ice) * forcing.lw_down
    )
    lw_net = config.emissivity_ice * forcing.lw_down - lw_up
    # Q_sfc is the net heat available at the surface skin.
    Q_sfc = sw_absorbed + lw_net - shflx - lhflx

    # 3. Conductive flux through snow+ice.
    F_cond = jnp.where(
        ice_mask,
        combined_conductive_flux(
            T_base=jnp.full_like(T_ice, config.T_freeze_ocean),
            T_surface=T_ice,
            h_ice=h,
            h_snow=h_snow,
            k_ice=config.k_ice,
            k_snow=config.snow.k_snow,
            h_ice_min=config.h_ice_min,
            h_snow_min=config.snow.h_snow_min,
        ),
        0.0,
    )

    # 4. Surface energy balance to update T_ice.
    skin_cap = (
        config.rho_ice * config.c_ice * h_eff * 0.5
        + config.snow.rho_snow * config.snow.c_snow * jnp.maximum(h_snow, 0.0)
    )
    skin_cap = jnp.maximum(skin_cap, 1.0)
    dT_dt = (Q_sfc + F_cond) / skin_cap
    T_trial = T_ice + dt * dT_dt
    T_new = jnp.where(
        ice_mask,
        jnp.clip(T_trial, config.T_ice_min, config.T_freeze_ocean),
        jnp.full(T_ice.shape, config.T_freeze_ocean, dtype=T_ice.dtype),
    )

    # 5. Excess surface energy → melt: snow first then ice.
    excess_W = jnp.maximum(T_trial - config.T_freeze_ocean, 0.0) * skin_cap / dt
    energy_for_melt = excess_W * dt  # [J/m²]
    h_snow_after_melt, h_after_melt, snow_melt_m, ice_melt_m = consume_from_snow_then_ice(
        energy_for_melt, h_snow, h, config.snow.rho_snow, config.rho_ice, config.L_f,
    )

    # 6. Basal exchange (ocean side).  After clipping ``h`` at zero
    #    we recover the *actual* basal mass change so the FW + salt
    #    budgets don't ship more ice than the column held.
    F_ocean = config.ocean_heat_transfer_coeff * jnp.maximum(
        ocean_sst - config.T_freeze_ocean, 0.0,
    )
    dh_dt_basal = (F_cond - F_ocean) / (config.rho_ice * config.L_f)
    h_after_basal = jnp.maximum(h_after_melt + dt * dh_dt_basal, 0.0)
    basal_delta = h_after_basal - h_after_melt
    basal_growth_m = jnp.maximum(basal_delta, 0.0)
    basal_melt_m = jnp.maximum(-basal_delta, 0.0)

    # 7. Sublimation: snow first, then ice.  ``lhflx > 0`` → mass loss.
    sublim_mass_per_area = (lhflx / constants.L_s) * dt
    h_snow_after_sub, h_after_sub, snow_sub_m, ice_sub_m = consume_sublimation_from_snow_then_ice(
        sublim_mass_per_area,
        h_snow_after_melt,
        h_after_basal,
        rho_snow=config.snow.rho_snow,
        rho_ice=config.rho_ice,
        sublim_partition=config.snow.sublim_partition,
    )

    # 8. Lead freezing (open-water freeze) — only when enabled (cat 0
    #    in multi-cat).  This deposits new ice at thickness
    #    ``h_new_ice`` into the lead area.  Partial-cover cells DO
    #    contribute: ``(1 − conc)`` of the cell is open water and can
    #    refreeze.
    freeze_flux_open = jnp.maximum(-Q_sfc, 0.0)
    lead_area = jnp.clip(
        (1.0 - conc) if open_water_fraction is None else open_water_fraction,
        0.0, 1.0,
    )
    # ``dh_dt_open`` is the local lead-ice thickening rate per unit
    # lead area, driven by the destabilising surface flux.
    dh_dt_open_raw = freeze_flux_open / (config.rho_ice * config.L_f)
    dh_dt_open = dh_dt_open_raw if enable_lead_freeze else jnp.zeros_like(dh_dt_open_raw)
    # Per-grid-cell new volume contributed by lead-freezing this step.
    delta_V_lead_freeze = jnp.maximum(dh_dt_open * lead_area * dt, 0.0)

    # Update concentration and mean thickness so the lead-freeze ice
    # volume is captured even in partial-cover cells:
    #   V_after = V_before + delta_V_lead_freeze
    #   conc_after = conc_before + dconc_growth
    #   h_after   = V_after / max(conc_after, eps)
    V_before_lead = h_after_sub * conc
    V_after_lead = V_before_lead + delta_V_lead_freeze

    dconc_growth = dh_dt_open * lead_area / config.h_new_ice
    dh_dt_melt_for_conc = (
        - (
            snow_melt_m * config.snow.rho_snow
            + ice_melt_m * config.rho_ice
            + basal_melt_m * config.rho_ice
            + ice_sub_m * config.rho_ice
        )
        / (config.rho_ice * dt)
    )
    dconc_melt = jnp.minimum(dh_dt_melt_for_conc, 0.0) * conc / jnp.maximum(h, config.h_ice_min)
    conc_new = jnp.clip(conc + dt * (dconc_growth + dconc_melt), 0.0, 1.0)

    h_new = jnp.where(
        conc_new > 1e-12,
        V_after_lead / jnp.maximum(conc_new, 1e-12),
        0.0,
    )

    # 10. Snow-ice flooding (white-ice formation).
    if config.snow.enabled and config.snow.flooding:
        h_ice_after_flood, h_snow_after_flood, h_si_formed = snow_ice_flooding(
            h_new, h_snow_after_sub,
            rho_ice=config.rho_ice,
            rho_snow=config.snow.rho_snow,
            rho_ocean=config.rho_ocean_ref,
        )
        h_new = h_ice_after_flood
        h_snow_new = h_snow_after_flood
        delta_V_white_ice = h_si_formed
    else:
        h_snow_new = h_snow_after_sub
        delta_V_white_ice = jnp.zeros_like(h_new)

    # 11. Pond update — ordered BEFORE the brine budget so refrozen
    #     pond water is included in ``V_new_cat`` when computing the
    #     salt-mass change.  Pond refreeze is liquid water (m of
    #     H₂O); converting to ice-thickness via ``rho_water/rho_ice``
    #     gives mass-equivalent ice gain.
    if config.ponds.enabled:
        rain_m_liquid = jnp.maximum(
            (forcing.precip_total - forcing.precip_snow) * dt / constants.rho_water,
            0.0,
        )
        # Surface-melt thickness is in *layer* meters (snow, ice).
        # Ponds collect liquid-water equivalent — convert with the
        # density ratio before passing to ``step_ponds``.
        melt_water_m_liquid = (
            snow_melt_m * config.snow.rho_snow / constants.rho_water
            + ice_melt_m * config.rho_ice / constants.rho_water
        )
        # Snow above ice blocks pond formation.  ``step_ponds`` uses
        # the same gate internally; we mirror it here to know how
        # much melt + rain water was actually captured (so we can
        # avoid double-routing the captured water to the ocean via
        # ``fw_from_melt``).
        snow_blocks = h_snow_new > config.ponds.snow_block_threshold
        pond_captured_m_liquid = jnp.where(
            ice_mask & (~snow_blocks),
            melt_water_m_liquid + rain_m_liquid,
            0.0,
        )
        # Of the captured water, the melt portion is the one that
        # would otherwise have appeared in ``fw_from_melt``.  Split
        # by source-water fraction.
        total_input_safe = jnp.maximum(
            melt_water_m_liquid + rain_m_liquid, 1.0e-30,
        )
        melt_fraction_of_capture = melt_water_m_liquid / total_input_safe
        pond_captured_melt_m_liquid = (
            pond_captured_m_liquid * melt_fraction_of_capture
        )

        pond_area_new, pond_depth_new, drain_to_ocean_m, refreeze_m = step_ponds(
            pond_area, pond_depth,
            melt_water_m=melt_water_m_liquid,
            rain_water_m=rain_m_liquid,
            ice_mask=ice_mask,
            h_snow=h_snow_new,
            T_air=forcing.T_lowest,
            dt=dt,
            drainage_timescale=config.ponds.drainage_timescale,
            refreeze_threshold=config.ponds.refreeze_threshold,
            pond_to_ice_max_area=config.ponds.pond_to_ice_max_area,
            depth_to_area_ratio=config.ponds.depth_to_area_ratio,
            snow_block_threshold=config.ponds.snow_block_threshold,
        )
        # Convert refrozen pond water [m of liquid] to equivalent ice
        # thickness [m of ice] using density.  Ice gain is
        # mass-neutral with the ocean: the water came from earlier
        # melt that we've already debited from the ice column.
        refreeze_ice_m = refreeze_m * constants.rho_water / config.rho_ice
        h_new = h_new + refreeze_ice_m
        pond_drain_to_ocean_kg_s = drain_to_ocean_m * constants.rho_water / dt
    else:
        pond_area_new = pond_area
        pond_depth_new = pond_depth
        pond_drain_to_ocean_kg_s = jnp.zeros_like(h_new)
        pond_captured_melt_m_liquid = jnp.zeros_like(h_new)

    # 12. Brine + salt budget update — uses the *final* per-cat
    #     volume (including white-ice and refrozen-pond contributions)
    #     so the salt budget is fully closed.
    if config.brine.enabled:
        V_old = h * conc
        V_new_cat = h_new * conc_new
        brine = update_salinity_and_salt_flux(
            S_ice_old=S_ice,
            V_ice_old=V_old,
            V_ice_new=V_new_cat,
            delta_V_lead_freeze=delta_V_lead_freeze,
            delta_V_white_ice=delta_V_white_ice * conc_new,
            rho_ice=config.rho_ice,
            dt=dt,
            S_lead_ice=config.brine.S_ice_new,
            S_white_ice=0.5 * config.brine.S_ocean_ref,
            S_ice_min=config.brine.S_ice_min,
            S_ice_max=config.brine.S_ice_max,
        )
        S_ice_new = brine.S_ice_new
        salt_flux_to_ocean = brine.salt_flux_to_ocean
    else:
        S_ice_new = S_ice
        salt_flux_to_ocean = jnp.zeros_like(h_new)

    # 13. Freshwater flux to ocean [kg/m²/s, PER-GRID-CELL-AREA].
    #
    #     Per-category-area contributions (melt of snow / ice /
    #     basal) are multiplied by ``conc`` to convert to per-cell
    #     mass loss.  Per-cell contributions (lead-freeze
    #     removal, open-water snowfall, pond drainage) flow through
    #     as-is.  When ponds are enabled, the meltwater captured
    #     into ponds is subtracted from the ocean FW channel — that
    #     mass leaves the ice column into ponds, not into the ocean
    #     (the ocean eventually receives it through ``pond_drain``).
    fw_from_melt_per_cell = (
        (snow_melt_m * config.snow.rho_snow
         + ice_melt_m * config.rho_ice
         + basal_melt_m * config.rho_ice)
        * conc / dt
    )
    if config.ponds.enabled:
        # ``pond_captured_melt_m_liquid`` and ``pond_drain_to_ocean_kg_s``
        # are per-ice-area quantities (pond state is stored relative to
        # ice area).  Multiply by ``conc`` to convert to per-grid-cell-
        # area before mixing with the per-cell fluxes here.
        fw_from_melt_per_cell = fw_from_melt_per_cell - (
            pond_captured_melt_m_liquid * constants.rho_water * conc / dt
        )
        pond_drain_per_cell = pond_drain_to_ocean_kg_s * conc
    else:
        pond_drain_per_cell = pond_drain_to_ocean_kg_s  # zero array
    fw_from_lead_freeze_per_cell = -delta_V_lead_freeze * config.rho_ice / dt
    freshwater_to_ocean = (
        fw_from_melt_per_cell
        + snow_to_ocean_open
        + pond_drain_per_cell
        + fw_from_lead_freeze_per_cell
    )

    # Ocean heat extraction [W/m², PER-GRID-CELL-AREA].
    #   * Turbulent basal heat (``F_ocean``) is per-cat-area
    #     → multiply by ``conc``.
    #   * Open-water lead freeze (``delta_V_lead_freeze`` already
    #     per-cell) releases latent heat L_f back to / removes from
    #     the ocean — sign convention: positive = ocean LOSES energy
    #     to the ice tile.  The lead-freeze contribution removes L_f
    #     per kg of ice formed from the ocean.
    ocean_heat_extraction = (
        F_ocean * conc
        + delta_V_lead_freeze * config.rho_ice * config.L_f / dt
    )

    # Pack diagnostics for the caller.
    return {
        "h": h_new,
        "T": T_new,
        "conc": conc_new,
        "h_snow": h_snow_new,
        "S_ice": S_ice_new,
        "pond_area": pond_area_new,
        "pond_depth": pond_depth_new,
        "alpha": alpha,
        "lw_up": lw_up,
        "shflx": shflx,
        "lhflx": lhflx,
        "tau_x": tau_x,
        "tau_y": tau_y,
        "freshwater_to_ocean": freshwater_to_ocean,
        "salt_flux_to_ocean": salt_flux_to_ocean,
        "ocean_heat_extraction": ocean_heat_extraction,
        "sw_penetrated_to_ocean": sw_penetrated,
        "delta_V_lead_freeze": delta_V_lead_freeze,
        "delta_V_white_ice": delta_V_white_ice,
    }


def _step_dynamic_v2(
    state: DynamicSeaIceState,
    forcing: AtmToSurface,
    ocean_sst: jnp.ndarray,
    ocean_u: jnp.ndarray,
    ocean_v: jnp.ndarray,
    config: SeaIceConfig,
    U_min: float,
    dt: float,
    grid=None,
) -> tuple[DynamicSeaIceState, TileResponse]:
    """Extended dynamic sea-ice step with Tier-1 + Tier-2 physics.

    Sequence: dynamics → transport (incl. snow + salt + pond
    tracers) → thermo (snow / brine / SW / ponds) → ITD remap
    (Lipscomb 2001) → ridging → coupler response.
    """
    h = state.h_ice.data
    T_ice = state.T_ice.data
    conc = state.concentration.data
    u_ice = state.u_ice.data
    v_ice = state.v_ice.data
    s11 = state.sigma_11.data
    s22 = state.sigma_22.data
    s12 = state.sigma_12.data
    h_snow = state.h_snow.data
    S_ice = state.S_ice.data
    pond_area = state.pond_area.data
    pond_depth = state.pond_depth.data

    # Pre-step concentration snapshot — used when normalising
    # per-grid-cell fluxes to per-ice-tile fluxes so a cell that
    # FULLY melts during the step (post-step conc = 0, pre-step conc
    # > 0) still divides by a finite denominator.  Note that the
    # downstream tile blender re-multiplies by post-step f_ice, so
    # terminal-melt cells will still lose their final pulse — see
    # README in this module + the pre-existing slab-path TODO.  The
    # max(pre, post) denominator at least keeps the divide finite and
    # consistent across pre→post.
    conc_pre = conc

    is_multicat = h.ndim > 3

    if is_multicat:
        h_agg, T_agg, conc_agg = aggregate_state(h, T_ice, conc)
    else:
        h_agg, T_agg, conc_agg = h, T_ice, conc

    # ---- 1. Dynamics ----
    if config.dynamics == "evp" and grid is not None:
        u_ice, v_ice, s11, s22, s12 = evp_solver(
            u_ice, v_ice, s11, s22, s12,
            h_agg, conc_agg,
            forcing.u_lowest, forcing.v_lowest,
            ocean_u, ocean_v,
            grid, dt,
            N_evp=config.N_evp, e_yield=config.e_yield,
            P_star=config.P_star, C_strength=config.C_strength,
            T_evp=config.T_evp, Delta_min=config.Delta_min,
            rho_ice=config.rho_ice, rho_air=config.rho_air_ref,
            rho_ocean=config.rho_ocean_ref,
            C_ai=config.drag_atm, C_oi=config.drag_ocean,
            differentiable=config.differentiable_dynamics,
        )
    elif config.dynamics == "mevp" and grid is not None:
        u_ice, v_ice, s11, s22, s12 = mevp_solver(
            u_ice, v_ice, s11, s22, s12,
            h_agg, conc_agg,
            forcing.u_lowest, forcing.v_lowest,
            ocean_u, ocean_v,
            grid, dt,
            N_mevp=config.N_mevp, e_yield=config.e_yield,
            P_star=config.P_star, C_strength=config.C_strength,
            alpha_mevp=config.alpha_mevp, beta_mevp=config.beta_mevp,
            Delta_min=config.Delta_min,
            rho_ice=config.rho_ice, rho_air=config.rho_air_ref,
            rho_ocean=config.rho_ocean_ref,
            C_ai=config.drag_atm, C_oi=config.drag_ocean,
            differentiable=config.differentiable_dynamics,
        )
    elif config.dynamics == "free_drift":
        u_ice, v_ice = free_drift_velocity(
            ocean_u, ocean_v,
            forcing.u_lowest, forcing.v_lowest,
            drag_ocean=config.drag_ocean, drag_atm=config.drag_atm,
            rho_air=config.rho_air_ref, rho_ocean=config.rho_ocean_ref,
        )

    # ---- 2. Transport ----
    # Advect ice tracers (h, conc, T) AND the new physics tracers
    # (h_snow, S_ice, pond_area, pond_depth) at the same velocity.
    if config.transport == "advect" and grid is not None:
        h, conc, T_ice = advect_ice_tracers(
            h, conc, T_ice, u_ice, v_ice, grid, dt,
            n_subcycles=config.transport_subcycles,
        )
        # Snow + S_ice + pond fields are advected as passive tracers
        # tied to the ice.  Reuse the same PPM transport — the
        # routine handles both 2D and multi-cat 3D inputs.
        h_snow, _, _ = advect_ice_tracers(
            h_snow, conc, T_ice, u_ice, v_ice, grid, dt,
            n_subcycles=config.transport_subcycles,
        )
        S_ice_temp, _, _ = advect_ice_tracers(
            S_ice, conc, T_ice, u_ice, v_ice, grid, dt,
            n_subcycles=config.transport_subcycles,
        )
        S_ice = S_ice_temp
        if config.ponds.enabled:
            pond_area_t, _, _ = advect_ice_tracers(
                pond_area, conc, T_ice, u_ice, v_ice, grid, dt,
                n_subcycles=config.transport_subcycles,
            )
            pond_area = jnp.clip(pond_area_t, 0.0, 1.0)
            pond_depth_t, _, _ = advect_ice_tracers(
                pond_depth, conc, T_ice, u_ice, v_ice, grid, dt,
                n_subcycles=config.transport_subcycles,
            )
            pond_depth = jnp.maximum(pond_depth_t, 0.0)

    # Snapshot post-transport for coupler bookkeeping.
    if is_multicat:
        h_agg_pt, _, _ = aggregate_state(h, T_ice, conc)
    else:
        h_agg_pt = h

    # ---- 3. Pre-thermo per-category state (save for ITD remap) ----
    h_old = h
    conc_old = conc

    # ---- 4. Thermodynamics per category ----
    if is_multicat:
        n_cat = h.shape[-1]
        open_water_agg = jnp.clip(1.0 - jnp.sum(conc, axis=-1), 0.0, 1.0)

        h_list = []
        T_list = []
        conc_list = []
        snow_list = []
        S_list = []
        pa_list = []
        pd_list = []
        salt_flux_list = []
        fw_flux_list = []
        ocean_heat_list = []
        sw_pen_list = []
        delta_lead_list = []
        delta_white_list = []
        shflx_aggsum_components = []
        lhflx_aggsum_components = []
        tau_x_components = []
        tau_y_components = []
        for k in range(n_cat):
            result = _thermo_v2(
                h[..., k], T_ice[..., k], conc[..., k],
                h_snow[..., k], S_ice[..., k],
                pond_area[..., k], pond_depth[..., k],
                forcing, ocean_sst, config, U_min, dt,
                open_water_fraction=open_water_agg,
                enable_lead_freeze=(k == 0),
            )
            h_list.append(result["h"])
            T_list.append(result["T"])
            conc_list.append(result["conc"])
            snow_list.append(result["h_snow"])
            S_list.append(result["S_ice"])
            pa_list.append(result["pond_area"])
            pd_list.append(result["pond_depth"])
            salt_flux_list.append(result["salt_flux_to_ocean"])
            fw_flux_list.append(result["freshwater_to_ocean"])
            ocean_heat_list.append(result["ocean_heat_extraction"])
            sw_pen_list.append(result["sw_penetrated_to_ocean"])
            delta_lead_list.append(result["delta_V_lead_freeze"])
            delta_white_list.append(result["delta_V_white_ice"])
            shflx_aggsum_components.append(result["shflx"] * result["conc"])
            lhflx_aggsum_components.append(result["lhflx"] * result["conc"])
            tau_x_components.append(result["tau_x"] * result["conc"])
            tau_y_components.append(result["tau_y"] * result["conc"])

        h = jnp.stack(h_list, axis=-1)
        T_ice = jnp.stack(T_list, axis=-1)
        conc = jnp.stack(conc_list, axis=-1)
        h_snow = jnp.stack(snow_list, axis=-1)
        S_ice = jnp.stack(S_list, axis=-1)
        pond_area = jnp.stack(pa_list, axis=-1)
        pond_depth = jnp.stack(pd_list, axis=-1)

        # Aggregate per-cell fluxes returned by ``_thermo_v2`` to a
        # single per-ice-tile flux for the coupler.  ``_thermo_v2``
        # already multiplied the per-cat-area melt / heat terms by
        # ``conc`` so summing across cats gives the per-grid-cell
        # total; dividing by the total ice fraction converts to the
        # per-ice-tile basis that ``blend_tiles`` re-multiplies by
        # ``f_ice``.  Use ``max(pre, post)`` so terminal-melt cells
        # still divide by a finite denominator (see ``conc_pre`` note
        # at the start of this function).
        sum_conc_pre = jnp.sum(conc_pre, axis=-1, keepdims=False)
        sum_conc_post = jnp.sum(conc, axis=-1, keepdims=False)
        sum_conc_safe = jnp.maximum(
            jnp.maximum(sum_conc_pre, sum_conc_post), 1e-30,
        )
        fw_per_cat = jnp.stack(fw_flux_list, axis=-1)
        heat_per_cat = jnp.stack(ocean_heat_list, axis=-1)
        fw_flux_total = jnp.sum(fw_per_cat, axis=-1) / sum_conc_safe
        ocean_heat_total = jnp.sum(heat_per_cat, axis=-1) / sum_conc_safe
        salt_flux_total = (
            jnp.sum(jnp.stack(salt_flux_list, axis=-1), axis=-1) / sum_conc_safe
        )
        # SW penetrated to the ocean is per-cat-area (per ice-tile
        # fraction within the cell); weight by ``conc`` for per-cell
        # then normalise.
        sw_pen_total = jnp.sum(
            jnp.stack(sw_pen_list, axis=-1) * conc, axis=-1,
        ) / sum_conc_safe
        ocean_heat_total = ocean_heat_total - sw_pen_total

        # Atmosphere-coupler fluxes (already concentration-weighted
        # because the per-cat values were stored multiplied by conc).
        shflx_resp = jnp.sum(jnp.stack(shflx_aggsum_components, axis=-1), axis=-1) / sum_conc_safe
        lhflx_resp = jnp.sum(jnp.stack(lhflx_aggsum_components, axis=-1), axis=-1) / sum_conc_safe
        tau_x_resp = jnp.sum(jnp.stack(tau_x_components, axis=-1), axis=-1) / sum_conc_safe
        tau_y_resp = jnp.sum(jnp.stack(tau_y_components, axis=-1), axis=-1) / sum_conc_safe

    else:
        result = _thermo_v2(
            h, T_ice, conc, h_snow, S_ice, pond_area, pond_depth,
            forcing, ocean_sst, config, U_min, dt,
        )
        h = result["h"]
        T_ice = result["T"]
        conc = result["conc"]
        h_snow = result["h_snow"]
        S_ice = result["S_ice"]
        pond_area = result["pond_area"]
        pond_depth = result["pond_depth"]
        # All fluxes returned by ``_thermo_v2`` are PER-GRID-CELL-
        # AREA (per-cat-area terms were already multiplied by
        # ``conc`` inside the kernel).  Convert to per-ice-tile-area
        # by dividing by the response ice fraction so ``blend_tiles``
        # can re-multiply by ``f_ice``.  ``response_conc`` =
        # max(pre, post) keeps the denominator finite when a cell
        # melts out within the step (terminal-melt edge case).
        response_conc = jnp.maximum(jnp.maximum(conc_pre, conc), 1e-30)
        salt_flux_total = result["salt_flux_to_ocean"] / response_conc
        fw_flux_total = result["freshwater_to_ocean"] / response_conc
        # SW penetration is per-cat-area (only valid where ice
        # exists); use the same response area for consistency.
        ocean_heat_total = (
            result["ocean_heat_extraction"]
            - result["sw_penetrated_to_ocean"] * conc
        ) / response_conc
        shflx_resp = result["shflx"]
        lhflx_resp = result["lhflx"]
        tau_x_resp = result["tau_x"]
        tau_y_resp = result["tau_y"]

    # ---- 5. ITD remap ----
    if is_multicat:
        n_cat = h.shape[-1]
        # Volume of snow per cat (h_snow * a) for remap.
        V_snow_cat = h_snow * conc
        if config.itd_remap == "lipscomb2001":
            remap = lipscomb_2001_remap(
                h_old=h_old, a_old=conc_old,
                h_new=h, a_new=conc,
                n_cat=n_cat, dt=dt,
                T_new=T_ice, S_new=S_ice,
                V_snow_new=V_snow_cat,
                V_pond_new=pond_area * pond_depth * conc,
            )
            h = remap["h"]
            conc = remap["a"]
            T_ice = remap["T"]
            S_ice = remap["S"]
            # Recover h_snow from V_snow and conc.
            conc_safe = jnp.maximum(conc, 1e-12)
            h_snow = jnp.where(conc > 1e-12, remap["V_snow"] / conc_safe, 0.0)
            V_pond = remap["V_pond"]
            # pond_area*pond_depth*a = V_pond  →  recover assuming
            # pond_area unchanged (legacy fallback when no explicit
            # pond remap is configured).
            pond_volume_cell = jnp.where(conc > 1e-12, V_pond / conc_safe, 0.0)
            # Split pond_volume into depth (assume depth follows existing
            # depth_to_area ratio).
            pond_depth = jnp.sqrt(
                jnp.maximum(config.ponds.depth_to_area_ratio * pond_volume_cell, 0.0),
            )
            pond_area = jnp.where(
                pond_depth > 1e-6,
                jnp.clip(pond_volume_cell / jnp.maximum(pond_depth, 1e-6), 0.0,
                         config.ponds.pond_to_ice_max_area),
                0.0,
            )
        else:
            # Fall back to legacy linear_remap for h, a, T only.
            h, conc, T_ice = linear_remap(
                h_old, conc_old, h, conc, n_cat, T_new=T_ice,
            )

    # ---- 6. Ridging ----
    if config.ridging.enabled and is_multicat and grid is not None:
        closing_rate = _closing_rate_from_velocity(
            u_ice, v_ice, grid, cap=config.ridging.closing_rate_max,
        )
        ridge_result = apply_ridging(
            conc, h, h_snow * conc, S_ice, closing_rate,
            n_cat=h.shape[-1], dt=dt,
            e_star=config.ridging.e_star,
            mu_rdg=config.ridging.mu_rdg,
            H_star=config.ridging.H_star,
            snow_fraction_retained=config.ridging.snow_fraction_retained,
        )
        conc = ridge_result["a"]
        h = ridge_result["h"]
        conc_safe = jnp.maximum(conc, 1e-12)
        h_snow = jnp.where(conc > 1e-12, ridge_result["V_snow"] / conc_safe, 0.0)
        S_ice = ridge_result["S_ice"]
        # Snow shed by ridging → ocean as freshwater (per-cat sum is
        # already collapsed to the cell value inside apply_ridging).
        fw_flux_total = fw_flux_total + ridge_result["snow_to_ocean"]

    # ---- 7. Build response ----
    if is_multicat:
        h_agg, T_agg, conc_agg = aggregate_state(h, T_ice, conc)
    else:
        h_agg, T_agg, conc_agg = h, T_ice, conc

    q_sfc = saturation_mixing_ratio_ice(T_agg, forcing.p_surface)
    lw_up_total = (
        config.emissivity_ice * constants.sigma_sb * T_agg ** 4
        + (1.0 - config.emissivity_ice) * forcing.lw_down
    )
    # Aggregate albedo (use compute_ice_sw on aggregated state for
    # diagnostic — matches what the atmosphere will see).
    sw_agg = compute_ice_sw(
        forcing.sw_down, T_agg, h_agg,
        jnp.sum(h_snow * conc, axis=-1) / jnp.maximum(conc_agg, 1e-12) if is_multicat else h_snow,
        jnp.sum(pond_area * conc, axis=-1) / jnp.maximum(conc_agg, 1e-12) if is_multicat else pond_area,
        jnp.sum(pond_depth * conc, axis=-1) / jnp.maximum(conc_agg, 1e-12) if is_multicat else pond_depth,
        scheme=config.shortwave_scheme,
        albedo_const=config.albedo_ice,
    )
    alpha_resp = sw_agg.albedo_eff

    # Ice → ocean back-reaction stress.
    du_oi = ocean_u - u_ice
    dv_oi = ocean_v - v_ice
    speed_oi = jnp.sqrt(du_oi ** 2 + dv_oi ** 2 + 1e-10)
    tau_oi_x = config.rho_ocean_ref * config.drag_ocean * speed_oi * du_oi
    tau_oi_y = config.rho_ocean_ref * config.drag_ocean * speed_oi * dv_oi
    ocean_stress_x = -tau_oi_x * conc_agg
    ocean_stress_y = -tau_oi_y * conc_agg

    response = TileResponse(
        T_surface=T_agg,
        albedo=alpha_resp,
        emissivity=jnp.full(h_agg.shape, config.emissivity_ice, dtype=h_agg.dtype),
        z0=jnp.full(h_agg.shape, config.z0_ice, dtype=h_agg.dtype),
        q_surface=q_sfc,
        shflx=shflx_resp,
        lhflx=lhflx_resp,
        tau_x=tau_x_resp,
        tau_y=tau_y_resp,
        lw_up=lw_up_total,
        u_ocean_sfc=u_ice,
        v_ocean_sfc=v_ice,
        co2_flux=jnp.zeros_like(h_agg),
        freshwater_flux=fw_flux_total,
        ocean_heat_extraction=ocean_heat_total,
        ocean_stress_x=ocean_stress_x,
        ocean_stress_y=ocean_stress_y,
        surface_mass_flux=lhflx_resp / constants.L_s,
        salt_flux=salt_flux_total,
    )

    new_state = DynamicSeaIceState(
        h_ice=state.h_ice.replace(data=h),
        T_ice=state.T_ice.replace(data=T_ice),
        concentration=state.concentration.replace(data=conc),
        u_ice=state.u_ice.replace(data=u_ice),
        v_ice=state.v_ice.replace(data=v_ice),
        sigma_11=state.sigma_11.replace(data=s11),
        sigma_22=state.sigma_22.replace(data=s22),
        sigma_12=state.sigma_12.replace(data=s12),
        h_snow=state.h_snow.replace(data=h_snow),
        S_ice=state.S_ice.replace(data=S_ice),
        pond_area=state.pond_area.replace(data=pond_area),
        pond_depth=state.pond_depth.replace(data=pond_depth),
    )
    return new_state, response
