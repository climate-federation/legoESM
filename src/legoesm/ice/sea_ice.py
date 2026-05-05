"""Sea ice model: thermodynamics, dynamics, and multi-category ice.

Modes (controlled by ``SeaIceConfig``):
- **Slab** (``dynamics="none"``, ``n_categories=1``): Original thermodynamic
  slab with diagnostic free-drift velocity. Fully backward compatible.
- **Free drift** (``dynamics="free_drift"``): Heuristic linear-combination
  velocity (not a force-balance solver) with optional tracer advection.
- **EVP** (``dynamics="evp"``): Elastic-Viscous-Plastic rheology with
  subcycled momentum solver (Hunke & Dukowicz 1997).

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
from legoesm.ice.dynamics import evp_solver, free_drift_velocity
from legoesm.ice.transport import advect_ice_tracers
from legoesm.ice.itd import aggregate_state, linear_remap
from legoesm.coupler.surface_energy import surface_radiation_fluxes
from legoesm.ice.config import SeaIceConfig
from legoesm.ice.dynamics import evp_solver, free_drift_velocity
from legoesm.ice.transport import advect_ice_tracers
from legoesm.ice.itd import aggregate_state, linear_remap
from legoesm.ice.state import (
    SeaIceState,
    DynamicSeaIceState,
    dynamic_to_slab,
)
from legoesm.surface_albedo import ice_albedo as compute_ice_albedo


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
    # Validate: dynamics/transport requiring grid must have grid != None
    if config.dynamics == "evp" and grid is None:
        raise ValueError(
            "dynamics='evp' requires a grid argument. "
            "Pass grid=<CubedSphereGrid> to step_sea_ice()."
        )
    if config.transport == "advect" and grid is None:
        raise ValueError(
            "transport='advect' requires a grid argument. "
            "Pass grid=<CubedSphereGrid> to step_sea_ice()."
        )

    if config.dynamics == "none" and config.n_categories == 1:
        # Original slab path — fully backward compatible
        if isinstance(state, DynamicSeaIceState):
            state = dynamic_to_slab(state)
        return _step_slab(state, forcing, ocean_sst, ocean_u, ocean_v,
                          config, U_min, dt)
    else:
        return _step_dynamic(state, forcing, ocean_sst, ocean_u, ocean_v,
                             config, U_min, dt, grid)


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

    # ---------- Surface fluxes (MOST dispatch stays in slab path) ----------
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
            L_latent=constants.L_s,  # sublimation over ice, not evaporation
        )
    else:
        tau_x, tau_y, shflx, lhflx = simple_bulk_fluxes(
            forcing.u_lowest, forcing.v_lowest,
            forcing.T_lowest, forcing.q_lowest,
            T_ice, q_sfc, rho, wind_speed,
            config.Cd_ice, config.Ch_ice,
            L_latent=constants.L_s,  # sublimation over ice
        )

    # ---------- Thermodynamics (delegate to shared routine) ----------
    h_new, T_ice_new, conc_new = _thermo_single(
        h, T_ice, conc, forcing, ocean_sst, config, U_min, dt,
        shflx=shflx, lhflx=lhflx,
    )

    # ---------- Ice velocity (free drift, diagnostic) ----------
    u_ice = (config.drag_ocean * ocean_u
             + config.drag_atm * (config.rho_air_ref / config.rho_ice) * forcing.u_lowest)
    v_ice = (config.drag_ocean * ocean_v
             + config.drag_atm * (config.rho_air_ref / config.rho_ice) * forcing.v_lowest)

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
            rho_ice=config.rho_ice,
        )

    # ---- 2. Transport ----
    # ``advect_ice_tracers`` handles both 2D ``(6, n, n)`` and multi-category
    # 3D ``(6, n, n, n_cat)`` shapes natively, with a single 4D vector halo
    # exchange across all quantities and categories — the prior vmap-over-
    # categories pattern issued ``n_cat × 3`` halo exchanges per timestep.
    if config.transport == "advect" and grid is not None:
        h, conc, T_ice = advect_ice_tracers(
            h, conc, T_ice, u_ice, v_ice, grid, dt,
        )

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

        def _thermo_cat(h_k, T_k, conc_k):
            return _thermo_single(
                h_k, T_k, conc_k,
                forcing, ocean_sst, config, U_min, dt,
            )

        h, T_ice, conc = jax.vmap(
            _thermo_cat, in_axes=-1, out_axes=-1,
        )(h, T_ice, conc)

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
        h, T_ice, conc = _thermo_single(
            h, T_ice, conc, forcing, ocean_sst, config, U_min, dt,
        )

    # Re-aggregate for coupler response
    if h.ndim > 3:
        h_agg, T_agg, conc_agg = aggregate_state(h, T_ice, conc)
    else:
        h_agg, T_agg, conc_agg = h, T_ice, conc

    # ---- Build response ----
    response = _build_response(
        h_agg, T_agg, conc_agg, u_ice, v_ice,
        forcing, config, U_min,
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
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Thermodynamic update for a single thickness class.

    Parameters
    ----------
    shflx, lhflx : jnp.ndarray or None
        Pre-computed sensible and latent heat fluxes.  When *None*
        (the default), ``simple_bulk_fluxes`` is called internally.
        Pass pre-computed values when the caller uses a different
        bulk-flux scheme (e.g. MOST).
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
    dh_dt_open = freeze_flux_open / (config.rho_ice * config.L_f)
    dh_dt = jnp.where(ice_mask, dh_dt_ice, dh_dt_open)
    h_new = jnp.maximum(h + dt * dh_dt, 0.0)

    # Concentration evolution.
    #   Growth (CICE convention): areal concentration only increases
    #   from NEW-ICE FORMATION in open-water portions of the cell —
    #   ``dh_dt_open · (1 − A) / h_new_ice`` with new floes appearing
    #   at thickness ``h_new_ice``.  Existing-ice basal growth thickens
    #   the floe (h_new > h) but does NOT spread it laterally.  The
    #   prior formulation used ``max(dh_dt, 0)`` for both branches,
    #   letting existing floes grow in area when basal freezing
    #   dominated — unphysical for any ice-covered cell.
    #   Melt:   areal concentration decreases as floes shrink in area
    #   while their thickness stays roughly constant —
    #   ``dh_dt · A / h_eff`` (sign carries through; dh_dt < 0).
    dconc_growth = jnp.where(
        ice_mask,
        0.0,
        dh_dt_open * (1.0 - conc) / config.h_new_ice,
    )
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
) -> TileResponse:
    """Build coupler response from aggregated ice fields."""
    q_sfc = saturation_mixing_ratio_ice(T_ice, forcing.p_surface)

    if config.temp_dependent_albedo:
        alpha_ice = compute_ice_albedo(T_ice, config.ice_albedo)
    else:
        alpha_ice = jnp.full(h.shape, config.albedo_ice, dtype=h.dtype)

    # Direct ``ε σ T⁴ + (1-ε)·lw_down`` instead of the full
    # ``surface_radiation_fluxes`` call (which discards ``sw_net``
    # / ``lw_net``).  Same direct-expression rewrite used for
    # ``ocean_tile_response`` and ``two_layer_lake``.
    lw_up = (
        config.emissivity_ice * constants.sigma_sb * T_ice ** 4
        + (1.0 - config.emissivity_ice) * forcing.lw_down
    )

    # Recompute surface fluxes from aggregated state, honoring bulk_scheme
    wind_speed = jnp.sqrt(
        forcing.u_lowest ** 2 + forcing.v_lowest ** 2 + U_min ** 2
    )
    rho = forcing.rho_lowest

    if config.bulk_scheme in ("most", "coare3", "large_yeager"):
        tau_x, tau_y, shflx, lhflx, _ = compute_most_fluxes(
            forcing.u_lowest, forcing.v_lowest,
            forcing.T_lowest, forcing.q_lowest,
            T_ice, q_sfc, rho,
            z_ref=config.z_ref,
            z0_init=config.z0_ice,
            scheme=config.bulk_scheme,
            n_iter=config.bulk_n_iter,
            L_latent=constants.L_s,  # sublimation over ice
        )
    else:
        tau_x, tau_y, shflx, lhflx = simple_bulk_fluxes(
            forcing.u_lowest, forcing.v_lowest,
            forcing.T_lowest, forcing.q_lowest,
            T_ice, q_sfc, rho, wind_speed,
            config.Cd_ice, config.Ch_ice,
            L_latent=constants.L_s,  # sublimation over ice
        )

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
        # Multi-cat aggregate: freshwater, ocean-heat-extraction, and
        # ice→ocean stress are constructed in the multi-cat step
        # path (above); the aggregator just exposes zero placeholders
        # and lets the multi-cat path overwrite if needed.
        freshwater_flux=jnp.zeros_like(h),
        ocean_heat_extraction=jnp.zeros_like(h),
        ocean_stress_x=jnp.zeros_like(h),
        ocean_stress_y=jnp.zeros_like(h),
        # Sublimation mass flux from the aggregated ice surface.
        surface_mass_flux=lhflx / constants.L_s,
    )
