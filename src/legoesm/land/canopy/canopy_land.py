"""Canopy energy balance land model — main step function.

Implements step_canopy_land(), a legoESM land component that uses the
DifferBESS-style two-leaf canopy energy balance coupled to legoESM's existing
multi-layer soil thermal and hydraulic solvers.

Physics sequence each time step:
  1.  SW decomposition: sw_down → PAR/NIR/UV (direct/diffuse)
  2.  Root-zone soil moisture stress → fStress_soil, fStress_vcmax
  3.  Canopy aerodynamics (z0m, displacement) from hc × PFT ratios
  4.  Two-leaf shortwave RT → APAR_Sun/Sh, ASW_Sun/Sh/Soil, Vcmax25 profiles
  5.  Newton-Raphson canopy closure (50 iters) over 7 state variables:
        [Tf_Sun, Tf_Sh, Ci_Sun, Ci_Sh, Ts, Tc, q_c]
  6.  Aggregate canopy fluxes: LE_tot, H_tot, GPP, Rn, G
  7.  Snow budget (reuse snow_budget.py)
  8.  Richards equation for soil hydraulics (reuse richards.py)
  9.  Soil thermal diffusion (reuse soil_thermal.py), G as surface BC
 10.  Build TileResponse for coupler

All operations are vectorised over (ncol,) columns.  The canopy closure runs
independently per column via jax.lax.scan (not MPI-partitioned at this stage).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio, saturation_mixing_ratio_ice
from legoesm.coupler.coupling_fields import AtmToSurface, TileResponse
from legoesm.land.carbon.config import CarbonState
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.snow_budget import update_snow
from legoesm.land.state import MultiLayerLandState
from legoesm.land.soil_grid import make_soil_grid
from legoesm.land.richards import solve_richards
from legoesm.land.soil_thermal import solve_soil_thermal
from legoesm.surface_albedo import land_albedo as compute_land_albedo

from legoesm.land.canopy.config import CanopyLandConfig, CanopyLandParams
from legoesm.land.canopy.radiative_transfer import split_sw_components, canopy_shortwave_rt
from legoesm.land.canopy.stability import (
    compute_aerodynamics,
    sat_specific_humidity,
)
from legoesm.land.canopy.solver import (
    CanopyForcingBundle,
    solve_canopy_closure,
    _canopy_forward,
)
from legoesm.land.canopy.energy_balance import saturation_specific_humidity


class CanopyDiagnostics(NamedTuple):
    """Per-column surface energy budget diagnostics (all [W m-2] unless noted).

    The canopy+soil system should satisfy ``Rn_int ≈ LE_tot + H_tot + G``
    exactly (internal closure); any drift is a model bug.  ``Rn_ext`` is
    recomputed from forcing and is the quantity a downstream diagnostic
    would see — the difference ``Rn_ext - Rn_int`` indicates RT / LW-accounting
    drift.
    """
    Rn_ext: jnp.ndarray          # (1-α) SW_down + ε (LW_down - σ T_surface^4)
    Rn_int: jnp.ndarray          # Rn_Sun + Rn_Sh + Rn_Soil (solver-internal)
    SW_net: jnp.ndarray          # (1-α) SW_down
    LW_net: jnp.ndarray          # ε (LW_down - σ T_surface^4)
    LE_tot: jnp.ndarray
    H_tot: jnp.ndarray
    G: jnp.ndarray               # positive = into soil
    LE_canopy: jnp.ndarray       # LE_Sun + LE_Sh
    LE_soil: jnp.ndarray
    H_canopy: jnp.ndarray
    H_soil: jnp.ndarray
    Rn_canopy: jnp.ndarray       # Rn_Sun + Rn_Sh (leaves, no G)
    Rn_soil: jnp.ndarray         # ASW_Soil + ALW_Soil
    residual_int: jnp.ndarray    # Rn_int - (LE_tot + H_tot + G)
    residual_ext: jnp.ndarray    # Rn_ext - (LE_tot + H_tot + G)
    GPP: jnp.ndarray             # [gC m-2 s-1]
    fSun: jnp.ndarray            # sunlit canopy fraction [0-1]
    n_iters: jnp.ndarray         # Newton-Raphson iterations used
    Tf_Sun: jnp.ndarray          # converged sunlit leaf T [K]
    Tf_Sh: jnp.ndarray           # converged shaded leaf T [K]
    Ts_solve: jnp.ndarray        # converged soil skin T from closure [K]
    T_surface: jnp.ndarray       # weighted emission T fed to coupler [K]


def _get(lp, name: str, fallback):
    if lp is None:
        return fallback
    return getattr(lp, name, fallback)


def _step_canopy_land_full(
    state: MultiLayerLandState,
    forcing: AtmToSurface,
    config: CanopyLandConfig,
    U_min: float,
    dt: float,
    lat: jnp.ndarray | None = None,
    carbon_state: CarbonState | None = None,
    doy: float = 0.0,
    land_params: CanopyLandParams | None = None,
) -> tuple[MultiLayerLandState, TileResponse, CarbonState | None, CanopyDiagnostics]:
    """Internal full-return step: advances state AND returns surface-budget diagnostics.

    Callers should use :func:`step_canopy_land` (3-tuple) or
    :func:`step_canopy_land_with_diagnostics` (4-tuple) instead.
    """
    cc = config.canopy
    mc = config.multilayer

    lp = land_params
    grid = make_soil_grid(mc.soil_grid)

    # ---- Unpack state ----
    T_soil = state.T_soil    # (ncol, n_layers)
    psi    = state.psi_soil  # (ncol, n_layers)
    theta  = state.theta_soil
    snow   = state.snow_depth
    snow_age = state.snow_age
    ncol   = T_soil.shape[0]

    # ---- Land surface parameters ----
    # Fall back to MultiLayerLandConfig scalars when lp is None
    emissivity = _get(lp, "emissivity", mc.emissivity_land)
    # Canopy-specific parameters: require lp or use scalar defaults
    LAI        = _get(lp, "LAI",      jnp.full(ncol, 1.5))
    hc         = _get(lp, "hc",       jnp.full(ncol, 5.0))
    fC4        = _get(lp, "fC4",      jnp.zeros(ncol))
    FNonVeg    = _get(lp, "FNonVeg",  jnp.zeros(ncol))
    CI         = _get(lp, "CI",       jnp.full(ncol, 0.75))
    kn         = _get(lp, "kn",       jnp.full(ncol, 0.3))
    Vc3_leaf   = _get(lp, "Vcmax25_C3_leaf", jnp.full(ncol, 60.0))
    Vc4_leaf   = _get(lp, "Vcmax25_C4_leaf", jnp.full(ncol, 40.0))
    m_C3       = _get(lp, "m_C3",     jnp.full(ncol, 9.0))
    m_C4       = _get(lp, "m_C4",     jnp.full(ncol, 4.0))
    b0_C3      = _get(lp, "b0_C3",    jnp.full(ncol, 0.01))
    b0_C4      = _get(lp, "b0_C4",    jnp.full(ncol, 0.04))
    alf        = _get(lp, "alf",      jnp.full(ncol, 0.3))
    TgC        = _get(lp, "TgC",      forcing.T_lowest - 273.15)
    ALB_VIS    = _get(lp, "ALB_VIS",  jnp.full(ncol, 0.1))
    ALB_NIR    = _get(lp, "ALB_NIR",  jnp.full(ncol, 0.2))
    rz0m       = _get(lp, "rz0m",     jnp.full(ncol, 0.055))
    rd         = _get(lp, "rd",       jnp.full(ncol, 0.67))

    theta_wp = _get(lp, "theta_wp", mc.theta_wp)
    theta_fc = _get(lp, "theta_fc", mc.theta_fc)
    root_depth = _get(lp, "root_depth", mc.root_depth)
    theta_r  = mc.hydraulics.theta_r

    # ---- Wind speed with minimum floor ----
    wind_speed = jnp.sqrt(forcing.u_lowest**2 + forcing.v_lowest**2 + U_min**2)
    wind_dir_x = forcing.u_lowest / jnp.maximum(wind_speed, 1e-6)
    wind_dir_y = forcing.v_lowest / jnp.maximum(wind_speed, 1e-6)

    # ---- Root distribution and soil moisture stress ----
    z_centers = grid.z_node  # (n_layers,) depth [m]
    if lp is not None and hasattr(lp, "root_depth") and lp.root_depth.ndim > 0:
        root_frac = jnp.exp(-z_centers[None, :] / root_depth[:, None])
        root_frac = root_frac / jnp.sum(root_frac, axis=-1, keepdims=True)
        beta_root = jnp.clip(
            (theta - theta_wp[:, None]) / (theta_fc[:, None] - theta_wp[:, None] + 1e-10),
            0.0, 1.0)
    else:
        root_frac = jnp.exp(-z_centers / root_depth)
        root_frac = root_frac / jnp.sum(root_frac)
        beta_root = jnp.clip(
            (theta - theta_wp) / (theta_fc - theta_wp + 1e-10), 0.0, 1.0)

    w_frac_rz = jnp.clip(jnp.sum(root_frac[None, :] * beta_root, axis=-1), 0.0, 1.0)
    fStress_soil  = w_frac_rz         # soil evaporation stress [0–1]
    fStress_vcmax = w_frac_rz         # Vcmax downregulation [0–1]

    # Apply Vcmax downregulation
    Vc3_leaf_stressed = Vc3_leaf * fStress_vcmax
    Vc4_leaf_stressed = Vc4_leaf * fStress_vcmax

    # Apply stomatal slope stress (same factor for both m and b0)
    m_eff  = m_C3  * fStress_vcmax  # stress also reduces gs slope
    b0_eff = b0_C3 * fStress_vcmax  # and intercept
    # weighted average for mixed C3/C4 canopy
    m_mix  = (1.0 - fC4) * m_eff  + fC4 * m_C4  * fStress_vcmax
    b0_mix = (1.0 - fC4) * b0_eff + fC4 * b0_C4 * fStress_vcmax

    # ---- Aerodynamics ----
    z0m, displa = compute_aerodynamics(hc, LAI, rz0m, rd)
    # Reference height must lie above the displacement height + roughness so
    # that log((z_ref - displa)/z0m) is finite for tall canopies.  Lift z_ref
    # above (displa + 2*z0m) when needed — matches DifferBESS's implicit
    # assumption that forcing is measured above the canopy.
    z_ref_base = jnp.full(ncol, mc.z_ref)
    z_ref = jnp.maximum(z_ref_base, displa + 10.0 * z0m + 2.0)

    # ---- SW decomposition ----
    cos_zenith = forcing.cos_zenith
    SZA = jnp.degrees(jnp.arccos(jnp.clip(cos_zenith, 0.0, 1.0)))
    PAR_dir, PAR_diff, NIR_dir, NIR_diff, UV = split_sw_components(
        forcing.sw_down, cos_zenith)

    # ---- Shortwave RT ----
    sw_rt = canopy_shortwave_rt(
        PAR_dir, PAR_diff, NIR_dir, NIR_diff, UV,
        SZA, LAI, CI, ALB_VIS, ALB_NIR,
        Vc3_leaf_stressed, Vc4_leaf_stressed, kn, FNonVeg)

    # ---- Thermodynamic properties (from lowest model level) ----
    Ta   = forcing.T_lowest
    Ps   = forcing.p_surface
    q_atm = forcing.q_lowest
    rhoa  = forcing.rho_lowest
    Tv_atm = Ta * (1.0 + 0.61 * q_atm)
    lam  = constants.L_v   # latent heat [J kg-1]
    Cp   = 1004.0          # specific heat [J kg-1 K-1]
    Ca   = forcing.co2_ppmv  # CO2 [μmol mol-1]

    # ---- Initial canopy state guess for Newton ----
    Ts_old   = T_soil[:, 0]                        # start-of-step skin T
    q_s_init = sat_specific_humidity(Ts_old, Ps)
    q_c_init = 0.5 * (q_s_init + q_atm)
    chi = 0.7 - 0.3 * fC4  # initial Ci/Ca ratio (0.7 C3, 0.4 C4)
    Ci_init = Ca * chi

    # 6-var Newton state: [Tf_Sun, Tf_Sh, Ci_Sun, Ci_Sh, Tc, q_c]
    initial_state = jnp.stack([
        Ta, Ta, Ci_init, Ci_init, Ta, q_c_init], axis=-1)  # (ncol, 6)

    # Build CanopyForcingBundle — broadcast scalar config values to (ncol,)
    def _bcast(v):
        if hasattr(v, "shape") and v.shape == (ncol,):
            return v
        return jnp.broadcast_to(jnp.asarray(v), (ncol,))

    def _build_bundle(Ts_bc):
        """Build a CanopyForcingBundle for a given prescribed skin T."""
        return CanopyForcingBundle(
            LAI=LAI, SZA=SZA, La=forcing.lw_down,
            epsf=_bcast(cc.epsf), epss=_bcast(cc.epss),
            fSun=sw_rt.fSun,
            APAR_Sun=sw_rt.APAR_Sun, APAR_Sh=sw_rt.APAR_Sh,
            Vcmax25_Sun=sw_rt.Vcmax25_C3Sun, Vcmax25_Sh=sw_rt.Vcmax25_C3Sh,
            Vcmax25_C4Sun=sw_rt.Vcmax25_C4Sun, Vcmax25_C4Sh=sw_rt.Vcmax25_C4Sh,
            ASW_Sun=sw_rt.ASW_Sun, ASW_Sh=sw_rt.ASW_Sh, ASW_Soil=sw_rt.ASW_Soil,
            Ts_bc=Ts_bc,
            Ca=Ca, Ps=Ps, Ta=Ta,
            lam=_bcast(lam), Cp=_bcast(Cp), rhoa=rhoa, Tv_atm=Tv_atm, q_atm=q_atm,
            m=m_mix, b0=b0_mix, alf=alf, TgC=TgC,
            fC4=fC4, fStress_soil=fStress_soil,
            ur=wind_speed, CI=CI, z0m=z0m, displa=displa, z0=z_ref,
        )

    def _solve_one_col(x0, bun):
        return solve_canopy_closure(x0, bun, cc)

    def _fwd_one_col(xf, bun):
        return _canopy_forward(xf, bun, cc.LE_module, cc.stomatal_model,
                               cc.use_ta_for_photosynthesis)

    # ---- Outer Picard loop: canopy closure ↔ soil thermal solver ----
    # The canopy turbulent fluxes (LE, H) respond to Ts on sub-minute
    # timescales, while the soil top layer has a ~1-hour thermal time
    # constant.  A single explicit pass (solve canopy with Ts_old, then
    # update soil with resulting G) has feedback gain up to ~7 for wet
    # moist soils with small aerodynamic resistance (LE sensitivity
    # ``dLE/dTs ≈ 200 W/m²/K`` dominates), leading to step-to-step
    # oscillation between "hot" and "cold" states — we observed this
    # as alternating ``LE ≈ 1000 / G = −500`` and ``LE ≈ 200 / G = +500``
    # for C4 savanna at peak sun.
    #
    # Picard iteration: re-solve the canopy at an updated ``Ts_bc``
    # between passes, with under-relaxation on the update.  The
    # stability condition for simple Picard under-relaxation is
    # ``|ω · G_Picard| < 1`` where G_Picard ≈ |dG/dTs| · dt/C_top ≈ 7.
    # Using ω = 0.15 gives effective gain 1.04 → barely stable;
    # 6 iterations compound the damping so the final residual is
    # ``(1 − ω · G)^6 · ε₀ ≈ 0.95^6 ε₀ ≈ 0.74 ε₀``.  For stiffer
    # cases (very wet soil, small resistance) this is still enough
    # to damp the oscillation to below the safety clamp.
    n_picard = 6
    omega = 0.15
    Ts_bc_k = Ts_old  # start from beginning-of-step skin T

    # ---- Pre-Picard preliminaries that don't depend on Ts ----
    # Snow / Richards setup that uses LE_tot will run AFTER Picard
    # converges.  Here we just need forcings that enter the bundle.

    for _picard_iter in range(n_picard):
        bundles_k = _build_bundle(Ts_bc_k)
        x_final, n_iters = jax.vmap(_solve_one_col)(initial_state, bundles_k)
        fluxes_per_col = jax.vmap(_fwd_one_col)(x_final, bundles_k)

        # Diagnose G and tentatively advance soil thermal to update Ts_bc.
        # Clamp to physical bounds so a non-convergent iter cannot
        # corrupt the update.
        G_k = jnp.clip(fluxes_per_col["G"], -500.0, 700.0)
        T_soil_tent = solve_soil_thermal(
            T_soil, theta, grid, mc.hydraulics, mc.thermal, G_k, dt)
        Ts_thermal = T_soil_tent[:, 0]
        # Under-relaxed update — the relaxation factor < 1 damps the
        # inner feedback loop (see module-level note above for the
        # stability analysis).
        Ts_bc_k = (1.0 - omega) * Ts_bc_k + omega * Ts_thermal

    # After Picard: x_final, fluxes_per_col, bundles_k reflect the
    # converged fast-canopy state paired with a slow-soil Ts_bc_k that
    # agrees (to O(ω^3) = O(0.125)) with what the thermal solver will
    # produce below.

    # Converged state — 6-var layout [Tf_Sun, Tf_Sh, Ci_Sun, Ci_Sh, Tc, q_c]
    Tf_Sun = x_final[:, 0]
    Tf_Sh  = x_final[:, 1]
    Ts_cvg = Ts_bc_k   # skin T used by the final canopy closure

    # ---- Aggregate fluxes ----
    fSun    = sw_rt.fSun
    LE_Sun  = fluxes_per_col["LE_Sun"]
    LE_Sh   = fluxes_per_col["LE_Sh"]
    LE_Soil = fluxes_per_col["LE_Soil"]
    H_Sun   = fluxes_per_col["H_Sun"]
    H_Sh    = fluxes_per_col["H_Sh"]
    H_Soil  = fluxes_per_col["H_Soil"]
    An_Sun  = fluxes_per_col["An_Sun"]
    An_Sh   = fluxes_per_col["An_Sh"]
    G       = fluxes_per_col["G"]
    Rsoil   = fluxes_per_col["Rsoil"]
    gs_Sun  = fluxes_per_col["gs_Sun"]
    gs_Sh   = fluxes_per_col["gs_Sh"]
    ustar   = fluxes_per_col["ustar"]

    # Rb_Sun/Rb_Sh are already scaled by LAI*fSun and LAI*(1-fSun), so the
    # per-component LE/H/An returned by the solver are already the full
    # sunlit / shaded canopy integrated fluxes — straight sum, no area
    # weighting (matches DifferBESS CarbonWaterFluxes.py line 387).
    LE_tot  = LE_Sun + LE_Sh + LE_Soil
    H_tot   = H_Sun  + H_Sh  + H_Soil
    # GPP [gC m-2 s-1] = An_canopy [μmol m-2 s-1] * 12 [g mol-1] * 1e-6
    GPP     = (An_Sun + An_Sh) * 12.0e-6

    # ---- Canopy albedo (diagnosed from RT) ----
    # ASW_Sun / ASW_Sh / ASW_Soil are already canopy-integrated ground-area
    # fluxes — the sunlit/shaded split is within the RT scheme, not a weight
    # the caller should re-apply.  Straight sum recovers (1-α)·SW_down.
    sw_absorbed = sw_rt.ASW_Sun + sw_rt.ASW_Sh + sw_rt.ASW_Soil
    sw_down_safe = jnp.maximum(forcing.sw_down, 1.0)
    alpha_canopy = jnp.clip(1.0 - sw_absorbed / sw_down_safe, 0.0, 1.0)
    alpha_canopy = jnp.where(forcing.sw_down < 1.0, mc.albedo_land, alpha_canopy)

    # ---- Land surface temperature (from emitted LW) ----
    # Use area-weighted mean leaf + soil emission
    a_soil = jnp.exp(-0.78 * LAI)  # diffuse transmittance to soil
    a_sun  = fSun * (1.0 - a_soil)
    a_sh   = (1.0 - fSun) * (1.0 - a_soil)
    Lw_up = (a_sun  * cc.epsf * constants.sigma_sb * Tf_Sun**4
           + a_sh   * cc.epsf * constants.sigma_sb * Tf_Sh**4
           + a_soil * cc.epss * constants.sigma_sb * Ts_cvg**4)
    eps_eff = a_sun * cc.epsf + a_sh * cc.epsf + a_soil * cc.epss
    T_surface = (Lw_up / jnp.maximum(eps_eff * constants.sigma_sb, 1e-12))**0.25

    # Physical safety clamp on G passed downstream: a single non-converged
    # Newton step at a stiff transition can return |G| ≫ 1000 W/m², which
    # the soil thermal solver then applies faithfully and corrupts
    # ``T_soil[0]`` for the next step.  Clamp to well beyond any
    # physically reasonable range (desert midday G ≲ 300 W/m², nocturnal
    # release ≳ -200 W/m²) — normal operation is unaffected, extreme
    # excursions are capped.
    G = jnp.clip(G, -500.0, 700.0)

    # ---- Snow budget ----
    has_snow = snow > 1e-6
    G_surface_snow = G  # pass canopy-diagnosed G to snow budget
    snow_new, snow_age_new, snow_melt = update_snow(
        snow, snow_age, T_surface, forcing.precip_snow, dt,
        Q_net=G_surface_snow,
        snow_melt_rate=mc.snow_melt_rate,
        T_snow_melt=mc.T_snow_melt,
    )
    melt_energy = snow_melt * constants.L_f / dt
    G_for_thermal = G - melt_energy

    # ---- Soil hydraulics (Richards) ----
    rho_w = constants.rho_water
    L_eff = jnp.where(has_snow, constants.L_s, constants.L_v)
    evap_rate_demand = LE_tot / L_eff

    # Snow sublimation
    snow_after_melt = snow_new
    max_sublim   = jnp.maximum(snow_after_melt / dt, 0.0)
    sublim_demand = jnp.where(has_snow, evap_rate_demand, 0.0)
    sublim_actual = jnp.minimum(sublim_demand, max_sublim)
    sublim_actual = jnp.where(sublim_demand < 0.0, sublim_demand, sublim_actual)
    snow_new = jnp.maximum(snow_new - sublim_actual * dt, 0.0)

    # Bare soil evaporation vs transpiration
    f_veg = jnp.clip(w_frac_rz, 0.0, 1.0)
    dz    = grid.dz
    extractable = jnp.sum(jnp.maximum(theta - theta_r, 0.0) * dz[None, :], axis=-1) * rho_w
    precip_rain = forcing.precip_total - forcing.precip_snow
    melt_rate   = snow_melt / dt
    soil_evap_demand = jnp.where(has_snow, 0.0, evap_rate_demand)
    max_soil_evap    = jnp.maximum(extractable / dt + precip_rain + melt_rate, 0.0)
    soil_evap        = jnp.minimum(soil_evap_demand, max_soil_evap)
    evap_rate        = jnp.where(has_snow, sublim_actual, soil_evap)
    evap_excess      = (evap_rate_demand - evap_rate) * L_eff
    lhflx_actual     = evap_rate * L_eff

    evap_bare   = evap_rate * (1.0 - f_veg)
    evap_transp = evap_rate * f_veg
    flux_top    = (precip_rain + melt_rate - evap_bare) / rho_w

    E_pot_transp = jnp.maximum(evap_transp, 0.0) / rho_w
    weight       = root_frac[None, :] * beta_root
    weight_sum   = jnp.sum(weight, axis=-1, keepdims=True)
    weight_norm  = weight / jnp.maximum(weight_sum, 1e-20)
    sink         = weight_norm * E_pot_transp[:, None] / dz[None, :]

    richards_out = solve_richards(
        psi, theta, grid, mc.hydraulics, mc.richards, flux_top, sink, dt)
    theta_corrected = jnp.clip(
        richards_out.theta_new, mc.hydraulics.theta_r, mc.hydraulics.theta_sat)
    richards_out = richards_out._replace(theta_new=theta_corrected)

    # ---- Soil thermal update ----
    G_for_thermal = G_for_thermal + evap_excess
    T_soil_new = solve_soil_thermal(
        T_soil, richards_out.theta_new, grid,
        mc.hydraulics, mc.thermal, G_for_thermal, dt)

    # ---- New state ----
    new_state = MultiLayerLandState(
        T_soil=T_soil_new,
        psi_soil=richards_out.psi_new,
        theta_soil=richards_out.theta_new,
        runoff_surface=richards_out.runoff_surface,
        runoff_subsurface=richards_out.runoff_subsurface,
        snow_depth=snow_new,
        snow_age=snow_age_new,
    )

    # ---- Surface temperature after soil thermal ----
    T_surface_new = T_soil_new[:, 0]

    # Albedo after snow update
    if mc.snow_albedo_feedback and lat is not None:
        alpha_new = compute_land_albedo(lat, snow_new, snow_age_new, mc.land_albedo)
    else:
        alpha_new = alpha_canopy

    # Updated upwelling LW
    lw_up_new = eps_eff * constants.sigma_sb * T_surface_new**4

    # ---- Surface specific humidity for coupler ----
    q_sat_liq_new = saturation_mixing_ratio(T_surface_new, forcing.p_surface)
    q_sat_ice_new = saturation_mixing_ratio_ice(T_surface_new, forcing.p_surface)
    has_snow_new  = snow_new > 1e-6
    q_sat_sfc_new = jnp.where(has_snow_new, q_sat_ice_new, q_sat_liq_new)

    # Post-step soil moisture stress for q_surface
    if lp is not None and hasattr(lp, "theta_wp") and lp.theta_wp.ndim > 0:
        beta_root_new = jnp.clip(
            (richards_out.theta_new - theta_wp[:, None])
            / (theta_fc[:, None] - theta_wp[:, None] + 1e-10),
            0.0, 1.0)
    else:
        beta_root_new = jnp.clip(
            (richards_out.theta_new - theta_wp) / (theta_fc - theta_wp + 1e-10),
            0.0, 1.0)
    w_frac_rz_new = jnp.clip(
        jnp.sum(root_frac[None, :] * beta_root_new, axis=-1), 0.0, 1.0)
    beta_new      = mc.beta_min + (1.0 - mc.beta_min) * w_frac_rz_new
    beta_eff_new  = jnp.where(has_snow_new, 1.0, beta_new)
    q_sfc_new     = beta_eff_new * q_sat_sfc_new

    # ---- Wind stress from MOST ustar ----
    tau_mag = rhoa * ustar**2
    tau_x   = tau_mag * wind_dir_x
    tau_y   = tau_mag * wind_dir_y

    # ---- CO2 flux: GPP is uptake (negative = down into surface) ----
    # co2_flux sign convention: positive = surface → atmosphere
    co2_flux = -GPP  # [gC m-2 s-1]; negative = land uptake

    # ---- TileResponse ----
    z0_sfc = z0m  # use canopy z0m as roughness for coupler
    response = TileResponse(
        T_surface=T_surface_new,
        albedo=alpha_new,
        emissivity=jnp.broadcast_to(jnp.asarray(emissivity), T_surface_new.shape),
        z0=z0_sfc,
        q_surface=q_sfc_new,
        shflx=H_tot,
        lhflx=lhflx_actual,
        tau_x=tau_x,
        tau_y=tau_y,
        lw_up=lw_up_new,
        u_ocean_sfc=jnp.zeros(ncol),
        v_ocean_sfc=jnp.zeros(ncol),
        co2_flux=co2_flux,
    )

    # ---- Surface-budget diagnostics ------------------------------------
    # Internal (solver-closed): each leaf satisfies Rn_leaf = LE_leaf + H_leaf;
    # soil satisfies Rn_soil = LE_soil + H_soil + G.  Sum → Rn_int = LE+H+G.
    Rn_Sun_d   = fluxes_per_col["Rn_Sun"]
    Rn_Sh_d    = fluxes_per_col["Rn_Sh"]
    Rn_Soil_d  = fluxes_per_col["Rn_Soil"]
    Rn_canopy_d = Rn_Sun_d + Rn_Sh_d
    Rn_int_d    = Rn_canopy_d + Rn_Soil_d
    LE_canopy_d = LE_Sun + LE_Sh
    H_canopy_d  = H_Sun  + H_Sh

    # External (boundary-condition): what a downstream observer sees from
    # forcing + canopy-mean surface temperature used for LW emission.
    eps_eff_ext = eps_eff
    SW_net_d = (1.0 - alpha_canopy) * forcing.sw_down
    LW_net_d = eps_eff_ext * forcing.lw_down - eps_eff_ext * constants.sigma_sb * T_surface**4
    Rn_ext_d = SW_net_d + LW_net_d

    residual_int_d = Rn_int_d - (LE_tot + H_tot + G)
    residual_ext_d = Rn_ext_d - (LE_tot + H_tot + G)

    diagnostics = CanopyDiagnostics(
        Rn_ext=Rn_ext_d,
        Rn_int=Rn_int_d,
        SW_net=SW_net_d,
        LW_net=LW_net_d,
        LE_tot=LE_tot,
        H_tot=H_tot,
        G=G,
        LE_canopy=LE_canopy_d,
        LE_soil=LE_Soil,
        H_canopy=H_canopy_d,
        H_soil=H_Soil,
        Rn_canopy=Rn_canopy_d,
        Rn_soil=Rn_Soil_d,
        residual_int=residual_int_d,
        residual_ext=residual_ext_d,
        GPP=GPP,
        fSun=fSun,
        n_iters=n_iters,
        Tf_Sun=Tf_Sun,
        Tf_Sh=Tf_Sh,
        Ts_solve=Ts_cvg,
        T_surface=T_surface_new,
    )

    return new_state, response, carbon_state, diagnostics


def step_canopy_land(
    state: MultiLayerLandState,
    forcing: AtmToSurface,
    config: CanopyLandConfig,
    U_min: float,
    dt: float,
    lat: jnp.ndarray | None = None,
    carbon_state: CarbonState | None = None,
    doy: float = 0.0,
    land_params: CanopyLandParams | None = None,
) -> tuple[MultiLayerLandState, TileResponse, CarbonState | None]:
    """Single time step of the canopy energy balance land model.

    Returns the standard 3-tuple (new_state, response, carbon_state).  For
    detailed surface-budget diagnostics use
    :func:`step_canopy_land_with_diagnostics`.
    """
    new_state, response, cs, _diag = _step_canopy_land_full(
        state, forcing, config, U_min, dt,
        lat=lat, carbon_state=carbon_state, doy=doy, land_params=land_params)
    return new_state, response, cs


def step_canopy_land_with_diagnostics(
    state: MultiLayerLandState,
    forcing: AtmToSurface,
    config: CanopyLandConfig,
    U_min: float,
    dt: float,
    lat: jnp.ndarray | None = None,
    carbon_state: CarbonState | None = None,
    doy: float = 0.0,
    land_params: CanopyLandParams | None = None,
) -> tuple[MultiLayerLandState, TileResponse, CarbonState | None, CanopyDiagnostics]:
    """Like :func:`step_canopy_land` but also returns a ``CanopyDiagnostics``.

    Intended for offline diagnostic runs — enables inspection of the full
    surface energy budget (Rn, LE, H, G, residual) without recomputing the
    Newton closure.  Adds ~dozen extra arrays to the return pytree; no
    performance cost beyond their allocation.
    """
    return _step_canopy_land_full(
        state, forcing, config, U_min, dt,
        lat=lat, carbon_state=carbon_state, doy=doy, land_params=land_params)


def init_canopy_land_state(
    ncol: int,
    config: CanopyLandConfig,
    T_init: float = 280.0,
    theta_init: float | None = None,
) -> MultiLayerLandState:
    """Initialise a MultiLayerLandState for the canopy land model.

    Delegates to the multilayer land initialiser (same soil structure).
    """
    from legoesm.land.multilayer_land import init_multilayer_land_state
    return init_multilayer_land_state(ncol, config.multilayer, T_init, theta_init)
