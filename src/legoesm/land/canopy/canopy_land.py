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

_SIGMA = 5.670373e-8  # Stefan-Boltzmann [W m-2 K-4]


def _get(lp, name: str, fallback):
    if lp is None:
        return fallback
    return getattr(lp, name, fallback)


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

    Parameters
    ----------
    state       : MultiLayerLandState — prognostic soil T, theta, psi, snow
    forcing     : AtmToSurface — coupler fields from atmosphere
    config      : CanopyLandConfig — physics configuration
    U_min       : minimum wind speed [m/s] for numerical stability
    dt          : time step [s]
    lat         : latitude [degrees], optional, for snow albedo
    carbon_state: CarbonState | None — reserved for Stage 2 (carbon pools)
    doy         : day of year [1–365]
    land_params : CanopyLandParams | None — spatially varying parameters

    Returns
    -------
    new_state   : MultiLayerLandState
    response    : TileResponse
    carbon_state: None (Stage 1 — no prognostic carbon pools)
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

    # ---- Build forcing bundles for solver (one element per column) ----
    # Initial canopy state guess
    Ts_init  = T_soil[:, 0]
    q_s_init = sat_specific_humidity(Ts_init, Ps)
    q_c_init = 0.5 * (q_s_init + q_atm)
    chi = 0.7 - 0.3 * fC4  # initial Ci/Ca ratio (0.7 C3, 0.4 C4)
    Ci_init = Ca * chi

    initial_state = jnp.stack([
        Ta, Ta, Ci_init, Ci_init, Ts_init, Ta, q_c_init], axis=-1)  # (ncol, 7)

    # Build CanopyForcingBundle — broadcast scalar config values to (ncol,)
    def _bcast(v):
        if hasattr(v, "shape") and v.shape == (ncol,):
            return v
        return jnp.broadcast_to(jnp.asarray(v), (ncol,))

    bundles = CanopyForcingBundle(
        LAI=LAI, SZA=SZA, La=forcing.lw_down,
        epsf=_bcast(cc.epsf), epss=_bcast(cc.epss),
        fSun=sw_rt.fSun,
        APAR_Sun=sw_rt.APAR_Sun, APAR_Sh=sw_rt.APAR_Sh,
        Vcmax25_Sun=sw_rt.Vcmax25_C3Sun, Vcmax25_Sh=sw_rt.Vcmax25_C3Sh,
        Vcmax25_C4Sun=sw_rt.Vcmax25_C4Sun, Vcmax25_C4Sh=sw_rt.Vcmax25_C4Sh,
        ASW_Sun=sw_rt.ASW_Sun, ASW_Sh=sw_rt.ASW_Sh, ASW_Soil=sw_rt.ASW_Soil,
        G_alpha=_bcast(cc.G_alpha),
        Ca=Ca, Ps=Ps, Ta=Ta,
        lam=_bcast(lam), Cp=_bcast(Cp), rhoa=rhoa, Tv_atm=Tv_atm, q_atm=q_atm,
        m=m_mix, b0=b0_mix, alf=alf, TgC=TgC,
        fC4=fC4, fStress_soil=fStress_soil,
        ur=wind_speed, CI=CI, z0m=z0m, displa=displa, z0=z_ref,
    )

    # ---- Newton-Raphson closure — vectorised over columns via vmap ----
    def _solve_one_col(x0, bun):
        return solve_canopy_closure(x0, bun, cc)

    x_final, n_iters = jax.vmap(_solve_one_col)(initial_state, bundles)

    # ---- Diagnostic forward pass — extract all fluxes ----
    def _fwd_one_col(xf, bun):
        return _canopy_forward(xf, bun, cc.coupling_scheme, cc.LE_module,
                               cc.use_ta_for_photosynthesis)

    fluxes_per_col = jax.vmap(_fwd_one_col)(x_final, bundles)

    # Converged state
    Tf_Sun = x_final[:, 0]
    Tf_Sh  = x_final[:, 1]
    Ts_cvg = x_final[:, 4]  # converged soil surface temperature

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
    # Reflected SW ≈ (1 - absorbed fraction) * sw_down
    # Conservative: use (ASW_Sun+ASW_Sh+ASW_Soil) / sw_down
    sw_absorbed = (sw_rt.ASW_Sun * fSun + sw_rt.ASW_Sh * (1.0 - fSun)
                   + sw_rt.ASW_Soil)
    sw_down_safe = jnp.maximum(forcing.sw_down, 1.0)
    alpha_canopy = jnp.clip(1.0 - sw_absorbed / sw_down_safe, 0.0, 1.0)
    alpha_canopy = jnp.where(forcing.sw_down < 1.0, mc.albedo_land, alpha_canopy)

    # ---- Land surface temperature (from emitted LW) ----
    # Use area-weighted mean leaf + soil emission
    a_soil = jnp.exp(-0.78 * LAI)  # diffuse transmittance to soil
    a_sun  = fSun * (1.0 - a_soil)
    a_sh   = (1.0 - fSun) * (1.0 - a_soil)
    Lw_up = (a_sun  * cc.epsf * _SIGMA * Tf_Sun**4
           + a_sh   * cc.epsf * _SIGMA * Tf_Sh**4
           + a_soil * cc.epss * _SIGMA * Ts_cvg**4)
    eps_eff = a_sun * cc.epsf + a_sh * cc.epsf + a_soil * cc.epss
    T_surface = (Lw_up / jnp.maximum(eps_eff * _SIGMA, 1e-12))**0.25

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
    lw_up_new = eps_eff * _SIGMA * T_surface_new**4

    # ---- Surface specific humidity for coupler ----
    q_sat_liq_new = saturation_mixing_ratio(T_surface_new, forcing.p_surface)
    q_sat_ice_new = saturation_mixing_ratio_ice(T_surface_new, forcing.p_surface)
    has_snow_new  = snow_new > 1e-6
    q_sat_sfc_new = jnp.where(has_snow_new, q_sat_ice_new, q_sat_liq_new)

    # Post-step soil moisture stress for q_surface
    if lp is not None and hasattr(lp, "theta_wp") and lp.theta_wp.ndim > 0:
        beta_root_new = jnp.clip(
            (T_soil_new[:, :] - theta_wp[:, None]) / (theta_fc[:, None] - theta_wp[:, None] + 1e-10),
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

    return new_state, response, carbon_state


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
