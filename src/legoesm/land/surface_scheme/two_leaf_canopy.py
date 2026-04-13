"""Two-leaf canopy surface scheme (DifferBESS-style).

Newton-Raphson closure of a 6-variable canopy state

    [Tf_Sun, Tf_Sh, Ci_Sun, Ci_Sh, Tc, q_c]

wrapped in an outer Picard loop that reconciles the canopy's
sub-minute turbulent response with the soil column's ~hour thermal
time constant.  Soil skin temperature ``Ts`` is prescribed and updated
between Picard passes by a caller-supplied ``soil_thermal_fn`` callback,
so this module is agnostic to whether the underlying land model is
slab (single explicit layer) or multilayer (implicit column thermal
solver).

``compute_two_leaf_canopy_fluxes`` returns a ``SurfaceFluxOutput`` that
the shared post-flux pipeline in ``step_multilayer_land`` / ``step_land``
consumes — the canopy scheme does not run its own snow / Richards /
soil-thermal post-processing, matching the design decision to keep
the post-flux block in one place.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.coupler.coupling_fields import AtmToSurface
from legoesm.land.canopy.config import CanopyConfig
from legoesm.land.canopy.radiative_transfer import (
    split_sw_components, canopy_shortwave_rt,
)
from legoesm.land.canopy.stability import (
    compute_aerodynamics, sat_specific_humidity,
)
from legoesm.land.canopy.solver import (
    CanopyForcingBundle, solve_canopy_closure, _canopy_forward,
)
from legoesm.land.surface_scheme.base import SurfaceFluxOutput


# =====================================================================
# Surface-scheme config — alias of the canopy biophysics config.
# =====================================================================
# ``TwoLeafCanopyConfig`` is the user-facing name for the canopy surface
# scheme dispatch.  It is exactly ``canopy.config.CanopyConfig`` (same
# NamedTuple, same fields) so ``isinstance`` dispatch in
# ``step_multilayer_land`` / ``step_land`` can tell it apart from
# ``SimpleSEBConfig`` while the canopy Newton solver continues to read
# its solver-level settings from the same object.  Per-column vegetation,
# physiology, radiative, and aerodynamic parameters live in
# ``CanopyLandParams`` — not here.
TwoLeafCanopyConfig = CanopyConfig

# Extend the default ``n_picard`` / ``picard_omega`` attributes lazily via
# ``getattr`` below — in the current ``CanopyConfig`` they do not exist as
# fields, so the surface scheme falls back to the values used by the
# original ``canopy_land.py`` implementation (n=6, ω=0.15).
_DEFAULT_N_PICARD = 6
_DEFAULT_PICARD_OMEGA = 0.15


def _get(lp, name: str, fallback):
    """Read from per-column params if available; else return fallback."""
    if lp is None:
        return fallback
    return getattr(lp, name, fallback)


def compute_two_leaf_canopy_fluxes(
    *,
    T_soil_top: jnp.ndarray,          # (ncol,) soil skin T start-of-step [K]
    forcing: AtmToSurface,
    canopy_config: TwoLeafCanopyConfig,
    land_config,                      # MultiLayerLandConfig or LandConfig (for z_ref, beta_min)
    canopy_params,                    # CanopyLandParams | None
    root_frac: jnp.ndarray,           # (ncol, n_layers) or (n_layers,)
    beta_root: jnp.ndarray,           # (ncol, n_layers) root-zone moisture stress
    w_frac_rz: jnp.ndarray,           # (ncol,) root-zone-weighted beta
    wind_speed: jnp.ndarray,          # (ncol,)
    wind_dir_x: jnp.ndarray,          # (ncol,)
    wind_dir_y: jnp.ndarray,          # (ncol,)
    soil_thermal_fn: Callable[[jnp.ndarray, float], jnp.ndarray],
    dt: float,
) -> SurfaceFluxOutput:
    """Compute surface fluxes via the two-leaf canopy Newton + Picard closure.

    Parameters
    ----------
    T_soil_top : start-of-step soil skin temperature, shape ``(ncol,)``.
    forcing    : atmospheric forcing fields.
    canopy_config : ``TwoLeafCanopyConfig`` — solver and biophysics settings.
    land_config : the enclosing ``MultiLayerLandConfig`` / ``LandConfig`` —
                  only ``z_ref``, ``emissivity_land``, ``albedo_land`` are
                  used from here (the surface scheme does not touch soil
                  hydraulics / thermal configs; those are the caller's
                  responsibility via ``soil_thermal_fn``).
    canopy_params : optional per-column ``CanopyLandParams``.  When
                    ``None``, scalar defaults are used.
    root_frac, beta_root, w_frac_rz : pre-computed root-zone stress
                    fields from the caller.  ``w_frac_rz`` is used for
                    soil evaporation stress and Vcmax down-regulation;
                    ``beta_root`` feeds the Richards root sink downstream
                    (via the caller, not here).
    wind_speed, wind_dir_x, wind_dir_y : pre-computed wind magnitude and
                    unit direction.
    soil_thermal_fn : ``(G [W/m^2], dt [s]) -> Ts_new [K]`` callback.
                    The canopy Picard loop calls this each iteration to
                    advance a tentative soil thermal step and update the
                    prescribed ``Ts_bc``.  Slab callers supply an explicit
                    single-layer update; multilayer callers wrap
                    ``solve_soil_thermal``.
    dt          : time step [s].

    Returns
    -------
    SurfaceFluxOutput with canopy diagnostics (``Tf_Sun``, ``gs_Sun``,
    ``n_iters``, ``f_veg``) populated.  ``stomatal_ratio`` is 1.0 because
    the canopy computes LE from leaf-level humidity gradients directly
    (no ``beta * q_sat`` proxy).
    """
    cc = canopy_config
    lp = canopy_params
    ncol = T_soil_top.shape[0]

    # ---- Resolve per-column parameters (fall back to scalar defaults) ----
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
    emissivity_per_col = _get(
        lp, "emissivity", jnp.full(ncol, land_config.emissivity_land))

    # ---- Soil moisture stress (canopy reuses the shared root-zone beta) ----
    fStress_soil  = w_frac_rz         # soil evaporation stress
    fStress_vcmax = w_frac_rz         # Vcmax down-regulation

    Vc3_leaf_stressed = Vc3_leaf * fStress_vcmax
    Vc4_leaf_stressed = Vc4_leaf * fStress_vcmax
    m_eff  = m_C3  * fStress_vcmax
    b0_eff = b0_C3 * fStress_vcmax
    m_mix  = (1.0 - fC4) * m_eff  + fC4 * m_C4  * fStress_vcmax
    b0_mix = (1.0 - fC4) * b0_eff + fC4 * b0_C4 * fStress_vcmax

    # ---- Aerodynamics ----
    z0m, displa = compute_aerodynamics(hc, LAI, rz0m, rd)
    # Lift reference height above the displacement + roughness sub-layer.
    z_ref_base = jnp.full(ncol, land_config.z_ref)
    z_ref = jnp.maximum(z_ref_base, displa + 10.0 * z0m + 2.0)

    # ---- SW decomposition ----
    cos_zenith = forcing.cos_zenith
    SZA = jnp.degrees(jnp.arccos(jnp.clip(cos_zenith, 0.0, 1.0)))
    PAR_dir, PAR_diff, NIR_dir, NIR_diff, UV = split_sw_components(
        forcing.sw_down, cos_zenith)

    sw_rt = canopy_shortwave_rt(
        PAR_dir, PAR_diff, NIR_dir, NIR_diff, UV,
        SZA, LAI, CI, ALB_VIS, ALB_NIR,
        Vc3_leaf_stressed, Vc4_leaf_stressed, kn, FNonVeg)

    # ---- Thermodynamic / atmosphere variables ----
    Ta    = forcing.T_lowest
    Ps    = forcing.p_surface
    q_atm = forcing.q_lowest
    rhoa  = forcing.rho_lowest
    Tv_atm = Ta * (1.0 + 0.61 * q_atm)
    lam   = constants.L_v
    Cp    = constants.c_pd
    Ca    = forcing.co2_ppmv

    # ---- Initial Newton state ----
    Ts_old   = T_soil_top
    q_s_init = sat_specific_humidity(Ts_old, Ps)
    q_c_init = 0.5 * (q_s_init + q_atm)
    chi = 0.7 - 0.3 * fC4
    Ci_init = Ca * chi

    initial_state = jnp.stack(
        [Ta, Ta, Ci_init, Ci_init, Ta, q_c_init], axis=-1)  # (ncol, 6)

    def _bcast(v):
        if hasattr(v, "shape") and v.shape == (ncol,):
            return v
        return jnp.broadcast_to(jnp.asarray(v), (ncol,))

    def _build_bundle(Ts_bc):
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

    # ---- Outer Picard loop: canopy closure ↔ soil thermal ----
    # See canopy_land.py module notes for the stability analysis.  The
    # soil_thermal_fn callback is responsible for advancing Ts tentatively
    # given the Picard-iterate G; slab and multilayer callers supply
    # different implementations.
    n_picard = getattr(cc, "n_picard", _DEFAULT_N_PICARD)
    omega    = getattr(cc, "picard_omega", _DEFAULT_PICARD_OMEGA)
    Ts_bc_k = Ts_old

    for _picard_iter in range(n_picard):
        bundles_k = _build_bundle(Ts_bc_k)
        x_final, n_iters = jax.vmap(_solve_one_col)(initial_state, bundles_k)
        fluxes_per_col = jax.vmap(_fwd_one_col)(x_final, bundles_k)

        G_k = jnp.clip(fluxes_per_col["G"], -500.0, 700.0)
        Ts_thermal = soil_thermal_fn(G_k, dt)
        Ts_bc_k = (1.0 - omega) * Ts_bc_k + omega * Ts_thermal

    # ---- Converged state ----
    Tf_Sun = x_final[:, 0]
    Tf_Sh  = x_final[:, 1]
    Ts_cvg = Ts_bc_k

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
    gs_Sun  = fluxes_per_col["gs_Sun"]
    gs_Sh   = fluxes_per_col["gs_Sh"]
    ustar   = fluxes_per_col["ustar"]
    Rn_Sun_d  = fluxes_per_col["Rn_Sun"]
    Rn_Sh_d   = fluxes_per_col["Rn_Sh"]
    Rn_Soil_d = fluxes_per_col["Rn_Soil"]

    LE_tot = LE_Sun + LE_Sh + LE_Soil
    H_tot  = H_Sun  + H_Sh  + H_Soil
    GPP    = (An_Sun + An_Sh) * 12.0e-6    # gC m-2 s-1

    # Per-component canopy fluxes (for offline diagnostic drivers).
    LE_canopy_d = LE_Sun + LE_Sh
    H_canopy_d  = H_Sun  + H_Sh
    Rn_canopy_d = Rn_Sun_d + Rn_Sh_d

    # Internal energy-balance closure diagnostic.
    Rn_int_d       = Rn_canopy_d + Rn_Soil_d
    residual_int_d = Rn_int_d - (LE_tot + H_tot + G)

    # ---- Canopy albedo diagnosed from SW absorption ----
    sw_absorbed = sw_rt.ASW_Sun + sw_rt.ASW_Sh + sw_rt.ASW_Soil
    sw_down_safe = jnp.maximum(forcing.sw_down, 1.0)
    alpha_canopy = jnp.clip(1.0 - sw_absorbed / sw_down_safe, 0.0, 1.0)
    alpha_canopy = jnp.where(
        forcing.sw_down < 1.0,
        jnp.broadcast_to(jnp.asarray(land_config.albedo_land), T_soil_top.shape),
        alpha_canopy)

    # ---- Emission-weighted surface T from canopy LW ----
    a_soil = jnp.exp(-0.78 * LAI)
    a_sun  = fSun * (1.0 - a_soil)
    a_sh   = (1.0 - fSun) * (1.0 - a_soil)
    Lw_up = (a_sun  * cc.epsf * constants.sigma_sb * Tf_Sun**4
           + a_sh   * cc.epsf * constants.sigma_sb * Tf_Sh**4
           + a_soil * cc.epss * constants.sigma_sb * Ts_cvg**4)
    eps_eff = a_sun * cc.epsf + a_sh * cc.epsf + a_soil * cc.epss
    T_surface = (Lw_up / jnp.maximum(eps_eff * constants.sigma_sb, 1e-12))**0.25

    # Downstream expects SW_net and LW_net: SW_net = (1-α) SW_down,
    # LW_net = ε (LW_down - σ T_surface^4).
    sw_net = (1.0 - alpha_canopy) * forcing.sw_down
    lw_net = eps_eff * forcing.lw_down - eps_eff * constants.sigma_sb * T_surface**4
    lw_up_out = Lw_up  # Emission-weighted upward LW (already multiplied by eps_eff).

    # External (boundary-condition) radiation balance — what a downstream
    # observer sees from forcing + canopy-mean emission T.
    Rn_ext_d       = sw_net + lw_net
    residual_ext_d = Rn_ext_d - (LE_tot + H_tot + G)

    # Safety clamp on G: extreme non-converged values corrupt T_soil.
    G_clamped = jnp.clip(G, -500.0, 700.0)

    # ---- Wind stress from MOST ustar ----
    tau_mag = rhoa * ustar**2
    tau_x = tau_mag * wind_dir_x
    tau_y = tau_mag * wind_dir_y

    # ---- Surface specific humidity (proxy; refined post-step by the caller) ----
    # The canopy solver tracks q_c (canopy-air humidity) as part of its
    # Newton state; here we report q_c as the surface humidity so the
    # coupler sees a consistent value when it queries the pre-Richards
    # surface state.  Post-Richards, the caller recomputes q_surface
    # from T_surface_new and the updated soil moisture.
    q_c_final = x_final[:, 5]

    # Vegetation cover fraction (for root-zone transpiration partition).
    f_veg = jnp.clip(w_frac_rz, 0.0, 1.0)

    return SurfaceFluxOutput(
        shflx=H_tot, lhflx=LE_tot,
        tau_x=tau_x, tau_y=tau_y,
        sw_net=sw_net, lw_net=lw_net, lw_up=lw_up_out,
        G_soil=G_clamped,
        T_surface=T_surface,
        q_surface=q_c_final,
        albedo=alpha_canopy,
        emissivity=jnp.broadcast_to(eps_eff, T_soil_top.shape),
        z0=z0m,
        gpp=GPP,
        Tf_Sun=Tf_Sun,
        Tf_Sh=Tf_Sh,
        gs_Sun=gs_Sun,
        gs_Sh=gs_Sh,
        n_iters=n_iters,
        f_veg=f_veg,
        fSun=fSun,
        Ts_solve=Ts_cvg,
        LE_canopy=LE_canopy_d,
        LE_soil=LE_Soil,
        H_canopy=H_canopy_d,
        H_soil=H_Soil,
        Rn_canopy=Rn_canopy_d,
        Rn_soil=Rn_Soil_d,
        Rn_int=Rn_int_d,
        residual_int=residual_int_d,
        Rn_ext=Rn_ext_d,
        residual_ext=residual_ext_d,
        stomatal_ratio=jnp.ones_like(T_soil_top),
    )
