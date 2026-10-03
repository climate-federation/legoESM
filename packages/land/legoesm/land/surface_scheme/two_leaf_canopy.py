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

from legoesm.core.bulk_flux import surface_reference_state
from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.canopy.config import (
    CanopyConfig,
    VCMAX25_C3_DEFAULT,
    VCMAX25_C4_DEFAULT,
)
from legoesm.land.canopy.photosynthesis import co2_compensation_point
from legoesm.land.canopy.radiative_transfer import (
    split_sw_components, canopy_shortwave_rt,
)
from legoesm.land.canopy.sif import two_leaf_canopy_sif
from legoesm.land.canopy.stability import (
    compute_aerodynamics, sat_specific_humidity,
)
from legoesm.land.canopy.solver import (
    CanopyForcingBundle, solve_canopy_closure_diag, canopy_forward,
    canopy_state_admissible,
)
from legoesm.land.canopy.energy_balance import soil_surface_evap_resistance
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

# Time constant of the growth-temperature exponential moving average [s].
# 30 days * 86400 s/day = 2.592e6 s.  At dt = 1800 s, ~76 days reach 95 %
# of the equilibrium.  Used by the caller (``step_*_land``) to advance
# the state-carried ``TgC`` field; this module exposes the constant so
# callers can stay consistent.
TGC_EMA_TAU_S: float = 30.0 * 86400.0

# Virtual-temperature coefficient (1−ε)/ε ≈ 0.608, derived from the repo epsilon
# (never hardcoded per CLAUDE.md constants hygiene).
_VIRT_T_COEF = (1.0 - constants.epsilon) / constants.epsilon

# Initial intercellular-CO2 ratio Ci/Ca: chi = _CI_CA_C3 − _CI_CA_C3_MINUS_C4·fC4
# → 0.7 for C3 (fC4=0), 0.4 for C4 (fC4=1).
_CI_CA_C3 = 0.7
_CI_CA_C3_MINUS_C4 = 0.3

# μmol CO2 → gC conversion (molar mass of carbon 12 g/mol × 1e-6 mol/μmol).
_G_C_PER_UMOL_CO2 = 12.0e-6

# --- Scalar fallback defaults for per-column canopy params (used when no
#     CanopyLandParams supplied; DifferBESS / Ryu et al. 2011 midranges) ---
_DEFAULT_LAI = 1.5          # [m2/m2] leaf area index
_DEFAULT_HC = 5.0           # [m] canopy height
_DEFAULT_CI = 0.75          # [-] clumping index
_DEFAULT_KN = 0.3           # [-] nitrogen extinction coefficient
_DEFAULT_M_C3 = 9.0         # [-] Ball-Berry slope C3
_DEFAULT_M_C4 = 4.0         # [-] Ball-Berry slope C4
_DEFAULT_B0_C3 = 0.01       # [mol m-2 s-1] Ball-Berry intercept C3
_DEFAULT_B0_C4 = 0.04       # [mol m-2 s-1] Ball-Berry intercept C4
_DEFAULT_ALF = 0.3          # [mol/mol] electron-transport quantum yield
_DEFAULT_ALB_VIS = 0.1      # [-] visible-band albedo
_DEFAULT_ALB_NIR = 0.2      # [-] NIR-band albedo
_DEFAULT_RZ0M = 0.055       # [-] z0m / hc ratio
_DEFAULT_RD = 0.67          # [-] displacement-height / hc ratio
_DEFAULT_D_LEAF = 0.025     # [m] characteristic leaf width (Schuepp 1993 midrange)


def compute_prognostic_lai(
    carbon_state,
    land_config,
    canopy_config: "TwoLeafCanopyConfig",
) -> jnp.ndarray | None:
    """Phase 6 / Stage 2b: prognostic LAI from the foliar carbon pool.

    Returns ``carbon_state.C_fol / LCMA`` when all three conditions hold:

    1. ``canopy_config.use_prognostic_lai`` is True (opt-in; defaults
       to False until the reverse-mode NaN through the MOST scan is
       resolved — see ``monin_obukhov_stability`` follow-up).
    2. ``land_config.carbon.scheme == "differland"``.
    3. ``carbon_state is not None``.

    Returns ``None`` otherwise, in which case callers fall through to
    the prescribed ``CanopyLandParams.LAI`` (or the scalar default).

    ``LCMA`` is read from ``land_config.carbon.LCMA`` — the leaf carbon
    mass per unit area [gC / m²] that maps the prognostic foliar pool
    onto a LAI value.  The typical DifferLand default is 50 gC/m²,
    giving ``LAI = 4`` at the ``C_fol_init = 200`` pool size.

    Notes
    -----
    - Canopy height ``hc`` stays prescribed in Phase 6 — there is no
      simple allometric mapping from the wood pool to canopy height,
      and the canopy closure is much less sensitive to ``hc`` than
      to ``LAI``.
    - The forward pass is fully differentiable wrt ``C_fol`` (``dLAI /
      dC_fol = 1 / LCMA``); ``jax.grad`` through the full
      ``C_fol → LAI → canopy Newton`` loop is finite + nonzero (guarded by
      ``test_prognostic_lai_jax_grad_through_feedback``).
    """
    if not getattr(canopy_config, "use_prognostic_lai", False):
        return None
    if carbon_state is None:
        return None
    if getattr(land_config.carbon, "scheme", "none") != "differland":
        return None
    LCMA = land_config.carbon.LCMA
    return carbon_state.C_fol / jnp.maximum(LCMA, 1e-6)


def advance_TgC_ema(
    TgC_old: jnp.ndarray,
    T_air_K: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """One step of the 30-day exponential moving average on T_air [°C].

    ``TgC_new = TgC_old + (dt / tau) * (T_air_C - TgC_old)``

    Parameters
    ----------
    TgC_old : current 30-day mean growth temperature [°C]
    T_air_K : instantaneous near-surface air temperature [K]
    dt      : time step [s]
    """
    T_air_C = T_air_K - constants.T_freeze
    alpha = dt / TGC_EMA_TAU_S
    return TgC_old + alpha * (T_air_C - TgC_old)


def static_canopy_roughness(lp, ncol):
    """(z0m, d) from the PRESCRIBED canopy geometry with exactly the defaults
    and the aerodynamics function of the canopy solve below."""
    _lp_lai = _get(lp, "LAI", None)
    LAI = jnp.full(ncol, _DEFAULT_LAI) if _lp_lai is None else _lp_lai
    return compute_aerodynamics(
        _get(lp, "hc", jnp.full(ncol, _DEFAULT_HC)), LAI,
        _get(lp, "rz0m", jnp.full(ncol, _DEFAULT_RZ0M)),
        _get(lp, "rd", jnp.full(ncol, _DEFAULT_RD)))


def _get(lp, name: str, fallback):
    """Read from per-column params if available; else return fallback.

    ``None``-valued fields count as absent: LandSurfaceParams declares every
    optional field with a ``None`` default, so ``getattr`` finds the attribute
    and hands back the ``None`` — which then detonates arithmetic ("float -
    NoneType") the first time a setup runs that never populated the field. The
    LAI read below had grown its own local guard for exactly this; the AMIP
    two-leaf run on the Voronoi mesh then hit the same trap on ``fC4``. One
    guard here covers every field the same way.
    """
    if lp is None:
        return fallback
    got = getattr(lp, name, None)
    return fallback if got is None else got


def compute_two_leaf_canopy_fluxes(
    *,
    T_soil_top: jnp.ndarray,          # (ncol,) soil skin T start-of-step [K]
    forcing: AtmToSurface,
    canopy_config: TwoLeafCanopyConfig,
    land_config,                      # MultiLayerLandConfig or LandConfig (for z_ref, beta_min)
    canopy_params,                    # CanopyLandParams | None
    w_frac_rz: jnp.ndarray,           # (ncol,) root-zone-weighted soil moisture beta
    wind_speed: jnp.ndarray,          # (ncol,)
    wind_dir_x: jnp.ndarray,          # (ncol,)
    wind_dir_y: jnp.ndarray,          # (ncol,)
    soil_thermal_fn: Callable[[jnp.ndarray, float], jnp.ndarray],
    dt: float,
    TgC_override: jnp.ndarray | None = None,
    LAI_override: jnp.ndarray | None = None,
    w_frac_soil_evap: jnp.ndarray | None = None,
    soil_surface_relsat: jnp.ndarray | None = None,
    fwet: jnp.ndarray | None = None,
    canopy_seed: jnp.ndarray | None = None,  # (ncol, 6) warm start; see below
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
    w_frac_rz   : root-zone-weighted soil moisture availability,
                    shape ``(ncol,)``.  Used as both ``fStress_soil``
                    (soil evaporation stress) and ``fStress_vcmax``
                    (photosynthesis down-regulation).  Provided by the
                    caller from its own soil model — multilayer computes
                    ``sum(root_frac * beta_root)`` over the soil column;
                    slab uses the bucket fraction ``W / W_max``.
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
    # Fail-early dispatch check on the static string fields (stomatal_model):
    # this runs at land-component setup, before any jitted canopy solve, so a
    # typo aborts here with a clear message rather than deep in the JAX kernel.
    cc = canopy_config.validate()
    lp = canopy_params
    ncol = T_soil_top.shape[0]

    # ---- Resolve per-column parameters (fall back to scalar defaults) ----
    # LAI priority: (1) caller-supplied ``LAI_override`` (prognostic from
    # ``C_fol / LCMA`` when carbon+canopy both active, Phase 6); (2)
    # ``CanopyLandParams.LAI`` (prescribed); (3) scalar default 1.5.
    if LAI_override is not None:
        LAI    = LAI_override
    else:
        # ``lp.LAI`` is now a real field on LandSurfaceParams (prescribed spatial
        # climatology); it is None only for non-CLM setups that never populated it,
        # in which case fall back to the scalar default (getattr alone would hand
        # back the None and break the canopy math).
        _lp_lai = _get(lp, "LAI",     None)
        LAI    = jnp.full(ncol, _DEFAULT_LAI) if _lp_lai is None else _lp_lai
    hc         = _get(lp, "hc",       jnp.full(ncol, _DEFAULT_HC))
    fC4        = _get(lp, "fC4",      jnp.zeros(ncol))
    CI         = _get(lp, "CI",       jnp.full(ncol, _DEFAULT_CI))
    kn         = _get(lp, "kn",       jnp.full(ncol, _DEFAULT_KN))

    # ---- Below-canopy soil-surface evaporation resistance [s/m] ----
    # Sellers (1992) r_ss + Sakaguchi-Zeng (2009) litter, added in SERIES with the
    # below-canopy aerodynamic resistance inside the soil energy balance (see
    # energy_balance.soil_surface_evap_resistance).  Only the MULTILAYER soil path
    # supplies ``soil_surface_relsat`` (top-layer theta_1/theta_sat); the slab path
    # leaves it None -> zero resistance (unchanged legacy behaviour).  The
    # ``soil_evap_series_resistance`` gate is a static Python bool on the config
    # (feature gate — not a traced select), and the short-circuit guarantees the
    # config field is only read when the multilayer caller opted in.
    # ``getattr`` default False so a slab ``LandConfig`` (which has no
    # ``soil_evap_series_resistance`` field) resolves to the documented
    # feature-off semantics instead of raising ``AttributeError`` if a caller
    # ever supplies ``soil_surface_relsat`` with a non-multilayer config
    # (matches the ``use_prognostic_lai`` getattr gate above; the multilayer
    # path always carries the real bool).  Still a static Python bool → the
    # ``if`` is a compile-time feature gate, not a traced select.
    if soil_surface_relsat is not None and getattr(
            land_config, "soil_evap_series_resistance", False):
        # Litter cover is driven by a persistent STRUCTURAL LAI (a deciduous forest
        # floor keeps its litter through the leaf-off season); fall back to the live
        # LAI when the caller does not supply ``litter_LAI``.
        _litter_LAI = getattr(lp, "litter_LAI", None) if lp is not None else None
        if _litter_LAI is None:
            _litter_LAI = LAI
        r_soil_surface = soil_surface_evap_resistance(
            soil_surface_relsat, LAI, land_config.soil_evap_litter_resistance_s_m,
            litter_LAI=_litter_LAI)
    else:
        r_soil_surface = jnp.zeros(ncol)
    # No-PFT fallback Vcmax25: DBF-temperate (C3) / mean C4 grass+crop (C4),
    # not a flat 60/40.  A driver should pre-assign per-column Vcmax25 from
    # ``lookup_vcmax25(pft, climate)`` (canopy.config) when PFTs are known.
    Vc3_leaf   = _get(lp, "Vcmax25_C3_leaf", jnp.full(ncol, VCMAX25_C3_DEFAULT))
    Vc4_leaf   = _get(lp, "Vcmax25_C4_leaf", jnp.full(ncol, VCMAX25_C4_DEFAULT))
    m_C3       = _get(lp, "m_C3",     jnp.full(ncol, _DEFAULT_M_C3))
    m_C4       = _get(lp, "m_C4",     jnp.full(ncol, _DEFAULT_M_C4))
    b0_C3      = _get(lp, "b0_C3",    jnp.full(ncol, _DEFAULT_B0_C3))
    b0_C4      = _get(lp, "b0_C4",    jnp.full(ncol, _DEFAULT_B0_C4))
    alf        = _get(lp, "alf",      jnp.full(ncol, _DEFAULT_ALF))
    # TgC is handled below — the priority order (state EMA > lp > forcing)
    # is set after all the other per-column params are resolved.
    ALB_VIS    = _get(lp, "ALB_VIS",  jnp.full(ncol, _DEFAULT_ALB_VIS))
    ALB_NIR    = _get(lp, "ALB_NIR",  jnp.full(ncol, _DEFAULT_ALB_NIR))
    rz0m       = _get(lp, "rz0m",     jnp.full(ncol, _DEFAULT_RZ0M))
    rd         = _get(lp, "rd",       jnp.full(ncol, _DEFAULT_RD))
    # Characteristic leaf width [m]; None (field absent or unset) -> 0.025 m
    # (Schuepp 1993 midrange), matching PFT_LEAF_WIDTH's default leaf class.
    _d_leaf_in = _get(lp, "d_leaf", None)
    d_leaf = jnp.full(ncol, _DEFAULT_D_LEAF) if _d_leaf_in is None else _d_leaf_in
    # NB: the two-stream ``canopy_longwave_rt`` resolves longwave from the
    # SEPARATE leaf and soil emissivities (cc.epsf / cc.epss) and exports the
    # conservative column ``eps_eff`` used as the surface emissivity below.  The
    # single broadband ``CanopyLandParams.emissivity`` has no unique mapping to
    # that leaf/soil pair, so this scheme uses the config-resolved pair and does
    # NOT consume the broadband field — that field drives the simpler SEB/slab
    # path instead (``simple_seb.compute_seb_fluxes``).  (Threading an OBSERVED
    # broadband emissivity into the two-stream RT would be a deliberate mapping
    # policy — a separate, validated change.)

    # ``TgC`` priority:
    #   1. Caller-supplied ``TgC_override`` (state-carried 30-day EMA from
    #      ``advance_TgC_ema``, threaded by ``step_*_land``).
    #   2. Per-column ``CanopyLandParams.TgC`` (externally prescribed).
    #   3. Instantaneous ``forcing.T_lowest - constants.T_freeze`` (degraded fallback;
    #      not a true 30-day mean — emits sensible but biased Vcmax
    #      acclimation outside any coupled / standalone driver).
    if TgC_override is not None:
        TgC = TgC_override
    else:
        TgC = _get(lp, "TgC", forcing.T_lowest - constants.T_freeze)

    # ---- Soil moisture stress ----
    # Photosynthesis/transpiration down-regulation uses the ROOT-ZONE beta.
    # Bare-soil evaporation is governed by the fast-drying SURFACE layer, not the
    # root zone: when ``w_frac_soil_evap`` (a top-layer availability) is supplied
    # by the caller, use it for ``fStress_soil``; otherwise fall back to the
    # root-zone beta (legacy behaviour).  Using the root-zone beta for soil evap
    # over-estimated forest-floor evaporation (it stays wet while the surface
    # dries), inflating LE and starving H.
    fStress_vcmax = w_frac_rz         # Vcmax / transpiration down-regulation
    fStress_soil  = (w_frac_rz if w_frac_soil_evap is None
                     else w_frac_soil_evap)   # soil evaporation stress

    Vc3_leaf_stressed = Vc3_leaf * fStress_vcmax
    Vc4_leaf_stressed = Vc4_leaf * fStress_vcmax
    m_eff  = m_C3  * fStress_vcmax
    # The Ball-Berry INTERCEPT b0 is the cuticular / residual minimum conductance.
    # ``cc.stress_b0`` (static Python bool) controls whether soil-moisture stress
    # down-regulates it too: True = legacy (b0 also stressed); False keeps the
    # cuticular leak alive under drought (baseline dry-season transpiration; the
    # photosynthesis-linked SLOPE m is still stressed).  ``_b0_stress`` is the
    # traced factor (1) or the plain scalar 1.0 — a compile-time branch, not a
    # traced select.
    _b0_stress = fStress_vcmax if cc.stress_b0 else 1.0
    b0_eff = b0_C3 * _b0_stress
    m_mix  = (1.0 - fC4) * m_eff  + fC4 * m_C4  * fStress_vcmax
    b0_mix = (1.0 - fC4) * b0_eff + fC4 * b0_C4 * _b0_stress

    # ---- Aerodynamics ----
    z0m, displa = compute_aerodynamics(hc, LAI, rz0m, rd)
    # Lift reference height above the displacement + roughness sub-layer.
    Ta, z_ref_base = surface_reference_state(
        forcing.T_lowest, land_config.z_ref, forcing.z_lowest)
    z_ref = jnp.maximum(z_ref_base, displa + 10.0 * z0m + 2.0)

    # ---- SW decomposition ----
    cos_zenith = forcing.cos_zenith
    # Clip strictly below 1 before arccos: d/dx arccos(x) = -1/sqrt(1-x^2)
    # diverges at x=1 (subsolar point) and clip zeroes the subgradient above 1,
    # so the reverse-mode VJP would give 0*inf = NaN in the cos_zenith cotangent.
    SZA = jnp.degrees(jnp.arccos(jnp.clip(cos_zenith, 0.0, 1.0 - 1e-7)))
    PAR_dir, PAR_diff, NIR_dir, NIR_diff, UV = split_sw_components(
        forcing.sw_down, cos_zenith)

    sw_rt = canopy_shortwave_rt(
        PAR_dir, PAR_diff, NIR_dir, NIR_diff, UV,
        SZA, LAI, CI, ALB_VIS, ALB_NIR,
        Vc3_leaf_stressed, Vc4_leaf_stressed, kn)

    # ---- Thermodynamic / atmosphere variables ----
    Ps    = forcing.p_surface
    q_atm = forcing.q_lowest
    rhoa  = forcing.rho_lowest
    Tv_atm = Ta * (1.0 + _VIRT_T_COEF * q_atm)
    lam   = constants.L_v
    Cp    = constants.c_pd
    Ca    = forcing.co2_ppmv

    # ---- Initial Newton state ----
    Ts_old   = T_soil_top
    q_s_init = sat_specific_humidity(Ts_old, Ps)
    q_c_init = 0.5 * (q_s_init + q_atm)
    chi = _CI_CA_C3 - _CI_CA_C3_MINUS_C4 * fC4
    Ci_init = Ca * chi

    cold_state = jnp.stack(
        [Ta, Ta, Ci_init, Ci_init, Ta, q_c_init], axis=-1)  # (ncol, 6)

    # ---- Warm start (numerical cache) ----
    # ``canopy_seed`` is the previous timestep's LAST CONVERGED solution, carried
    # by the caller.  A column whose seed is not finite (never solved yet, or the
    # previous solve failed) OR lies outside the physical box a root may occupy
    # (a cache written before that box was enforced -- restarts heal here) falls
    # back to the cold start above.  The seed is sanitised BEFORE the select:
    # ``where`` evaluates both branches, and a NaN in the discarded one would
    # poison a reverse-mode tangent.
    #
    # The solve's adjoint returns a ZERO cotangent for its seed (``solver.
    # solve_bwd``), but that does NOT make the answer seed-independent: the
    # residual has spurious roots, and which one a far-off seed reaches is the
    # defect the admissibility box exists to reject.
    if canopy_seed is None:
        initial_state = cold_state
    else:
        _seed = jnp.asarray(canopy_seed, cold_state.dtype)
        _ok = canopy_state_admissible(_seed)[:, None]
        initial_state = jnp.where(_ok, jnp.nan_to_num(_seed), cold_state)

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
            cv=_bcast(cc.cv), d_leaf=d_leaf,
            r_soil_surface=r_soil_surface,
            fwet=(jnp.zeros_like(w_frac_rz) if fwet is None
                  else jnp.broadcast_to(fwet, w_frac_rz.shape)),
        )

    def _solve_one_col(x0, bun):
        # Diagnostic entry point: identical solve, plus the terminal residual /
        # damping / cap-exit flag the caller needs to tell a stalled column from
        # a nearly-solved one.  Gradients are unchanged (zero cotangent on every
        # diagnostic output).
        return solve_canopy_closure_diag(x0, bun, cc)

    def _fwd_one_col(xf, bun):
        return canopy_forward(xf, bun, cc.LE_module, cc.stomatal_model,
                               cc.le_cap_mode, cc.use_ta_for_photosynthesis)

    # ---- Outer Picard loop: canopy closure ↔ soil thermal ----
    # See canopy_land.py module notes for the stability analysis.  The
    # soil_thermal_fn callback is responsible for advancing Ts tentatively
    # given the Picard-iterate G; slab and multilayer callers supply
    # different implementations.
    n_picard = getattr(cc, "n_picard", _DEFAULT_N_PICARD)
    omega    = getattr(cc, "picard_omega", _DEFAULT_PICARD_OMEGA)
    Ts_bc_k = Ts_old

    # Each Picard pass RESEEDS from the previous pass's converged solution
    # instead of restarting cold.  Successive passes differ only in ``Ts_bc``,
    # which the relaxation moves by a fraction of a kelvin, so the previous
    # solution is a far better seed than the air temperature — same fixed point,
    # fewer iterations to reach it.  Only a CONVERGED pass earns the right to
    # seed the next one: a failed solve's last iterate is not a root, and
    # reusing it would let one bad pass cascade through the rest.
    # ``x_conv`` holds the last CONVERGED solution per column, NaN until one
    # exists; it is both the next pass's seed and (after the loop) the cache
    # handed back to the caller.
    # A column with no converged solution yet alternates its seed across passes
    # between the caller's seed and the COLD state, so a warm seed that led the
    # solve astray gets a cold retry inside the same call.  The passes run over
    # every column anyway (vmap), so this costs nothing; for a column that never
    # converges, its frozen soil boundary makes later passes repeat passes 0/1.
    # A pass that did not converge must not move the soil boundary either: its
    # ground flux is not physics.  Acceptance stays tied to the LAST pass.
    x_conv = jnp.full_like(initial_state, jnp.nan)
    for _picard_iter in range(n_picard):
        bundles_k = _build_bundle(Ts_bc_k)
        _fallback = initial_state if _picard_iter % 2 == 0 else cold_state
        _seed_k = jnp.where(jnp.all(jnp.isfinite(x_conv), axis=-1, keepdims=True),
                            jnp.nan_to_num(x_conv), _fallback)
        (x_final, n_iters, converged, resid_sq, resid_rel, lam_f,
         hit_cap) = jax.vmap(_solve_one_col)(_seed_k, bundles_k)
        x_conv = jnp.where(converged[:, None], x_final, x_conv)
        # A rejected iterate may be non-finite: evaluate the fluxes of a failed
        # column at the (finite) cold state instead, so no NaN enters a
        # reverse-mode tangent.  Those fluxes are not physics either way -- the
        # column reports converged=False and the caller holds it.
        x_final = jnp.where(converged[:, None], x_final, cold_state)
        fluxes_per_col = jax.vmap(_fwd_one_col)(x_final, bundles_k)

        G_k = jnp.clip(fluxes_per_col["G"], -500.0, 700.0)  # coeff-ok: physical range clamp on ground heat flux [W m-2]
        Ts_thermal = soil_thermal_fn(jnp.where(converged, G_k, 0.0), dt)
        Ts_bc_k = jnp.where(converged,
                            (1.0 - omega) * Ts_bc_k + omega * Ts_thermal,
                            Ts_bc_k)

    # ---- Converged state ----
    # State vector order: [Tf_Sun, Tf_Sh, Ci_Sun, Ci_Sh, Tc, q_c].
    Tf_Sun = x_final[:, 0]
    Tf_Sh  = x_final[:, 1]
    Tc_cvg = x_final[:, 4]   # canopy air-space temperature (aerodynamic node)
    Ts_cvg = Ts_bc_k

    fSun    = sw_rt.fSun
    LE_Sun  = fluxes_per_col["LE_Sun"]
    LE_Sh   = fluxes_per_col["LE_Sh"]
    # Wet-leaf evaporation (interception loss) — the store-sourced share of the
    # canopy latent flux; the caller routes it to the canopy-water store.
    LE_wet_canopy_d = fluxes_per_col["LE_wet_Sun"] + fluxes_per_col["LE_wet_Sh"]
    LE_Soil = fluxes_per_col["LE_Soil"]
    H_Sun   = fluxes_per_col["H_Sun"]
    H_Sh    = fluxes_per_col["H_Sh"]
    H_Soil  = fluxes_per_col["H_Soil"]
    An_Sun  = fluxes_per_col["An_Sun"]        # NET (drives SIF + leaf coupling)
    An_Sh   = fluxes_per_col["An_Sh"]
    Agross_Sun = fluxes_per_col["Agross_Sun"]  # GROSS (carbon-model GPP)
    Agross_Sh  = fluxes_per_col["Agross_Sh"]
    G       = fluxes_per_col["G"]
    gs_Sun  = fluxes_per_col["gs_Sun"]
    gs_Sh   = fluxes_per_col["gs_Sh"]
    ustar   = fluxes_per_col["ustar"]
    Rn_Sun_d  = fluxes_per_col["Rn_Sun"]
    Rn_Sh_d   = fluxes_per_col["Rn_Sh"]
    Rn_Soil_d = fluxes_per_col["Rn_Soil"]

    LE_tot = LE_Sun + LE_Sh + LE_Soil
    # Positive UPWARD: leaf/soil heat warms air; land loses H_tot in Rn-H-LE.
    H_tot  = H_Sun  + H_Sh  + H_Soil
    # GPP is GROSS carbon uptake (BEFORE leaf dark respiration).  The carbon
    # model (carbon_cycle.step_carbon) re-charges foliar MAINTENANCE
    # respiration r_maint_fol*C_fol separately, so exporting NET An here would
    # double-count leaf respiration (once as Rd folded into An, once as
    # r_maint_fol) and bias carbon-use efficiency (NPP/GPP) low.  This matches
    # the SimpleSEB path (carbon/stomata.py: gpp = max(A_gross, 0)*_MC).  NET
    # An_Sun/An_Sh still drive stomatal coupling, the leaf energy/CO2 flux, and
    # SIF (below); only the carbon-facing GPP is gross.
    GPP    = (Agross_Sun + Agross_Sh) * _G_C_PER_UMOL_CO2    # gC m-2 s-1 (GROSS)

    # ---- Optional solar-induced fluorescence (passive TOC diagnostic) ----
    # cc.sif is a static config leaf, so this Python gate does not double-trace.
    # An_Sun/An_Sh and APAR_Sun/APAR_Sh are canopy-integrated per leaf-class
    # (per ground area), so the sunlit+shaded sum is the canopy total.
    # Gamma* MUST use the same temperature the Farquhar An used — the solver
    # takes T_phot = Ta if use_ta_for_photosynthesis else Tf (solver.py) — else
    # the je inversion is inconsistent with the assimilation it inverts.
    # NOTE: the BEPS-SIF je inversion is C3-style (uses Gamma*); for a mixed
    # canopy (fC4 > 0) it is applied to the blended C3/C4 An as a documented
    # BEPS-parity approximation (no separate C4 fluorescence path).
    if cc.sif is not None:
        T_phot_Sun = Ta if cc.use_ta_for_photosynthesis else Tf_Sun
        T_phot_Sh  = Ta if cc.use_ta_for_photosynthesis else Tf_Sh
        sif_out = two_leaf_canopy_sif(
            An_Sun, x_final[:, 2], co2_compensation_point(T_phot_Sun), sw_rt.APAR_Sun,
            An_Sh, x_final[:, 3], co2_compensation_point(T_phot_Sh), sw_rt.APAR_Sh,
            cc.sif)
    else:
        sif_out = None

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

    # ---- Surface emissivity + radiometric temperature (atmosphere-equivalent) ----
    # The conservative ``canopy_longwave_rt`` exports the column LW emissivity
    # ``eps_col = 1 - R_col`` (R_col = column LW reflectance) and the emission-only
    # upward flux ``LW_emit``, chosen so the atmosphere's property-coupling LW
    # boundary ``eps_col*sigma*T_surface^4 + (1-eps_col)*La`` reproduces the
    # canopy's conservative ``LW_out`` EXACTLY (verified to ~1e-13 W/m2 across
    # LAI/emissivities).  These are the physically-correct surface radiative
    # properties: the coupler tile-blends ``eps_eff`` (the column emissivity) and
    # ``T_surface`` and threads them into RRTMGP as the dynamic surface emissivity
    # + skin temperature, so the land->atmosphere LW boundary carries no static-
    # emissivity mismatch.  The raw upward flux is still ``lw_up = LW_out`` with
    # ``lw_net = La - LW_out``.
    LW_out_col = fluxes_per_col["LW_out"]
    eps_eff    = fluxes_per_col["eps_col"]
    LW_emit    = fluxes_per_col["LW_emit"]
    T_surface = (LW_emit / jnp.maximum(eps_eff * constants.sigma_sb, 1e-12)) ** 0.25

    # SW_net and the (conservative) external LW_net = La − LW_out.
    sw_net = (1.0 - alpha_canopy) * forcing.sw_down
    lw_net = forcing.lw_down - LW_out_col
    lw_up_out = LW_out_col   # true top-of-canopy upward LW (emission + reflection)

    # External (boundary-condition) radiation balance — what a downstream
    # observer sees from forcing + canopy-mean emission T.
    Rn_ext_d       = sw_net + lw_net
    residual_ext_d = Rn_ext_d - (LE_tot + H_tot + G)

    # Safety clamp on G: extreme non-converged values corrupt T_soil.
    G_clamped = jnp.clip(G, -500.0, 700.0)  # coeff-ok: physical range clamp on ground heat flux [W m-2]

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
        sif=sif_out,
        # Warm-start cache for the NEXT call: the last CONVERGED solve per
        # column (NaN where this column has never converged, which the seed
        # consumer reads as "cold start").  A numerical cache only — see the
        # ``canopy_seed`` note above.
        canopy_x=x_conv,
        canopy_resid_sq=resid_sq,
        canopy_resid_rel=resid_rel,
        canopy_hit_cap=hit_cap,
        Tf_Sun=Tf_Sun,
        Tf_Sh=Tf_Sh,
        T_canopy_air=Tc_cvg,
        gs_Sun=gs_Sun,
        gs_Sh=gs_Sh,
        n_iters=n_iters,
        # Whether the Newton closure actually reached a root on this column.
        # False means the fluxes above are a stopped iterate, not a solution —
        # the caller decides what to do with the column; it must not simply
        # spend them.  Carried from the LAST Picard pass.
        converged=converged,
        f_veg=f_veg,
        fSun=fSun,
        Ts_solve=Ts_cvg,
        LE_canopy=LE_canopy_d,
        LE_wet_canopy=LE_wet_canopy_d,
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
