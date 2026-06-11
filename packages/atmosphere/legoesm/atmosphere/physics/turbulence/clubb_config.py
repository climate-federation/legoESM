"""Configuration for the fuller CLUBB turbulence scheme (CAM-default flavour).

Three NamedTuples (all JAX pytrees, all snake_case-or-symbol fields per the
legoESM naming rule for capitalized physics symbols):

  * :class:`CLUBBFlags`   — STATIC integer/boolean model flags. Used for
    compile-time feature gating via Python ``if`` on these static fields (NOT
    ``jnp.where``), per the CLAUDE.md feature-gating exception. Defaults are the
    **CAM namelist** values (``namelist_defaults_cam.xml`` ``clubb_*``), which
    OVERRIDE the CLUBB library defaults in several places — see ``PORT_CLUBB.md``.
  * :class:`CLUBBParams`  — the tunable closure coefficients (the CLUBB
    ``clubb_params`` vector). Defaults are the CLUBB library ``_DEFAULTS``
    (``parameters_tunable.F90:set_default_parameters``) with the CAM namelist
    overrides applied (CAM-effective values). These are scheme-tunable params,
    so they live here, NOT in ``legoesm.constants`` (CLAUDE.md).
  * :class:`CLUBBConfig`  — the top-level scheme config bundling flags, params,
    surface-layer config, numerical tolerances, and the CLUBB sub-step.

Physical constants (g, c_pd, L_v, R_d, R_v, kappa, p_ref, ...) are deliberately
ABSENT here: the implementation modules pull them from ``legoesm.constants``.

Reference values are documented inline as ``# lib X -> CAM Y`` where the CAM
namelist overrides the library default. ``cam7``/``silhs`` variant deltas are
noted but NOT applied (we target the base CAM-default config).
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax
from legoesm.atmosphere.physics.turbulence.config import SurfaceLayerConfig


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class CLUBBFlags:
    """STATIC CLUBB model flags, defaulted to the CAM namelist base values.

    Registered as a *static* pytree node (``jax.tree_util.register_static``):
    instances contribute NO array leaves and are carried as hashable aux data
    in the treedef. This is what lets the CLUBB kernels branch on a flag with a
    plain Python ``if`` (the CLAUDE.md feature-gating exception) even when a
    ``CLUBBConfig`` is threaded through a ``jax.jit``/``grad`` boundary —
    ``cfg.flags.l_*`` stays a concrete Python bool, never a tracer. Changing a
    flag changes the treedef and (correctly) triggers recompilation.

    Defaults are the **CAM namelist base** values (``namelist_defaults_cam.xml``
    ``clubb_*``); where a flag is absent from the namelist it falls back to the
    CLUBB library default (``model_flags.F90``) — those are marked
    ``# absent->lib``. Several namelist values OVERRIDE the library default;
    those are marked ``# lib X -> CAM Y``. Integer flags use the enum codes from
    ``model_flags.F90``. See ``PORT_CLUBB.md`` for the full table.
    """

    # ── Integer / enum flags ─────────────────────────────────────────────
    iiPDF_type: int = 1            # ADG1                 (absent->lib)
    ipdf_call_placement: int = 2   # ipdf_post_advance_fields
    saturation_formula: int = 3    # flatau               (absent->lib)
    penta_solve_method: int = 1    # lib 2 -> CAM 1
    tridiag_solve_method: int = 1  # lib 2 -> CAM 1
    grid_remap_method: int = 1     # lib 2(ppm) -> CAM 1
    grid_adapt_in_time_method: int = 0   # no grid adaptation
    fill_holes_type: int = 2       # sliding_window

    # ── Logical flags — CAM namelist base values (⚠ several differ from lib) ──
    l_use_precip_frac: bool = True
    l_predict_upwp_vpwp: bool = False          # lib True -> CAM False
    l_min_wp2_from_corr_wx: bool = False
    l_min_xp2_from_corr_wx: bool = True
    l_C2_cloud_frac: bool = False
    l_diffuse_rtm_and_thlm: bool = False
    l_stability_correct_Kh_N2_zm: bool = False
    l_calc_thlp2_rad: bool = True
    l_upwind_xpyp_ta: bool = True
    l_upwind_xm_ma: bool = True
    l_uv_nudge: bool = False                    # absent->lib
    l_rtm_nudge: bool = False
    l_tke_aniso: bool = True
    l_vert_avg_closure: bool = True            # lib False -> CAM True
    l_trapezoidal_rule_zt: bool = True         # lib False -> CAM True
    l_trapezoidal_rule_zm: bool = True         # lib False -> CAM True
    l_call_pdf_closure_twice: bool = True      # lib False -> CAM True
    l_standard_term_ta: bool = False
    l_partial_upwind_wp3: bool = False
    l_godunov_upwind_wpxp_ta: bool = False
    l_godunov_upwind_xpyp_ta: bool = False
    l_use_cloud_cover: bool = True             # lib False -> CAM True
    l_diagnose_correlations: bool = False
    l_calc_w_corr: bool = False
    l_const_Nc_in_cloud: bool = False
    l_fix_w_chi_eta_correlations: bool = True
    l_stability_correct_tau_zm: bool = True    # lib False -> CAM True
    l_damp_wp2_using_em: bool = False          # lib True -> CAM False
    l_do_expldiff_rtm_thlm: bool = False
    l_Lscale_plume_centered: bool = False
    l_diag_Lscale_from_tau: bool = False       # lib True -> CAM False
    l_use_C7_Richardson: bool = False          # lib True -> CAM False
    l_use_C11_Richardson: bool = False
    l_use_shear_Richardson: bool = False
    l_brunt_vaisala_freq_moist: bool = False
    l_use_thvm_in_bv_freq: bool = False
    l_rcm_supersat_adj: bool = False           # lib True -> CAM False
    l_damp_wp3_Skw_squared: bool = False       # lib True -> CAM False
    l_prescribed_avg_deltaz: bool = False
    l_lmm_stepping: bool = False
    l_e3sm_config: bool = False
    l_vary_convect_depth: bool = False
    l_use_tke_in_wp3_pr_turb_term: bool = False   # lib True -> CAM False
    l_use_tke_in_wp2_wp3_K_dfsn: bool = False
    l_smooth_Heaviside_tau_wpxp: bool = False
    l_enable_relaxed_clipping: bool = False
    l_mono_flux_lim_thlm: bool = True
    l_mono_flux_lim_rtm: bool = True
    l_mono_flux_lim_um: bool = True
    l_mono_flux_lim_vm: bool = True
    l_mono_flux_lim_spikefix: bool = True
    l_host_applies_sfc_fluxes: bool = False    # absent->lib
    l_wp2_fill_holes_tke: bool = True          # absent->lib
    l_add_dycore_grid: bool = False
    l_ascending_grid: bool = False             # CAM namelist
    l_c14_ml: bool = False                     # CAM namelist (neural C14, OFF)
    l_intr_sfc_flux_smooth: bool = False       # CAM namelist (clubb_intr smoothing)


class CLUBBParams(NamedTuple):
    """CLUBB tunable closure coefficients (CAM-effective defaults).

    Library defaults from ``parameters_tunable.F90:set_default_parameters``
    (the ``_DEFAULTS`` dict in the CLUBB-JAX port), with CAM namelist overrides
    applied. ``# lib X -> Y`` marks a CAM override of the library default.
    """

    # ── Return-to-isotropy / pressure-correlation C-coefficients ──────────
    C1: float = 1.0
    C1b: float = 1.0
    C1c: float = 1.0
    C2rt: float = 1.0          # lib 2.0 -> 1.0
    C2thl: float = 1.0         # lib 2.0 -> 1.0
    C2rtthl: float = 1.3       # lib 2.0 -> 1.3
    C4: float = 5.2            # lib 2.0 -> 5.2
    C_uu_shr: float = 0.3      # lib 0.4 -> 0.3   (cam7: 0.1)
    C_uu_buoy: float = 0.3
    C6rt: float = 4.0          # lib 2.0 -> 4.0
    C6rtb: float = 6.0         # lib 2.0 -> 6.0
    C6rtc: float = 1.0
    C6thl: float = 4.0         # lib 2.0 -> 4.0
    C6thlb: float = 6.0        # lib 2.0 -> 6.0
    C6thlc: float = 1.0
    C7: float = 0.5            # (cam7: 0.1)
    C7b: float = 0.5
    C7c: float = 0.5
    C8: float = 4.2            # lib 0.5 -> 4.2   (cam7: 4.6)
    C8b: float = 0.0           # lib 0.02 -> 0.0
    C10: float = 3.3
    C11: float = 0.7           # lib 0.4 -> 0.7
    C11b: float = 0.35         # lib 0.4 -> 0.35
    C11c: float = 0.5
    C12: float = 1.0
    C13: float = 0.1
    C14: float = 2.2           # lib 1.0 -> 2.2
    C_wp2_pr_dfsn: float = 0.0
    C_wp3_pr_tp: float = 0.0
    C_wp3_pr_turb: float = 0.4    # lib 0.0 -> 0.4
    C_wp3_pr_dfsn: float = 0.0
    C_wp2_splat: float = 0.0      # lib 2.0 -> 0.0

    # ── Lscale-zero blending coefficients ─────────────────────────────────
    C6rt_Lscale0: float = 14.0
    C6thl_Lscale0: float = 14.0
    C7_Lscale0: float = 0.85
    wpxp_L_thresh: float = 60.0

    # ── Eddy-diffusion (c_K*) and background-diffusion (nu*) coefficients ──
    c_K: float = 0.2
    c_K1: float = 0.75         # lib 0.2 -> 0.75
    nu1: float = 20.0
    c_K2: float = 0.125        # lib 0.025 -> 0.125
    nu2: float = 5.0           # lib 1.0 -> 5.0
    c_K6: float = 0.375
    nu6: float = 5.0
    c_K8: float = 1.25         # lib 5.0 -> 1.25
    nu8: float = 20.0
    c_K9: float = 0.25         # lib 0.1 -> 0.25
    nu9: float = 20.0          # lib 10.0 -> 20.0
    nu10: float = 0.0
    c_K10: float = 0.5         # lib 1.0 -> 0.5
    c_K10h: float = 0.3        # lib 1.0 -> 0.3  (cam7: 0.280)

    # ── Hydrometeor-diffusion (OFF in CAM default; kept for completeness) ──
    c_K_hm: float = 0.75
    c_K_hmb: float = 0.75
    K_hm_min_coef: float = 0.1
    nu_hm: float = 1.5

    # ── PDF (ADG1) spread / skewness coefficients ─────────────────────────
    slope_coef_spread_DG_means_w: float = 21.0
    pdf_component_stdev_factor_w: float = 1.0
    coef_spread_DG_means_rt: float = 0.8
    coef_spread_DG_means_thl: float = 0.8
    gamma_coef: float = 0.308       # lib 0.25 -> 0.308 (cam7: 0.3)
    gamma_coefb: float = 0.32       # lib 0.25 -> 0.32  (cam7: 0.3)
    gamma_coefc: float = 5.0
    Skw_denom_coef: float = 0.0     # lib 4.0 -> 0.0
    Skw_max_mag: float = 4.5        # lib 10.0 -> 4.5

    # ── Mixing length / time scale ────────────────────────────────────────
    mu: float = 1.0e-3
    beta: float = 2.4               # lib 1.0 -> 2.4
    lmin_coef: float = 0.1          # lib 0.5 -> 0.1
    Lscale_mu_coef: float = 2.0
    Lscale_pert_coef: float = 0.1
    lambda0_stability_coef: float = 0.04   # lib 0.03 -> 0.04
    mult_coef: float = 1.0          # lib 0.5 -> 1.0
    taumin: float = 90.0
    taumax: float = 3600.0
    alpha_corr: float = 0.15

    # ── Inverse-tau (dissipation time scale) coefficients ─────────────────
    C_invrs_tau_bkgnd: float = 1.0          # lib 1.1 -> 1.0
    C_invrs_tau_sfc: float = 0.1
    C_invrs_tau_shear: float = 0.02         # lib 0.15 -> 0.02
    C_invrs_tau_N2: float = 0.1             # lib 0.4 -> 0.1
    C_invrs_tau_N2_wp2: float = 0.2
    C_invrs_tau_N2_xp2: float = 0.2         # lib 0.05 -> 0.2
    C_invrs_tau_N2_wpxp: float = 0.0
    C_invrs_tau_N2_clear_wp3: float = 0.0   # lib 1.0 -> 0.0
    C_invrs_tau_wpxp_Ri: float = 0.35
    C_invrs_tau_wpxp_N2_thresh: float = 3.3e-4

    # ── Misc closure / clipping / Richardson coefficients ─────────────────
    omicron: float = 0.5
    zeta_vrnce_rat: float = 0.0
    upsilon_precip_frac_rat: float = 0.55
    thlp2_rad_coef: float = 1.0
    thlp2_rad_cloud_frac_thresh: float = 0.1
    up2_sfc_coef: float = 2.0               # lib 4.0 -> 2.0
    xp3_coef_base: float = 0.25
    xp3_coef_slope: float = 0.01
    altitude_threshold: float = 100.0
    rtp2_clip_coef: float = 0.5
    Cx_min: float = 0.33
    Cx_max: float = 0.95
    Richardson_num_min: float = 0.25
    Richardson_num_max: float = 400.0
    a3_coef_min: float = 1.0
    a_const: float = 1.8
    bv_efold: float = 5.0
    wpxp_Ri_exp: float = 0.5
    z_displace: float = 25.0


class CLUBBConfig(NamedTuple):
    """Top-level configuration for the fuller CLUBB turbulence scheme.

    Fields
    ------
    flags : CLUBBFlags
        Static model flags (CAM-default).
    params : CLUBBParams
        Tunable closure coefficients (CAM-effective defaults).
    surface : SurfaceLayerConfig
        Bulk surface-flux configuration (shared with the other turbulence
        schemes).
    clubb_dt : float
        CLUBB internal sub-step [s] (``clubb_timestep`` namelist default
        300 s). The host physics ``dt`` may be sub-cycled to this.
    w_tol : float
        Tolerance / floor for w-moments ``sqrt(wp2)`` [m/s] (``w_tol``).
    rt_tol : float
        Tolerance for total water mixing ratio [kg/kg] (``rt_tol``).
    thl_tol : float
        Tolerance for liquid-water potential temperature [K] (``thl_tol``).
    wp2_max : float
        Upper clip for ``wp2`` [m^2/s^2] (``wp2_max``).
    tke_min : float
        Floor for the carried TKE/wp2 state slot [m^2/s^2] (mirrors the other
        prognostic-moment schemes so ``integration.py`` can seed the state).
    T0 : float
        Reference temperature [K] for the dry Brunt-Vaisala frequency
        ``N^2 = (g/T0) d(thlm)/dz`` (CLUBB ``T0``; CAM standard 300 K). A fixed
        reference (not a tunable closure coefficient), passed to
        ``calc_brunt_vaisala_freq_sqd``.
    """

    flags: CLUBBFlags = CLUBBFlags()
    params: CLUBBParams = CLUBBParams()
    surface: SurfaceLayerConfig = SurfaceLayerConfig()
    clubb_dt: float = 300.0
    w_tol: float = 2.0e-2
    rt_tol: float = 1.0e-8
    thl_tol: float = 1.0e-2
    wp2_max: float = 1000.0
    tke_min: float = 1.0e-6
    T0: float = 300.0


# ---------------------------------------------------------------------------
# Derived parameters (recomputed from base config, never stored as magic
# numbers — Oracle-recipe fidelity doctrine). Faithful to
# ``parameters_tunable.F90:setup_parameters`` / ``CLUBB-JAX`` derivation.
# ---------------------------------------------------------------------------
import math as _math  # noqa: E402

# Reference sigma_sqd_w used in the mixt_frac cap derivation (NOT a tunable):
_MIXT_FRAC_CAP_SIGMA_REF = 0.4
# Reference layer depth [m] scaling lmin (parameters_tunable.F90 ``lmin_deltaz``).
_LMIN_DELTAZ = 40.0


def derive_mixt_frac_max_mag(Skw_max_mag: float) -> float:
    """Maximum |mixt_frac - 0.5| + 0.5 cap, derived from ``Skw_max_mag``.

    ``1 - 0.5*(1 - Skw_max/sqrt(4*(1-0.4)^3 + Skw_max^2))`` — the ADG1 mixture
    fraction evaluated at the maximum allowed w-skewness (so the runtime
    ``mixt_frac`` clip is consistent with ``Skw_max_mag``). With the CAM default
    ``Skw_max_mag = 4.5`` this is ~0.9897.
    """
    inner = 4.0 * (1.0 - _MIXT_FRAC_CAP_SIGMA_REF) ** 3 + Skw_max_mag ** 2
    return 1.0 - 0.5 * (1.0 - Skw_max_mag / _math.sqrt(inner))


def derive_lmin(lmin_coef: float) -> float:
    """Minimum mixing length [m] = ``lmin_coef * lmin_deltaz`` (40 m).

    With the CAM default ``lmin_coef = 0.1`` this is 4 m.
    """
    return lmin_coef * _LMIN_DELTAZ
