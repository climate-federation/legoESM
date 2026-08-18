"""CLUBB higher-order turbulence closure (fuller port; ``scheme="clubb"``).

This is the single-file home for the fuller CLUBB port tracked in
``docs/dev-notes/clubb.md`` — substantially richer than :mod:`clubb_lite` — restricted to
the call tree exercised by the **CAM-default CLUBB flags** (every piece
golden-locked or parity-tested against CLUBB-JAX). Per the legoESM
one-file-per-scheme convention, the remaining ``clubb_*.py`` helper modules are
being absorbed here section by section (see the table of contents below); the
CAM-default model-flag values are recorded as comments at the end of the file.

Table of contents (sections, in order; flag reference table at line 6000)
-----------------------------------------------------------------------------
  1.  [line   328] Diagnostic ADG1-PDF closure (``diagnose_cloud_and_buoyancy``)
  2.  [line   411] Configuration (``CLUBBParams`` / ``CLUBBConfig`` + derived params;
      model flags fixed at CAM defaults — reference table at file end)
  3.  [line   641] Staggered CLUBB grid (``CLUBBGrid`` / zm-zt operators /
      ``make_clubb_grid[_from_levels]`` / ``flip_vertical``)
  4.  [line   972] Flatau saturation adapters (``sat_mixrat_liq``/``sat_mixrat_ice`` over
      the canonical ``legoesm.thermo`` curves)
  5.  [line  1026] Closure helpers (``safe_sqrt`` / ``compute_sigma_sqd_w`` /
      ``calc_brunt_vaisala_freq_sqd``)
  6.  [line  1200] Parcel buoyant-sorting mixing length (``compute_mixing_length`` /
      ``set_Lscale_max``)
  7.  [line  1635] Implicit band solvers (``tridiag_solve`` / ``penta_solve``)
  8.  [line  1763] Mass-conserving hole filling (``fill_holes_vertical`` /
      ``fill_holes_wp2_from_horz_tke``)
  9.  [line  1944] Skewness diagnostics (``Skx_func`` / ``compute_gamma_Skw`` / LG05 /
      ``compute_skewness_diagnostics``)
  10. [line  2130] Dissipation time-scale family (``compute_tke`` / ``compute_tau_family``)
  11. [line  2211] ADG1 assumed-PDF parameter closure (``ADG1_pdf_driver`` + the liquid
      cloud-fraction closure)
  12. [line  2533] ADG1 PDF moment integrals + buoyancy-flux assembly
      (``calc_pdf_higher_order_moments`` / ``calc_pdf_xprcp_fluxes`` /
      ``calc_xpthvp_terms``)
  13. [line  2777] Moment-advance building blocks + the xp2_xpyp / windm advances
      (diffusion/mean-advection LHS builders, Cauchy-Schwarz clips,
      ``advance_xp2_xpyp`` / ``advance_windm_edsclrm``)
  14. [line  3602] Skewness-dependent C-coefficient family (``compute_skw_fnc`` users:
      ``damp_coefficient`` / ``compute_C6_C7_Skw_fnc``)
  15. [line  3663] Coupled wp2/wp3 advance (``advance_wp2_wp3`` + penta LHS/RHS builders +
      ``clip_skewness``)
  16. [line  4321] Monotonic turbulent-flux limiter (``monotonic_turbulent_flux_limit`` +
      ``calc_turb_adv_range``)
  17. [line  4601] Coupled xm/wpxp advance (``advance_xm_wpxp`` + the monotonic-flux-limiter
      coupling + ``solve_xm_wpxp_with_single_lhs``)
  18. [line  4975] Core orchestration (``compute_clubb_diagnostics`` /
      ``compute_pdf_closure`` / ``advance_clubb_core`` + the
      ``CLUBBMomentState``/``CLUBBForcing`` carry types and pack/unpack)
  19. [line  5371] Scheme entries (``clubb_turbulence`` diagnostic default /
      ``clubb_turbulence_prognostic`` opt-in / ``clubb_step`` bridge /
      ``integrate_clubb_column`` SCM driver)

Phasing (the scheme is wired in and runnable now; fidelity deepens per phase):

  * **Phase 1 (diagnostic default):** uses CLUBB's exact **parcel
    buoyant-sorting length scale** ``Lscale`` (``compute_mixing_length``,
    golden-locked vs CLUBB-JAX) to set the eddy diffusivity
    ``Km = c_K · Lscale · sqrt(em)`` — the distinctive CLUBB feature, replacing
    clubb_lite's Blackadar length — and diagnoses the cloud fraction from the
    ADG1 PDF. Mean fields (u, v, T, q_v) are advanced by the implicit eddy
    diffusion shared with the other legoESM schemes; ``wp2`` is carried (via
    the ``tke`` slot) with a production / dissipation / diffusion budget whose
    dissipation time scale is ``tau = Lscale / sqrt(em)``.
  * **Phase 2 (prognostic, opt-in via ``CLUBBConfig.prognostic=True``):** the
    full prognostic higher-order moment transport — ``advance_wp2_wp3``,
    ``advance_xp2_xpyp``, ``advance_xm_wpxp`` — coupled through the ADG1 PDF
    buoyancy flux ``wpthvp`` and the implicit tridiag/penta solves, with the
    15-field :class:`CLUBBMomentState` carried in
    ``PhysicsState.clubb_moments``.

Faithfulness status vs CLUBB (larson-group/clubb_release CLUBB_core Fortran,
via the CLUBB-JAX port) — updated 2026-07-17
-----------------------------------------------------------------------------
FAITHFUL (mechanically pinned):

* Every piece on the CAM-default call tree is golden-locked or round-off
  parity-tested against CLUBB-JAX (committed ``tests/unit/clubb_fixtures/``
  golden ``.npz`` + live parity when the sibling checkout is present):
  Lscale, ADG1 PDF closure + higher-order moments, xp2_xpyp / windm /
  xm_wpxp / wp2_wp3 advances, band solvers, hole filling, clipping.
* The CAM-default branches CLUBB-JAX does NOT implement (it carries the
  ARM/True paths) are pinned DIRECTLY against independent transcriptions of
  the CLUBB Fortran in ``tests/unit/test_clubb_cam_branch_fortran_pins.py``
  (rel 1e-12): ``compute_skw_fnc``, ``damp_coefficient``,
  ``compute_C6_C7_Skw_fnc`` (advance_xm_wpxp_module.F90:640-700, 5990-6048),
  ``wp2_term_dp1_rhs`` (``l_damp_wp2_using_em=.false.``), and
  ``wp3_term_pr_turb_rhs`` (``l_use_tke_in_wp3_pr_turb_term=.false.``,
  advance_wp2_wp3_module.F90:5350-5410).

KNOWN GAPS / DEPARTURES (documented, deliberate):

* No closed-form WHOLE-SCHEME oracle exists: CLUBB is an iterated implicit
  PDF closure, so scheme-level behaviour is verified via the per-piece pins
  plus behavioural invariants (even-moment non-negativity, budget checks),
  not a single end-to-end golden run — a structural property of the scheme,
  not a defect.
* Restricted to the CAM-default flag tree (table at file end); non-default
  CLUBB branches are not ported.
* ``conserves: none`` — the column moment budgets are open by construction
  in the diagnostic default (Phase 1); see ``__physics_contract__``.

The ``TurbulenceOutput`` contract carries no ``cloud_fraction`` field (see the
clubb_lite docstring), so the PDF cloud fraction is computed and exposed only
through diagnostics for now; the eddy-diffusion tendencies are the live output.

No ``__all__`` is declared (intentional): the public scheme API is the four
entries in the final section; everything else is closure machinery kept
importable for the per-piece parity/oracle unit tests, which address symbols
explicitly. (The absorbed helper modules' ``__all__`` lists were dropped with
the modules.)
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from legoesm.atmosphere.physics._shared import (
    buoyancy_coefficient,
    exner_function,
    virtual_temperature,
)
from legoesm.atmosphere.physics.turbulence.config import SurfaceLayerConfig
from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput
from legoesm.atmosphere.physics.turbulence.pbl_height import diagnose_pbl_height
from legoesm.atmosphere.physics.turbulence.surface_layer import compute_surface_fluxes
from legoesm.atmosphere.physics.turbulence.vertical_diffusion import (
    implicit_vertical_diffusion,
    implicit_vertical_diffusion_theta,
)
from legoesm.thermo import (
    saturation_vapor_pressure_flatau,
    saturation_vapor_pressure_ice_flatau,
)
from legoesm.timestepping.tridiagonal import thomas_solve

from legoesm import constants

# Machine-checked scheme contract for the public entries in section 19 (see
# tests/test_physics_contracts.py). The architect pins units/signs/reference;
# the body must honour it.
__physics_contract__ = {
    "summary": (
        "CLUBB higher-order turbulence closure (phase-1 diagnostic default "
        "clubb_turbulence): down-gradient eddy diffusion using CLUBB's parcel "
        "buoyant-sorting length scale Lscale and an ADG1 double-Gaussian PDF "
        "buoyancy flux; the wp2 (w'^2) moment is carried and advanced."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "T": "K", "q_v": "kg/kg",
        "tke": "m^2/s^2 (carries wp2 = w'^2)",
        "p_full": "Pa", "p_half": "Pa", "z_full": "m", "z_half": "m",
        "T_sfc": "K", "q_sfc": "kg/kg", "rho": "kg/m^3", "dt": "s",
    },
    "outputs": {
        "du_dt": "m/s^2", "dv_dt": "m/s^2", "dT_dt": "K/s", "dq_v_dt": "kg/kg/s",
        "Km": "m^2/s", "Kh": "m^2/s", "shflx": "W/m^2", "lhflx": "W/m^2",
        "ustar": "m/s", "h_pbl": "m", "wp2_new": "m^2/s^2 (updated w'^2)",
    },
    "sign_convention": (
        "Down-gradient eddy diffusion, Km >= 0, Kh = Km/Pr_t >= 0; PDF "
        "buoyancy flux wpthvp produces wp2 in unstable layers. The column "
        "budget is OPEN: the surface flux (shflx > 0 upward, lhflx > 0 "
        "upward/moistening) is injected as the bottom boundary condition and "
        "the top is zero-flux; z increases upward."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Golaz, Larson & Cotton (2002), J. Atmos. Sci. 59, 3540-3551; "
        "Larson & Golaz (2005), J. Atmos. Sci. 62, 3620-3649; "
        "Larson (2022) arXiv:1711.03675"
    ),
    "idealized_test": (
        "no surface flux + well-mixed neutral column -> near-zero interior "
        "tendency; Km, Kh >= 0; wp2 stays in [tke_min, wp2_max]; per-piece "
        "parity vs CLUBB-JAX for Lscale and the ADG1 liquid cloud fraction."
    ),
}

# Machine-readable tunable/fixed split (see tests/test_param_specs.py). The
# tunable closure coefficients live on the NESTED CLUBBParams tuple
# (CLUBBConfig.params), so the spec is keyed by BOTH classes: CLUBBParams carries
# the trainable pressure/PDF/diffusion/dissipation coefficients (scheme_key
# "atm.turb.CLUBBParams"), while CLUBBConfig's own float fields are tolerances /
# sub-step / a reference temperature — all excluded (numerics/fixed). Because the
# coefficients are one level down, a consumer applies overrides to config.params
# and re-wraps (CLUBBConfig._replace(params=...)); see the nested-override path in
# the SCM parameter-tuning drivers.
__param_spec__ = {
    "CLUBBParams": {
        "scheme_key": "atm.turb.CLUBBParams",
        "excluded": {
            "C8b": "off (0.0) in the CAM default flag tree; sigmoid cannot seed a lower-bound default — enable via a dedicated activation study",
            "C_invrs_tau_N2_clear_wp3": "off (0.0) in the CAM default flag tree; sigmoid cannot seed a lower-bound default — enable via a dedicated activation study",
            "C_invrs_tau_N2_wpxp": "off (0.0) in the CAM default flag tree; sigmoid cannot seed a lower-bound default — enable via a dedicated activation study",
            "C_wp2_pr_dfsn": "off (0.0) in the CAM default flag tree; sigmoid cannot seed a lower-bound default — enable via a dedicated activation study",
            "C_wp2_splat": "off (0.0) in the CAM default flag tree; sigmoid cannot seed a lower-bound default — enable via a dedicated activation study",
            "C_wp3_pr_dfsn": "off (0.0) in the CAM default flag tree; sigmoid cannot seed a lower-bound default — enable via a dedicated activation study",
            "C_wp3_pr_tp": "off (0.0) in the CAM default flag tree; sigmoid cannot seed a lower-bound default — enable via a dedicated activation study",
            "Cx_max": "numerics: Cx pressure-coefficient clip upper bound",
            "Cx_min": "numerics: Cx pressure-coefficient clip lower bound",
            "K_hm_min_coef": "hydrometeor diffusion off in the CAM default flag tree (not read)",
            "Richardson_num_max": "numerics: Richardson-number clip upper bound",
            "Richardson_num_min": "numerics: Richardson-number clip lower bound",
            "Skw_denom_coef": "numerics: skewness denominator regularizer (off, 0.0, in CAM default)",
            "a3_coef_min": "numerics: a3 PDF-integral coefficient floor",
            "altitude_threshold": "numerics: altitude threshold [m] for near-surface treatment",
            "c_K_hm": "hydrometeor diffusion off in the CAM default flag tree (not read)",
            "c_K_hmb": "hydrometeor diffusion off in the CAM default flag tree (not read)",
            "nu10": "off (0.0) in the CAM default flag tree; sigmoid cannot seed a lower-bound default — enable via a dedicated activation study",
            "nu_hm": "hydrometeor diffusion off in the CAM default flag tree (not read)",
            "rtp2_clip_coef": "numerics: rtp2 Cauchy-Schwarz clip coefficient",
            "taumax": "numerics: dissipation-timescale upper clip [s]",
            "taumin": "numerics: dissipation-timescale lower clip [s]",
            "thlp2_rad_cloud_frac_thresh": "numerics: cloud-fraction gate threshold for the radiative thlp2 term",
            "zeta_vrnce_rat": "off (0.0) in the CAM default flag tree; sigmoid cannot seed a lower-bound default — enable via a dedicated activation study",
        },
        "params": {
            "C1": {"units": "1", "bounds": (0.3, 3.0), "tunable_tier": 1, "transform": "sigmoid", "category": "pressure_correlation", "reference": "Golaz, Larson & Cotton (2002) CLUBB pressure/return-to-isotropy C-coefficients (parameters_tunable.F90)", "shape": None},
            "C10": {"units": "1", "bounds": (1.0, 6.0), "tunable_tier": 2, "transform": "sigmoid", "category": "pressure_correlation", "reference": "Golaz, Larson & Cotton (2002) CLUBB pressure/return-to-isotropy C-coefficients (parameters_tunable.F90)", "shape": None},
            "C11": {"units": "1", "bounds": (0.1, 1.0), "tunable_tier": 1, "transform": "sigmoid", "category": "pressure_correlation", "reference": "Golaz, Larson & Cotton (2002) CLUBB pressure/return-to-isotropy C-coefficients (parameters_tunable.F90)", "shape": None},
            "C11b": {"units": "1", "bounds": (0.1, 1.0), "tunable_tier": 3, "transform": "sigmoid", "category": "pressure_correlation", "reference": "Golaz, Larson & Cotton (2002) CLUBB pressure/return-to-isotropy C-coefficients (parameters_tunable.F90)", "shape": None},
            "C11c": {"units": "1", "bounds": (0.1, 1.0), "tunable_tier": 3, "transform": "sigmoid", "category": "pressure_correlation", "reference": "Golaz, Larson & Cotton (2002) CLUBB pressure/return-to-isotropy C-coefficients (parameters_tunable.F90)", "shape": None},
            "C12": {"units": "1", "bounds": (0.3, 3.0), "tunable_tier": 2, "transform": "sigmoid", "category": "pressure_correlation", "reference": "Golaz, Larson & Cotton (2002) CLUBB pressure/return-to-isotropy C-coefficients (parameters_tunable.F90)", "shape": None},
            "C13": {"units": "1", "bounds": (0.02, 1.0), "tunable_tier": 3, "transform": "sigmoid", "category": "pressure_correlation", "reference": "Golaz, Larson & Cotton (2002) CLUBB pressure/return-to-isotropy C-coefficients (parameters_tunable.F90)", "shape": None},
            "C14": {"units": "1", "bounds": (0.5, 6.0), "tunable_tier": 1, "transform": "sigmoid", "category": "pressure_correlation", "reference": "Golaz, Larson & Cotton (2002) CLUBB pressure/return-to-isotropy C-coefficients (parameters_tunable.F90)", "shape": None},
            "C1b": {"units": "1", "bounds": (0.3, 3.0), "tunable_tier": 3, "transform": "sigmoid", "category": "pressure_correlation", "reference": "Golaz, Larson & Cotton (2002) CLUBB pressure/return-to-isotropy C-coefficients (parameters_tunable.F90)", "shape": None},
            "C1c": {"units": "1", "bounds": (0.3, 3.0), "tunable_tier": 3, "transform": "sigmoid", "category": "pressure_correlation", "reference": "Golaz, Larson & Cotton (2002) CLUBB pressure/return-to-isotropy C-coefficients (parameters_tunable.F90)", "shape": None},
            "C2rt": {"units": "1", "bounds": (0.3, 3.0), "tunable_tier": 2, "transform": "sigmoid", "category": "pressure_correlation", "reference": "Golaz, Larson & Cotton (2002) CLUBB pressure/return-to-isotropy C-coefficients (parameters_tunable.F90)", "shape": None},
            "C2rtthl": {"units": "1", "bounds": (0.3, 3.0), "tunable_tier": 2, "transform": "sigmoid", "category": "pressure_correlation", "reference": "Golaz, Larson & Cotton (2002) CLUBB pressure/return-to-isotropy C-coefficients (parameters_tunable.F90)", "shape": None},
            "C2thl": {"units": "1", "bounds": (0.3, 3.0), "tunable_tier": 2, "transform": "sigmoid", "category": "pressure_correlation", "reference": "Golaz, Larson & Cotton (2002) CLUBB pressure/return-to-isotropy C-coefficients (parameters_tunable.F90)", "shape": None},
            "C4": {"units": "1", "bounds": (1.0, 10.0), "tunable_tier": 2, "transform": "sigmoid", "category": "pressure_correlation", "reference": "Golaz, Larson & Cotton (2002) CLUBB pressure/return-to-isotropy C-coefficients (parameters_tunable.F90)", "shape": None},
            "C6rt": {"units": "1", "bounds": (1.0, 10.0), "tunable_tier": 2, "transform": "sigmoid", "category": "pressure_correlation", "reference": "Golaz, Larson & Cotton (2002) CLUBB pressure/return-to-isotropy C-coefficients (parameters_tunable.F90)", "shape": None},
            "C6rt_Lscale0": {"units": "1", "bounds": (2.0, 30.0), "tunable_tier": 3, "transform": "sigmoid", "category": "lscale_zero_blend", "reference": "CLUBB Lscale->0 pressure-term blending coefficients (parameters_tunable.F90)", "shape": None},
            "C6rtb": {"units": "1", "bounds": (1.0, 12.0), "tunable_tier": 2, "transform": "sigmoid", "category": "pressure_correlation", "reference": "Golaz, Larson & Cotton (2002) CLUBB pressure/return-to-isotropy C-coefficients (parameters_tunable.F90)", "shape": None},
            "C6rtc": {"units": "1", "bounds": (0.3, 3.0), "tunable_tier": 3, "transform": "sigmoid", "category": "pressure_correlation", "reference": "Golaz, Larson & Cotton (2002) CLUBB pressure/return-to-isotropy C-coefficients (parameters_tunable.F90)", "shape": None},
            "C6thl": {"units": "1", "bounds": (1.0, 10.0), "tunable_tier": 2, "transform": "sigmoid", "category": "pressure_correlation", "reference": "Golaz, Larson & Cotton (2002) CLUBB pressure/return-to-isotropy C-coefficients (parameters_tunable.F90)", "shape": None},
            "C6thl_Lscale0": {"units": "1", "bounds": (2.0, 30.0), "tunable_tier": 3, "transform": "sigmoid", "category": "lscale_zero_blend", "reference": "CLUBB Lscale->0 pressure-term blending coefficients (parameters_tunable.F90)", "shape": None},
            "C6thlb": {"units": "1", "bounds": (1.0, 12.0), "tunable_tier": 2, "transform": "sigmoid", "category": "pressure_correlation", "reference": "Golaz, Larson & Cotton (2002) CLUBB pressure/return-to-isotropy C-coefficients (parameters_tunable.F90)", "shape": None},
            "C6thlc": {"units": "1", "bounds": (0.3, 3.0), "tunable_tier": 3, "transform": "sigmoid", "category": "pressure_correlation", "reference": "Golaz, Larson & Cotton (2002) CLUBB pressure/return-to-isotropy C-coefficients (parameters_tunable.F90)", "shape": None},
            "C7": {"units": "1", "bounds": (0.1, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "pressure_correlation", "reference": "Golaz, Larson & Cotton (2002) CLUBB pressure/return-to-isotropy C-coefficients (parameters_tunable.F90)", "shape": None},
            "C7_Lscale0": {"units": "1", "bounds": (0.1, 1.0), "tunable_tier": 3, "transform": "sigmoid", "category": "lscale_zero_blend", "reference": "CLUBB Lscale->0 pressure-term blending coefficients (parameters_tunable.F90)", "shape": None},
            "C7b": {"units": "1", "bounds": (0.1, 1.0), "tunable_tier": 3, "transform": "sigmoid", "category": "pressure_correlation", "reference": "Golaz, Larson & Cotton (2002) CLUBB pressure/return-to-isotropy C-coefficients (parameters_tunable.F90)", "shape": None},
            "C7c": {"units": "1", "bounds": (0.1, 1.0), "tunable_tier": 3, "transform": "sigmoid", "category": "pressure_correlation", "reference": "Golaz, Larson & Cotton (2002) CLUBB pressure/return-to-isotropy C-coefficients (parameters_tunable.F90)", "shape": None},
            "C8": {"units": "1", "bounds": (1.0, 8.0), "tunable_tier": 1, "transform": "sigmoid", "category": "pressure_correlation", "reference": "Golaz, Larson & Cotton (2002) CLUBB pressure/return-to-isotropy C-coefficients (parameters_tunable.F90)", "shape": None},
            "C_invrs_tau_N2": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "dissipation_timescale", "reference": "CLUBB inverse dissipation-time-scale coefficients C_invrs_tau_* (Guo et al. 2021 tau reformulation)", "shape": None},
            "C_invrs_tau_N2_wp2": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "dissipation_timescale", "reference": "CLUBB inverse dissipation-time-scale coefficients C_invrs_tau_* (Guo et al. 2021 tau reformulation)", "shape": None},
            "C_invrs_tau_N2_xp2": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "dissipation_timescale", "reference": "CLUBB inverse dissipation-time-scale coefficients C_invrs_tau_* (Guo et al. 2021 tau reformulation)", "shape": None},
            "C_invrs_tau_bkgnd": {"units": "1", "bounds": (0.1, 3.0), "tunable_tier": 2, "transform": "sigmoid", "category": "dissipation_timescale", "reference": "CLUBB inverse dissipation-time-scale coefficients C_invrs_tau_* (Guo et al. 2021 tau reformulation)", "shape": None},
            "C_invrs_tau_sfc": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "dissipation_timescale", "reference": "CLUBB inverse dissipation-time-scale coefficients C_invrs_tau_* (Guo et al. 2021 tau reformulation)", "shape": None},
            "C_invrs_tau_shear": {"units": "1", "bounds": (0.0, 0.3), "tunable_tier": 2, "transform": "sigmoid", "category": "dissipation_timescale", "reference": "CLUBB inverse dissipation-time-scale coefficients C_invrs_tau_* (Guo et al. 2021 tau reformulation)", "shape": None},
            "C_invrs_tau_wpxp_N2_thresh": {"units": "1/s^2", "bounds": (1e-05, 0.001), "tunable_tier": 3, "transform": "sigmoid", "category": "dissipation_timescale", "reference": "CLUBB inverse dissipation-time-scale coefficients C_invrs_tau_* (Guo et al. 2021 tau reformulation)", "shape": None},
            "C_invrs_tau_wpxp_Ri": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 3, "transform": "sigmoid", "category": "dissipation_timescale", "reference": "CLUBB inverse dissipation-time-scale coefficients C_invrs_tau_* (Guo et al. 2021 tau reformulation)", "shape": None},
            "C_uu_buoy": {"units": "1", "bounds": (0.05, 0.8), "tunable_tier": 2, "transform": "sigmoid", "category": "pressure_correlation", "reference": "Golaz, Larson & Cotton (2002) CLUBB pressure/return-to-isotropy C-coefficients (parameters_tunable.F90)", "shape": None},
            "C_uu_shr": {"units": "1", "bounds": (0.05, 0.8), "tunable_tier": 2, "transform": "sigmoid", "category": "pressure_correlation", "reference": "Golaz, Larson & Cotton (2002) CLUBB pressure/return-to-isotropy C-coefficients (parameters_tunable.F90)", "shape": None},
            "C_wp3_pr_turb": {"units": "1", "bounds": (0.0, 1.5), "tunable_tier": 2, "transform": "sigmoid", "category": "pressure_correlation", "reference": "Golaz, Larson & Cotton (2002) CLUBB pressure/return-to-isotropy C-coefficients (parameters_tunable.F90)", "shape": None},
            "Lscale_mu_coef": {"units": "1", "bounds": (0.5, 5.0), "tunable_tier": 2, "transform": "sigmoid", "category": "mixing_length", "reference": "CLUBB parcel buoyant-sorting mixing-length coefficients (Larson et al.; parameters_tunable.F90)", "shape": None},
            "Lscale_pert_coef": {"units": "1", "bounds": (0.0, 0.5), "tunable_tier": 3, "transform": "sigmoid", "category": "mixing_length", "reference": "CLUBB parcel buoyant-sorting mixing-length coefficients (Larson et al.; parameters_tunable.F90)", "shape": None},
            "Skw_max_mag": {"units": "1", "bounds": (2.0, 10.0), "tunable_tier": 3, "transform": "sigmoid", "category": "pdf_closure", "reference": "Larson & Golaz (2005) ADG1 assumed-PDF spread/skewness coefficients", "shape": None},
            "a_const": {"units": "1", "bounds": (1.0, 3.0), "tunable_tier": 3, "transform": "sigmoid", "category": "pdf_closure", "reference": "CLUBB tunable closure coefficient (parameters_tunable.F90:set_default_parameters)", "shape": None},
            "alpha_corr": {"units": "1", "bounds": (0.0, 0.5), "tunable_tier": 3, "transform": "sigmoid", "category": "pdf_closure", "reference": "Larson & Golaz (2005) ADG1 assumed-PDF spread/skewness coefficients", "shape": None},
            "beta": {"units": "1", "bounds": (1.0, 4.0), "tunable_tier": 1, "transform": "sigmoid", "category": "pdf_closure", "reference": "Larson & Golaz (2005) ADG1 assumed-PDF spread/skewness coefficients", "shape": None},
            "bv_efold": {"units": "1", "bounds": (1.0, 20.0), "tunable_tier": 3, "transform": "sigmoid", "category": "dissipation_timescale", "reference": "CLUBB tunable closure coefficient (parameters_tunable.F90:set_default_parameters)", "shape": None},
            "c_K": {"units": "1", "bounds": (0.05, 0.6), "tunable_tier": 1, "transform": "sigmoid", "category": "diffusivity", "reference": "CLUBB eddy-diffusivity coefficients c_K* (Km = c_K*·L·sqrt(TKE); parameters_tunable.F90)", "shape": None},
            "c_K1": {"units": "1", "bounds": (0.1, 2.0), "tunable_tier": 2, "transform": "sigmoid", "category": "diffusivity", "reference": "CLUBB eddy-diffusivity coefficients c_K* (Km = c_K*·L·sqrt(TKE); parameters_tunable.F90)", "shape": None},
            "c_K10": {"units": "1", "bounds": (0.05, 2.0), "tunable_tier": 2, "transform": "sigmoid", "category": "diffusivity", "reference": "CLUBB eddy-diffusivity coefficients c_K* (Km = c_K*·L·sqrt(TKE); parameters_tunable.F90)", "shape": None},
            "c_K10h": {"units": "1", "bounds": (0.05, 1.5), "tunable_tier": 2, "transform": "sigmoid", "category": "diffusivity", "reference": "CLUBB eddy-diffusivity coefficients c_K* (Km = c_K*·L·sqrt(TKE); parameters_tunable.F90)", "shape": None},
            "c_K2": {"units": "1", "bounds": (0.01, 0.6), "tunable_tier": 2, "transform": "sigmoid", "category": "diffusivity", "reference": "CLUBB eddy-diffusivity coefficients c_K* (Km = c_K*·L·sqrt(TKE); parameters_tunable.F90)", "shape": None},
            "c_K6": {"units": "1", "bounds": (0.05, 1.5), "tunable_tier": 2, "transform": "sigmoid", "category": "diffusivity", "reference": "CLUBB eddy-diffusivity coefficients c_K* (Km = c_K*·L·sqrt(TKE); parameters_tunable.F90)", "shape": None},
            "c_K8": {"units": "1", "bounds": (0.1, 5.0), "tunable_tier": 2, "transform": "sigmoid", "category": "diffusivity", "reference": "CLUBB eddy-diffusivity coefficients c_K* (Km = c_K*·L·sqrt(TKE); parameters_tunable.F90)", "shape": None},
            "c_K9": {"units": "1", "bounds": (0.02, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "diffusivity", "reference": "CLUBB eddy-diffusivity coefficients c_K* (Km = c_K*·L·sqrt(TKE); parameters_tunable.F90)", "shape": None},
            "coef_spread_DG_means_rt": {"units": "1", "bounds": (0.1, 2.0), "tunable_tier": 2, "transform": "sigmoid", "category": "pdf_closure", "reference": "Larson & Golaz (2005) ADG1 assumed-PDF spread/skewness coefficients", "shape": None},
            "coef_spread_DG_means_thl": {"units": "1", "bounds": (0.1, 2.0), "tunable_tier": 2, "transform": "sigmoid", "category": "pdf_closure", "reference": "Larson & Golaz (2005) ADG1 assumed-PDF spread/skewness coefficients", "shape": None},
            "gamma_coef": {"units": "1", "bounds": (0.1, 0.5), "tunable_tier": 1, "transform": "sigmoid", "category": "pdf_closure", "reference": "Larson & Golaz (2005) ADG1 assumed-PDF spread/skewness coefficients", "shape": None},
            "gamma_coefb": {"units": "1", "bounds": (0.1, 0.5), "tunable_tier": 2, "transform": "sigmoid", "category": "pdf_closure", "reference": "Larson & Golaz (2005) ADG1 assumed-PDF spread/skewness coefficients", "shape": None},
            "gamma_coefc": {"units": "1", "bounds": (1.0, 10.0), "tunable_tier": 3, "transform": "sigmoid", "category": "pdf_closure", "reference": "Larson & Golaz (2005) ADG1 assumed-PDF spread/skewness coefficients", "shape": None},
            "lambda0_stability_coef": {"units": "1", "bounds": (0.01, 0.2), "tunable_tier": 2, "transform": "sigmoid", "category": "mixing_length", "reference": "CLUBB parcel buoyant-sorting mixing-length coefficients (Larson et al.; parameters_tunable.F90)", "shape": None},
            "lmin_coef": {"units": "1", "bounds": (0.02, 0.5), "tunable_tier": 2, "transform": "sigmoid", "category": "mixing_length", "reference": "CLUBB parcel buoyant-sorting mixing-length coefficients (Larson et al.; parameters_tunable.F90)", "shape": None},
            "mu": {"units": "1/m", "bounds": (0.0001, 0.01), "tunable_tier": 1, "transform": "sigmoid", "category": "mixing_length", "reference": "CLUBB parcel buoyant-sorting mixing-length coefficients (Larson et al.; parameters_tunable.F90)", "shape": None},
            "mult_coef": {"units": "1", "bounds": (0.3, 2.0), "tunable_tier": 2, "transform": "sigmoid", "category": "mixing_length", "reference": "CLUBB parcel buoyant-sorting mixing-length coefficients (Larson et al.; parameters_tunable.F90)", "shape": None},
            "nu1": {"units": "m^2/s", "bounds": (1.0, 100.0), "tunable_tier": 2, "transform": "sigmoid", "category": "background_diffusion", "reference": "CLUBB background/diffusion coefficients nu* (Larson & Golaz 2005; parameters_tunable.F90)", "shape": None},
            "nu2": {"units": "m^2/s", "bounds": (0.5, 50.0), "tunable_tier": 2, "transform": "sigmoid", "category": "background_diffusion", "reference": "CLUBB background/diffusion coefficients nu* (Larson & Golaz 2005; parameters_tunable.F90)", "shape": None},
            "nu6": {"units": "m^2/s", "bounds": (0.5, 50.0), "tunable_tier": 2, "transform": "sigmoid", "category": "background_diffusion", "reference": "CLUBB background/diffusion coefficients nu* (Larson & Golaz 2005; parameters_tunable.F90)", "shape": None},
            "nu8": {"units": "m^2/s", "bounds": (1.0, 100.0), "tunable_tier": 2, "transform": "sigmoid", "category": "background_diffusion", "reference": "CLUBB background/diffusion coefficients nu* (Larson & Golaz 2005; parameters_tunable.F90)", "shape": None},
            "nu9": {"units": "m^2/s", "bounds": (1.0, 100.0), "tunable_tier": 2, "transform": "sigmoid", "category": "background_diffusion", "reference": "CLUBB background/diffusion coefficients nu* (Larson & Golaz 2005; parameters_tunable.F90)", "shape": None},
            "omicron": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 3, "transform": "sigmoid", "category": "pdf_closure", "reference": "CLUBB tunable closure coefficient (parameters_tunable.F90:set_default_parameters)", "shape": None},
            "pdf_component_stdev_factor_w": {"units": "1", "bounds": (0.5, 2.0), "tunable_tier": 3, "transform": "sigmoid", "category": "pdf_closure", "reference": "Larson & Golaz (2005) ADG1 assumed-PDF spread/skewness coefficients", "shape": None},
            "slope_coef_spread_DG_means_w": {"units": "1", "bounds": (5.0, 50.0), "tunable_tier": 2, "transform": "sigmoid", "category": "pdf_closure", "reference": "Larson & Golaz (2005) ADG1 assumed-PDF spread/skewness coefficients", "shape": None},
            "thlp2_rad_coef": {"units": "1", "bounds": (0.0, 2.0), "tunable_tier": 3, "transform": "sigmoid", "category": "radiation_coupling", "reference": "CLUBB tunable closure coefficient (parameters_tunable.F90:set_default_parameters)", "shape": None},
            "up2_sfc_coef": {"units": "1", "bounds": (0.5, 4.0), "tunable_tier": 3, "transform": "sigmoid", "category": "pressure_correlation", "reference": "CLUBB tunable closure coefficient (parameters_tunable.F90:set_default_parameters)", "shape": None},
            "upsilon_precip_frac_rat": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 3, "transform": "sigmoid", "category": "pdf_closure", "reference": "CLUBB tunable closure coefficient (parameters_tunable.F90:set_default_parameters)", "shape": None},
            "wpxp_L_thresh": {"units": "m", "bounds": (10.0, 200.0), "tunable_tier": 3, "transform": "sigmoid", "category": "lscale_zero_blend", "reference": "CLUBB Lscale->0 pressure-term blending coefficients (parameters_tunable.F90)", "shape": None},
            "wpxp_Ri_exp": {"units": "1", "bounds": (0.0, 2.0), "tunable_tier": 3, "transform": "sigmoid", "category": "dissipation_timescale", "reference": "CLUBB tunable closure coefficient (parameters_tunable.F90:set_default_parameters)", "shape": None},
            "xp3_coef_base": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 3, "transform": "sigmoid", "category": "pdf_closure", "reference": "CLUBB tunable closure coefficient (parameters_tunable.F90:set_default_parameters)", "shape": None},
            "xp3_coef_slope": {"units": "1", "bounds": (0.0, 0.1), "tunable_tier": 3, "transform": "sigmoid", "category": "pdf_closure", "reference": "CLUBB tunable closure coefficient (parameters_tunable.F90:set_default_parameters)", "shape": None},
            "z_displace": {"units": "m", "bounds": (0.0, 100.0), "tunable_tier": 3, "transform": "sigmoid", "category": "mixing_length", "reference": "CLUBB tunable closure coefficient (parameters_tunable.F90:set_default_parameters)", "shape": None},
        },
    },
    "CLUBBConfig": {
        "scheme_key": "atm.turb.CLUBBConfig",
        "excluded": {
            "T0": "fixed reference temperature [K] for dry N^2 = (g/T0) d(thlm)/dz (a reference, not a closure coefficient)",
            "clubb_dt": "numerics: CLUBB internal sub-step [s]",
            "rt_tol": "numerics: total-water mixing-ratio tolerance [kg/kg]",
            "thl_tol": "numerics: liquid-water potential-temperature tolerance [K]",
            "tke_min": "numerics: carried-TKE/wp2 state floor [m^2/s^2]",
            "w_tol": "numerics: w-moment tolerance/floor [m/s]",
            "wp2_max": "numerics: wp2 upper clip [m^2/s^2]",
        },
        "params": {},
    },
}

# Eddy-diffusivity and dissipation coefficients are read from CLUBBParams
# (c_K, beta, ...) — no hardcoded tunables here.
_PR_T = 1.0   # phase-1 turbulent Prandtl number (Kh = Km/_PR_T); refined in P2

_EP1 = (1.0 - constants.epsilon) / constants.epsilon
_EP2 = 1.0 / constants.epsilon
_HUNDRED = 100.0


# ===========================================================================
# 1. Diagnostic ADG1-PDF closure
# ===========================================================================
# Given the column mean state and the carried ``wp2`` on the CLUBB grid, this
# diagnoses the second moments with standard mixing-length / down-gradient
# closures, runs the ADG1 double-Gaussian assumed-PDF closure — the
# distinctive CLUBB feature absent from clubb_lite (single Gaussian) — and
# returns the liquid cloud fraction, cloud water ``rcm``, and the moist
# buoyancy flux ``wpthvp`` (including the cloud-water latent-heat term that
# makes a cloudy layer more buoyant). This is the *diagnostic* coupling used
# by the phase-1 runnable ``clubb_turbulence`` entry: skewness is taken
# symmetric (``Skw = 0`` → ``mixt_frac = 1/2``) and the variances are
# mixing-length closures. All fields are on thermodynamic (zt) levels of the
# ascending CLUBB grid.


def _grad_zt(field_zt, gr: CLUBBGrid):
    """d/dz of a zt-level field, returned on zt (``zm2zt(ddzt(.))``)."""
    return zm2zt(ddzt(field_zt, gr), gr)


def diagnose_cloud_and_buoyancy(thlm, rtm, wp2, exner, p_in_Pa, thv_ds, Kh, Lscale,
                                gr: CLUBBGrid, config: CLUBBConfig):
    """ADG1-PDF cloud fraction, cloud water, and moist buoyancy flux (zt levels).

    Parameters (all ``(ncol, nzt)`` on the ascending CLUBB grid)
    ----------
    thlm, rtm : jax.Array
        Liquid-water potential temperature [K] and total water [kg/kg].
    wp2 : jax.Array
        Carried ``w'^2`` [m^2/s^2].
    exner, p_in_Pa, thv_ds : jax.Array
        Exner, pressure [Pa], dry-static virtual potential temperature [K].
    Kh : jax.Array
        Eddy diffusivity for scalars [m^2/s].
    Lscale : jax.Array
        CLUBB parcel mixing length [m].
    gr : CLUBBGrid
    config : CLUBBConfig

    Returns
    -------
    tuple of jax.Array
        ``(cloud_frac, rcm, wpthvp)`` on zt levels — liquid cloud fraction [-],
        cloud water [kg/kg], and the buoyancy flux ``w'thv'`` [K m/s].
    """
    params = config.params
    wp2 = jnp.maximum(wp2, config.tke_min)
    sqrt_wp2 = jnp.sqrt(wp2)

    ddz_thl = _grad_zt(thlm, gr)
    ddz_rt = _grad_zt(rtm, gr)

    # Down-gradient second-order fluxes and mixing-length variances.
    wpthlp = -Kh * ddz_thl
    wprtp = -Kh * ddz_rt
    thlp2 = jnp.maximum((Lscale * ddz_thl) ** 2, config.thl_tol ** 2)
    rtp2 = jnp.maximum((Lscale * ddz_rt) ** 2, config.rt_tol ** 2)
    rtpthlp = Lscale ** 2 * ddz_thl * ddz_rt
    up2 = vp2 = jnp.maximum(wp2, config.w_tol ** 2)

    # sigma_sqd_w (Skw = 0 -> gamma = gamma_coef), computed directly on zt.
    denom_thl = jnp.sqrt(wp2 * thlp2) + _HUNDRED * config.w_tol * config.thl_tol
    denom_rt = jnp.sqrt(wp2 * rtp2) + _HUNDRED * config.w_tol * config.rt_tol
    max_corr = jnp.maximum((wpthlp / denom_thl) ** 2, (wprtp / denom_rt) ** 2)
    sigma_sqd_w = jnp.clip(params.gamma_coef * (1.0 - jnp.minimum(max_corr, 1.0)), 0.0, 0.99)

    z = jnp.zeros_like(wp2)
    mfmm = derive_mixt_frac_max_mag(params.Skw_max_mag)
    adg1 = ADG1_pdf_driver(
        z, rtm, thlm, z, z, wp2, rtp2, thlp2, up2, vp2, z,
        wprtp, wpthlp, z, z, sqrt_wp2, sigma_sqd_w, params.beta, mfmm)

    rcm, cloud_frac = calc_pdf_liquid_cloud_frac(adg1, rtpthlp, rtm, thlm, exner, p_in_Pa)

    # Moist buoyancy flux: wpthvp = wpthlp + ep1*thv_ds*wprtp + rc_coef*wprcp,
    # with a down-gradient cloud-water flux wprcp (rc_coef = Lv/(exner*Cp) - ep2*thv).
    wprcp = -Kh * _grad_zt(rcm, gr)
    rc_coef = constants.L_v / (exner * constants.c_pd) - _EP2 * thv_ds
    wpthvp = wpthlp + _EP1 * thv_ds * wprtp + rc_coef * wprcp
    return cloud_frac, rcm, wpthvp


# ===========================================================================
# 2. Configuration (CAM-default flavour)
# ===========================================================================
# CLUBBParams = the tunable closure coefficients (the CLUBB ``clubb_params``
# vector): library defaults (parameters_tunable.F90:set_default_parameters)
# with the CAM namelist overrides applied (``# lib X -> Y`` marks an
# override). Scheme-tunable params live HERE, not in legoesm.constants
# (CLAUDE.md). CLUBBConfig = the top-level scheme config (params, shared
# surface-layer config, tolerances, CLUBB sub-step, prognostic switch).
# Physical constants are deliberately absent — pulled from legoesm.constants.
# The model FLAGS are fixed at their CAM defaults (table at end of file).


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

    The CLUBB model FLAGS are not configurable: only the CAM-default flag
    tree is implemented (see the reference table at the end of this file).

    Fields
    ------
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
    prognostic : bool
        Select the FULL prognostic higher-order moment closure
        (``advance_clubb_core`` via ``clubb_step``, carrying ``CLUBBMomentState``
        in ``PhysicsState.clubb_moments``) instead of the default diagnostic
        phase-1 path (parcel ``Lscale`` eddy diffusion + ADG1-PDF buoyancy). Opt-
        in (default ``False``) so existing ``scheme="clubb"`` runs are unchanged.
        Read only at setup/dispatch time (a static Python branch), never in
        traced code, so it stays a valid plain pytree-leaf field.
    """

    params: CLUBBParams = CLUBBParams()
    surface: SurfaceLayerConfig = SurfaceLayerConfig()
    clubb_dt: float = 300.0
    w_tol: float = 2.0e-2
    rt_tol: float = 1.0e-8
    thl_tol: float = 1.0e-2
    wp2_max: float = 1000.0
    tke_min: float = 1.0e-6
    T0: float = 300.0
    prognostic: bool = False


# Derived parameters (recomputed from base config, never stored as magic
# numbers). Faithful to ``parameters_tunable.F90:setup_parameters``.
# Reference sigma_sqd_w used in the mixt_frac cap derivation (NOT a tunable):
_MIXT_FRAC_CAP_SIGMA_REF = 0.4
# Reference layer depth [m] scaling lmin (parameters_tunable.F90 ``lmin_deltaz``).
_LMIN_DELTAZ = 40.0
# CAM ``fill_holes_type = 2`` (sliding window) — the only ported hole-fill
# dispatch value (see the flag table at the end of the file).
_CAM_FILL_HOLES_TYPE = 2


def derive_mixt_frac_max_mag(Skw_max_mag: float) -> float:
    """Maximum |mixt_frac - 0.5| + 0.5 cap, derived from ``Skw_max_mag``.

    ``1 - 0.5*(1 - Skw_max/sqrt(4*(1-0.4)^3 + Skw_max^2))`` — the ADG1 mixture
    fraction evaluated at the maximum allowed w-skewness (so the runtime
    ``mixt_frac`` clip is consistent with ``Skw_max_mag``). With the CAM default
    ``Skw_max_mag = 4.5`` this is ~0.9897.
    """
    inner = 4.0 * (1.0 - _MIXT_FRAC_CAP_SIGMA_REF) ** 3 + Skw_max_mag ** 2
    # jnp.sqrt, not math.sqrt: ``Skw_max_mag`` is a REGISTERED trainable
    # (__param_spec__, aggressive tier), so once a calibration or a trained
    # override splices a traced JAX scalar here, math.sqrt raises on the
    # Python-scalar conversion and the whole trace dies (codex review
    # 2026-08-05). jnp.sqrt is identical on a Python float.
    return 1.0 - 0.5 * (1.0 - Skw_max_mag / jnp.sqrt(inner))


def derive_lmin(lmin_coef: float) -> float:
    """Minimum mixing length [m] = ``lmin_coef * lmin_deltaz`` (40 m).

    With the CAM default ``lmin_coef = 0.1`` this is 4 m.
    """
    return lmin_coef * _LMIN_DELTAZ


# ===========================================================================
# 3. Staggered CLUBB grid
# ===========================================================================
# The ascending zt/zm staggered-grid pytree (CLUBBGrid) with the
# interpolation (zm2zt/zt2zm), derivative (ddzm/ddzt) and smoothing
# operators, the constructors (make_clubb_grid[_from_levels]) and the
# top-down <-> ascending flip helper. Pure grid plumbing (grid_class.F90).

class CLUBBGrid(NamedTuple):
    """Ascending staggered vertical grid for the CLUBB closure (a JAX pytree).

    All arrays are shape ``(ngrdcol, nz*)`` with ascending levels (index 0 is
    the lowest, nearest the surface). ``nzm == nzt + 1`` in the standard CLUBB
    layout (momentum levels bracket thermodynamic levels), but the operators
    here only require the shape relationships noted per function, so the
    container does not hard-enforce it.

    Attributes
    ----------
    zm : jax.Array
        Momentum-level heights [m], shape ``(ngrdcol, nzm)``.
    zt : jax.Array
        Thermodynamic-level heights [m], shape ``(ngrdcol, nzt)``.
    invrs_dzm : jax.Array
        ``1 / (zt[k] - zt[k-1])`` evaluated at momentum levels [1/m], shape
        ``(ngrdcol, nzm)``. Used by :func:`ddzt`.
    invrs_dzt : jax.Array
        ``1 / (zm[k+1] - zm[k])`` evaluated at thermodynamic levels [1/m],
        shape ``(ngrdcol, nzt)``. Used by :func:`ddzm`.
    dzm : jax.Array
        Momentum-level spacing [m], shape ``(ngrdcol, nzm)`` (reciprocal of
        ``invrs_dzm`` away from degenerate levels). Consumed by the
        mixing-length parcel integrals.
    dzt : jax.Array
        Thermodynamic-level spacing ``zm[k+1] - zm[k]`` [m], shape
        ``(ngrdcol, nzt)``.
    """

    zm: jax.Array
    zt: jax.Array
    invrs_dzm: jax.Array
    invrs_dzt: jax.Array
    dzm: jax.Array
    dzt: jax.Array


def zm2zt(azm: jax.Array, gr: CLUBBGrid) -> jax.Array:
    """Interpolate a momentum-level field to thermodynamic levels (linear).

    Faithful port of Fortran ``linear_interpolated_azt_2D``. For each thermo
    level ``k`` (ascending grid)::

        azt[k] = w_above * azm[k+1] + w_below * azm[k]
        w_above = (zt[k] - zm[k])   / (zm[k+1] - zm[k])
        w_below = (zm[k+1] - zt[k]) / (zm[k+1] - zm[k])

    Parameters
    ----------
    azm : jax.Array
        Field on momentum levels, shape ``(ngrdcol, nzm)``.
    gr : CLUBBGrid
        Grid; uses ``gr.zm`` ``(ngrdcol, nzm)`` and ``gr.zt`` ``(ngrdcol,
        nzt)`` with ``nzt = nzm - 1``.

    Returns
    -------
    jax.Array
        Field on thermodynamic levels, shape ``(ngrdcol, nzt)``.
    """
    zm = gr.zm
    zt = gr.zt
    dzt = zm[:, 1:] - zm[:, :-1]          # (ngrdcol, nzt), = nzm-1
    w_above = (zt - zm[:, :-1]) / dzt
    w_below = (zm[:, 1:] - zt) / dzt
    return w_above * azm[:, 1:] + w_below * azm[:, :-1]


def zt2zm(azt: jax.Array, gr: CLUBBGrid, zm_min: float | None = None) -> jax.Array:
    """Interpolate a thermodynamic-level field to momentum levels (linear).

    Faithful port of Fortran ``linear_interpolated_azm_2D`` (ascending grid).

    Interior ``k = 1 .. nzm-2``::

        azm[k] = w_above * azt[k] + w_below * azt[k-1]
        w_above = (zm[k] - zt[k-1]) / (zt[k] - zt[k-1])
        w_below = (zt[k] - zm[k])   / (zt[k] - zt[k-1])

    Boundaries (ascending grid):
      * lower ``k=0``: ``azm[0] = azt[0]`` (Fortran sets it directly);
      * upper ``k=nzm-1``: linear extension from ``zt[-2], zt[-1]``.

    Parameters
    ----------
    azt : jax.Array
        Field on thermodynamic levels, shape ``(ngrdcol, nzt)``.
    gr : CLUBBGrid
        Grid; uses ``gr.zm`` ``(ngrdcol, nzm)`` and ``gr.zt`` ``(ngrdcol,
        nzt)`` with ``nzm = nzt + 1``.
    zm_min : float, optional
        Lower clamp applied after interpolation (e.g. positivity floor for
        variances). ``None`` leaves the field unclamped.

    Returns
    -------
    jax.Array
        Field on momentum levels, shape ``(ngrdcol, nzm)``.
    """
    zm = gr.zm
    zt = gr.zt

    # Interior k = 1 .. nzm-2 (zt[k-1] < zm[k] < zt[k] for ascending grids).
    denom_int = zt[:, 1:] - zt[:, :-1]       # (ngrdcol, nzt-1) = (ngrdcol, nzm-2)
    zm_int = zm[:, 1:-1]                       # (ngrdcol, nzm-2)
    w_above_int = (zm_int - zt[:, :-1]) / denom_int
    w_below_int = (zt[:, 1:] - zm_int) / denom_int
    azm_int = w_above_int * azt[:, 1:] + w_below_int * azt[:, :-1]

    # Lower boundary: azm[0] = azt[0].
    azm_bot = azt[:, :1]

    # Upper boundary: linear extension above zt[-1].
    denom_top = zt[:, -1:] - zt[:, -2:-1]
    w_above_top = (zm[:, -1:] - zt[:, -2:-1]) / denom_top
    w_below_top = (zt[:, -1:] - zm[:, -1:]) / denom_top
    azm_top = w_above_top * azt[:, -1:] + w_below_top * azt[:, -2:-1]

    azm = jnp.concatenate([azm_bot, azm_int, azm_top], axis=1)
    if zm_min is not None:
        azm = jnp.maximum(azm, zm_min)
    return azm


def ddzm(azm: jax.Array, gr: CLUBBGrid) -> jax.Array:
    """Vertical derivative of a momentum-level field, at thermodynamic levels.

    Faithful port of Fortran ``gradzm_2D``::

        dazm_dz[k] = (azm[k+1] - azm[k]) * invrs_dzt[k]   for k = 0 .. nzt-1

    Parameters
    ----------
    azm : jax.Array
        Field on momentum levels, shape ``(ngrdcol, nzm)``.
    gr : CLUBBGrid
        Grid; uses ``gr.invrs_dzt`` ``(ngrdcol, nzt)``.

    Returns
    -------
    jax.Array
        Derivative on thermodynamic levels, shape ``(ngrdcol, nzt)``.
    """
    return (azm[:, 1:] - azm[:, :-1]) * gr.invrs_dzt


def ddzt(azt: jax.Array, gr: CLUBBGrid) -> jax.Array:
    """Vertical derivative of a thermodynamic-level field, at momentum levels.

    Faithful port of Fortran ``gradzt_2D``::

        interior k = 1 .. nzm-2: dazt_dz[k] = (azt[k] - azt[k-1]) * invrs_dzm[k]
        boundaries: dazt_dz[0] = dazt_dz[1],  dazt_dz[nzm-1] = dazt_dz[nzm-2]

    Parameters
    ----------
    azt : jax.Array
        Field on thermodynamic levels, shape ``(ngrdcol, nzt)``.
    gr : CLUBBGrid
        Grid; uses ``gr.invrs_dzm`` ``(ngrdcol, nzm)``.

    Returns
    -------
    jax.Array
        Derivative on momentum levels, shape ``(ngrdcol, nzm)``.
    """
    interior = (azt[:, 1:] - azt[:, :-1]) * gr.invrs_dzm[:, 1:-1]  # (ngrdcol, nzm-2)
    bottom = interior[:, :1]
    top = interior[:, -1:]
    return jnp.concatenate([bottom, interior, top], axis=1)


def zm2zt2zm(azm: jax.Array, gr: CLUBBGrid, zm_min: float | None = None) -> jax.Array:
    """Round-trip smoother ``zm -> zt -> zm`` (Fortran ``zm2zt2zm``)."""
    return zt2zm(zm2zt(azm, gr), gr, zm_min=zm_min)


def zt2zm2zt(azt: jax.Array, gr: CLUBBGrid, zt_min: float | None = None) -> jax.Array:
    """Round-trip smoother ``zt -> zm -> zt`` (Fortran ``zt2zm2zt``)."""
    result = zm2zt(zt2zm(azt, gr), gr)
    if zt_min is not None:
        result = jnp.maximum(result, zt_min)
    return result


def _safe_invrs(dz: jax.Array, floor: float = 1.0e-30) -> jax.Array:
    """Reciprocal with the reference's zero-spacing guard (``setup_grid``).

    Faithful to ``_calc_grid_spacings``: ``invrs = 1/dz`` where ``|dz| >
    floor`` else ``0``. Degenerate (zero) spacings therefore yield ``0`` rather
    than ``inf`` — the same graceful failure mode as the Fortran/JAX reference,
    and JIT/autodiff-safe (no data-dependent host control flow).
    """
    return jnp.where(jnp.abs(dz) > floor, 1.0 / jnp.where(jnp.abs(dz) > floor, dz, 1.0), 0.0)


def make_clubb_grid(zm: jax.Array, zt: jax.Array) -> CLUBBGrid:
    """Build a :class:`CLUBBGrid` from ascending ``zm``/``zt`` height arrays.

    Faithful to the canonical CLUBB grid construction
    (``grid_class.F90:setup_grid_heights`` /
    ``derived_types/grid_class.py:_calc_grid_spacings``) for an **ascending**
    grid:

      * ``dzt[k]  = zm[k+1] - zm[k]``                       (thermo levels, nzt)
      * interior ``dzm[k] = zt[k] - zt[k-1]``, ``k = 1 .. nzm-2``
      * lower boundary ``dzm[0]    = 2 * (zt[0] - zm[0])``  (NOT a copy of the
        adjacent interior spacing — the host grid's lowest thermo level need
        not be the midpoint of the two lowest momentum levels)
      * upper boundary ``dzm[nzm-1] = dzm[nzm-2]``          (copy adjacent)

    Inverses use the reference zero-spacing guard (:func:`_safe_invrs`). Only
    the *structural* preconditions (2-D, ``nzt == nzm - 1``, ``nzt >= 2``) are
    enforced here — these are static (shape-level) so they remain JIT-safe and
    fail before tracing array values. Strict-ascending monotonicity is a
    documented precondition; a violated (e.g. duplicate) level yields a ``0``
    inverse spacing there rather than a spurious ``inf`` (matching the
    reference), instead of a data-dependent runtime exception.

    Parameters
    ----------
    zm : jax.Array
        Ascending momentum-level heights [m], shape ``(ngrdcol, nzm)``.
    zt : jax.Array
        Ascending thermodynamic-level heights [m], shape ``(ngrdcol, nzt)``,
        with ``nzt = nzm - 1`` and ``nzt >= 2``.

    Returns
    -------
    CLUBBGrid
    """
    if zm.ndim != 2 or zt.ndim != 2:
        raise ValueError(
            f"zm and zt must be 2-D (ngrdcol, nz); got zm.ndim={zm.ndim}, "
            f"zt.ndim={zt.ndim}."
        )
    ngrdcol_m, nzm = zm.shape
    ngrdcol_t, nzt = zt.shape
    if ngrdcol_m != ngrdcol_t:
        raise ValueError(
            f"zm and zt must share ngrdcol; got {ngrdcol_m} vs {ngrdcol_t}."
        )
    if nzt != nzm - 1:
        raise ValueError(
            f"CLUBB staggered grid requires nzt == nzm - 1; got nzm={nzm}, "
            f"nzt={nzt}."
        )
    if nzt < 2:
        raise ValueError(
            f"CLUBB grid needs nzt >= 2 (nzm >= 3) for the staggered "
            f"interpolation/derivative stencils; got nzt={nzt}."
        )

    dzt = zm[:, 1:] - zm[:, :-1]                          # (ngrdcol, nzt)
    dzm_int = zt[:, 1:] - zt[:, :-1]                       # (ngrdcol, nzt-1) = (nzm-2)
    dzm_lower = 2.0 * (zt[:, :1] - zm[:, :1])              # (ngrdcol, 1)
    dzm_upper = dzm_int[:, -1:]                            # (ngrdcol, 1), copy adjacent
    dzm = jnp.concatenate([dzm_lower, dzm_int, dzm_upper], axis=1)  # (ngrdcol, nzm)

    return CLUBBGrid(
        zm=zm,
        zt=zt,
        invrs_dzm=_safe_invrs(dzm),
        invrs_dzt=_safe_invrs(dzt),
        dzm=dzm,
        dzt=dzt,
    )


# ---------------------------------------------------------------------------
# legoESM <-> CLUBB orientation bridge
# ---------------------------------------------------------------------------
# legoESM stores columns TOP-DOWN (index 0 = model top, index -1 = surface;
# ``z_half[:, -1] == 0``). CLUBB stores them ASCENDING (index 0 = surface).
# The flip is its own inverse, so one helper serves both directions.


def flip_vertical(field: jax.Array) -> jax.Array:
    """Flip a column field along the vertical axis (axis 1).

    Converts between legoESM top-down ordering and CLUBB ascending ordering.
    Self-inverse: ``flip_vertical(flip_vertical(x)) == x``. Operates on the
    last axis of a ``(ngrdcol, nz)`` array.
    """
    return field[:, ::-1]


def make_clubb_grid_from_levels(z_full: jax.Array, z_half: jax.Array) -> CLUBBGrid:
    """Build an ascending :class:`CLUBBGrid` from legoESM level heights.

    legoESM level semantics (see ``_shared.compute_heights_from_sigma``):
      * ``z_full`` — full-level (layer-midpoint) heights [m], shape
        ``(ncol, nlev)``, TOP-DOWN (index 0 = top).
      * ``z_half`` — half-level (interface) heights [m], shape
        ``(ncol, nlev+1)``, TOP-DOWN, with ``z_half[:, -1] == 0`` (surface).

    CLUBB staggering: thermodynamic levels ``zt`` carry means (T, q, u, v) ->
    legoESM full levels; momentum levels ``zm`` carry fluxes/w-moments ->
    legoESM half levels. Hence ``nzt = nlev`` and ``nzm = nlev + 1`` (so
    ``nzt == nzm - 1`` as :func:`make_clubb_grid` requires), with the surface
    momentum level ``zm[0] == 0``. Each legoESM full level lands exactly at the
    midpoint of its two bracketing half levels, so ``zt[k]`` lies between
    ``zm[k]`` and ``zm[k+1]`` — the CLUBB interior staggering.

    Parameters
    ----------
    z_full : jax.Array
        Top-down full-level heights [m], shape ``(ncol, nlev)``.
    z_half : jax.Array
        Top-down half-level heights [m], shape ``(ncol, nlev+1)``.

    Returns
    -------
    CLUBBGrid
        Ascending staggered grid with ``zt`` from ``z_full`` and ``zm`` from
        ``z_half``.
    """
    zt = flip_vertical(z_full)   # ascending thermodynamic levels (nlev)
    zm = flip_vertical(z_half)   # ascending momentum levels (nlev+1), zm[0]=surface
    return make_clubb_grid(zm, zt)


# ===========================================================================
# 4. Flatau saturation adapters
# ===========================================================================
# Thin adapters over the CANONICAL Flatau (1992) saturation curves in
# legoesm.thermo (per the CLAUDE.md no-saturation-reimpl rule): saturation
# mixing ratio over liquid / ice with CLUBB's denominator guard
# (saturation.F90, CAM ``saturation_formula = flatau``).

_SAT_DENOM_MIN_PA = 1.0  # p - esat floor [Pa] (saturation.F90 convention)


def _mixrat_from_esat(p: jax.Array, esat: jax.Array) -> jax.Array:
    """Assemble rsat = ep*esat/(p-esat) with CLUBB's AD-safe denominator guard."""
    safe = (p - esat) >= _SAT_DENOM_MIN_PA
    denom_safe = jnp.where(safe, p - esat, 1.0)
    return jnp.where(safe, constants.epsilon * esat / denom_safe, constants.epsilon)


def sat_mixrat_liq(p: jax.Array, T: jax.Array) -> jax.Array:
    """Saturation mixing ratio over liquid water (Flatau), [kg/kg].

    Parameters
    ----------
    p : jax.Array
        Pressure [Pa].
    T : jax.Array
        Temperature [K] (same shape as ``p``).

    Returns
    -------
    jax.Array
        Saturation mixing ratio over liquid [kg/kg].
    """
    return _mixrat_from_esat(p, saturation_vapor_pressure_flatau(T))


def sat_mixrat_ice(p: jax.Array, T: jax.Array) -> jax.Array:
    """Saturation mixing ratio over ice (Flatau), [kg/kg].

    Parameters
    ----------
    p : jax.Array
        Pressure [Pa].
    T : jax.Array
        Temperature [K] (same shape as ``p``).

    Returns
    -------
    jax.Array
        Saturation mixing ratio over ice [kg/kg].
    """
    return _mixrat_from_esat(p, saturation_vapor_pressure_ice_flatau(T))


# ===========================================================================
# 5. Closure helpers (sigma_sqd_w, Brunt-Vaisala, safe_sqrt)
# ===========================================================================
# The AD-safe square root shared across the scheme (safe_sqrt; the MFL
# section keeps its deliberate NaN-propagating local variant), the PDF-width
# parameter sigma_sqd_w (CAM ``l_gamma_Skw = .true.`` path), and the moist
# Brunt-Vaisala frequency N^2 (T0-referenced, CAM-default form).

_ONE_HUNDRED = 100.0
# bv_mixed clip: Fortran min(bv, 1e8*|bv|^3) (advance_helper_module.F90).
_BV_CLIP_COEF = 1.0e8


def safe_sqrt(x: jax.Array) -> jax.Array:
    """``sqrt(max(x,0))`` with a finite (0) gradient at ``x<=0`` (double-where).

    The canonical AD-safe square root shared across the CLUBB modules: the
    ``jnp.sqrt`` is never evaluated at ``<=0`` in either the primal or the VJP, so
    the gradient stays finite (0) at the boundary instead of the ``+inf`` slope of
    a bare ``sqrt`` at 0. A ``NaN`` input maps to 0 (``NaN > 0`` is False).
    (NOTE: the MFL section of :mod:`clubb` keeps a local ``_safe_sqrt`` that is a
    deliberate variant PROPAGATING ``NaN`` instead; do not collapse it into this one.)
    """
    xp = jnp.maximum(x, 0.0)
    safe = jnp.where(xp > 0.0, xp, 1.0)
    return jnp.where(xp > 0.0, jnp.sqrt(safe), 0.0)


def compute_sigma_sqd_w(
    gamma_Skw_fnc: jax.Array,
    wp2: jax.Array,
    thlp2: jax.Array,
    rtp2: jax.Array,
    wpthlp: jax.Array,
    wprtp: jax.Array,
    gr: CLUBBGrid,
    *,
    w_tol: float,
    thl_tol: float,
    rt_tol: float,
) -> jax.Array:
    """PDF width parameter ``sigma_sqd_w`` (CAM default, l_predict_upwp_vpwp=F).

    ``sigma_sqd_w = gamma_Skw_fnc * (1 - min(max_x corr_wx^2, 1))`` smoothed
    zm->zt->zm with a zero floor (``sigma_sqd_w_module.F90``). All fields are on
    momentum (zm) levels, shape ``(ngrdcol, nzm)``.

    Parameters
    ----------
    gamma_Skw_fnc : jax.Array
        Skewness-dependent gamma coefficient on zm levels.
    wp2, thlp2, rtp2 : jax.Array
        w, thl, rt variances on zm levels.
    wpthlp, wprtp : jax.Array
        w'thl', w'rt' fluxes on zm levels.
    gr : CLUBBGrid
        CLUBB staggered grid (for the zm->zt->zm smoother).
    w_tol, thl_tol, rt_tol : float
        Tolerances (from ``CLUBBConfig``) regularizing the correlation
        denominators (``100 * w_tol * x_tol``).

    Returns
    -------
    jax.Array
        ``sigma_sqd_w`` on zm levels, shape ``(ngrdcol, nzm)``.
    """
    denom_thl = jnp.sqrt(wp2 * thlp2) + _ONE_HUNDRED * w_tol * thl_tol
    denom_rtp = jnp.sqrt(wp2 * rtp2) + _ONE_HUNDRED * w_tol * rt_tol

    corr_thl_sqd = (wpthlp / denom_thl) ** 2
    corr_rtp_sqd = (wprtp / denom_rtp) ** 2
    max_corr = jnp.maximum(corr_thl_sqd, corr_rtp_sqd)

    sigma_sqd_w_tmp = gamma_Skw_fnc * (1.0 - jnp.minimum(max_corr, 1.0))
    return zm2zt2zm(sigma_sqd_w_tmp, gr, zm_min=_ZERO_THRESHOLD)


def calc_brunt_vaisala_freq_sqd(
    thlm: jax.Array,
    exner: jax.Array,
    rtm: jax.Array,
    rcm: jax.Array,
    p_in_Pa: jax.Array,
    ice_supersat_frac: jax.Array,
    bv_efold: jax.Array | float,
    T0: float,
    gr: CLUBBGrid,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Brunt-Vaisala frequency squared (CAM-default branch).

    CAM defaults assumed: ``l_use_thvm_in_bv_freq = .false.``,
    ``l_brunt_vaisala_freq_moist = .false.``,
    ``l_modify_limiters_for_cnvg_test = .false.``. The returned
    ``brunt_vaisala_freq_sqd`` is therefore the dry form ``(g/T0) d(thlm)/dz``;
    ``bv_moist``/``bv_mixed``/``bv_smth`` are still computed (downstream
    mixing-length/Ri consume them).

    All thermodynamic inputs are on thermodynamic (zt) levels,
    shape ``(ngrdcol, nzt)``; outputs are on momentum (zm) levels,
    shape ``(ngrdcol, nzm)`` (the ``ddzt``/``zt2zm`` operators move zt->zm).

    Parameters
    ----------
    thlm : jax.Array
        Liquid-water potential temperature [K] (zt).
    exner : jax.Array
        Exner function [-] (zt).
    rtm : jax.Array
        Total water mixing ratio [kg/kg] (zt).
    rcm : jax.Array
        Cloud water mixing ratio [kg/kg] (zt) (from the PDF closure).
    p_in_Pa : jax.Array
        Pressure [Pa] (zt).
    ice_supersat_frac : jax.Array
        Ice supersaturation fraction [-] (zt).
    bv_efold : jax.Array or float
        e-folding coefficient for the dry<->moist blend [-]; per-column
        ``(ngrdcol,)`` or scalar (``CLUBBParams.bv_efold``).
    T0 : float
        Reference absolute temperature [K] for the dry BV frequency.
    gr : CLUBBGrid
        CLUBB staggered grid.

    Returns
    -------
    tuple of jax.Array
        ``(brunt_vaisala_freq_sqd, bv_mixed, bv_smth, bv_dry, bv_moist)``,
        each on zm levels [1/s^2].
    """
    g = constants.g
    cp = constants.c_pd
    lv = constants.L_v
    rd = constants.R_d
    ep = constants.epsilon

    ddzt_thlm = ddzt(thlm, gr)
    bv_dry_main = (g / T0) * ddzt_thlm   # l_use_thvm_in_bv_freq = False

    T_in_K = thlm * exner + (lv / cp) * rcm
    T_in_K_zm = zt2zm(T_in_K, gr, zm_min=_ZERO_THRESHOLD)

    rsat = sat_mixrat_liq(p_in_Pa, T_in_K)
    rsat_zm = zt2zm(rsat, gr, zm_min=_ZERO_THRESHOLD)
    ddzt_rsat = ddzt(rsat, gr)

    thm = thlm + (lv / (cp * exner)) * rcm
    thm_zm = zt2zm(thm, gr, zm_min=_ZERO_THRESHOLD)
    ddzt_thm = ddzt(thm, gr)
    ddzt_rtm = ddzt(rtm, gr)

    bv_dry = (g / thm_zm) * ddzt_thm

    num_fac = 1.0 + lv * rsat_zm / (rd * T_in_K_zm)
    den_fac = 1.0 + ep * lv ** 2 * rsat_zm / (cp * rd * T_in_K_zm ** 2)
    bv_moist = g * (
        (num_fac / den_fac) * (ddzt_thm / thm_zm + (lv / (cp * T_in_K_zm)) * ddzt_rsat)
        - ddzt_rtm
    )

    bv_efold_arr = jnp.asarray(bv_efold)
    if bv_efold_arr.ndim == 1:
        bv_efold_arr = bv_efold_arr[:, None]
    ice_supersat_frac_zm = zt2zm(ice_supersat_frac, gr, zm_min=_ZERO_THRESHOLD)
    bv_mixed = bv_moist + jnp.exp(-bv_efold_arr * ice_supersat_frac_zm) * (bv_dry - bv_moist)

    # l_modify_limiters_for_cnvg_test = False -> clip then smooth (no min clamp).
    bv_clipped = jnp.minimum(bv_mixed, _BV_CLIP_COEF * jnp.abs(bv_mixed) ** 3)
    bv_smth = zm2zt2zm(bv_clipped, gr)

    # l_brunt_vaisala_freq_moist = False -> return the dry form.
    brunt_vaisala_freq_sqd = bv_dry_main
    return brunt_vaisala_freq_sqd, bv_mixed, bv_smth, bv_dry, bv_moist


# ===========================================================================
# 6. Parcel buoyant-sorting mixing length (Lscale)
# ===========================================================================
# CLUBB's nonlocal parcel buoyant-sorting length scale (mixing_length.F90,
# golden-locked vs CLUBB-JAX) — the distinctive CLUBB feature replacing
# Blackadar-type lengths: an entraining parcel ascends/descends until its
# buoyancy is exhausted; Lscale_up/down are averaged geometrically.
# NOTE: DIFFERENT numerics from the simple mixing length in
# atmosphere/physics/_shared.py used by clubb_lite (issue: the parcel length
# could eventually supplant the _shared one — see
# docs/dev-notes/issues/clubb_parcel_lscale_vs_shared_mixing_length.md).

# Derived thermodynamic ratios (legoESM constants).
_EP = constants.epsilon
_LV2_COEF = _EP * constants.L_v ** 2 / (constants.R_d * constants.c_pd)  # K^2

_ZLMIN = 0.1                   # minimum Lscale [m]
_LSCALE_SFCLYR_DEPTH = 500.0   # surface-layer depth for lminh [m]




def _bounded_while(cond_fn, body_fn, init_state, max_iters):
    """Reverse-mode-differentiable fixed-length replacement for ``lax.while_loop``.

    Runs ``max_iters`` ``lax.scan`` steps; each applies ``body_fn`` only where
    ``cond_fn`` is still true (pytree select), freezing the state otherwise. For
    a body that no-ops once its ``done`` flag is set this reproduces the
    ``while_loop`` final state bit-exactly when ``max_iters >= true trip count``;
    unlike ``while_loop`` it supports ``jax.grad``.
    """
    def step(state, _):
        run = cond_fn(state)
        new = body_fn(state)
        state2 = jax.tree_util.tree_map(lambda o, n: jnp.where(run, n, o), state, new)
        return state2, None

    final, _ = jax.lax.scan(step, init_state, None, length=max_iters)
    return final


def set_Lscale_max(l_implemented: bool, host_dx, host_dy, ngrdcol: int) -> jax.Array:
    """Maximum allowable ``Lscale`` [m] (``mixing_length.F90:set_Lscale_max``).

    In a host model the cap is ``0.25 * min(host_dx, host_dy)``; standalone it is
    ``1e5``. Returns a ``(ngrdcol,)`` array.
    """
    if l_implemented:
        return 0.25 * jnp.minimum(jnp.asarray(host_dx), jnp.asarray(host_dy))
    return jnp.full((ngrdcol,), 1.0e5)


def _parcel_thv(thl_par, rt_par, exner_j, p_j, thv_ds_j, Lv_coef_j):
    """Virtual potential temperature of the parcel at one level (Lewellen-Yoh 1993)."""
    tl_j = thl_par * exner_j
    rsat_j = sat_mixrat_liq(p_j, tl_j)
    tl_sqd = tl_j ** 2
    s_j = (rt_par - rsat_j) * tl_sqd / (tl_sqd + _LV2_COEF * rsat_j)
    rc_j = jnp.maximum(s_j, _ZERO)
    return thl_par + _EP1 * thv_ds_j * rt_par + Lv_coef_j * rc_j


def _upward_inner_while(
    k_py, tke_0, thl_init, rt_init, dCAPE_init,
    thl_precalc_up, rt_precalc_up, exp_mu_dzm,
    grav_on_thvm, Lv_coef, thv_ds, exner, p, thvm,
    dzm, invrs_dzm, zt, k_ub_zt_py,
):
    """Upward parcel trajectory from ``k_py+2`` (ascending). See module docstring."""
    init_state = (
        k_py + 2, tke_0, thl_init, rt_init, dCAPE_init,
        jnp.bool_(False), k_py + 1, tke_0, dCAPE_init,
        jnp.zeros((), dtype=tke_0.dtype),
    )

    def cond_fn(state):
        j, _tke, _thl, _rt, _dcp, done, _jl, _tex, _dep, _dej = state
        return ~done & (j < k_ub_zt_py)

    def body_fn(state):
        j, tke, thl, rt, dCAPE_prev, done, j_last, tke_exit, dep, dej = state
        thl_new = thl_precalc_up[j] + thl * exp_mu_dzm[j]
        rt_new = rt_precalc_up[j] + rt * exp_mu_dzm[j]
        thv_new = _parcel_thv(thl_new, rt_new, exner[j], p[j], thv_ds[j], Lv_coef[j])
        dCAPE_j = grav_on_thvm[j] * (thv_new - thvm[j])
        CAPE_incr = 0.5 * (dCAPE_j + dCAPE_prev) * dzm[j]

        new_tke = tke + CAPE_incr
        exhausted = new_tke <= 0.0
        newly_ex = exhausted & ~done

        tke_exit_out = jnp.where(newly_ex, tke, tke_exit)
        dep_out = jnp.where(newly_ex, dCAPE_prev, dep)
        dej_out = jnp.where(newly_ex, dCAPE_j, dej)

        j_out = jnp.where(exhausted, j, j + 1)
        tke_out = jnp.where(exhausted, tke, new_tke)
        thl_out = jnp.where(exhausted, thl, thl_new)
        rt_out = jnp.where(exhausted, rt, rt_new)
        dCAPE_out = jnp.where(exhausted, dCAPE_prev, dCAPE_j)
        j_last_out = jnp.where(exhausted, j_last, j)
        done_out = done | exhausted
        return (j_out, tke_out, thl_out, rt_out, dCAPE_out, done_out,
                j_last_out, tke_exit_out, dep_out, dej_out)

    final = _bounded_while(cond_fn, body_fn, init_state, k_ub_zt_py)
    j_final, _tke, _thl, _rt, _dcp, done_final, j_last, tke_exit, dep, dej = final
    return j_last, done_final, j_final, tke_exit, dep, dej


def _compute_lscale_up_col(
    tke_i_col, thl_par_1_up, rt_par_1_up, dCAPE_dz_1_up, CAPE_incr_1_up,
    thl_precalc_up, rt_precalc_up, exp_mu_dzm,
    grav_on_thvm, Lv_coef, thv_ds, exner, p, thvm,
    dzm, invrs_dzm, zt, k_ub_zt_py, nzt,
):
    """``Lscale_up`` for a single column (outer scan over launch levels)."""
    def outer_step(max_alt, k_py):
        tke_i_k = tke_i_col[k_py]
        tke_0 = tke_i_k + CAPE_incr_1_up[k_py + 1]

        # Case A: TKE exhausted before reaching k+1.
        dCAPE_1_kp1 = dCAPE_dz_1_up[k_py + 1]
        safe_dCAPE_a = jnp.where(jnp.abs(dCAPE_1_kp1) > 0.0, dCAPE_1_kp1, 1.0)
        frac_a = -safe_sqrt(-2.0 * tke_i_k * dzm[k_py + 1] * dCAPE_1_kp1) / safe_dCAPE_a

        # Case B/C: parcel survives the initial step -> inner ascent.
        j_last, exited_early, j_final, tke_exit, dCAPE_exit_prev, dCAPE_exit_j = (
            _upward_inner_while(
                k_py, tke_0, thl_par_1_up[k_py + 1], rt_par_1_up[k_py + 1],
                dCAPE_dz_1_up[k_py + 1], thl_precalc_up, rt_precalc_up, exp_mu_dzm,
                grav_on_thvm, Lv_coef, thv_ds, exner, p, thvm,
                dzm, invrs_dzm, zt, k_ub_zt_py,
            )
        )

        base_dist = zt[j_last] - zt[k_py]
        dCAPE_diff = dCAPE_exit_j - dCAPE_exit_prev
        linear_case = (jnp.abs(dCAPE_diff) * 2.0
                       <= jnp.abs(dCAPE_exit_j + dCAPE_exit_prev) * _EPS)
        safe_dCAPE_j = jnp.where(jnp.abs(dCAPE_exit_j) > 0.0, dCAPE_exit_j, 1.0)
        frac_linear = -tke_exit / safe_dCAPE_j
        safe_diff = jnp.where(jnp.abs(dCAPE_diff) > 0.0, dCAPE_diff, 1.0)
        invrs_diff = 1.0 / safe_diff
        disc = dCAPE_exit_prev ** 2 - 2.0 * tke_exit * invrs_dzm[j_final] * dCAPE_diff
        frac_quad = (-dCAPE_exit_prev * invrs_diff * dzm[j_final]
                     - safe_sqrt(disc) * invrs_diff * dzm[j_final])
        frac_inner = jnp.where(linear_case, frac_linear, frac_quad)
        frac_bc = jnp.where(exited_early, frac_inner, 0.0)

        Lscale_up_k = jnp.where(tke_0 > 0.0, _ZLMIN + base_dist + frac_bc, _ZLMIN + frac_a)

        k_alt = zt[k_py] + Lscale_up_k
        Lscale_up_k_smooth = jnp.where(k_alt < max_alt, max_alt - zt[k_py], Lscale_up_k)
        new_max_alt = jnp.where(k_alt < max_alt, max_alt, k_alt)
        return new_max_alt, Lscale_up_k_smooth

    _, vals = jax.lax.scan(
        outer_step, jnp.zeros((), dtype=zt.dtype), jnp.arange(nzt - 2))
    return jnp.concatenate([vals, jnp.full(2, _ZLMIN, dtype=zt.dtype)])


def _downward_inner_while(
    k_py, tke_0, thl_init, rt_init, dCAPE_init,
    thl_precalc_down, rt_precalc_down, exp_mu_dzm,
    grav_on_thvm, Lv_coef, thv_ds, exner, p, thvm,
    dzm, invrs_dzm, zt, k_lb_zt_py, max_iters,
):
    """Downward parcel trajectory from ``k_py-2`` (ascending). See module docstring."""
    init_state = (
        k_py - 2, tke_0, thl_init, rt_init, dCAPE_init,
        jnp.bool_(False), k_py - 1, tke_0, dCAPE_init,
        jnp.zeros((), dtype=tke_0.dtype),
    )

    def cond_fn(state):
        j, _tke, _thl, _rt, _dcp, done, _jl, _tex, _dep1, _dej = state
        return ~done & (j >= k_lb_zt_py)

    def body_fn(state):
        j, tke, thl, rt, dCAPE_plus1, done, j_last, tex, dep1, dej = state
        thl_new = thl_precalc_down[j] + thl * exp_mu_dzm[j + 1]
        rt_new = rt_precalc_down[j] + rt * exp_mu_dzm[j + 1]
        thv_new = _parcel_thv(thl_new, rt_new, exner[j], p[j], thv_ds[j], Lv_coef[j])
        dCAPE_j = grav_on_thvm[j] * (thv_new - thvm[j])
        CAPE_incr = 0.5 * (dCAPE_j + dCAPE_plus1) * dzm[j + 1]

        new_tke = tke - CAPE_incr
        exhausted = new_tke <= 0.0
        newly_ex = exhausted & ~done

        tex_out = jnp.where(newly_ex, tke, tex)
        dep1_out = jnp.where(newly_ex, dCAPE_plus1, dep1)
        dej_out = jnp.where(newly_ex, dCAPE_j, dej)

        j_out = jnp.where(exhausted, j, j - 1)
        tke_out = jnp.where(exhausted, tke, new_tke)
        thl_out = jnp.where(exhausted, thl, thl_new)
        rt_out = jnp.where(exhausted, rt, rt_new)
        dCAPE_out = jnp.where(exhausted, dCAPE_plus1, dCAPE_j)
        j_last_out = jnp.where(exhausted, j_last, j)
        done_out = done | exhausted
        return (j_out, tke_out, thl_out, rt_out, dCAPE_out, done_out,
                j_last_out, tex_out, dep1_out, dej_out)

    final = _bounded_while(cond_fn, body_fn, init_state, max_iters)
    j_final, _tke, _thl, _rt, _dcp, done_final, j_last, tke_exit, dep1, dej = final
    return j_last, done_final, j_final, tke_exit, dep1, dej


def _compute_lscale_down_col(
    tke_i_col, thl_par_1_down, rt_par_1_down, dCAPE_dz_1_down, CAPE_incr_1_down,
    thl_precalc_down, rt_precalc_down, exp_mu_dzm,
    grav_on_thvm, Lv_coef, thv_ds, exner, p, thvm,
    dzm, invrs_dzm, zt, k_ub_zt_py, k_lb_zt_py, nzt,
):
    """``Lscale_down`` for a single column (outer scan descending from the top)."""
    def outer_step(min_alt, i):
        k_py = nzt - 1 - i

        tke_i_k = tke_i_col[k_py]
        tke_0 = tke_i_k - CAPE_incr_1_down[k_py - 1]

        dCAPE_1_km1 = dCAPE_dz_1_down[k_py - 1]
        safe_dCAPE_a = jnp.where(jnp.abs(dCAPE_1_km1) > 0.0, dCAPE_1_km1, 1.0)
        frac_a = safe_sqrt(2.0 * tke_i_k * dzm[k_py] * dCAPE_1_km1) / safe_dCAPE_a

        j_last, exited_early, j_final, tke_exit, dCAPE_exit_plus1, dCAPE_exit_j = (
            _downward_inner_while(
                k_py, tke_0, thl_par_1_down[k_py - 1], rt_par_1_down[k_py - 1],
                dCAPE_dz_1_down[k_py - 1], thl_precalc_down, rt_precalc_down, exp_mu_dzm,
                grav_on_thvm, Lv_coef, thv_ds, exner, p, thvm,
                dzm, invrs_dzm, zt, k_lb_zt_py, k_ub_zt_py,
            )
        )

        base_dist = zt[k_py] - zt[j_last]
        dCAPE_diff = dCAPE_exit_j - dCAPE_exit_plus1
        linear_case = (jnp.abs(dCAPE_diff) * 2.0
                       <= jnp.abs(dCAPE_exit_j + dCAPE_exit_plus1) * _EPS)
        safe_dCAPE_j = jnp.where(jnp.abs(dCAPE_exit_j) > 0.0, dCAPE_exit_j, 1.0)
        frac_linear = tke_exit / safe_dCAPE_j
        safe_diff = jnp.where(jnp.abs(dCAPE_diff) > 0.0, dCAPE_diff, 1.0)
        invrs_diff = 1.0 / safe_diff
        disc = dCAPE_exit_plus1 ** 2 + 2.0 * tke_exit * invrs_dzm[j_final + 1] * dCAPE_diff
        frac_quad = (-dCAPE_exit_plus1 * invrs_diff * dzm[j_final + 1]
                     + safe_sqrt(disc) * invrs_diff * dzm[j_final + 1])
        frac_inner = jnp.where(linear_case, frac_linear, frac_quad)
        frac_bc = jnp.where(exited_early, frac_inner, 0.0)

        Lscale_down_k = jnp.where(tke_0 > 0.0, _ZLMIN + base_dist + frac_bc, _ZLMIN + frac_a)

        k_alt = zt[k_py] - Lscale_down_k
        Lscale_down_k_smooth = jnp.where(k_alt > min_alt, zt[k_py] - min_alt, Lscale_down_k)
        new_min_alt = jnp.where(k_alt > min_alt, min_alt, k_alt)
        return new_min_alt, (k_py, Lscale_down_k_smooth)

    init_min_alt = zt[k_ub_zt_py]
    _, (k_indices, vals) = jax.lax.scan(outer_step, init_min_alt, jnp.arange(nzt - 1))
    col = jnp.full(nzt, _ZLMIN, dtype=zt.dtype)
    return col.at[k_indices].set(vals)


def compute_mixing_length(
    thvm, thlm, rtm, em, Lscale_max, p_in_Pa, exner, thv_ds,
    mu, lmin, l_implemented, gr: CLUBBGrid,
):
    """Nonlocal parcel mixing length (``mixing_length.F90:compute_mixing_length``).

    Parameters
    ----------
    thvm, thlm, rtm, p_in_Pa, exner, thv_ds : jax.Array
        Virtual potential temp, liquid-water potential temp, total water,
        pressure [Pa], Exner, dry-static virtual potential temp; all on
        thermodynamic (zt) levels, shape ``(ngrdcol, nzt)``.
    em : jax.Array
        TKE on momentum (zm) levels, shape ``(ngrdcol, nzm)``.
    Lscale_max : jax.Array
        Per-column cap [m], shape ``(ngrdcol,)`` (see :func:`set_Lscale_max`).
    mu : jax.Array
        Entrainment rate [1/m], shape ``(ngrdcol,)`` (``CLUBBParams.mu``).
    lmin : float
        Surface-layer minimum length [m] (``CLUBBParams.lmin_coef``-scaled).
    l_implemented : bool
        True when CLUBB runs inside a host model (surface layer above ground).
    gr : CLUBBGrid
        Ascending CLUBB staggered grid (provides ``zt``/``zm``/``dzm``/
        ``invrs_dzm``).

    Returns
    -------
    tuple of jax.Array
        ``(Lscale, Lscale_up, Lscale_down)`` on zt levels, ``(ngrdcol, nzt)``.
    """
    ngrdcol, nzt = thvm.shape
    k_ub_zt_py = nzt - 1
    k_lb_zt_py = 0
    # Working float dtype = the thermodynamic-state dtype. Normalize EVERY float
    # input (state, grid, params) to it up front so the whole compute path — scan
    # carries, padded concatenations, the returned Lscale — stays in one dtype.
    # Without this a float32 column silently promotes to float64 (breaking
    # float32/Metal) AND, under JAX_ENABLE_X64, makes lax.scan reject a float32
    # carry against a float64 body. Anchoring to a single dtype also covers MIXED
    # inputs (e.g. float32 state with a float64 grid, or vice-versa). Under an
    # all-float64 column every cast is a no-op → byte-identical to before (the
    # golden-parity Lscale tests still pass).
    dt_f = thlm.dtype
    thvm = thvm.astype(dt_f)
    thlm = thlm.astype(dt_f)
    rtm = rtm.astype(dt_f)
    em = em.astype(dt_f)
    p_in_Pa = p_in_Pa.astype(dt_f)
    exner = exner.astype(dt_f)
    thv_ds = thv_ds.astype(dt_f)
    Lscale_max = jnp.asarray(Lscale_max).astype(dt_f)
    mu = jnp.asarray(mu).astype(dt_f)
    lmin = jnp.asarray(lmin).astype(dt_f)   # scalar surface-layer floor coeff
    gr = gr._replace(
        zm=gr.zm.astype(dt_f), zt=gr.zt.astype(dt_f),
        invrs_dzm=gr.invrs_dzm.astype(dt_f), invrs_dzt=gr.invrs_dzt.astype(dt_f),
        dzm=gr.dzm.astype(dt_f), dzt=gr.dzt.astype(dt_f))

    # ---- Shared precomputations (vectorized over columns) ----
    tke_i = zm2zt(em, gr)                                  # (ngrdcol, nzt)
    grav_on_thvm = buoyancy_coefficient(thvm)
    Lv_coef = constants.L_v / (exner * constants.c_pd) - _EP2 * thv_ds

    exp_mu_dzm = jnp.exp(-mu[:, None] * gr.dzm)            # (ngrdcol, nzm)
    entrain_coef = (1.0 - exp_mu_dzm) * gr.invrs_dzm / mu[:, None]

    _pad0 = jnp.zeros((ngrdcol, 1), dtype=dt_f)
    _pad2 = jnp.zeros((ngrdcol, 2), dtype=dt_f)

    # Upward precalcs (parcel-from-below recurrence coefficients).
    thl_mid, thl_blw = thlm[:, 1:nzt - 1], thlm[:, 0:nzt - 2]
    rt_mid, rt_blw = rtm[:, 1:nzt - 1], rtm[:, 0:nzt - 2]
    emu_up, ec_up = exp_mu_dzm[:, 1:nzt - 1], entrain_coef[:, 1:nzt - 1]
    thl_precalc_up = jnp.concatenate(
        [_pad0, thl_mid - thl_blw * emu_up - (thl_mid - thl_blw) * ec_up, _pad2], axis=1)
    rt_precalc_up = jnp.concatenate(
        [_pad0, rt_mid - rt_blw * emu_up - (rt_mid - rt_blw) * ec_up, _pad2], axis=1)

    ec_init_up = entrain_coef[:, 1:nzt]
    thl_par_1_up_int = thlm[:, 1:] - (thlm[:, 1:] - thlm[:, :-1]) * ec_init_up
    rt_par_1_up_int = rtm[:, 1:] - (rtm[:, 1:] - rtm[:, :-1]) * ec_init_up
    tl_par_1_up_int = thl_par_1_up_int * exner[:, 1:]
    rsat_1_up_int = sat_mixrat_liq(p_in_Pa[:, 1:], tl_par_1_up_int)
    tl_sqd_up = tl_par_1_up_int ** 2
    s_1_up = (rt_par_1_up_int - rsat_1_up_int) * tl_sqd_up / (tl_sqd_up + _LV2_COEF * rsat_1_up_int)
    rc_1_up = jnp.maximum(s_1_up, _ZERO)
    thv_1_up = (thl_par_1_up_int + _EP1 * thv_ds[:, 1:] * rt_par_1_up_int
                + Lv_coef[:, 1:] * rc_1_up)
    dCAPE_dz_1_up_int = grav_on_thvm[:, 1:] * (thv_1_up - thvm[:, 1:])
    CAPE_incr_1_up_int = 0.5 * dCAPE_dz_1_up_int * gr.dzm[:, 1:nzt]

    thl_par_1_up = jnp.concatenate([_pad0, thl_par_1_up_int], axis=1)
    rt_par_1_up = jnp.concatenate([_pad0, rt_par_1_up_int], axis=1)
    dCAPE_dz_1_up = jnp.concatenate([_pad0, dCAPE_dz_1_up_int], axis=1)
    CAPE_incr_1_up = jnp.concatenate([_pad0, CAPE_incr_1_up_int], axis=1)

    # Downward precalcs (parcel-from-above recurrence coefficients).
    thl_abv, thl_at_j = thlm[:, 1:], thlm[:, :-1]
    rt_abv, rt_at_j = rtm[:, 1:], rtm[:, :-1]
    emu_dn, ec_dn = exp_mu_dzm[:, 1:nzt], entrain_coef[:, 1:nzt]
    thl_precalc_down = jnp.concatenate(
        [thl_at_j - thl_abv * emu_dn - (thl_at_j - thl_abv) * ec_dn, _pad2], axis=1)
    rt_precalc_down = jnp.concatenate(
        [rt_at_j - rt_abv * emu_dn - (rt_at_j - rt_abv) * ec_dn, _pad2], axis=1)

    ec_init_dn = entrain_coef[:, 1:nzt]
    thl_par_1_dn_int = thlm[:, :-1] - (thlm[:, :-1] - thlm[:, 1:]) * ec_init_dn
    rt_par_1_dn_int = rtm[:, :-1] - (rtm[:, :-1] - rtm[:, 1:]) * ec_init_dn
    tl_par_1_dn_int = thl_par_1_dn_int * exner[:, :-1]
    rsat_1_dn_int = sat_mixrat_liq(p_in_Pa[:, :-1], tl_par_1_dn_int)
    tl_sqd_dn = tl_par_1_dn_int ** 2
    s_1_dn = (rt_par_1_dn_int - rsat_1_dn_int) * tl_sqd_dn / (tl_sqd_dn + _LV2_COEF * rsat_1_dn_int)
    rc_1_dn = jnp.maximum(s_1_dn, _ZERO)
    thv_1_dn = (thl_par_1_dn_int + _EP1 * thv_ds[:, :-1] * rt_par_1_dn_int
                + Lv_coef[:, :-1] * rc_1_dn)
    dCAPE_dz_1_dn_int = grav_on_thvm[:, :-1] * (thv_1_dn - thvm[:, :-1])
    CAPE_incr_1_dn_int = 0.5 * dCAPE_dz_1_dn_int * gr.dzm[:, 1:nzt]

    thl_par_1_down = jnp.concatenate([thl_par_1_dn_int, _pad0], axis=1)
    rt_par_1_down = jnp.concatenate([rt_par_1_dn_int, _pad0], axis=1)
    dCAPE_dz_1_down = jnp.concatenate([dCAPE_dz_1_dn_int, _pad0], axis=1)
    CAPE_incr_1_down = jnp.concatenate([CAPE_incr_1_dn_int, _pad0], axis=1)

    # ---- Per-column up/down via vmap over the column axis ----
    def up_col(args):
        (tke_i_c, thl1_c, rt1_c, dcap1_c, cap1_c, thlpc_c, rtpc_c, emu_c,
         got_c, lvc_c, thvds_c, exner_c, p_c, thvm_c, dzm_c, idzm_c, zt_c) = args
        return _compute_lscale_up_col(
            tke_i_c, thl1_c, rt1_c, dcap1_c, cap1_c, thlpc_c, rtpc_c, emu_c,
            got_c, lvc_c, thvds_c, exner_c, p_c, thvm_c, dzm_c, idzm_c, zt_c,
            k_ub_zt_py, nzt)

    def down_col(args):
        (tke_i_c, thl1_c, rt1_c, dcap1_c, cap1_c, thlpc_c, rtpc_c, emu_c,
         got_c, lvc_c, thvds_c, exner_c, p_c, thvm_c, dzm_c, idzm_c, zt_c) = args
        return _compute_lscale_down_col(
            tke_i_c, thl1_c, rt1_c, dcap1_c, cap1_c, thlpc_c, rtpc_c, emu_c,
            got_c, lvc_c, thvds_c, exner_c, p_c, thvm_c, dzm_c, idzm_c, zt_c,
            k_ub_zt_py, k_lb_zt_py, nzt)

    up_args = (tke_i, thl_par_1_up, rt_par_1_up, dCAPE_dz_1_up, CAPE_incr_1_up,
               thl_precalc_up, rt_precalc_up, exp_mu_dzm,
               grav_on_thvm, Lv_coef, thv_ds, exner, p_in_Pa, thvm,
               gr.dzm, gr.invrs_dzm, gr.zt)
    down_args = (tke_i, thl_par_1_down, rt_par_1_down, dCAPE_dz_1_down, CAPE_incr_1_down,
                 thl_precalc_down, rt_precalc_down, exp_mu_dzm,
                 grav_on_thvm, Lv_coef, thv_ds, exner, p_in_Pa, thvm,
                 gr.dzm, gr.invrs_dzm, gr.zt)

    Lscale_up_all = jax.vmap(up_col)(up_args)
    Lscale_down_all = jax.vmap(down_col)(down_args)

    # ---- Surface-layer floor lminh + Lscale_max cap ----
    invrs_sfclyr = 1.0 / _LSCALE_SFCLYR_DEPTH
    if l_implemented:
        zm_sfc = gr.zm[:, 0]   # ascending: bottom zm level = ground
        lminh = (jnp.maximum(0.0, _LSCALE_SFCLYR_DEPTH - (gr.zt - zm_sfc[:, None]))
                 * lmin * invrs_sfclyr)
    else:
        lminh = jnp.maximum(0.0, _LSCALE_SFCLYR_DEPTH - gr.zt) * lmin * invrs_sfclyr

    Lscale_up = jnp.maximum(lminh, Lscale_up_all)
    Lscale_down = jnp.maximum(lminh, Lscale_down_all)
    Lscale = safe_sqrt(Lscale_up * Lscale_down)

    # Upper boundary: Lscale[k_ub] = Lscale[k_ub - 1].
    Lscale = Lscale.at[:, k_ub_zt_py].set(Lscale[:, k_ub_zt_py - 1])
    Lscale = jnp.minimum(Lscale, Lscale_max[:, None])
    return Lscale, Lscale_up, Lscale_down


# ===========================================================================
# 7. Implicit band solvers
# ===========================================================================
# ``tridiag_solve`` adapts CLUBB's band storage to the shared legoESM Thomas
# solver (legoesm.timestepping.tridiagonal.thomas_solve); ``penta_solve`` is a
# verbatim, bit-exact port of CLUBB's pentadiagonal LU (penta_lu_solve).

def tridiag_solve(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
    """Solve the CLUBB-band tridiagonal system ``lhs @ x = rhs`` per column.

    Parameters
    ----------
    lhs : jax.Array
        Band-stored LHS, shape ``(3, ngrdcol, ndim)`` with rows
        ``[super, main, sub]`` (see module docstring).
    rhs : jax.Array
        Right-hand side, shape ``(ngrdcol, ndim)``.

    Returns
    -------
    jax.Array
        Solution, shape ``(ngrdcol, ndim)``.
    """
    sup = lhs[0]   # super (couples to k+1) -> Thomas c
    mid = lhs[1]   # main diagonal          -> Thomas b
    sub = lhs[2]   # sub (couples to k-1)   -> Thomas a
    return thomas_solve(sub, mid, sup, rhs)


def penta_solve(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
    """Solve a pentadiagonal system ``lhs @ x = rhs`` via LU (CLUBB band storage).

    Faithful port of ``penta_lu_solver.F90`` /
    ``CLUBB-JAX/.../penta_lu_solver.py`` (``penta_lu_solve``). legoESM has no
    pentadiagonal solver; this is the one used by the coupled moment advances
    (``advance_wp2_wp3``, ``advance_xm_wpxp``), whose interleaved 2-field systems
    are pentadiagonal of size ``2*nzm-1``. Pure ``lax.scan`` LU (no module-scope
    JIT — the caller JITs the physics step); reverse-mode differentiable.

    Parameters
    ----------
    lhs : jax.Array
        Band-stored LHS, shape ``(5, ngrdcol, ndim)`` with rows
        ``[super2, super1, diag, sub1, sub2]`` (Fortran ``lhs(-2:2)``):
        ``super2``/``super1`` couple level ``k`` to ``k+2``/``k+1``;
        ``sub1``/``sub2`` couple to ``k-1``/``k-2``. Requires ``ndim >= 3``.
    rhs : jax.Array
        Right-hand side, shape ``(ngrdcol, ndim)``.

    Returns
    -------
    jax.Array
        Solution, shape ``(ngrdcol, ndim)``.
    """
    super2_t = lhs[0].T   # (ndim, ngrdcol)
    super1_t = lhs[1].T
    diag_t = lhs[2].T
    sub1_t = lhs[3].T
    sub2_t = lhs[4].T
    rhs_t = rhs.T
    ndim = lhs.shape[2]

    # ---- LU decomposition ----
    ldi_0 = 1.0 / diag_t[0]
    u1_0 = ldi_0 * super1_t[0]
    u2_0 = ldi_0 * super2_t[0]
    l1_0 = jnp.zeros_like(ldi_0)
    l2_0 = jnp.zeros_like(ldi_0)

    l1_1 = sub1_t[1]
    l2_1 = jnp.zeros_like(ldi_0)
    ldi_1 = 1.0 / (diag_t[1] - l1_1 * u1_0)
    u1_1 = ldi_1 * (super1_t[1] - l1_1 * u2_0)
    u2_1 = ldi_1 * super2_t[1]

    def lu_scan_step(carry, x):
        u1_km1, u1_km2, u2_km1, u2_km2 = carry
        s2, s1, d, sb1, sb2 = x
        l2 = sb2
        l1 = sb1 - l2 * u1_km2
        ldi = 1.0 / (d - l2 * u2_km2 - l1 * u1_km1)
        u1 = ldi * (s1 - l1 * u2_km1)
        u2 = ldi * s2
        return (u1, u1_km1, u2, u2_km1), (ldi, l1, l2, u1, u2)

    _, (ldi_rest, l1_rest, l2_rest, u1_rest, u2_rest) = lax.scan(
        lu_scan_step, (u1_1, u1_0, u2_1, u2_0),
        (super2_t[2:], super1_t[2:], diag_t[2:], sub1_t[2:], sub2_t[2:]))

    ldi_t = jnp.concatenate([ldi_0[None], ldi_1[None], ldi_rest], axis=0)
    l1_t = jnp.concatenate([l1_0[None], l1_1[None], l1_rest], axis=0)
    l2_t = jnp.concatenate([l2_0[None], l2_1[None], l2_rest], axis=0)
    u1_t = jnp.concatenate([u1_0[None], u1_1[None], u1_rest], axis=0)
    u2_t = jnp.concatenate([u2_0[None], u2_1[None], u2_rest], axis=0)

    # ---- Forward substitution: L y = rhs ----
    soln_0 = ldi_t[0] * rhs_t[0]
    soln_1 = ldi_t[1] * (rhs_t[1] - l1_t[1] * soln_0)

    def fwd_scan_step(carry, x):
        soln_km2, soln_km1 = carry
        rhs_k, ldi_k, l1_k, l2_k = x
        soln_k = ldi_k * (rhs_k - l2_k * soln_km2 - l1_k * soln_km1)
        return (soln_km1, soln_k), soln_k

    _, soln_rest = lax.scan(
        fwd_scan_step, (soln_0, soln_1),
        (rhs_t[2:], ldi_t[2:], l1_t[2:], l2_t[2:]))
    soln_t = jnp.concatenate([soln_0[None], soln_1[None], soln_rest], axis=0)

    # ---- Backward substitution: U x = y ----
    soln_nm2 = soln_t[ndim - 2] - u1_t[ndim - 2] * soln_t[ndim - 1]

    def bwd_scan_step(carry, x):
        soln_kp1, soln_kp2 = carry
        soln_k_fwd, u1_k, u2_k = x
        soln_k = soln_k_fwd - u1_k * soln_kp1 - u2_k * soln_kp2
        return (soln_k, soln_kp1), soln_k

    _, soln_bwd_rev = lax.scan(
        bwd_scan_step, (soln_nm2, soln_t[ndim - 1]),
        (soln_t[:ndim - 2][::-1], u1_t[:ndim - 2][::-1], u2_t[:ndim - 2][::-1]))

    soln_final_t = jnp.concatenate(
        [soln_bwd_rev[::-1], soln_nm2[None], soln_t[ndim - 1:ndim]], axis=0)
    return soln_final_t.T


# ===========================================================================
# 8. Mass-conserving hole filling
# ===========================================================================
# fill_holes.F90 ports: the sliding-window / global hole fillers used on the
# advanced means (CAM ``fill_holes_type = 2``) and the TKE-conserving
# wp2-from-horizontal-TKE fill (``l_wp2_fill_holes_tke = .true.``).

_NUM_HF_DRAW = 2       # num_hf_draw_points (constants_clubb): sliding-window half-width


def fill_holes_global(field, rho_dz, threshold, lower_k, upper_k):
    """Mass-conserving global hole-fill over ``[lower_k, upper_k]`` (``fill_holes_global``).

    Redistributes mass across the whole column-interval so every level reaches
    at least ``threshold`` while preserving ``sum(rho_dz * field)``. ``field``,
    ``rho_dz`` are ``(ngrdcol, nz)``; ``lower_k``/``upper_k`` are static 0-based
    inclusive bounds. Returns the filled field (unchanged in columns with no
    hole).
    """
    nz = field.shape[1]
    k_idx = jnp.arange(nz)[None, :]
    mask = (k_idx >= lower_k) & (k_idx <= upper_k)

    rho_dz_m = jnp.where(mask, rho_dz, 0.0)
    denom = jnp.sum(rho_dz_m, axis=1, keepdims=True)
    field_avg = jnp.sum(rho_dz_m * field, axis=1, keepdims=True) / denom

    field_clipped = jnp.where(
        field_avg >= threshold,
        jnp.maximum(threshold, field),
        jnp.minimum(threshold, field),
    )
    field_clipped_avg = jnp.sum(rho_dz_m * field_clipped, axis=1, keepdims=True) / denom

    safe = (jnp.abs(field_clipped_avg - threshold)
            > jnp.abs(field_clipped_avg + threshold) * _EPS / 2.0)
    mass_frac = jnp.where(
        safe,
        (field_avg - threshold) / jnp.where(safe, field_clipped_avg - threshold, 1.0),
        1.0,
    )
    field_new = jnp.where(mask,
                          threshold + mass_frac * (field_clipped - threshold),
                          field)

    any_hole = jnp.any(jnp.where(mask, field < threshold, False),
                       axis=1, keepdims=True)
    return jnp.where(any_hole, field_new, field)


def fill_holes_sliding_window(field, rho_dz, threshold, lower_k, upper_k,
                              num_draw=_NUM_HF_DRAW):
    """Sliding-window fill with global fallback (``fill_holes_type = 2``).

    Sweeps a window of width ``2*num_draw + 1`` over the interior, locally
    redistributing mass to fill holes; if any hole survives the sweep the
    mass-conserving :func:`fill_holes_global` runs over the full interval. The
    window length is static (compile-time) so the ``fori_loop`` body has a fixed
    ``dynamic_slice`` shape. ``lower_k``/``upper_k`` are static 0-based inclusive
    bounds.
    """
    wlen = 2 * num_draw + 1

    def body(k, field_carry):
        start = k - num_draw
        field_win = jax.lax.dynamic_slice(
            field_carry, (0, start), (field_carry.shape[0], wlen))
        rho_dz_win = jax.lax.dynamic_slice(
            rho_dz, (0, start), (rho_dz.shape[0], wlen))

        denom = jnp.sum(rho_dz_win, axis=1, keepdims=True)
        field_avg = jnp.sum(rho_dz_win * field_win, axis=1, keepdims=True) / denom
        any_hole = jnp.any(field_win < threshold, axis=1, keepdims=True)

        field_clipped = jnp.where(
            field_avg >= threshold,
            jnp.maximum(threshold, field_win),
            jnp.minimum(threshold, field_win),
        )
        field_clipped_avg = jnp.sum(rho_dz_win * field_clipped, axis=1, keepdims=True) / denom

        safe = (jnp.abs(field_clipped_avg - threshold)
                > jnp.abs(field_clipped_avg + threshold) * _EPS / 2.0)
        mass_frac = jnp.where(
            safe,
            (field_avg - threshold) / jnp.where(safe, field_clipped_avg - threshold, 1.0),
            1.0,
        )
        field_win_new = threshold + mass_frac * (field_clipped - threshold)
        field_win_out = jnp.where(any_hole, field_win_new, field_win)
        return jax.lax.dynamic_update_slice(field_carry, field_win_out, (0, start))

    start_k = lower_k + num_draw
    end_k = upper_k - num_draw + 1
    field_sw = jax.lax.fori_loop(start_k, end_k, body, field)

    return jax.lax.cond(
        jnp.any(field_sw < threshold),
        lambda f: fill_holes_global(f, rho_dz, threshold, lower_k, upper_k),
        lambda f: f,
        field_sw,
    )


def fill_holes_vertical(field, rho_ds, dz, threshold, lower_k, upper_k,
                        fill_holes_type, grid_dir_indx=1):
    """Mass-conserving vertical hole-fill (``fill_holes_vertical_api``).

    Dispatches on the static ``fill_holes_type``: 1 = global, 2 = sliding-window
    + global fallback (CAM default). ``field``/``rho_ds``/``dz`` are
    ``(ngrdcol, nz)``; ``lower_k``/``upper_k`` are static 0-based inclusive
    bounds. Returns a filled copy (input not mutated). Unknown types raise
    (no silent default).

    JIT contract: ``fill_holes_type``, ``lower_k``, ``upper_k`` and
    ``grid_dir_indx`` are **compile-time static** (they drive Python branching
    and the window/slice shapes). In normal use they come from the static
    fixed CAM flag values/grid config, closed over by the enclosing ``jax.jit``;
    if this function is jitted directly they must be passed via
    ``static_argnums=(4, 5, 6, 7)`` (or the matching ``static_argnames``). The
    array inputs (``field``/``rho_ds``/``dz``) and ``threshold`` are traced and
    differentiable. ``grid_dir_indx`` is accepted for reference-signature parity
    but currently unused (the CAM-default ascending grid is grid_dir = +1).
    """
    rho_dz = rho_ds * dz
    if fill_holes_type == 1:
        return fill_holes_global(field, rho_dz, threshold, lower_k, upper_k)
    elif fill_holes_type == 2:
        return fill_holes_sliding_window(field, rho_dz, threshold, lower_k, upper_k)
    raise ValueError(f"fill_holes_type={fill_holes_type} not supported "
                     "(CAM-default tree implements 1 and 2)")




def fill_holes_wp2_from_horz_tke(wp2, up2, vp2, threshold, lower_k, upper_k):
    """TKE-conserving wp2 hole-fill from the horizontal variances (CAM default).

    Faithful port of ``fill_holes.F90:fill_holes_wp2_from_horz_tke``
    (``l_wp2_fill_holes_tke = .true.``): where ``wp2 < threshold`` and there is
    available TKE in ``up2``/``vp2`` (``> threshold``), borrow from up2/vp2 to
    fill the wp2 hole, conserving total ``wp2 + up2 + vp2``. If the available
    TKE is insufficient (case 1) wp2 takes all of it (up2/vp2 floored to
    ``threshold``); otherwise (case 2) the deficit is drawn proportionally, with
    one-sided fallbacks when a component has no surplus. Only levels in the
    static ``[lower_k, upper_k]`` range are modified. ``wp2``/``up2``/``vp2`` are
    ``(ncol, nzm)``. Returns ``(wp2, up2, vp2)``.
    """
    nzm = wp2.shape[1]
    k_idx = jnp.arange(nzm)[None, :]
    in_range = (k_idx >= lower_k) & (k_idx <= upper_k)

    do_fill = in_range & (wp2 < threshold) & ((up2 > threshold) | (vp2 > threshold))
    missing = threshold - wp2
    up2_avail = jnp.maximum(up2 - threshold, 0.0)
    vp2_avail = jnp.maximum(vp2 - threshold, 0.0)
    total_avail = up2_avail + vp2_avail

    case1 = do_fill & (missing >= total_avail)        # not enough TKE
    wp2_c1 = wp2 + total_avail
    up2_c1 = jnp.minimum(up2, threshold)
    vp2_c1 = jnp.minimum(vp2, threshold)

    case2 = do_fill & (missing < total_avail)          # enough TKE
    eps_thr = _F64_EPS * 1000.0
    case2a = case2 & (jnp.abs(up2_avail) < eps_thr)    # take all from vp2
    case2b = case2 & (~case2a) & (jnp.abs(vp2_avail) < eps_thr)  # take all from up2
    ratio = jnp.where(
        total_avail > 0.0, missing / jnp.where(total_avail > 0.0, total_avail, 1.0), 0.0)

    up2_2c = threshold + up2_avail * (1.0 - ratio)
    vp2_2c = threshold + vp2_avail * (1.0 - ratio)
    up2_c2 = jnp.where(case2a, up2, jnp.where(case2b, up2 - missing, up2_2c))
    vp2_c2 = jnp.where(case2a, vp2 - missing, jnp.where(case2b, vp2, vp2_2c))

    wp2_new = jnp.where(case1, wp2_c1, jnp.where(case2, threshold, wp2))
    up2_new = jnp.where(case1, up2_c1, jnp.where(case2, up2_c2, up2))
    vp2_new = jnp.where(case1, vp2_c1, jnp.where(case2, vp2_c2, vp2))
    return wp2_new, up2_new, vp2_new


# ===========================================================================
# 9. Skewness diagnostics
# ===========================================================================
# Skx_func / gamma(Skw) / the LG 2005 xp3 ansatz (CAM
# ``l_advance_xp3 = .false.`` → xp3 diagnosed, not prognosed) and the
# smoothed wp3_on_wp2 ratio bundle feeding the closure.

# CLUBB ``eps`` = max(1e-10, machine-eps); used only in the degenerate-gamma
# guard below. A safety tolerance, not a physical constant.
_WP3_ON_WP2_CLIP = 1000.0   # bound on the wp3/wp2 ratio (calc_wp3_on_wp2)


def Skx_func(
    xp2: jax.Array,
    xp3: jax.Array,
    x_tol: float,
    Skw_denom_coef: float,
) -> jax.Array:
    """Skewness of ``x`` with the LG05 sensitivity-reduction denominator.

    ``Skx = xp3 * (xp2 + Skw_denom_coef * x_tol^2)^(-3/2)``
    (``Skx_module.F90:Skx_func``). With the CAM default ``Skw_denom_coef = 0``
    this reduces to ``xp3 / xp2^(3/2)``.

    Parameters
    ----------
    xp2, xp3 : jax.Array
        Second and third moments of ``x``.
    x_tol : float
        Tolerance for ``x`` (e.g. ``w_tol``/``rt_tol``/``thl_tol``).
    Skw_denom_coef : float
        Sensitivity-reduction coefficient (``CLUBBParams.Skw_denom_coef``).

    Returns
    -------
    jax.Array
        Skewness of ``x``.
    """
    denom_tol = Skw_denom_coef * x_tol ** 2
    return xp3 * (xp2 + denom_tol) ** (-1.5)


def compute_gamma_Skw(
    Skw: jax.Array,
    gamma_coef: float,
    gamma_coefb: float,
    gamma_coefc: float,
    l_gamma_Skw: bool = True,
) -> jax.Array:
    """Gamma coefficient as a Gaussian function of w-skewness.

    ``Skx_module.F90:compute_gamma_Skw``. With ``l_gamma_Skw`` on and the two
    coefficients meaningfully different::

        gamma = gamma_coefb + (gamma_coef - gamma_coefb)
                              * exp(-0.5 * (Skw / gamma_coefc)^2)

    otherwise ``gamma = gamma_coef`` (constant). The degenerate-coefficient
    branch is data-independent (depends only on the coefficients), so it is a
    ``jnp.where`` rather than a Python ``if`` — keeping the coefficients
    differentiable. ``l_gamma_Skw`` is a static model flag (Python ``if``).

    Parameters
    ----------
    Skw : jax.Array
        Skewness of w (zm or zt levels), shape ``(ngrdcol, nz)``.
    gamma_coef, gamma_coefb, gamma_coefc : float
        Tunable gamma coefficients (``CLUBBParams``).
    l_gamma_Skw : bool, default True
        Static flag; when False, returns the constant ``gamma_coef``.

    Returns
    -------
    jax.Array
        ``gamma_Skw_fnc`` with ``Skw``'s shape.
    """
    if not l_gamma_Skw:
        return gamma_coef + jnp.zeros_like(Skw)
    gc = jnp.asarray(gamma_coef)
    gb = jnp.asarray(gamma_coefb)
    gcf = jnp.asarray(gamma_coefc)
    cond = jnp.abs(gc - gb) > jnp.abs(gc + gb) * _EPS / 2.0
    varying = gb + (gc - gb) * jnp.exp(-0.5 * (Skw / gcf) ** 2)
    return jnp.where(cond, varying, gc + jnp.zeros_like(Skw))


def LG_2005_ansatz(
    Skw: jax.Array,
    wpxp: jax.Array,
    wp2: jax.Array,
    xp2: jax.Array,
    sigma_sqd_w: jax.Array,
    beta: float,
    x_tol: float,
    w_tol: float,
) -> jax.Array:
    """Skewness of ``x`` from skewness of ``w`` (LG05 eqs. 11, 16, 33).

    ``Skx_module.F90:LG_2005_ansatz``.

    Parameters
    ----------
    Skw : jax.Array
        Skewness of w.
    wpxp, wp2, xp2 : jax.Array
        ``w'x'`` flux, w-variance, x-variance.
    sigma_sqd_w : jax.Array
        PDF width parameter (< 1).
    beta : float
        Tunable LG05 coefficient (``CLUBBParams.beta``).
    x_tol, w_tol : float
        Tolerances for ``x`` and ``w`` (floors on the variances).

    Returns
    -------
    jax.Array
        Skewness of ``x``.
    """
    one_minus_ssw = 1.0 - sigma_sqd_w
    nrmlzd_corr_wx = wpxp / jnp.sqrt(
        jnp.maximum(wp2, w_tol ** 2) * jnp.maximum(xp2, x_tol ** 2) * one_minus_ssw
    )
    nrmlzd_Skw = Skw / (one_minus_ssw * jnp.sqrt(one_minus_ssw))
    return nrmlzd_Skw * nrmlzd_corr_wx * (beta + (1.0 - beta) * nrmlzd_corr_wx ** 2)


def xp3_LG_2005_ansatz(
    Skw_zt: jax.Array,
    wpxp_zt: jax.Array,
    wp2_zt: jax.Array,
    xp2_zt: jax.Array,
    sigma_sqd_w_zt: jax.Array,
    beta: float,
    x_tol: float,
    w_tol: float,
    Skw_denom_coef: float,
) -> jax.Array:
    """``<x'^3>`` from the LG05 skewness ansatz (inverse of :func:`Skx_func`).

    ``Skx_module.F90:xp3_LG_2005_ansatz``: ``xp3 = Skx * (xp2 + denom_tol)^(3/2)``
    with ``Skx`` from :func:`LG_2005_ansatz`. Used to diagnose ``xp3`` when
    ``l_advance_xp3 = .false.`` (CAM default).
    """
    Skx_denom_tol = Skw_denom_coef * x_tol ** 2
    Skx_zt = LG_2005_ansatz(
        Skw_zt, wpxp_zt, wp2_zt, xp2_zt, sigma_sqd_w_zt, beta, x_tol, w_tol
    )
    xp2_safe = xp2_zt + Skx_denom_tol
    return Skx_zt * xp2_safe * jnp.sqrt(xp2_safe)


def calc_wp3_on_wp2(wp2, wp3, w_tol, gr: CLUBBGrid):
    """Smoothed ``wp3/wp2`` ratio on zm and zt levels (``calc_wp3_on_wp2``).

    ``wp2`` is floored to ``w_tol^2`` on zt, the ratio clipped to ``[-1000,
    1000]``, then round-tripped zt->zm->zt to suppress spikes. ``wp2`` is
    zm-level, ``wp3`` zt-level. Returns ``(wp3_on_wp2, wp3_on_wp2_zt)``.
    """
    w_tol_sqd = w_tol ** 2
    wp2_zt = jnp.maximum(zm2zt(wp2, gr), w_tol_sqd)
    wp3_on_wp2_zt = jnp.clip(wp3 / jnp.maximum(wp2_zt, w_tol_sqd),
                             -_WP3_ON_WP2_CLIP, _WP3_ON_WP2_CLIP)
    wp3_on_wp2 = zt2zm(wp3_on_wp2_zt, gr)
    wp3_on_wp2_zt = zm2zt(wp3_on_wp2, gr)
    return wp3_on_wp2, wp3_on_wp2_zt


def compute_skewness_diagnostics(wp2, wp3, w_tol, Skw_denom_coef, gr: CLUBBGrid):
    """Skewness + wp3/wp2-ratio diagnostics for the moment advances.

    Assembles ``Skw`` on both grids (:func:`Skx_func`) and the smoothed
    ``wp3_on_wp2`` ratio (:func:`calc_wp3_on_wp2`) from the carried ``wp2`` (zm)
    and ``wp3`` (zt). Returns a dict with ``Skw_zm``, ``Skw_zt``, ``wp2_zt``
    (floored), ``wp3_zm``, ``wp3_on_wp2``, ``wp3_on_wp2_zt`` — the diagnostics the
    wp2/wp3, xp2/xpyp and xm/wpxp advances consume.
    """
    w_tol_sqd = w_tol ** 2
    wp2_zt = jnp.maximum(zm2zt(wp2, gr), w_tol_sqd)
    wp3_zm = zt2zm(wp3, gr)
    Skw_zt = Skx_func(wp2_zt, wp3, w_tol, Skw_denom_coef)
    Skw_zm = Skx_func(wp2, wp3_zm, w_tol, Skw_denom_coef)
    wp3_on_wp2, wp3_on_wp2_zt = calc_wp3_on_wp2(wp2, wp3, w_tol, gr)
    return dict(Skw_zm=Skw_zm, Skw_zt=Skw_zt, wp2_zt=wp2_zt, wp3_zm=wp3_zm,
                wp3_on_wp2=wp3_on_wp2, wp3_on_wp2_zt=wp3_on_wp2_zt)


# ===========================================================================
# 10. Dissipation time-scale (tau) family
# ===========================================================================
# CAM tau family (``l_diag_Lscale_from_tau = .false.`` → SIMPLE
# ``tau = Lscale/sqrt(em)`` with the N^2 stability correction,
# ``l_stability_correct_tau_zm = .true.``): TKE em, invrs_tau_C1/C4/C6/C14/
# xp2_zm and invrs_tau_wp3_zt.

_MAX_STABILITY_CORR = 3.0   # cap on the N2 stability enhancement (advance_helper)
_EM_MIN_COEF = 1.5          # em_min = 1.5 * w_tol^2 (constants_clubb)


def compute_tke(wp2, up2, vp2, gr: CLUBBGrid, config):
    """Turbulent kinetic energy ``em`` (zm) and ``sqrt_em_zt`` (zt).

    CAM ``l_tke_aniso = .true.`` → ``em = 0.5·(wp2 + vp2 + up2)`` (the anisotropic
    TKE); the ``.false.`` branch uses ``em = 1.5·wp2``. ``sqrt_em_zt =
    sqrt(max(zm2zt(em), em_min))`` with ``em_min = 1.5·w_tol^2``. All moment
    inputs are zm-level. Returns ``(em, sqrt_em_zt)`` — the TKE the tau model and
    MFL consume.
    """
    # CAM l_tke_aniso = True (fixed; flags table at file end). The isotropic
    # False branch (em = 1.5*wp2) is not ported.
    em = 0.5 * (wp2 + vp2 + up2)
    em_min = _EM_MIN_COEF * config.w_tol ** 2
    sqrt_em_zt = jnp.sqrt(jnp.maximum(zm2zt(em, gr), em_min))
    return em, sqrt_em_zt


def calc_stability_correction(brunt_vaisala_freq_sqd, Lscale_zm, em,
                              lambda0_stability_coef):
    """Brunt-Vaisala stability correction factor (``calc_stability_correction``).

    ``1 + min(lambda0·N^2·Lscale_zm^2/em, 3)`` where ``lambda0`` is zeroed in
    unstable layers (``N^2 <= 0``). All zm-level ``(ncol, nzm)``;
    ``lambda0_stability_coef`` is the tunable coefficient (scalar or per-column).
    """
    lambda0_eff = jnp.where(brunt_vaisala_freq_sqd > 0.0, lambda0_stability_coef, 0.0)
    return 1.0 + jnp.minimum(
        lambda0_eff * brunt_vaisala_freq_sqd * Lscale_zm ** 2 / em, _MAX_STABILITY_CORR)


def compute_tau_family(Lscale, em, sqrt_em_zt, brunt_vaisala_freq_sqd, gr: CLUBBGrid,
                       config):
    """CAM-default ``invrs_tau_*`` family from the parcel ``Lscale`` and TKE.

    ``tau_zt = min(Lscale/sqrt_em_zt, taumax)``, ``tau_zm =
    min(Lscale_zm/sqrt(max(em_min, em)), taumax)`` with ``Lscale_zm = max(zt2zm
    (Lscale), 0)``. The stability correction (``l_stability_correct_tau_zm =
    True``) scales the C1/C6 branch; C4/C14/xp2 use the plain wp2 tau
    (``l_use_invrs_tau_N2_iso = False``); wp3 uses the zt tau. ``Lscale``/
    ``sqrt_em_zt`` are zt-level; ``em`` (TKE) / ``brunt_vaisala_freq_sqd`` are
    zm-level. Returns a dict of the inverse time-scales the advances consume.
    """
    params = config.params
    taumax = params.taumax
    em_min = _EM_MIN_COEF * config.w_tol ** 2

    tau_zt = jnp.minimum(Lscale / sqrt_em_zt, taumax)
    Lscale_zm = jnp.maximum(zt2zm(Lscale, gr), 0.0)
    tau_zm = jnp.minimum(Lscale_zm / jnp.sqrt(jnp.maximum(em_min, em)), taumax)
    invrs_tau_zm = 1.0 / tau_zm
    invrs_tau_zt = 1.0 / tau_zt

    # em is floored to em_min here for the stability-correction division (em is
    # physically TKE >= em_min, so this is forward-identical to the reference's
    # raw-em form, but keeps the 1/em gradient finite at the floor — AD safety).
    stability_correction = calc_stability_correction(
        brunt_vaisala_freq_sqd, Lscale_zm, jnp.maximum(em, em_min),
        params.lambda0_stability_coef)
    invrs_tau_N2_zm = invrs_tau_zm * stability_correction

    return dict(
        invrs_tau_zm=invrs_tau_zm, invrs_tau_zt=invrs_tau_zt,
        invrs_tau_C1_zm=invrs_tau_N2_zm, invrs_tau_C6_zm=invrs_tau_N2_zm,
        invrs_tau_C4_zm=invrs_tau_zm, invrs_tau_C14_zm=invrs_tau_zm,
        invrs_tau_xp2_zm=invrs_tau_zm, invrs_tau_wp3_zt=invrs_tau_zt,
        tau_zm=tau_zm, tau_zt=tau_zt, stability_correction=stability_correction,
    )


# ===========================================================================
# 11. ADG1 assumed-PDF parameter closure
# ===========================================================================
# The ADG1 double-Gaussian PDF parameters (CAM ``iiPDF_type = ADG1``):
# component means/variances/mixture fraction, the binormal component
# correlations, and the liquid cloud-fraction / cloud-water closure (Flatau
# saturation via clubb_saturation -> legoesm.thermo).

_ZERO = 0.0
_SKW_TOL = 1.0e-5   # |Skw| below which mixt_frac is pinned to 0.5

# Numerical tolerances / smoothing magnitudes (CLUBB constants_clubb.F90) — not
# physical constants and not tunable scheme params (safety/smoothing scales).
_CHI_TOL = max(1.0e-8, jnp.finfo(jnp.float64).eps)   # chi tolerance [kg/kg]
_MIN_MAX_SMTH_MAG = 1.0e-9        # smoothing magnitude for smooth_max
_MAX_NUM_STDEVS = 5.0             # PDF truncation range for cloud-frac limits




def ADG1_w_closure(wm, wp2, Skw, sigma_sqd_w, sqrt_wp2, mixt_frac_max_mag):
    """Mixture fraction and w PDF component parameters (``ADG1_w_closure``).

    Returns ``(w_1, w_2, w_1_n, w_2_n, varnce_w_1, varnce_w_2, mixt_frac)``.
    ``w_1_n``/``w_2_n`` are the normalized component means
    (``w_i = wm + sqrt_wp2 * w_i_n``). By construction the two Gaussians
    reproduce ``wm``, ``wp2``, and ``Skw`` exactly.

    Parameters
    ----------
    wm, wp2, Skw, sigma_sqd_w, sqrt_wp2 : jax.Array
        Mean w, w-variance, w-skewness, PDF width parameter, ``sqrt(wp2)``.
    mixt_frac_max_mag : float
        Cap on the mixture fraction (``derive_mixt_frac_max_mag``).
    """
    denom_sq = 4.0 * (1.0 - sigma_sqd_w) ** 3 + Skw ** 2
    mf_formula = 0.5 * (1.0 - Skw / jnp.sqrt(denom_sq))
    mixt_frac = jnp.where(jnp.abs(Skw) <= _SKW_TOL, 0.5, mf_formula)
    mixt_frac = jnp.clip(mixt_frac, 1.0 - mixt_frac_max_mag, mixt_frac_max_mag)

    one_minus_mf = 1.0 - mixt_frac
    sigma_factor = 1.0 - sigma_sqd_w
    # safe_sqrt: 1-sigma_sqd_w -> 0 in well-mixed/surface layers -> bare sqrt
    # has an inf reverse-mode gradient there (forward-identical, arg >= 0).
    w_1_n = safe_sqrt(one_minus_mf / mixt_frac * sigma_factor)
    w_2_n = -safe_sqrt(mixt_frac / one_minus_mf * sigma_factor)

    w_1 = wm + sqrt_wp2 * w_1_n
    w_2 = wm + sqrt_wp2 * w_2_n
    varnce_w_1 = sigma_sqd_w * wp2
    varnce_w_2 = sigma_sqd_w * wp2
    return w_1, w_2, w_1_n, w_2_n, varnce_w_1, varnce_w_2, mixt_frac


def ADG1_ADG2_responder_params(xm, xp2, wp2, sqrt_wp2, wpxp,
                               w_1_n, w_2_n, mixt_frac, sigma_sqd_w, beta):
    """Bi-normal component params for a responder ``x`` (rt/thl/u/v).

    ``ADG1_ADG2_responder_params``. Returns
    ``(x_1, x_2, varnce_x_1, varnce_x_2, alpha_x)``. By construction the
    bi-normal reproduces ``xm`` and the covariance ``w'x'`` (``wpxp``) exactly.

    Parameters
    ----------
    xm, xp2, wp2, sqrt_wp2, wpxp : jax.Array
        Mean/variance of x, w-variance, ``sqrt(wp2)``, ``w'x'`` covariance.
    w_1_n, w_2_n, mixt_frac, sigma_sqd_w : jax.Array
        ADG1 w-closure outputs.
    beta : float or jax.Array
        Tunable parameter (``CLUBBParams.beta``); scalar or per-column
        ``(ngrdcol,)``.
    """
    x_1 = xm - wpxp / (sqrt_wp2 * w_2_n)
    x_2 = xm - wpxp / (sqrt_wp2 * w_1_n)

    alpha_x = 0.5 * (1.0 - wpxp ** 2 / ((1.0 - sigma_sqd_w) * wp2 * xp2))
    alpha_x = jnp.clip(alpha_x, _ZERO, 1.0)

    beta_arr = jnp.asarray(beta)
    beta_bc = beta_arr[:, None] if beta_arr.ndim == 1 else beta_arr
    two_thirds_beta = (2.0 / 3.0) * beta_bc
    width_factor_1 = two_thirds_beta + 2.0 * mixt_frac * (1.0 - two_thirds_beta)

    varnce_x_1 = width_factor_1 * xp2 * alpha_x / mixt_frac
    varnce_x_2 = (2.0 - width_factor_1) * xp2 * alpha_x / (1.0 - mixt_frac)
    return x_1, x_2, varnce_x_1, varnce_x_2, alpha_x


def ADG1_pdf_driver(wm, rtm, thlm, um, vm, wp2, rtp2, thlp2, up2, vp2,
                    Skw, wprtp, wpthlp, upwp, vpwp, sqrt_wp2, sigma_sqd_w,
                    beta, mixt_frac_max_mag):
    """Top-level ADG1 PDF parameter driver (``ADG1_pdf_driver``).

    Closes the ``w`` PDF then the rt/thl/u/v responders. All inputs on zt
    levels, shape ``(ngrdcol, nzt)``.

    Returns
    -------
    dict
        Component means/variances and mixture fraction for w and each
        responder (keys ``w_1``/``w_2``/``varnce_w_*``/``mixt_frac``,
        ``rt_1``/``rt_2``/``varnce_rt_*``/``alpha_rt``, and likewise for
        ``thl``, ``u``, ``v``).
    """
    (w_1, w_2, w_1_n, w_2_n,
     varnce_w_1, varnce_w_2, mixt_frac) = ADG1_w_closure(
        wm, wp2, Skw, sigma_sqd_w, sqrt_wp2, mixt_frac_max_mag)

    rt_1, rt_2, varnce_rt_1, varnce_rt_2, alpha_rt = ADG1_ADG2_responder_params(
        rtm, rtp2, wp2, sqrt_wp2, wprtp, w_1_n, w_2_n, mixt_frac, sigma_sqd_w, beta)
    thl_1, thl_2, varnce_thl_1, varnce_thl_2, alpha_thl = ADG1_ADG2_responder_params(
        thlm, thlp2, wp2, sqrt_wp2, wpthlp, w_1_n, w_2_n, mixt_frac, sigma_sqd_w, beta)
    u_1, u_2, varnce_u_1, varnce_u_2, alpha_u = ADG1_ADG2_responder_params(
        um, up2, wp2, sqrt_wp2, upwp, w_1_n, w_2_n, mixt_frac, sigma_sqd_w, beta)
    v_1, v_2, varnce_v_1, varnce_v_2, alpha_v = ADG1_ADG2_responder_params(
        vm, vp2, wp2, sqrt_wp2, vpwp, w_1_n, w_2_n, mixt_frac, sigma_sqd_w, beta)

    return {
        "w_1": w_1, "w_2": w_2,
        "w_1_n": w_1_n, "w_2_n": w_2_n,
        "varnce_w_1": varnce_w_1, "varnce_w_2": varnce_w_2,
        "mixt_frac": mixt_frac,
        "rt_1": rt_1, "rt_2": rt_2,
        "varnce_rt_1": varnce_rt_1, "varnce_rt_2": varnce_rt_2, "alpha_rt": alpha_rt,
        "thl_1": thl_1, "thl_2": thl_2,
        "varnce_thl_1": varnce_thl_1, "varnce_thl_2": varnce_thl_2, "alpha_thl": alpha_thl,
        "u_1": u_1, "u_2": u_2,
        "varnce_u_1": varnce_u_1, "varnce_u_2": varnce_u_2, "alpha_u": alpha_u,
        "v_1": v_1, "v_2": v_2,
        "varnce_v_1": varnce_v_1, "varnce_v_2": varnce_v_2, "alpha_v": alpha_v,
    }


# ===========================================================================
# Liquid cloud fraction / cloud water from the ADG1 PDF (pdf_closure_module)
# ===========================================================================


def smooth_corr_quotient(numerator, denominator, denom_thresh):
    """Smoothly bounded correlation quotient ``num/den`` (``pdf_utilities.F90``).

    Two ``smooth_max`` lifts keep the result a valid correlation: the
    denominator is raised to at least ``|num|/max_mag_correlation`` (so
    ``|quotient| <= max_mag_correlation``) and then to ``denom_thresh`` (never
    divides by ~0). Pure-jnp, differentiable.
    """
    num = jnp.asarray(numerator)
    den = jnp.asarray(denominator)
    coef = jnp.minimum(_MIN_MAX_SMTH_MAG, denom_thresh)

    def _smax(a, b):
        return 0.5 * ((a + b) + jnp.sqrt((a - b) ** 2 + coef ** 2))

    tmp = _smax(jnp.abs(num) / _MAX_MAG_CORRELATION, den)
    tmp = _smax(tmp, denom_thresh)
    return num / tmp


def calc_comp_corrs_binormal(xpyp, xm, ym, mu_x_1, mu_x_2, mu_y_1, mu_y_2,
                             sigma_x_1_sqd, sigma_x_2_sqd, sigma_y_1_sqd,
                             sigma_y_2_sqd, mixt_frac):
    """Shared PDF-component correlation of two bi-normal variables x, y.

    ``pdf_utilities.F90:calc_comp_corrs_binormal``. Both components share one
    correlation, solved from the overall covariance
    ``<x'y'> = sum_i w_i[(mu_x_i-<x>)(mu_y_i-<y>) + corr*sigma_x_i*sigma_y_i]``
    and bounded by :func:`smooth_corr_quotient`. Returns ``(corr, corr)``.
    """
    a = jnp.asarray(mixt_frac)
    numerator = (xpyp - a * (mu_x_1 - xm) * (mu_y_1 - ym)
                 - (1.0 - a) * (mu_x_2 - xm) * (mu_y_2 - ym))
    denominator = (a * safe_sqrt(sigma_x_1_sqd * sigma_y_1_sqd)
                   + (1.0 - a) * safe_sqrt(sigma_x_2_sqd * sigma_y_2_sqd))
    corr = smooth_corr_quotient(numerator, denominator, _EPS)
    return corr, corr


def transform_pdf_chi_eta_component(tl, rsatl, rt, exner_in,
                                    varnce_rt, varnce_thl, corr_rt_thl):
    """Sommeria-Deardorff (rt, thl) -> (chi, eta) transform for one PDF component.

    ``pdf_closure_module.F90:transform_pdf_chi_eta_component``. ``chi`` is the
    extended liquid water (saturation excess). The local ``cc_slope`` is the
    Clausius-Clapeyron sensitivity ``eps*L_v^2/(R_d*c_pd*tl^2)`` (NOT the tunable
    ``beta``). Returns ``(chi, crt, cthl, stdev_chi, stdev_eta, covar_chi_eta,
    corr_chi_eta)``.
    """
    cc_slope = constants.epsilon * constants.L_v ** 2 / (constants.R_d * constants.c_pd * tl ** 2)
    invrs = 1.0 / (1.0 + cc_slope * rsatl)
    chi = (rt - rsatl) * invrs
    crt = invrs
    cthl = ((1.0 + cc_slope * rt) * invrs ** 2
            * (constants.c_pd / constants.L_v) * cc_slope * rsatl * exner_in)
    vrnc_rt_t = crt ** 2 * varnce_rt
    vrnc_thl_t = cthl ** 2 * varnce_thl
    # safe_sqrt: component variances can be exactly 0 (e.g. alpha_x clipped to 0
    # for perfectly-correlated columns), and a bare sqrt has a singular VJP
    # there. Forward-identical (variances >= 0).
    corr_t = 2.0 * corr_rt_thl * crt * cthl * safe_sqrt(varnce_rt * varnce_thl)
    vrnc_chi = vrnc_rt_t - corr_t + vrnc_thl_t
    vrnc_eta = vrnc_rt_t + corr_t + vrnc_thl_t
    stdev_chi = safe_sqrt(vrnc_chi)
    stdev_eta = safe_sqrt(vrnc_eta)
    covar_chi_eta = vrnc_rt_t - vrnc_thl_t
    # smooth_corr_quotient (pdf_utilities) bounding corr_chi_eta to [-0.99, 0.99].
    corr_chi_eta = smooth_corr_quotient(covar_chi_eta, stdev_chi * stdev_eta, _CHI_TOL ** 2)
    return chi, crt, cthl, stdev_chi, stdev_eta, covar_chi_eta, corr_chi_eta


def calc_liquid_cloud_frac_component(mean_chi, stdev_chi):
    """Liquid cloud fraction + cloud water of one PDF component (Gaussian CDF of chi).

    ``pdf_closure_module.F90:calc_liquid_cloud_frac_component``, with
    +-``max_num_stdevs`` truncation to the clear / fully-cloudy limits.
    Returns ``(cloud_frac, rc)``.
    """
    mean_chi = jnp.asarray(mean_chi)
    stdev_chi = jnp.asarray(stdev_chi)
    is_clear = (((jnp.abs(mean_chi) <= _EPS) & (stdev_chi <= _CHI_TOL))
                | (mean_chi < -_MAX_NUM_STDEVS * stdev_chi))
    is_full = mean_chi > _MAX_NUM_STDEVS * stdev_chi
    # Double-where denominator guard (AD-safe in any precision). Where the PDF
    # component has resolvable width (``stdev_chi > floor``) the quotient uses
    # the true stdev (in float64 ``floor`` is below the reference 1e-100, so
    # active cells are bit-identical to the reference); elsewhere it uses a
    # dummy denominator of 1, so neither the quotient nor its tangent
    # ``-mean/safe_s^2`` can overflow regardless of ``mean_chi`` magnitude. Such
    # zero-width cells are always selected as clear or fully-cloudy below, so the
    # forward result is unchanged. ``floor = sqrt(tiny)`` keeps ``stdev_chi^2``
    # representable for the resolvable cells too.
    dt = stdev_chi.dtype
    floor = jnp.sqrt(jnp.finfo(dt).tiny)
    resolvable = stdev_chi > floor
    safe_s = jnp.where(resolvable, stdev_chi, jnp.asarray(1.0, dt))
    zeta = mean_chi / safe_s
    # The Gaussian (cf_mid/rc_mid) is the SELECTED output only on partial-cloud
    # cells (``partial``), where |zeta| <= max_num_stdevs already — there it uses
    # the RAW zeta, so the active-cell value AND gradient are reference-identical
    # (including exactly at the cutoff mean_chi = +-max_num_stdevs*stdev_chi,
    # which the strict clear/full comparisons keep in the partial branch).
    # On masked clear/full cells the Gaussian is discarded, so zeta is clamped
    # there only to stop ``exp(-0.5*zeta^2)`` from overflowing (-> NaN VJP) in
    # float32 where the masked zeta can be enormous.
    partial = ~(is_clear | is_full)
    zeta_g = jnp.where(partial, zeta, jnp.clip(zeta, -_MAX_NUM_STDEVS, _MAX_NUM_STDEVS))
    cf_mid = 0.5 * (1.0 + jax.scipy.special.erf(zeta_g / _SQRT_2))
    rc_mid = mean_chi * cf_mid + stdev_chi * jnp.exp(-0.5 * zeta_g ** 2) / _SQRT_2PI
    cf = jnp.where(is_clear, 0.0, jnp.where(is_full, 1.0, cf_mid))
    rc = jnp.where(is_clear, 0.0, jnp.where(is_full, mean_chi, rc_mid))
    return cf, rc


def calc_pdf_liquid_cloud_frac(adg1, rtpthlp, rtm, thlm, exner, p_in_Pa):
    """Liquid cloud fraction and cloud water from the ADG1 PDF components.

    Derives the bi-normal rt-thl component correlation from the resolved
    covariance ``rtpthlp``, applies the chi-eta transform per component, the
    Gaussian cloud-fraction integral per component, and combines by mixture
    fraction (``pdf_closure_module.F90`` pre-advance path). Saturation is Flatau
    (CAM default) via :mod:`clubb_saturation`.

    Parameters
    ----------
    adg1 : dict
        Output of :func:`ADG1_pdf_driver` (component means/variances + mixt_frac).
    rtpthlp, rtm, thlm, exner, p_in_Pa : jax.Array
        ``r_t' theta_l'`` covariance, mean total water, mean theta_l, Exner, and
        pressure [Pa], all on zt levels, shape ``(ngrdcol, nzt)``.

    Returns
    -------
    tuple of jax.Array
        ``(rcm, cloud_frac)`` on zt levels — cloud water [kg/kg] and liquid
        cloud fraction [-].
    """
    comp = calc_pdf_liquid_cloud_frac_components(
        adg1, rtpthlp, rtm, thlm, exner, p_in_Pa)
    return comp["rcm"], comp["cloud_frac"]


def calc_pdf_liquid_cloud_frac_components(adg1, rtpthlp, rtm, thlm, exner, p_in_Pa):
    """Per-component liquid cloud-fraction PDF closure (the full intermediates).

    Like :func:`calc_pdf_liquid_cloud_frac` but returns every per-component
    intermediate (``chi``/``crt``/``cthl``/``stdev_chi``/``stdev_eta``/
    ``corr_ce``/``cf``/``rc`` for components 1 and 2, plus ``mixt_frac``,
    ``cloud_frac`` and ``rcm``). The cloud-water flux assembly
    (:func:`calc_pdf_xprcp_fluxes`) consumes these.
    Mirrors ``pdf_closure_module.F90:calc_pdf_liquid_cloud_frac_components``.
    """
    corr_1, corr_2 = calc_comp_corrs_binormal(
        rtpthlp, rtm, thlm, adg1["rt_1"], adg1["rt_2"], adg1["thl_1"], adg1["thl_2"],
        adg1["varnce_rt_1"], adg1["varnce_rt_2"], adg1["varnce_thl_1"],
        adg1["varnce_thl_2"], adg1["mixt_frac"])

    mf = adg1["mixt_frac"]
    tl_1 = adg1["thl_1"] * exner
    tl_2 = adg1["thl_2"] * exner
    rsatl_1 = sat_mixrat_liq(p_in_Pa, tl_1)
    rsatl_2 = sat_mixrat_liq(p_in_Pa, tl_2)

    (chi_1, crt_1, cthl_1, schi_1, seta_1, _, corr_ce_1) = transform_pdf_chi_eta_component(
        tl_1, rsatl_1, adg1["rt_1"], exner, adg1["varnce_rt_1"], adg1["varnce_thl_1"], corr_1)
    (chi_2, crt_2, cthl_2, schi_2, seta_2, _, corr_ce_2) = transform_pdf_chi_eta_component(
        tl_2, rsatl_2, adg1["rt_2"], exner, adg1["varnce_rt_2"], adg1["varnce_thl_2"], corr_2)

    cf_1, rc_1 = calc_liquid_cloud_frac_component(chi_1, schi_1)
    cf_2, rc_2 = calc_liquid_cloud_frac_component(chi_2, schi_2)

    cloud_frac = mf * cf_1 + (1.0 - mf) * cf_2
    rcm = jnp.maximum(0.0, mf * rc_1 + (1.0 - mf) * rc_2)
    return {
        "mixt_frac": mf, "cloud_frac": cloud_frac, "rcm": rcm,
        "chi_1": chi_1, "chi_2": chi_2,
        "crt_1": crt_1, "crt_2": crt_2, "cthl_1": cthl_1, "cthl_2": cthl_2,
        "stdev_chi_1": schi_1, "stdev_chi_2": schi_2,
        "stdev_eta_1": seta_1, "stdev_eta_2": seta_2,
        "corr_ce_1": corr_ce_1, "corr_ce_2": corr_ce_2,
        "cf_1": cf_1, "cf_2": cf_2, "rc_1": rc_1, "rc_2": rc_2,
    }


# ===========================================================================
# 12. ADG1 PDF moment integrals + buoyancy-flux assembly
# ===========================================================================
# PDF moment integrals over the ADG1 components: higher-order velocity
# moments (wp4, wp2up2, ...), cloud-water turbulent fluxes (x'rc'), and the
# buoyancy-flux assembly wpthvp / wp2thvp / rtpthvp / thlpthvp.





# ---------------------------------------------------------------------------
# Two-/tri-normal PDF moment integrals (pure; no physical constants)
# ---------------------------------------------------------------------------

def calc_wp2xp_pdf(wm, xm, w_1, w_2, x_1, x_2, varnce_w_1, varnce_w_2,
                   varnce_x_1, varnce_x_2, corr_w_x_1, corr_w_x_2, mixt_frac):
    """``<w'^2 x'>`` over the binormal (w, x) PDF (``calc_wp2xp_pdf``)."""
    a = mixt_frac
    dw1, dw2 = w_1 - wm, w_2 - wm
    dx1, dx2 = x_1 - xm, x_2 - xm
    return (a * ((dw1 ** 2 + varnce_w_1) * dx1
                 + 2.0 * corr_w_x_1 * safe_sqrt(varnce_w_1 * varnce_x_1) * dw1)
            + (1.0 - a) * ((dw2 ** 2 + varnce_w_2) * dx2
                           + 2.0 * corr_w_x_2 * safe_sqrt(varnce_w_2 * varnce_x_2) * dw2))


def calc_wpxp2_pdf(wm, xm, w_1, w_2, x_1, x_2, varnce_w_1, varnce_w_2,
                   varnce_x_1, varnce_x_2, corr_w_x_1, corr_w_x_2, mixt_frac):
    """``<w'x'^2>`` over the binormal (w, x) PDF (``calc_wpxp2_pdf``)."""
    a = mixt_frac
    dw1, dw2 = w_1 - wm, w_2 - wm
    dx1, dx2 = x_1 - xm, x_2 - xm
    return (a * (dw1 * (dx1 ** 2 + varnce_x_1)
                 + 2.0 * corr_w_x_1 * safe_sqrt(varnce_w_1 * varnce_x_1) * dx1)
            + (1.0 - a) * (dw2 * (dx2 ** 2 + varnce_x_2)
                           + 2.0 * corr_w_x_2 * safe_sqrt(varnce_w_2 * varnce_x_2) * dx2))


def calc_wp2xp2_pdf(wm, xm, w_1, w_2, x_1, x_2, varnce_w_1, varnce_w_2,
                    varnce_x_1, varnce_x_2, corr_w_x_1, corr_w_x_2, mixt_frac):
    """``<w'^2 x'^2>`` over the binormal (w, x) PDF (``calc_wp2xp2_pdf``)."""
    a = mixt_frac
    dw1, dw2 = w_1 - wm, w_2 - wm
    dx1, dx2 = x_1 - xm, x_2 - xm
    term1 = (dw1 ** 2 * (dx1 ** 2 + varnce_x_1)
             + 4.0 * corr_w_x_1 * safe_sqrt(varnce_w_1 * varnce_x_1) * dx1 * dw1
             + (dx1 ** 2 + (1.0 + 2.0 * corr_w_x_1 ** 2) * varnce_x_1) * varnce_w_1)
    term2 = (dw2 ** 2 * (dx2 ** 2 + varnce_x_2)
             + 4.0 * corr_w_x_2 * safe_sqrt(varnce_w_2 * varnce_x_2) * dx2 * dw2
             + (dx2 ** 2 + (1.0 + 2.0 * corr_w_x_2 ** 2) * varnce_x_2) * varnce_w_2)
    return a * term1 + (1.0 - a) * term2


def calc_wp4_pdf(wm, w_1, w_2, varnce_w_1, varnce_w_2, mixt_frac):
    """``<w'^4>`` over the two-component normal w-PDF (``calc_wp4_pdf``)."""
    a = mixt_frac
    d1, d2 = w_1 - wm, w_2 - wm
    return (a * (3.0 * varnce_w_1 ** 2 + 6.0 * d1 ** 2 * varnce_w_1 + d1 ** 4)
            + (1.0 - a) * (3.0 * varnce_w_2 ** 2 + 6.0 * d2 ** 2 * varnce_w_2 + d2 ** 4))


def calc_wpxpyp_pdf(wm, xm, ym, w_1, w_2, x_1, x_2, y_1, y_2,
                    varnce_w_1, varnce_w_2, varnce_x_1, varnce_x_2, varnce_y_1, varnce_y_2,
                    corr_w_x_1, corr_w_x_2, corr_w_y_1, corr_w_y_2,
                    corr_x_y_1, corr_x_y_2, mixt_frac):
    """``<w'x'y'>`` over the trinormal (w, x, y) PDF (``calc_wpxpyp_pdf``)."""
    a = mixt_frac
    dw1, dw2 = w_1 - wm, w_2 - wm
    dx1, dx2 = x_1 - xm, x_2 - xm
    dy1, dy2 = y_1 - ym, y_2 - ym
    comp1 = (dw1 * dx1 * dy1 + corr_x_y_1 * safe_sqrt(varnce_x_1 * varnce_y_1) * dw1
             + corr_w_y_1 * safe_sqrt(varnce_w_1 * varnce_y_1) * dx1
             + corr_w_x_1 * safe_sqrt(varnce_w_1 * varnce_x_1) * dy1)
    comp2 = (dw2 * dx2 * dy2 + corr_x_y_2 * safe_sqrt(varnce_x_2 * varnce_y_2) * dw2
             + corr_w_y_2 * safe_sqrt(varnce_w_2 * varnce_y_2) * dx2
             + corr_w_x_2 * safe_sqrt(varnce_w_2 * varnce_x_2) * dy2)
    return a * comp1 + (1.0 - a) * comp2


# ---------------------------------------------------------------------------
# Higher-order-moment orchestration (ADG1: velocity-scalar corrs = 0)
# ---------------------------------------------------------------------------

def calc_pdf_higher_order_moments(adg1, wm_zt, rtm, thlm, um, vm,
                                  corr_rt_thl_1, corr_rt_thl_2, gr: CLUBBGrid):
    """Integrate the ADG1 PDF for the velocity-scalar higher-order moments.

    Returns a dict with the Fortran moment names (``wp2rtp``, ``wp2thlp``,
    ``wp2up``, ``wpup2``, ``wpvp2``, ``wp2up2_zm``, ``wp2vp2_zm``, ``wp4_zm``,
    ``wprtp2``, ``wpthlp2``, ``wprtpthlp``). For ADG1 the w-scalar correlations
    are zero; only ``corr_rt_thl`` (per component) is nonzero.
    """
    mf = adg1["mixt_frac"]
    z = jnp.zeros_like(mf)
    w1, w2 = adg1["w_1"], adg1["w_2"]
    vw1, vw2 = adg1["varnce_w_1"], adg1["varnce_w_2"]
    nzm = gr.zm.shape[1]
    k_ub = nzm - 1
    k_lb = 0

    def _wp2xp(xm, x1, x2, vx1, vx2):
        return calc_wp2xp_pdf(wm_zt, xm, w1, w2, x1, x2, vw1, vw2, vx1, vx2, z, z, mf)

    def _wpxp2(xm, x1, x2, vx1, vx2):
        return calc_wpxp2_pdf(wm_zt, xm, w1, w2, x1, x2, vw1, vw2, vx1, vx2, z, z, mf)

    def _wp2xp2(xm, x1, x2, vx1, vx2):
        return calc_wp2xp2_pdf(wm_zt, xm, w1, w2, x1, x2, vw1, vw2, vx1, vx2, z, z, mf)

    wp2rtp = _wp2xp(rtm, adg1["rt_1"], adg1["rt_2"], adg1["varnce_rt_1"], adg1["varnce_rt_2"])
    wp2thlp = _wp2xp(thlm, adg1["thl_1"], adg1["thl_2"], adg1["varnce_thl_1"], adg1["varnce_thl_2"])
    wp2up = _wp2xp(um, adg1["u_1"], adg1["u_2"], adg1["varnce_u_1"], adg1["varnce_u_2"])
    wpup2 = _wpxp2(um, adg1["u_1"], adg1["u_2"], adg1["varnce_u_1"], adg1["varnce_u_2"])
    wpvp2 = _wpxp2(vm, adg1["v_1"], adg1["v_2"], adg1["varnce_v_1"], adg1["varnce_v_2"])

    wp2up2_zt = _wp2xp2(um, adg1["u_1"], adg1["u_2"], adg1["varnce_u_1"], adg1["varnce_u_2"])
    wp2vp2_zt = _wp2xp2(vm, adg1["v_1"], adg1["v_2"], adg1["varnce_v_1"], adg1["varnce_v_2"])
    wp2up2_zm = zt2zm(wp2up2_zt, gr).at[:, k_ub].set(0.0)
    wp2vp2_zm = zt2zm(wp2vp2_zt, gr).at[:, k_ub].set(0.0)

    wp4_zt = calc_wp4_pdf(wm_zt, w1, w2, vw1, vw2, mf)
    wp4_zm = zt2zm(wp4_zt, gr, zm_min=0.0).at[:, k_lb].set(0.0).at[:, k_ub].set(0.0)

    wprtp2 = _wpxp2(rtm, adg1["rt_1"], adg1["rt_2"], adg1["varnce_rt_1"], adg1["varnce_rt_2"])
    wpthlp2 = _wpxp2(thlm, adg1["thl_1"], adg1["thl_2"], adg1["varnce_thl_1"], adg1["varnce_thl_2"])

    wprtpthlp = calc_wpxpyp_pdf(
        wm_zt, rtm, thlm, w1, w2, adg1["rt_1"], adg1["rt_2"], adg1["thl_1"], adg1["thl_2"],
        vw1, vw2, adg1["varnce_rt_1"], adg1["varnce_rt_2"],
        adg1["varnce_thl_1"], adg1["varnce_thl_2"],
        z, z, z, z, corr_rt_thl_1, corr_rt_thl_2, mf)

    return {
        "wp2rtp": wp2rtp, "wp2thlp": wp2thlp, "wp2up": wp2up,
        "wpup2": wpup2, "wpvp2": wpvp2,
        "wp2up2_zm": wp2up2_zm, "wp2vp2_zm": wp2vp2_zm, "wp4_zm": wp4_zm,
        "wprtp2": wprtp2, "wpthlp2": wpthlp2, "wprtpthlp": wprtpthlp,
    }


# ---------------------------------------------------------------------------
# Cloud-water fluxes x'rc' (ADG1: corr_w_chi = 0)
# ---------------------------------------------------------------------------

def calc_xprcp_component(wm, rtm, thlm, um, vm, rcm,
                         w_i, rt_i, thl_i, u_i, v_i, varnce_w_i,
                         stdev_chi_i, stdev_eta_i, corr_chi_eta_i, crt_i, cthl_i,
                         rc_i, cloud_frac_i):
    """Per-component cloud-water covariances (``calc_xprcp_component``, ADG1).

    Returns ``(wprcp, wp2rcp, rtprcp, thlprcp, uprcp, vprcp)``. ``crt_i``/``cthl_i``
    are the chi sensitivities from the chi/eta transform; the ``cthl=0`` (rsatl=0)
    limit is guarded (the result is masked by ``cloud_frac=0`` there).
    """
    drc = rc_i - rcm
    wprcp = (w_i - wm) * drc
    wp2rcp = ((w_i - wm) ** 2 + varnce_w_i) * drc
    crt_safe = jnp.where(crt_i == 0.0, 1.0, crt_i)
    rtprcp = ((rt_i - rtm) * drc
              + (corr_chi_eta_i * stdev_eta_i + stdev_chi_i) / (2.0 * crt_safe)
              * stdev_chi_i * cloud_frac_i)
    cthl_safe = jnp.where(cthl_i == 0.0, 1.0, cthl_i)
    thlprcp = ((thl_i - thlm) * drc
               + (corr_chi_eta_i * stdev_eta_i - stdev_chi_i) / (2.0 * cthl_safe)
               * stdev_chi_i * cloud_frac_i)
    uprcp = (u_i - um) * drc
    vprcp = (v_i - vm) * drc
    return wprcp, wp2rcp, rtprcp, thlprcp, uprcp, vprcp


def calc_pdf_xprcp_fluxes(adg1, comp, wm_zt, rtm, thlm, um, vm, rcm_zt, gr: CLUBBGrid):
    """Mixed cloud-water turbulent fluxes from the ADG1 PDF (``calc_pdf_xprcp_fluxes``).

    Calls :func:`calc_xprcp_component` for each PDF component, mixes the six
    fluxes (``w'rc'``, ``w'^2 rc'``, ``rt'rc'``, ``thl'rc'``, ``u'rc'``,
    ``v'rc'``) by ``mixt_frac`` on zt, then regrids the five zm-output fluxes
    zt->zm with the top momentum level (``k_ub_zm``) zeroed (corr_w_chi = 0 for
    ADG1). ``comp`` is the per-component dict from
    :func:`calc_pdf_liquid_cloud_frac_components`.

    Returns a dict with the zt-grid fluxes (consumed by the buoyancy-flux
    assembly, which wants the native pdf-grid values) and the regridded zm fluxes.
    """
    mf = comp["mixt_frac"]
    k_ub = gr.zm.shape[1] - 1

    c1 = calc_xprcp_component(
        wm_zt, rtm, thlm, um, vm, rcm_zt,
        adg1["w_1"], adg1["rt_1"], adg1["thl_1"], adg1["u_1"], adg1["v_1"],
        adg1["varnce_w_1"], comp["stdev_chi_1"], comp["stdev_eta_1"],
        comp["corr_ce_1"], comp["crt_1"], comp["cthl_1"], comp["rc_1"], comp["cf_1"])
    c2 = calc_xprcp_component(
        wm_zt, rtm, thlm, um, vm, rcm_zt,
        adg1["w_2"], adg1["rt_2"], adg1["thl_2"], adg1["u_2"], adg1["v_2"],
        adg1["varnce_w_2"], comp["stdev_chi_2"], comp["stdev_eta_2"],
        comp["corr_ce_2"], comp["crt_2"], comp["cthl_2"], comp["rc_2"], comp["cf_2"])

    wprcp_zt, wp2rcp_zt, rtprcp_zt, thlprcp_zt, uprcp_zt, vprcp_zt = (
        mf * a + (1.0 - mf) * b for a, b in zip(c1, c2))

    def _to_zm(field_zt):
        return zt2zm(field_zt, gr).at[:, k_ub].set(0.0)

    return {
        "wprcp_zt": wprcp_zt, "wp2rcp_zt": wp2rcp_zt, "rtprcp_zt": rtprcp_zt,
        "thlprcp_zt": thlprcp_zt, "uprcp_zt": uprcp_zt, "vprcp_zt": vprcp_zt,
        "wprcp_zm": _to_zm(wprcp_zt), "rtprcp_zm": _to_zm(rtprcp_zt),
        "thlprcp_zm": _to_zm(thlprcp_zt), "uprcp_zm": _to_zm(uprcp_zt),
        "vprcp_zm": _to_zm(vprcp_zt),
    }


# ---------------------------------------------------------------------------
# Buoyancy fluxes x'thv' (uses legoesm.constants)
# ---------------------------------------------------------------------------

def calc_xpthvp_terms(exner, thv_ds_zt, wprcp_zt, wp2rcp_zt, rtprcp_zt, thlprcp_zt,
                      wpthlp_zt, wprtp_zt, wp2thlp_zt, wp2rtp_zt,
                      rtpthlp_zt, rtp2_zt, thlp2_zt, gr: CLUBBGrid):
    """Virtual-potential-temperature (buoyancy) fluxes (``calc_xpthvp_terms``).

    ``rc_coef = L_v/(exner*c_pd) - ep2*thv_ds`` and
    ``x'thv' = x'thl' + ep1*thv_ds*x'rt' + rc_coef*x'rc'`` for ``x in {w, w^2,
    rt, thl}``. The three zm-output fluxes (and ``rc_coef``) are regridded zt->zm
    with the top momentum level zeroed; ``wp2thvp`` stays on zt.

    Returns ``(wpthvp_zm, wp2thvp_zt, rtpthvp_zm, thlpthvp_zm, rc_coef_zt,
    rc_coef_zm)``.
    """
    lv, cp = constants.L_v, constants.c_pd
    rc_coef_zt = lv / (exner * cp) - _EP2 * thv_ds_zt
    wpthvp_zt = wpthlp_zt + _EP1 * thv_ds_zt * wprtp_zt + rc_coef_zt * wprcp_zt
    wp2thvp_zt = wp2thlp_zt + _EP1 * thv_ds_zt * wp2rtp_zt + rc_coef_zt * wp2rcp_zt
    rtpthvp_zt = rtpthlp_zt + _EP1 * thv_ds_zt * rtp2_zt + rc_coef_zt * rtprcp_zt
    thlpthvp_zt = thlp2_zt + _EP1 * thv_ds_zt * rtpthlp_zt + rc_coef_zt * thlprcp_zt
    k_ub = gr.zm.shape[1] - 1
    wpthvp_zm = zt2zm(wpthvp_zt, gr).at[:, k_ub].set(0.0)
    rtpthvp_zm = zt2zm(rtpthvp_zt, gr).at[:, k_ub].set(0.0)
    thlpthvp_zm = zt2zm(thlpthvp_zt, gr).at[:, k_ub].set(0.0)
    rc_coef_zm = zt2zm(rc_coef_zt, gr).at[:, k_ub].set(0.0)
    return wpthvp_zm, wp2thvp_zt, rtpthvp_zm, thlpthvp_zm, rc_coef_zt, rc_coef_zm


# ===========================================================================
# 13. Moment-advance building blocks + xp2_xpyp / windm advances
# ===========================================================================
# The shared implicit-advance machinery (diffusion/mean-advection LHS
# builders, Cauchy-Schwarz clipping family) plus two of the four CAM-order
# advances: ``advance_xp2_xpyp`` (rtp2/thlp2/rtpthlp/up2/vp2 — per-moment C2
# dissipation, CAM's 3 distinct C2 values, UPWIND turbulent advection) and
# ``advance_windm_edsclrm`` (um/vm eddy-diffusion advance with implicit
# surface momentum flux + diagnosed upwp/vpwp; CAM
# ``l_predict_upwp_vpwp = .false.``). Round-off parity-tested vs CLUBB-JAX.

_MAX_MAG_CORRELATION = 0.99   # Cauchy-Schwarz correlation bound (constants_clubb)
_MAX_MAG_CORRELATION_FLUX = 0.99  # flux correlation bound (constants_clubb)
_ZERO_THRESHOLD = 0.0
_ONE_THIRD = 1.0 / 3.0
_GAMMA_OVER_IMPLICIT_TS = 1.5   # over-implicit weight (constants_clubb)




# ---------------------------------------------------------------------------
# LHS band builders
# ---------------------------------------------------------------------------

def diffusion_zt_lhs(K_zm, nu, invrs_rho_ds_zt, rho_ds_zm, gr: CLUBBGrid):
    """Tridiagonal LHS for implicit eddy diffusion of a zt-level variable.

    Faithful port of ``diffusion.F90:diffusion_zt_lhs`` (non-upwind path):
    discretizes ``d/dz[(K_zm + nu) d(var_zt)/dz]`` at zt levels with zero-flux
    boundaries. Returns ``(3, ngrdcol, nzt)`` = ``[super, main, sub]``.
    """
    K_zm_nu = K_zm + nu[:, None]
    invrs_dzt = gr.invrs_dzt
    invrs_dzm = gr.invrs_dzm

    common_bot = (invrs_dzt[:, :1] * invrs_rho_ds_zt[:, :1]
                  * K_zm_nu[:, 1:2] * rho_ds_zm[:, 1:2] * invrs_dzm[:, 1:2])
    super_bot, main_bot, sub_bot = -common_bot, common_bot, jnp.zeros_like(common_bot)

    scale_int = invrs_dzt[:, 1:-1] * invrs_rho_ds_zt[:, 1:-1]
    super_int = -scale_int * K_zm_nu[:, 2:-1] * rho_ds_zm[:, 2:-1] * invrs_dzm[:, 2:-1]
    sub_int = -scale_int * K_zm_nu[:, 1:-2] * rho_ds_zm[:, 1:-2] * invrs_dzm[:, 1:-2]
    main_int = -(super_int + sub_int)

    common_top = (invrs_dzt[:, -1:] * invrs_rho_ds_zt[:, -1:]
                  * K_zm_nu[:, -2:-1] * rho_ds_zm[:, -2:-1] * invrs_dzm[:, -2:-1])
    super_top, sub_top, main_top = jnp.zeros_like(common_top), -common_top, common_top

    superdiag = jnp.concatenate([super_bot, super_int, super_top], axis=1)
    maindiag = jnp.concatenate([main_bot, main_int, main_top], axis=1)
    subdiag = jnp.concatenate([sub_bot, sub_int, sub_top], axis=1)
    return jnp.stack([superdiag, maindiag, subdiag], axis=0)


def diffusion_zm_lhs(K_zt, nu, invrs_rho_ds_zm, rho_ds_zt, gr: CLUBBGrid):
    """Tridiagonal LHS for implicit eddy diffusion of a zm-level variable.

    Faithful port of ``diffusion.F90:diffusion_zm_lhs`` (non-upwind): discretizes
    ``d/dz[(K_zt + nu) d(var_zm)/dz]`` at zm levels with zero-flux boundaries.
    Returns ``(3, ngrdcol, nzm)`` = ``[super, main, sub]``. (The k=0 row is not
    used by the solver, per the Fortran note, but is filled for shape.)
    """
    K_zt_nu = K_zt + nu[:, None]
    invrs_dzm = gr.invrs_dzm
    invrs_dzt = gr.invrs_dzt

    common_bot = (invrs_dzm[:, :1] * invrs_rho_ds_zm[:, :1]
                  * K_zt_nu[:, :1] * rho_ds_zt[:, :1] * invrs_dzt[:, :1])
    super_bot, main_bot, sub_bot = -common_bot, common_bot, jnp.zeros_like(common_bot)

    scale_int = invrs_dzm[:, 1:-1] * invrs_rho_ds_zm[:, 1:-1]
    super_int = -scale_int * K_zt_nu[:, 1:] * rho_ds_zt[:, 1:] * invrs_dzt[:, 1:]
    sub_int = -scale_int * K_zt_nu[:, :-1] * rho_ds_zt[:, :-1] * invrs_dzt[:, :-1]
    main_int = -(super_int + sub_int)

    common_top = (invrs_dzm[:, -1:] * invrs_rho_ds_zm[:, -1:]
                  * K_zt_nu[:, -1:] * rho_ds_zt[:, -1:] * invrs_dzt[:, -1:])
    super_top, sub_top, main_top = jnp.zeros_like(common_top), -common_top, common_top

    superdiag = jnp.concatenate([super_bot, super_int, super_top], axis=1)
    maindiag = jnp.concatenate([main_bot, main_int, main_top], axis=1)
    subdiag = jnp.concatenate([sub_bot, sub_int, sub_top], axis=1)
    return jnp.stack([superdiag, maindiag, subdiag], axis=0)


def term_ma_zt_lhs_upwind(wm_zt, gr: CLUBBGrid):
    """Upwind mean-advection LHS for a zt-level variable (CAM ``l_upwind_xm_ma``).

    Faithful port of the upwind branch of ``mean_adv.F90:term_ma_zt_lhs``
    (ascending grid). ``(3, ngrdcol, nzt)`` = ``[super, main, sub]``. The
    centered branch is out of the CAM-default tree and not ported.
    """
    ngrdcol, nzt = wm_zt.shape
    invrs_dzm = gr.invrs_dzm

    wm_int = wm_zt[:, 1:-1]
    idzm_k = invrs_dzm[:, 1:-2]
    idzm_kp1 = invrs_dzm[:, 2:-1]
    mask = wm_int >= 0.0
    sup_int = jnp.where(mask, 0.0, wm_int * idzm_kp1)
    mid_int = jnp.where(mask, wm_int * idzm_k, -wm_int * idzm_kp1)
    sub_int = jnp.where(mask, -wm_int * idzm_k, 0.0)

    # Lower boundary k=0 (Fortran k=1): uses invrs_dzm[1]; the upward-wind
    # contribution is dropped (zero-flux from below), matching mean_adv.F90.
    wm0 = wm_zt[:, 0]
    m0 = wm0 >= 0.0
    idzm_2 = invrs_dzm[:, 1]
    sup0 = jnp.where(m0, 0.0, wm0 * idzm_2)
    mid0 = jnp.where(m0, 0.0, -wm0 * idzm_2)
    sub0 = jnp.zeros_like(wm0)

    # Upper boundary k=nzt-1 (Fortran k=nzt): uses invrs_dzm[nzm-2]=invrs_dzm[nzt-1];
    # the downward-wind contribution is dropped (zero-flux from above).
    wmt = wm_zt[:, -1]
    mt = wmt >= 0.0
    idzm_top = invrs_dzm[:, nzt - 1]
    supt = jnp.zeros_like(wmt)
    midt = jnp.where(mt, wmt * idzm_top, 0.0)
    subt = jnp.where(mt, -wmt * idzm_top, 0.0)

    sup = jnp.concatenate([sup0[:, None], sup_int, supt[:, None]], axis=1)
    mid = jnp.concatenate([mid0[:, None], mid_int, midt[:, None]], axis=1)
    sub = jnp.concatenate([sub0[:, None], sub_int, subt[:, None]], axis=1)
    return jnp.stack([sup, mid, sub], axis=0)


def term_ma_zm_lhs(wm_zm, gr: CLUBBGrid):
    """Centered mean-advection LHS for a zm-level variable (``term_ma_zm_lhs``).

    Faithful port of ``mean_adv.F90:term_ma_zm_lhs``: discretizes
    ``w·d(var_zm)/dz`` implicitly at interior momentum levels with the
    zm→zt interpolation weights. The xp2/xpyp moments live on zm, so their
    mean advection always uses this centered form (the ``l_upwind_xm_ma`` flag
    gates only the *zt*-level scalar/wind advance, not the zm-level moments).

    The zm→zt weights are computed inline from the grid geometry — exactly
    ``calc_zm2zt_weights`` (``grid_class.F90``), ascending grid (grid_dir=+1):

      ``w_above[k] = (zt[k] - zm[k])   / (zm[k+1] - zm[k])``  (weight of zm[k]),
      ``w_below[k] = (zm[k+1] - zt[k]) / (zm[k+1] - zm[k])``  (weight of zm[k+1]),

    for ``k = 0 .. nzt-1``. On a uniform grid both are 1/2. Boundary rows
    (k=0, k=nzm-1) are zero (fixed-value BCs applied by the assembler).
    ``(3, ngrdcol, nzm)`` = ``[super, main, sub]``.
    """
    invrs_dzm = gr.invrs_dzm                       # (ng, nzm)
    total_dist = (gr.zm[:, 1:] - gr.zm[:, :-1]) + 1.0e-30   # (ng, nzt)
    w_above = (gr.zt - gr.zm[:, :-1]) / total_dist  # M_ABOVE, (ng, nzt)
    w_below = (gr.zm[:, 1:] - gr.zt) / total_dist   # M_BELOW, (ng, nzt)

    # Interior momentum levels k = 1 .. nzm-2 (Fortran k = 2 .. nzm-1).
    fac = wm_zm[:, 1:-1] * invrs_dzm[:, 1:-1]       # (ng, nzm-2)
    super_int = fac * w_above[:, 1:]                # weights_zm2zt[:, 1:, M_ABOVE]
    main_int = fac * (w_below[:, 1:] - w_above[:, :-1])
    sub_int = -fac * w_below[:, :-1]                # weights_zm2zt[:, :-1, M_BELOW]

    zeros_bnd = jnp.zeros((wm_zm.shape[0], 1), dtype=wm_zm.dtype)
    superdiag = jnp.concatenate([zeros_bnd, super_int, zeros_bnd], axis=1)
    maindiag = jnp.concatenate([zeros_bnd, main_int, zeros_bnd], axis=1)
    subdiag = jnp.concatenate([zeros_bnd, sub_int, zeros_bnd], axis=1)
    return jnp.stack([superdiag, maindiag, subdiag], axis=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def calc_xpwp(Km_zm, xm, invrs_dzm):
    """Down-gradient eddy flux ``x'w'`` on momentum levels (``calc_xpwp``).

    ``xpwp[k] = Km_zm[k]*invrs_dzm[k]*(xm[k]-xm[k-1])`` for interior k; top and
    bottom levels are zero. ``Km_zm``/``invrs_dzm`` are ``(ngrdcol, nzm)``;
    ``xm`` is ``(ngrdcol, nzt)``.
    """
    ng, nzm = Km_zm.shape
    interior = Km_zm[:, 1:nzm - 1] * invrs_dzm[:, 1:nzm - 1] * (xm[:, 1:] - xm[:, :-1])
    return jnp.zeros((ng, nzm), dtype=xm.dtype).at[:, 1:nzm - 1].set(interior)


def clip_covar(wpxp, wp2, xp2, max_mag_corr=_MAX_MAG_CORRELATION):
    """Cauchy-Schwarz clip of a covariance after the solve (``clip_covar``).

    Clips ``wpxp`` to ``±max_mag_corr·sqrt(wp2·xp2)`` at interior levels; the
    top/bottom boundaries are left unchanged. ``safe_sqrt`` keeps the gradient
    finite where a variance is zero (forward-identical, variances ≥ 0).
    """
    bound = max_mag_corr * safe_sqrt(wp2 * xp2)
    clipped = jnp.clip(wpxp, -bound, bound)
    clipped = clipped.at[:, 0].set(wpxp[:, 0])
    clipped = clipped.at[:, -1].set(wpxp[:, -1])
    return clipped


def clip_covars_denom(wprtp, wpthlp, upwp, vpwp, wp2, rtp2, thlp2, up2, vp2,
                      l_tke_aniso=True):
    """Cauchy-Schwarz clip of the four w-fluxes after the solves (``clip_covars_denom``).

    Clips ``wprtp``/``wpthlp`` against ``wp2``·rtp2/thlp2 with the flux
    correlation bound, and the momentum fluxes ``upwp``/``vpwp`` against
    ``wp2``·up2/vp2 (CAM ``l_tke_aniso = True``; the ``False`` branch clips
    against ``wp2``·wp2) with the (default) correlation bound. Applied between the
    advances in ``advance_clubb_core``. Returns ``(wprtp, wpthlp, upwp, vpwp)``.
    """
    wprtp_new = clip_covar(wprtp, wp2, rtp2, _MAX_MAG_CORRELATION_FLUX)
    wpthlp_new = clip_covar(wpthlp, wp2, thlp2, _MAX_MAG_CORRELATION_FLUX)
    if l_tke_aniso:
        upwp_new = clip_covar(upwp, wp2, up2)
        vpwp_new = clip_covar(vpwp, wp2, vp2)
    else:
        upwp_new = clip_covar(upwp, wp2, wp2)
        vpwp_new = clip_covar(vpwp, wp2, wp2)
    return wprtp_new, wpthlp_new, upwp_new, vpwp_new


def compute_uv_tndcy(fcor, ug, vg, um, vm, um_forcing, vm_forcing):
    """Coriolis + geostrophic + prescribed-forcing wind tendencies (``compute_uv_tndcy``).

    ``d(um)/dt = -fcor·vg + fcor·vm + um_forcing``;
    ``d(vm)/dt = +fcor·ug - fcor·um + vm_forcing``. ``fcor`` is ``(ngrdcol,)``.
    """
    f = fcor[:, None]
    return (-f * vg + f * vm + um_forcing, f * ug - f * um + vm_forcing)


def windm_edsclrm_rhs(lhs_diff, xm, xm_tndcy, dt):
    """RHS of the um/vm tridiagonal solve (``windm_edsclrm_rhs``, implicit sfc flux).

    ``rhs[k] = 0.5·explicit_diffusion[k] + xm_tndcy[k] + xm[k]/dt`` (no explicit
    surface-flux term — it is implicit in the LHS).
    """
    invrs_dt = 1.0 / dt
    rhs_bot = (0.5 * (-lhs_diff[1, :, 0] * xm[:, 0] - lhs_diff[0, :, 0] * xm[:, 1])
               + xm_tndcy[:, 0] + invrs_dt * xm[:, 0])[:, None]
    rhs_int = (0.5 * (-lhs_diff[2, :, 1:-1] * xm[:, :-2]
                      - lhs_diff[1, :, 1:-1] * xm[:, 1:-1]
                      - lhs_diff[0, :, 1:-1] * xm[:, 2:])
               + xm_tndcy[:, 1:-1] + invrs_dt * xm[:, 1:-1])
    rhs_top = (0.5 * (-lhs_diff[2, :, -1] * xm[:, -2] - lhs_diff[1, :, -1] * xm[:, -1])
               + xm_tndcy[:, -1] + invrs_dt * xm[:, -1])[:, None]
    return jnp.concatenate([rhs_bot, rhs_int, rhs_top], axis=1)


def windm_edsclrm_lhs(lhs_diff, lhs_ma_zt, dt, invrs_rho_ds_zt, rho_ds_zm,
                      u_star_sqd, wind_speed, gr: CLUBBGrid):
    """Assemble the windm/edsclrm tridiagonal LHS (``windm_edsclrm_lhs``).

    CN diffusion (0.5·lhs_diff) + 1/dt accumulation + mean advection (interior
    only) + the implicit surface-momentum-flux term at the bottom level
    (``l_imp_sfc_momentum_flux = .true.``). ``k_lb_zt = k_lb_zm = 0`` (ascending).
    """
    lhs = 0.5 * lhs_diff
    lhs = lhs.at[1].add(1.0 / dt)
    lhs = lhs.at[:, :, :-1].add(lhs_ma_zt[:, :, :-1])
    sfc_term = (invrs_rho_ds_zt[:, 0] * gr.invrs_dzt[:, 0] * rho_ds_zm[:, 0]
                * (u_star_sqd / wind_speed[:, 0]))
    return lhs.at[1, :, 0].add(sfc_term)


# ---------------------------------------------------------------------------
# advance_windm_edsclrm
# ---------------------------------------------------------------------------

def advance_windm_edsclrm(um, vm, upwp, vpwp, wp2, up2, vp2, wm_zt, Kh_zm,
                          ug, vg, um_forcing, vm_forcing,
                          rho_ds_zm, rho_ds_zt, invrs_rho_ds_zt, fcor,
                          c_K10, nu10, dt, gr: CLUBBGrid, l_tke_aniso=True):
    """Advance um/vm (and the diagnostic upwp/vpwp) via eddy diffusion.

    Faithful port of ``advance_windm_edsclrm`` for the CAM ``l_predict_upwp_vpwp
    = .false.`` path (standalone, ascending grid, upwind MA, implicit surface
    momentum flux). Two Crank-Nicholson half-steps update the momentum fluxes
    around the implicit um/vm tridiagonal solve, then the fluxes are
    Cauchy-Schwarz clipped (``l_tke_aniso`` selects up2/vp2 vs wp2).

    Parameters (all ascending CLUBB grid)
    ----------
    um, vm, wm_zt, ug, vg, um_forcing, vm_forcing, rho_ds_zt, invrs_rho_ds_zt :
        zt-level fields ``(ngrdcol, nzt)``.
    upwp, vpwp, wp2, up2, vp2, Kh_zm, rho_ds_zm : zm-level fields ``(ngrdcol, nzm)``.
    fcor : Coriolis parameter ``(ngrdcol,)``.
    c_K10 : float
        Momentum-diffusivity coefficient (``CLUBBParams.c_K10``; ``Km = c_K10·Kh``).
    nu10 : float
        Background momentum diffusivity (``CLUBBParams.nu10``).
    dt : float
        Time step [s].
    l_tke_aniso : bool
        CAM default True → clip upwp/vpwp against up2/vp2 (else against wp2).

    Returns
    -------
    tuple of jax.Array
        ``(um_new, vm_new, upwp_new, vpwp_new)``.
    """
    nzm = gr.zm.shape[1]
    k_ub_zm = nzm - 1

    Km_zm = Kh_zm * c_K10
    Km_zm_p_nu10 = Km_zm + nu10

    nu10_arr = jnp.full((um.shape[0],), nu10, dtype=um.dtype)
    lhs_diff = diffusion_zt_lhs(Km_zm, nu10_arr, invrs_rho_ds_zt, rho_ds_zm, gr)
    lhs_ma_zt = term_ma_zt_lhs_upwind(wm_zt, gr)

    um_tndcy, vm_tndcy = compute_uv_tndcy(fcor, ug, vg, um, vm, um_forcing, vm_forcing)

    # sqrt(max(s^2, eps^2)) is forward-identical to the reference's
    # max(sqrt(s^2), eps) but AD-safe: it never differentiates through sqrt(0),
    # so calm-wind columns (um=vm=0, which feed the implicit sfc-flux LHS term)
    # keep finite gradients.
    wind_speed = jnp.sqrt(jnp.maximum(um ** 2 + vm ** 2, _EPS ** 2))
    u_star_sqd = safe_sqrt(upwp[:, 0] ** 2 + vpwp[:, 0] ** 2)

    # First Crank-Nicholson half (explicit) for upwp/vpwp.
    xpwp_u = calc_xpwp(Km_zm_p_nu10, um, gr.invrs_dzm)[:, 1:-1]
    upwp_new = upwp.at[:, 1:-1].set(-0.5 * xpwp_u).at[:, k_ub_zm].set(0.0)
    xpwp_v = calc_xpwp(Km_zm_p_nu10, vm, gr.invrs_dzm)[:, 1:-1]
    vpwp_new = vpwp.at[:, 1:-1].set(-0.5 * xpwp_v).at[:, k_ub_zm].set(0.0)

    lhs = windm_edsclrm_lhs(lhs_diff, lhs_ma_zt, dt, invrs_rho_ds_zt, rho_ds_zm,
                            u_star_sqd, wind_speed, gr)
    rhs_um = windm_edsclrm_rhs(lhs_diff, um, um_tndcy, dt)
    rhs_vm = windm_edsclrm_rhs(lhs_diff, vm, vm_tndcy, dt)
    um_new = tridiag_solve(lhs, rhs_um)
    vm_new = tridiag_solve(lhs, rhs_vm)

    # Second Crank-Nicholson half (implicit component) for upwp/vpwp.
    xpwp_u_new = calc_xpwp(Km_zm_p_nu10, um_new, gr.invrs_dzm)[:, 1:-1]
    xpwp_v_new = calc_xpwp(Km_zm_p_nu10, vm_new, gr.invrs_dzm)[:, 1:-1]
    upwp_new = upwp_new.at[:, 1:-1].add(-0.5 * xpwp_u_new)
    vpwp_new = vpwp_new.at[:, 1:-1].add(-0.5 * xpwp_v_new)

    xp2_u = up2 if l_tke_aniso else wp2
    xp2_v = vp2 if l_tke_aniso else wp2
    upwp_new = clip_covar(upwp_new, wp2, xp2_u)
    vpwp_new = clip_covar(vpwp_new, wp2, xp2_v)
    return um_new, vm_new, upwp_new, vpwp_new


# ---------------------------------------------------------------------------
# advance_xp2_xpyp term builders (scalar/horizontal-velocity variance equations)
# ---------------------------------------------------------------------------

def term_dp1_lhs(Cn, invrs_tau_zm):
    """Main-diagonal dissipation-term-1 coefficient for x_a'x_b' (``term_dp1_lhs``).

    Implicit ``+(C_n/tau_zm)·x_a'x_b'(t+1)`` — main diagonal only, interior
    levels; boundaries zero. ``Cn``/``invrs_tau_zm`` are ``(ngrdcol, nzm)``.
    """
    interior = Cn[:, 1:-1] * invrs_tau_zm[:, 1:-1]
    zeros_bnd = jnp.zeros((Cn.shape[0], 1), dtype=Cn.dtype)
    return jnp.concatenate([zeros_bnd, interior, zeros_bnd], axis=1)


def term_dp1_rhs(Cn, invrs_tau_zm, threshold):
    """Explicit dissipation-term-1 RHS for x'y' (``term_dp1_rhs``), all levels.

    The explicit part of ``-(C_n/tau_zm)·(x'y' - threshold)`` is
    ``+(C_n/tau_zm)·threshold``.
    """
    return Cn * invrs_tau_zm * threshold


def term_tp_rhs(xam, xbm, wpxap, wpxbp, invrs_dzm):
    """Turbulent production of x_a'x_b' (explicit) on interior zm levels (``term_tp_rhs``).

    ``rhs = -w'x_b'·d(x_am)/dz - w'x_a'·d(x_bm)/dz``. ``x_am``/``x_bm`` are zt
    (nzt); returns the interior slice ``(ngrdcol, nzm-2)``.
    """
    return (-wpxbp[:, 1:-1] * invrs_dzm[:, 1:-1] * (xam[:, 1:] - xam[:, :-1])
            - wpxap[:, 1:-1] * invrs_dzm[:, 1:-1] * (xbm[:, 1:] - xbm[:, :-1]))


def term_pr1(C4, C14, xbp2, wp2, invrs_tau_C4_zm, invrs_tau_C14_zm, w_tol_sqd):
    """Explicit pressure/dissipation term 1 for up2/vp2 (``term_pr1``), interior.

    ``rhs = (1/3)C4(xbp2+wp2)/tau_C4 - (1/3)C14(xbp2+wp2)/tau_C14
            + C14·w_tol²/tau_C14``; ``xbp2`` is the *other* horizontal variance.
    Returns the interior slice ``(ngrdcol, nzm-2)``.
    """
    return (_ONE_THIRD * C4 * (xbp2[:, 1:-1] + wp2[:, 1:-1]) * invrs_tau_C4_zm[:, 1:-1]
            - _ONE_THIRD * C14 * (xbp2[:, 1:-1] + wp2[:, 1:-1]) * invrs_tau_C14_zm[:, 1:-1]
            + C14 * invrs_tau_C14_zm[:, 1:-1] * w_tol_sqd)


def term_pr2(C_uu_shr, C_uu_buoy, thv_ds_zm, wpthvp, upwp, vpwp, um, vm, gr: CLUBBGrid):
    """Explicit pressure term 2 (PR2) for up2/vp2 (``term_pr2``), interior, floored ≥0.

    ``rhs = (2/3)[C_uu_buoy·(g/thv_ds)·w'thv' + C_uu_shr·(-u'w'·d(um)/dz
            - v'w'·d(vm)/dz)]`` clamped to 0. Uses ``constants.g``. Returns the
    interior slice ``(ngrdcol, nzm-2)``.
    """
    invrs_dzm = gr.invrs_dzm
    du_dz = invrs_dzm[:, 1:-1] * (um[:, 1:] - um[:, :-1])
    dv_dz = invrs_dzm[:, 1:-1] * (vm[:, 1:] - vm[:, :-1])
    pr2 = (2.0 / 3.0) * (
        C_uu_buoy * buoyancy_coefficient(thv_ds_zm[:, 1:-1]) * wpthvp[:, 1:-1]
        + C_uu_shr * (-upwp[:, 1:-1] * du_dz - vpwp[:, 1:-1] * dv_dz))
    return jnp.maximum(pr2, _ZERO_THRESHOLD)


# ---------------------------------------------------------------------------
# Turbulent advection of xp2/xpyp (CAM l_upwind_xpyp_ta = True; ascending grid)
# ---------------------------------------------------------------------------
# CAM uses the upwind (Godunov-style one-sided) turbulent-advection operator on
# the ascending grid (grid_dir = +1). The centered branch (needs weights_zm2zt)
# is out of the CAM-default tree and not ported.


def _xpyp_ta_pdf_lhs_upwind(coef_zm, sgn, rho_ds_zm, invrs_rho_ds_zm, gr: CLUBBGrid):
    """Upwind turbulent-advection LHS for xp2/xpyp (``xpyp_term_ta_pdf_lhs``).

    One-sided stencil keyed on ``sgn`` (grid_dir=+1): ``(3, ngrdcol, nzm)`` =
    ``[super, main, sub]``; boundaries zero.
    """
    invrs_dzt = gr.invrs_dzt
    irho = invrs_rho_ds_zm[:, 1:-1]
    s = sgn[:, 1:-1]
    rho_k, coef_k = rho_ds_zm[:, 1:-1], coef_zm[:, 1:-1]
    rho_km1, coef_km1 = rho_ds_zm[:, :-2], coef_zm[:, :-2]
    rho_kp1, coef_kp1 = rho_ds_zm[:, 2:], coef_zm[:, 2:]
    idzt_km1, idzt_k = invrs_dzt[:, :-1], invrs_dzt[:, 1:]
    zint = jnp.zeros_like(rho_k)

    sup_up = zint
    main_up = irho * idzt_km1 * rho_k * coef_k
    sub_up = -irho * idzt_km1 * rho_km1 * coef_km1
    sup_dn = irho * idzt_k * rho_kp1 * coef_kp1
    main_dn = -irho * idzt_k * rho_k * coef_k
    sub_dn = zint
    is_up = s > 0.0
    super_int = jnp.where(is_up, sup_up, sup_dn)
    main_int = jnp.where(is_up, main_up, main_dn)
    sub_int = jnp.where(is_up, sub_up, sub_dn)

    zb = jnp.zeros((rho_ds_zm.shape[0], 1), dtype=rho_ds_zm.dtype)
    return jnp.stack([jnp.concatenate([zb, super_int, zb], axis=1),
                      jnp.concatenate([zb, main_int, zb], axis=1),
                      jnp.concatenate([zb, sub_int, zb], axis=1)], axis=0)


def _xpyp_ta_pdf_rhs_upwind(term_zm, sgn, rho_ds_zm, invrs_rho_ds_zm, gr: CLUBBGrid):
    """Upwind turbulent-advection explicit RHS for xp2/xpyp (``xpyp_term_ta_pdf_rhs``).

    Returns ``(ngrdcol, nzm)``; boundaries zero.
    """
    invrs_dzt = gr.invrs_dzt
    irho = invrs_rho_ds_zm[:, 1:-1]
    s = sgn[:, 1:-1]
    rho_k, term_k = rho_ds_zm[:, 1:-1], term_zm[:, 1:-1]
    rho_km1, term_km1 = rho_ds_zm[:, :-2], term_zm[:, :-2]
    rho_kp1, term_kp1 = rho_ds_zm[:, 2:], term_zm[:, 2:]
    idzt_km1, idzt_k = invrs_dzt[:, :-1], invrs_dzt[:, 1:]

    rhs_up = -irho * idzt_km1 * (rho_k * term_k - rho_km1 * term_km1)
    rhs_dn = -irho * idzt_k * (rho_kp1 * term_kp1 - rho_k * term_k)
    rhs_int = jnp.where(s > 0.0, rhs_up, rhs_dn)
    zb = jnp.zeros((rho_ds_zm.shape[0], 1), dtype=rho_ds_zm.dtype)
    return jnp.concatenate([zb, rhs_int, zb], axis=1)


def calc_xp2_xpyp_ta_lhs(wp3_on_wp2, sigma_sqd_w, beta, rho_ds_zm, invrs_rho_ds_zm,
                         gr: CLUBBGrid):
    """Shared implicit turbulent-advection LHS for xp2/xpyp (ADG1 upwind path).

    The operator depends only on the w-PDF, so it is the SAME ``(3, ngrdcol,
    nzm)`` for all five moments. ``beta`` is a scalar/per-column param.
    """
    beta_c = jnp.asarray(beta)
    beta_c = beta_c[:, None] if beta_c.ndim == 1 else beta_c
    a1 = 1.0 / (1.0 - sigma_sqd_w)
    sgn = jnp.where(wp3_on_wp2 >= 0.0, 1.0, -1.0)
    coef_zm = _ONE_THIRD * beta_c * a1 * wp3_on_wp2
    return _xpyp_ta_pdf_lhs_upwind(coef_zm, sgn, rho_ds_zm, invrs_rho_ds_zm, gr)


def calc_xp2_xpyp_ta_rhs(wp3_on_wp2, sigma_sqd_w, wp2, beta, flux_a_zm, flux_b_zm,
                         rho_ds_zm, invrs_rho_ds_zm, gr: CLUBBGrid):
    """Turbulent-advection explicit RHS for one xp2/xpyp moment (ADG1 upwind).

    Explicit term ``wp_coef·<w'a'><w'b'>`` with ``wp_coef = (1 - beta/3)·a1²·
    wp3_on_wp2 / wp2``; variance uses ``flux_a = flux_b``, covariance uses the
    two distinct fluxes (e.g. rtpthlp: wprtp, wpthlp).
    """
    beta_c = jnp.asarray(beta)
    beta_c = beta_c[:, None] if beta_c.ndim == 1 else beta_c
    a1 = 1.0 / (1.0 - sigma_sqd_w)
    sgn = jnp.where(wp3_on_wp2 >= 0.0, 1.0, -1.0)
    wp_coef = (1.0 - _ONE_THIRD * beta_c) * a1 ** 2 * wp3_on_wp2 / wp2
    return _xpyp_ta_pdf_rhs_upwind(wp_coef * flux_a_zm * flux_b_zm, sgn,
                                   rho_ds_zm, invrs_rho_ds_zm, gr)


def xp2_xpyp_lhs(lhs_ta, lhs_ma, lhs_diff, lhs_dp1, dt, gamma=_GAMMA_OVER_IMPLICIT_TS):
    """Assemble the full xp2/xpyp tridiagonal LHS (``xp2_xpyp_lhs``).

    Interior: ``diff + ma + gamma·ta`` (+ ``lhs_dp1`` + ``1/dt`` on the main
    diagonal); boundaries are fixed-value BCs ``[0, 1, 0]``. ``lhs_dp1`` is the
    dissipation main-diagonal pre-scaled by the caller. ``(3, ngrdcol, nzm)``.
    """
    super_int = lhs_diff[0, :, 1:-1] + lhs_ma[0, :, 1:-1] + lhs_ta[0, :, 1:-1] * gamma
    main_int = (lhs_diff[1, :, 1:-1] + lhs_ma[1, :, 1:-1] + lhs_ta[1, :, 1:-1] * gamma
                + lhs_dp1[:, 1:-1] + 1.0 / dt)
    sub_int = lhs_diff[2, :, 1:-1] + lhs_ma[2, :, 1:-1] + lhs_ta[2, :, 1:-1] * gamma

    ng = lhs_ta.shape[1]
    zb = jnp.zeros((ng, 1), dtype=lhs_ta.dtype)
    ob = jnp.ones((ng, 1), dtype=lhs_ta.dtype)
    return jnp.stack([jnp.concatenate([zb, super_int, zb], axis=1),
                      jnp.concatenate([ob, main_int, ob], axis=1),
                      jnp.concatenate([zb, sub_int, zb], axis=1)], axis=0)


def xp2_xpyp_rhs(lhs_ta, rhs_ta, Cn, invrs_tau_zm, threshold, xapxbp, xam, xbm,
                 wpxap, wpxbp, invrs_dzm, xpyp_forcing, dt, gamma=_GAMMA_OVER_IMPLICIT_TS):
    """Explicit RHS of the x'^2 / x'y' equations (``xp2_xpyp_rhs``).

    Interior: ``rhs_ta + (1-gamma)·(over-implicit TA) + turbulent-production
    + Cn/tau·threshold + (1-gamma)·(over-implicit DP1) + forcing + xapxbp/dt``.
    BCs: lower carries the current value, upper is set to ``threshold``.
    ``(ngrdcol, nzm)``.
    """
    one_minus_gamma = 1.0 - gamma
    rhs_tp_int = term_tp_rhs(xam, xbm, wpxap, wpxbp, invrs_dzm)
    rhs_dp1_int = Cn[:, 1:-1] * invrs_tau_zm[:, 1:-1] * threshold
    lhs_dp1_int = Cn[:, 1:-1] * invrs_tau_zm[:, 1:-1]

    rhs_int = (rhs_ta[:, 1:-1]
               + one_minus_gamma * (-lhs_ta[0, :, 1:-1] * xapxbp[:, 2:]
                                    - lhs_ta[1, :, 1:-1] * xapxbp[:, 1:-1]
                                    - lhs_ta[2, :, 1:-1] * xapxbp[:, :-2])
               + rhs_tp_int + rhs_dp1_int
               + one_minus_gamma * (-lhs_dp1_int * xapxbp[:, 1:-1])
               + xpyp_forcing[:, 1:-1] + (1.0 / dt) * xapxbp[:, 1:-1])

    rhs_lb = xapxbp[:, 0:1]
    rhs_ub = jnp.full((Cn.shape[0], 1), threshold, dtype=Cn.dtype)
    return jnp.concatenate([rhs_lb, rhs_int, rhs_ub], axis=1)


def calc_xp2_xpyp_lhs(lhs_ta, lhs_ma, Kh_zt, c_K2, nu2, invrs_rho_ds_zm,
                      rho_ds_zt, Cn, invrs_tau_xp2_zm, gamma, dt, gr: CLUBBGrid):
    """Shared implicit LHS for rtp2/thlp2/rtpthlp (``calc_xp2_xpyp_lhs``).

    ``Kw2 = c_K2·Kh_zt`` eddy diffusion + the ``Cn`` pressure-damping (dp1) term,
    combined with the shared turbulent-advection (``lhs_ta``) and mean-advection
    (``lhs_ma``) operators. The SAME LHS solves all three second moments under
    ADG1. ``Cn``/``invrs_tau_xp2_zm`` are ``(ngrdcol, nzm)``; ``nu2`` is
    ``(ngrdcol,)``. Returns ``(lhs, lhs_diff, dp1)`` — ``lhs_diff``/``dp1`` are
    reused by the budget diagnostics.
    """
    Kw2 = c_K2 * Kh_zt
    lhs_diff = diffusion_zm_lhs(Kw2, nu2, invrs_rho_ds_zm, rho_ds_zt, gr)
    dp1 = term_dp1_lhs(Cn, invrs_tau_xp2_zm)
    lhs = xp2_xpyp_lhs(lhs_ta, lhs_ma, lhs_diff, dp1 * gamma, dt)
    return lhs, lhs_diff, dp1


def calc_up2_vp2_lhs(lhs_ta, lhs_ma, Kh_zt, c_K9, nu9, invrs_rho_ds_zm,
                     rho_ds_zt, C4, C14, invrs_tau_C4_zm, invrs_tau_C14_zm,
                     gamma, dt, gr: CLUBBGrid):
    """Shared implicit LHS for up2/vp2 (``calc_up2_vp2_lhs``).

    The shared TA/MA operators plus the up2/vp2-specific ``Kw9 = c_K9·Kh_zt``
    eddy diffusion and the ``C4``/``C14`` pressure-damping (dp1) terms scaled by
    ``gamma``. The same LHS solves both up2 and vp2 (ADG1). ``c_K9`` may be a
    scalar or per-column ``(ngrdcol,)``. Returns
    ``(lhs, lhs_diff, lhs_dp1_C4, lhs_dp1_C14)`` — the latter three are reused by
    the up2/vp2 RHS build and the dp2 budget diagnostic.
    """
    c_K9 = jnp.asarray(c_K9)
    c_K9 = c_K9[:, None] if c_K9.ndim == 1 else c_K9
    Kw9_zt = c_K9 * Kh_zt
    lhs_diff = diffusion_zm_lhs(Kw9_zt, nu9, invrs_rho_ds_zm, rho_ds_zt, gr)

    ng, nzm = invrs_tau_C4_zm.shape
    c4_1d = (2.0 / 3.0) * C4 * jnp.ones((ng, nzm), dtype=invrs_tau_C4_zm.dtype)
    c14_1d = _ONE_THIRD * C14 * jnp.ones((ng, nzm), dtype=invrs_tau_C14_zm.dtype)
    lhs_dp1_C4 = term_dp1_lhs(c4_1d, invrs_tau_C4_zm)
    lhs_dp1_C14 = term_dp1_lhs(c14_1d, invrs_tau_C14_zm)
    lhs_dp1 = (lhs_dp1_C4 + lhs_dp1_C14) * gamma
    lhs = xp2_xpyp_lhs(lhs_ta, lhs_ma, lhs_diff, lhs_dp1, dt)
    return lhs, lhs_diff, lhs_dp1_C4, lhs_dp1_C14


def xp2_xpyp_uv_rhs(rhs_ta_this, this_pre, other_pre, this_wp, this_dvel_dz,
                    lhs_splat, wp2, lhs_ta, C_uu_shr, C4, C14,
                    invrs_tau_C4_zm, invrs_tau_C14_zm, lhs_dp1_C4, lhs_dp1_C14,
                    pr2, omg, dt, w_tol_sqd, l_coriolis=False, fcor_y_col=None):
    """Explicit RHS for the up2 (or vp2) equation (``xp2_xpyp_uv_rhs``).

    The pressure-rotation (C_uu) form. Symmetric: for vp2 pass the v-quantities
    as ``this_*``/``this_wp``/``this_dvel_dz`` and the *pre-solve* up2 as
    ``other_pre`` (the C4/C14 isotropization couples the two horizontal
    variances). Interior terms: shared turbulent advection (over-implicit
    ``omg``), shear production ``(1-C_uu_shr)·(-2 u'w' d(vel)/dz)``, ``term_pr1``
    (C4/C14 isotropization), over-implicit dp1 damping, ``pr2`` (buoyancy/shear
    pressure), and ``xp2/dt``. The splat term ``½·lhs_splat·wp2`` is zero in the
    CAM default (``C_wp2_splat = 0``) but kept for faithfulness.

    ``l_coriolis`` (CAM default ``l_ho_nontrad_coriolis = .false.``) is a
    **compile-time static** feature gate (Python branch, not ``jnp.where``):
    when on, subtracts ``2·fcor_y·this_wp`` (``fcor_y_col`` is ``(ncol, 1)``).
    In normal use it is a fixed CAM-default value closed over by the
    enclosing ``jax.jit``; if this helper is jitted directly, pass it via
    ``static_argnums=(19,)`` / ``static_argnames=("l_coriolis",)``. BCs: lower
    row carries the current value, upper row is ``w_tol_sqd``.
    ``this_dvel_dz``/``pr2`` are interior slices ``(ncol, nzm-2)``. Returns
    ``(ncol, nzm)``.
    """
    ng = this_pre.shape[0]
    rhs_int = (
        rhs_ta_this[:, 1:-1]
        + 0.5 * lhs_splat[:, 1:-1] * wp2[:, 1:-1]
        + omg * (-lhs_ta[0, :, 1:-1] * this_pre[:, 2:]
                 - lhs_ta[1, :, 1:-1] * this_pre[:, 1:-1]
                 - lhs_ta[2, :, 1:-1] * this_pre[:, :-2])
        + (1.0 - C_uu_shr) * (-this_wp[:, 1:-1] * this_dvel_dz
                              - this_wp[:, 1:-1] * this_dvel_dz)
        + term_pr1(C4, C14, other_pre, wp2, invrs_tau_C4_zm, invrs_tau_C14_zm, w_tol_sqd)
        + omg * (-lhs_dp1_C4[:, 1:-1] - lhs_dp1_C14[:, 1:-1]) * this_pre[:, 1:-1]
        + pr2
        + (1.0 / dt) * this_pre[:, 1:-1])
    if l_coriolis:
        rhs_int = rhs_int - 2.0 * fcor_y_col * this_wp[:, 1:-1]

    rhs_lb = this_pre[:, 0:1]
    rhs_ub = jnp.full((ng, 1), w_tol_sqd, dtype=this_pre.dtype)
    return jnp.concatenate([rhs_lb, rhs_int, rhs_ub], axis=1)


def pos_definite_variances(field, rho_ds_zm, dzm, threshold, hf_lower, hf_upper,
                           fill_holes_type):
    """Mass-conserving hole-fill of one variance field (``pos_definite_variances``).

    Thin wrapper over :func:`fill_holes_vertical` (CAM default
    ``fill_holes_type = 2``): restores ``field >= threshold`` over the zm
    interior ``[hf_lower, hf_upper]`` while conserving ``sum(rho_ds·dz·field)``.
    ``hf_lower``/``hf_upper``/``fill_holes_type`` are **compile-time static**
    (closed over by the enclosing ``jax.jit``; if jitted directly, pass via
    ``static_argnums=(4, 5, 6)``).
    """
    return fill_holes_vertical(field, rho_ds_zm, dzm, threshold,
                               hf_lower, hf_upper, fill_holes_type)


def clip_variance(xp2, threshold_lo, threshold_hi=None):
    """Clamp a variance to ``[threshold_lo, threshold_hi]`` (``clip_variance``).

    Faithful port of ``clip_explicit.F90:clip_variance``: floors (and optionally
    caps) ``xp2`` over levels ``0 .. nzm-2`` — the bottom boundary is included,
    the top level ``nzm-1`` is left unchanged. ``threshold_lo`` may be a scalar
    or ``(ncol, nzm)`` array (the ``l_min_xp2_from_corr_wx`` boosted floor).
    """
    nzm = xp2.shape[1]
    mask = jnp.arange(nzm)[None, :] < (nzm - 1)
    out = jnp.where(mask, jnp.maximum(threshold_lo, xp2), xp2)
    if threshold_hi is not None:
        out = jnp.where(mask, jnp.minimum(threshold_hi, out), out)
    return out


def solve_xp2_xpyp(lhs_assembled, lhs_ta, rhs_ta, Cn, invrs_tau_zm, threshold,
                   xapxbp, xam, xbm, wpxap, wpxbp, invrs_dzm, xpyp_forcing, dt,
                   gamma=_GAMMA_OVER_IMPLICIT_TS):
    """Build the explicit RHS and tridiag-solve one xp2/xpyp moment (``solve_xp2_xpyp``).

    Combines :func:`xp2_xpyp_rhs` (turbulent advection + production + dissipation
    + forcing + over-implicit terms) with the pre-assembled shared LHS via the
    CLUBB-band Thomas solve. Returns the solution on zm levels ``(ncol, nzm)``.
    """
    rhs = xp2_xpyp_rhs(lhs_ta, rhs_ta, Cn, invrs_tau_zm, threshold, xapxbp, xam,
                       xbm, wpxap, wpxbp, invrs_dzm, xpyp_forcing, dt, gamma)
    return tridiag_solve(lhs_assembled, rhs)


def advance_xp2_xpyp(rtm, thlm, um, vm, rtp2, thlp2, rtpthlp, up2, vp2,
                     wprtp, wpthlp, wpthvp, upwp, vpwp,
                     wp2, wp2_zt, wp3_on_wp2, wp3_on_wp2_zt,
                     sigma_sqd_w, thv_ds_zm, Kh_zt,
                     invrs_tau_xp2_zm, invrs_tau_C4_zm, invrs_tau_C14_zm,
                     rho_ds_zm, rho_ds_zt, invrs_rho_ds_zm, wm_zm,
                     rtp2_forcing, thlp2_forcing, rtpthlp_forcing,
                     nu2, nu9, dt, gr: CLUBBGrid, config):
    """Advance the five second moments rtp2/thlp2/rtpthlp/up2/vp2 one step.

    Faithful port of the core (non-budget) path of
    ``advance_xp2_xpyp_module.F90:advance_xp2_xpyp`` for the CAM-default tree.
    Ties together the builders from section 2:

      * shared turbulent-advection LHS (:func:`calc_xp2_xpyp_ta_lhs`, ADG1
        upwind) and centered mean-advection LHS (:func:`term_ma_zm_lhs`);
      * rtp2/thlp2/rtpthlp: each solved with its OWN dissipation coefficient
        (``C2rt``/``C2thl``/``C2rtthl``) in the dp1 pressure-damping term, so
        each gets its own implicit LHS (:func:`calc_xp2_xpyp_lhs`) + solve
        (:func:`solve_xp2_xpyp`). CAM defaults ``C2rt = C2thl = 1.0`` but
        ``C2rtthl = 1.3`` differ, so the single shared-LHS solve (valid only when
        all three are equal, ``advance_xp2_xpyp_module.F90:836``) is NOT taken;
        ``l_C2_cloud_frac = .false.`` (CAM default) → the C2's are plain
        constants (F90:595, 625-627). Then :func:`pos_definite_variances`
        (hole-fill) + :func:`clip_variance` with the ``l_min_xp2_from_corr_wx``
        boosted floor on the variances, and :func:`clip_covar` (Cauchy-Schwarz)
        on rtpthlp;
      * up2/vp2: the up2/vp2 LHS (:func:`calc_up2_vp2_lhs`) with the
        pressure-rotation RHS (:func:`xp2_xpyp_uv_rhs`, with :func:`term_pr2`),
        solved with the shared LHS, then hole-fill + clip.

    CAM gating (static): ``l_lmm_stepping = False`` (no LMM blend),
    ``C_wp2_splat = 0`` (no splat), ``l_ho_nontrad_coriolis = False`` (no
    nontraditional Coriolis), ``l_min_xp2_from_corr_wx = True``,
    ``fill_holes_type = 2``. All means are on zt; all moments/fluxes/forcings on
    zm; ``Kh_zt`` on zt. The budget/stats diagnostics (``l_sample`` branch) are
    not part of the live tendency path and are omitted.

    Returns ``(rtp2, thlp2, rtpthlp, up2, vp2)`` on zm levels.
    """
    params = config.params
    beta = params.beta
    gamma = _GAMMA_OVER_IMPLICIT_TS
    rt_thr = config.rt_tol ** 2
    thl_thr = config.thl_tol ** 2
    w_tol_sqd = config.w_tol ** 2

    ng, nzm = wp2.shape
    invrs_dzm = gr.invrs_dzm
    hf_lower, hf_upper = 1, nzm - 2     # ascending grid: k_lb_zm+dir, k_ub_zm-dir

    # ---- shared operators (w-PDF only → same for all five moments) ----
    lhs_ma = term_ma_zm_lhs(wm_zm, gr)
    lhs_ta = calc_xp2_xpyp_ta_lhs(wp3_on_wp2, sigma_sqd_w, beta,
                                  rho_ds_zm, invrs_rho_ds_zm, gr)

    # ---- rtp2 / thlp2 / rtpthlp (shared LHS) ----
    rhs_ta_rtp2 = calc_xp2_xpyp_ta_rhs(wp3_on_wp2, sigma_sqd_w, wp2, beta,
                                       wprtp, wprtp, rho_ds_zm, invrs_rho_ds_zm, gr)
    rhs_ta_thlp2 = calc_xp2_xpyp_ta_rhs(wp3_on_wp2, sigma_sqd_w, wp2, beta,
                                        wpthlp, wpthlp, rho_ds_zm, invrs_rho_ds_zm, gr)
    rhs_ta_rtpthlp = calc_xp2_xpyp_ta_rhs(wp3_on_wp2, sigma_sqd_w, wp2, beta,
                                          wprtp, wpthlp, rho_ds_zm, invrs_rho_ds_zm, gr)

    # CAM uses 3 distinct dissipation coefficients (C2rt for rtp2, C2thl for
    # thlp2, C2rtthl for rtpthlp). l_C2_cloud_frac = .false. (CAM default) → plain
    # constants. Because C2rtthl (1.3) != C2rt = C2thl (1.0) the single shared-LHS
    # solve is invalid (advance_xp2_xpyp_module.F90:836), so each moment gets its
    # own dp1 term in BOTH the LHS assembly and the over-implicit solve correction.
    Cn_rt = jnp.full((ng, nzm), params.C2rt, dtype=wp2.dtype)
    Cn_thl = jnp.full((ng, nzm), params.C2thl, dtype=wp2.dtype)
    Cn_rtthl = jnp.full((ng, nzm), params.C2rtthl, dtype=wp2.dtype)

    def _scalar_lhs(Cn_x):
        lhs_x2, _lhs_diff, _dp1 = calc_xp2_xpyp_lhs(
            lhs_ta, lhs_ma, Kh_zt, params.c_K2, nu2, invrs_rho_ds_zm, rho_ds_zt,
            Cn_x, invrs_tau_xp2_zm, gamma, dt, gr)
        return lhs_x2

    soln_rtp2 = solve_xp2_xpyp(_scalar_lhs(Cn_rt), lhs_ta, rhs_ta_rtp2, Cn_rt,
                               invrs_tau_xp2_zm, rt_thr, rtp2, rtm, rtm, wprtp,
                               wprtp, invrs_dzm, rtp2_forcing, dt)
    soln_thlp2 = solve_xp2_xpyp(_scalar_lhs(Cn_thl), lhs_ta, rhs_ta_thlp2, Cn_thl,
                                invrs_tau_xp2_zm, thl_thr, thlp2, thlm, thlm,
                                wpthlp, wpthlp, invrs_dzm, thlp2_forcing, dt)
    soln_rtpthlp = solve_xp2_xpyp(_scalar_lhs(Cn_rtthl), lhs_ta, rhs_ta_rtpthlp,
                                  Cn_rtthl, invrs_tau_xp2_zm, _ZERO_THRESHOLD,
                                  rtpthlp, rtm, thlm, wprtp, wpthlp, invrs_dzm,
                                  rtpthlp_forcing, dt)

    rtp2_fh = pos_definite_variances(soln_rtp2, rho_ds_zm, gr.dzm, rt_thr,
                                     hf_lower, hf_upper, _CAM_FILL_HOLES_TYPE)
    thlp2_fh = pos_definite_variances(soln_thlp2, rho_ds_zm, gr.dzm, thl_thr,
                                      hf_lower, hf_upper, _CAM_FILL_HOLES_TYPE)

    # CAM l_min_xp2_from_corr_wx = True (fixed): variance floors from the
    # maximum-correlation bound.
    #
    # The denominator is floored at w_tol^2, which is what the reference gets
    # for free from its CALL ORDER: advance_wp2_wp3 floors wp2 at w_tol^2 and
    # runs BEFORE this solve in the Fortran, so the Fortran never divides by a
    # smaller wp2. This port runs the scalar-variance solve first, so on the
    # very first step it saw the seeded wp2 = tke_min = 1e-6 -- 400x below the
    # floor -- and turned a perfectly ordinary surface flux into an absurd
    # variance that nothing afterwards lowered.
    #
    # It stayed invisible for as long as the surface flux was zero, because the
    # numerator is wpthlp^2: 0/1e-6 is 0. The moment a prescribed-flux case
    # actually delivered its surface flux to the closure, the dry convective
    # column got thr_thlp2 = 0.0601^2 / (1e-6 * 0.99^2) = 3685 K^2 -- a 61 K RMS
    # temperature fluctuation -- which drove a CONSTANT spurious tendency and a
    # LINEAR temperature drift of 4.5 K per step. See #1508.
    #
    # The same ratio is already floored this way in nrmlzd_corr_wx (the
    # skewness helper), so this makes the two treatments agree. Note the
    # numerator is SQUARED, so the sign of the surface flux is irrelevant: a
    # stable, cooling case is hit exactly as hard as a convective one.
    max_corr2 = _MAX_MAG_CORRELATION_FLUX ** 2
    wp2_denom = jnp.maximum(wp2, w_tol_sqd) * max_corr2
    thr_thlp2 = jnp.maximum(thl_thr, wpthlp ** 2 / wp2_denom)
    thr_rtp2 = jnp.maximum(rt_thr, wprtp ** 2 / wp2_denom)
    thlp2_cv = clip_variance(thlp2_fh, thr_thlp2)
    rtp2_cv = clip_variance(rtp2_fh, thr_rtp2)

    # rtpthlp is a COVARIANCE (sign-indefinite): the reference applies neither
    # pos_definite_variances nor clip_variance to it (those force positivity,
    # valid only for variances) — only the Cauchy-Schwarz magnitude clip against
    # the post-clipped variances (advance_xp2_xpyp_module.F90:824). Interior
    # levels are bounded by |rtpthlp| <= 0.99·sqrt(rtp2·thlp2); the boundary
    # levels carry the solve's BC values (lower = prior value, upper = 0),
    # left unchanged by clip_covar, exactly as in the reference.
    rtpthlp_clip = clip_covar(soln_rtpthlp, rtp2_cv, thlp2_cv, _MAX_MAG_CORRELATION)

    # ---- up2 / vp2 (shared LHS; pressure-rotation RHS) ----
    lhs_uv, _lhs_diff_uv, lhs_dp1_C4, lhs_dp1_C14 = calc_up2_vp2_lhs(
        lhs_ta, lhs_ma, Kh_zt, params.c_K9, nu9, invrs_rho_ds_zm, rho_ds_zt,
        params.C4, params.C14, invrs_tau_C4_zm, invrs_tau_C14_zm, gamma, dt, gr)
    rhs_ta_up2 = calc_xp2_xpyp_ta_rhs(wp3_on_wp2, sigma_sqd_w, wp2, beta,
                                      upwp, upwp, rho_ds_zm, invrs_rho_ds_zm, gr)
    rhs_ta_vp2 = calc_xp2_xpyp_ta_rhs(wp3_on_wp2, sigma_sqd_w, wp2, beta,
                                      vpwp, vpwp, rho_ds_zm, invrs_rho_ds_zm, gr)
    pr2 = term_pr2(params.C_uu_shr, params.C_uu_buoy, thv_ds_zm, wpthvp,
                   upwp, vpwp, um, vm, gr)
    du_dz = invrs_dzm[:, 1:-1] * (um[:, 1:] - um[:, :-1])
    dv_dz = invrs_dzm[:, 1:-1] * (vm[:, 1:] - vm[:, :-1])
    lhs_splat = jnp.zeros((ng, nzm), dtype=wp2.dtype)   # C_wp2_splat = 0
    omg = 1.0 - gamma

    rhs_up2 = xp2_xpyp_uv_rhs(rhs_ta_up2, up2, vp2, upwp, du_dz, lhs_splat, wp2,
                              lhs_ta, params.C_uu_shr, params.C4, params.C14,
                              invrs_tau_C4_zm, invrs_tau_C14_zm, lhs_dp1_C4,
                              lhs_dp1_C14, pr2, omg, dt, w_tol_sqd, False, None)
    rhs_vp2 = xp2_xpyp_uv_rhs(rhs_ta_vp2, vp2, up2, vpwp, dv_dz, lhs_splat, wp2,
                              lhs_ta, params.C_uu_shr, params.C4, params.C14,
                              invrs_tau_C4_zm, invrs_tau_C14_zm, lhs_dp1_C4,
                              lhs_dp1_C14, pr2, omg, dt, w_tol_sqd, False, None)
    soln_up2 = tridiag_solve(lhs_uv, rhs_up2)
    soln_vp2 = tridiag_solve(lhs_uv, rhs_vp2)

    up2_fh = pos_definite_variances(soln_up2, rho_ds_zm, gr.dzm, w_tol_sqd,
                                    hf_lower, hf_upper, _CAM_FILL_HOLES_TYPE)
    vp2_fh = pos_definite_variances(soln_vp2, rho_ds_zm, gr.dzm, w_tol_sqd,
                                    hf_lower, hf_upper, _CAM_FILL_HOLES_TYPE)
    up2_cv = clip_variance(up2_fh, w_tol_sqd)
    vp2_cv = clip_variance(vp2_fh, w_tol_sqd)

    return rtp2_cv, thlp2_cv, rtpthlp_clip, up2_cv, vp2_cv


# ===========================================================================
# 14. Skewness-dependent C-coefficient family (CAM-default tree)
# ===========================================================================
# The xm/wpxp advance needs the pressure-term coefficients ``C6rt_Skw_fnc``,
# ``C6thl_Skw_fnc`` and ``C7_Skw_fnc``. For the CAM-default flags
# (``l_diag_Lscale_from_tau = .false.`` and ``l_use_C7_Richardson = .false.``)
# these are skewness functions of ``Skw_zm`` (NOT the ARM Richardson/constant
# that the CLUBB-JAX reference hard-wires), ported from the CLUBB Fortran
# (``advance_xm_wpxp_module.F90``):
#   * C6rt/C6thl: ``Cb + (C - Cb)*exp(-0.5(Skw/Cc)^2)`` (compute_skw_fnc) then
#     the Lscale-based stable-region damping (damp_coefficient);
#   * C7: the same skewness function, no damping (CAM default C7 = C7b reduces
#     it to the constant C7b).
# The C1/C11 skewness functions (wp2/wp3) are computed inside advance_wp2_wp3.
# The CAM branch has no CLUBB-JAX oracle (the reference is ARM) — pinned
# against independent CLUBB-Fortran transcriptions in
# tests/unit/test_clubb_cam_branch_fortran_pins.py (rel 1e-12).

def damp_coefficient(coefficient, Cx_Skw_fnc, max_coeff_value, altitude_threshold,
                     threshold, Lscale_zm, gr: CLUBBGrid):
    """Lscale-based damping of a skewness coefficient (``damp_coefficient``).

    In stably stratified regions (``Lscale_zm < threshold`` AND ``zm >
    altitude_threshold``) the coefficient is ramped linearly toward
    ``max_coeff_value`` as ``Lscale_zm -> 0``:
    ``max_coeff_value + (coefficient - max_coeff_value)/threshold · Lscale_zm``;
    elsewhere the input ``Cx_Skw_fnc`` is unchanged. ``coefficient``/
    ``max_coeff_value``/``altitude_threshold``/``threshold`` are ``(ncol,)``;
    ``Cx_Skw_fnc``/``Lscale_zm`` are ``(ncol, nzm)``.
    """
    thr = threshold[:, None]
    mx = max_coeff_value[:, None]
    cond = (Lscale_zm < thr) & (gr.zm > altitude_threshold[:, None])
    damped = mx + ((coefficient[:, None] - mx) / thr) * Lscale_zm
    return jnp.where(cond, damped, Cx_Skw_fnc)


def compute_C6_C7_Skw_fnc(Skw_zm, Lscale_zm, config, gr: CLUBBGrid):
    """The xm/wpxp ``C6rt``/``C6thl``/``C7`` skewness coefficients (CAM branch).

    ``C6rt``/``C6thl`` are skewness functions of ``Skw_zm`` then Lscale-damped;
    ``C7`` is the skewness function (no damping; CAM ``C7 = C7b`` → constant).
    All zm-level ``(ncol, nzm)``. Returns ``(C6rt_Skw_fnc, C6thl_Skw_fnc,
    C7_Skw_fnc)``.
    """
    p = config.params
    ng = Skw_zm.shape[0]

    def col(v):
        return jnp.full((ng,), v, dtype=Skw_zm.dtype)

    C6rt_raw = compute_skw_fnc(col(p.C6rt), col(p.C6rtb), col(p.C6rtc), Skw_zm)
    C6rt = damp_coefficient(col(p.C6rt), C6rt_raw, col(p.C6rt_Lscale0),
                            col(p.altitude_threshold), col(p.wpxp_L_thresh), Lscale_zm, gr)
    C6thl_raw = compute_skw_fnc(col(p.C6thl), col(p.C6thlb), col(p.C6thlc), Skw_zm)
    C6thl = damp_coefficient(col(p.C6thl), C6thl_raw, col(p.C6thl_Lscale0),
                             col(p.altitude_threshold), col(p.wpxp_L_thresh), Lscale_zm, gr)
    C7 = compute_skw_fnc(col(p.C7), col(p.C7b), col(p.C7c), Skw_zm)
    return C6rt, C6thl, C7


# ===========================================================================
# 15. Coupled wp2/wp3 advance (pentadiagonal)
# ===========================================================================
# The coupled wp2 (zm) / wp3 (zt) advance: interleaved band-matrix LHS
# builders + RHS terms solved with the verbatim-port pentadiagonal LU
# (penta_solve), CAM-default flags throughout (UPWIND wp3 mean advection —
# ``l_upwind_xm_ma = .true.`` —, ``l_damp_wp3_Skw_squared = .false.`` → C8b=0,
# ``l_tke_aniso = .true.``). Each builder is round-off parity-tested vs
# CLUBB-JAX; clip_skewness applies the Skw_max_mag clip after the solve.

_TWO_THIRDS = 2.0 / 3.0
_GAMMA = 1.5   # gamma_over_implicit_ts (constants_clubb); shared by the
               # wp2/wp3 and xm/wpxp advances (identical value in both)
_EPS = 1.0e-10  # constants_clubb eps = max(1e-10, machine eps): branch threshold floor

# clip_skewness (clip_explicit) algorithm constants (CLUBB; not tunable):
_WP3_MAX = 100.0           # absolute |wp3| limit [m^3/s^3] ("known magic number")
_SFC_AGL_THRESH_M = 100.0  # surface-layer threshold for the tighter skewness limit [m AGL]
_SFC_SKW_FACTOR = 0.0021   # surface-layer wp3_lim_sqd factor (clip_explicit.F90)




def clip_skewness(wp3, wp2_zt, zt, sfc_elevation, Skw_max_mag):
    """Limit ``|Sk_w| = |wp3|/wp2_zt^(3/2)`` (``clip_skewness``, CAM branch).

    Faithful port of the ``l_use_wp3_lim_with_smth_Heaviside = .false.`` branch
    of ``clip_explicit.F90:clip_skewness_core`` (the CAM default — the smooth
    Heaviside path is the conv-test variant): a sharp 100 m-AGL threshold caps
    ``wp3^2`` to ``Skw_max_mag^2·wp2_zt^3`` aloft and to
    ``0.0021·Skw_max_mag^2·wp2_zt^3`` in the surface layer, then clips
    ``|wp3| <= 100``. ``wp2_zt`` (``>= 0``) and ``wp3`` are zt-level
    ``(ncol, nzt)``; ``sfc_elevation``/``Skw_max_mag`` are ``(ncol,)``. Pure /
    JIT-safe / differentiable.
    """
    wp2_zt_cubed = wp2_zt ** 3
    zagl = zt - sfc_elevation[:, None]
    skw_sq = Skw_max_mag[:, None] ** 2
    wp3_lim_sqd = jnp.where(zagl <= _SFC_AGL_THRESH_M,
                            _SFC_SKW_FACTOR * skw_sq * wp2_zt_cubed,
                            skw_sq * wp2_zt_cubed)
    exceed = wp3 ** 2 > wp3_lim_sqd
    wp3 = jnp.where(exceed, jnp.sign(wp3) * safe_sqrt(wp3_lim_sqd), wp3)
    return jnp.clip(wp3, -_WP3_MAX, _WP3_MAX)


def weights_zt2zm(gr: CLUBBGrid):
    """zt->zm interpolation weights ``(ncol, nzm, 2)`` (``calc_zt2zm_weights``).

    Ascending grid (grid_dir = +1): for interior momentum level ``k`` the two
    columns are ``[T_ABOVE, T_BELOW]`` weighting ``zt[k-1]`` and ``zt[k]`` onto
    ``zm[k]``; the boundary levels (``k=0``, ``k=nzm-1``) use the reference's
    linear extension. On a uniform grid both interior weights are ``1/2``.
    """
    zm, zt = gr.zm, gr.zt
    denom = (zt[:, 1:] - zt[:, :-1]) + 1.0e-30        # (ncol, nzm-2), interior k=1..nzm-2
    t_above_int = (zm[:, 1:-1] - zt[:, :-1]) / denom
    t_below_int = (zt[:, 1:] - zm[:, 1:-1]) / denom

    denom0 = (zt[:, 1] - zt[:, 0]) + 1.0e-30
    t_above_0 = (zm[:, 0] - zt[:, 0]) / denom0
    t_below_0 = (zt[:, 1] - zm[:, 0]) / denom0

    denomn = (zt[:, -1] - zt[:, -2]) + 1.0e-30
    t_above_n = (zm[:, -1] - zt[:, -2]) / denomn
    t_below_n = (zt[:, -1] - zm[:, -1]) / denomn

    t_above = jnp.concatenate([t_above_0[:, None], t_above_int, t_above_n[:, None]], axis=1)
    t_below = jnp.concatenate([t_below_0[:, None], t_below_int, t_below_n[:, None]], axis=1)
    return jnp.stack([t_above, t_below], axis=-1)


def wp2_term_ta_lhs(invrs_rho_ds_zm, rho_ds_zt, gr: CLUBBGrid):
    """Turbulent-advection LHS for wp2 (``wp2_term_ta_lhs``), ``(2, ncol, nzm)``.

    The wp2 (zm) turbulent advection couples to wp3 (zt): band 0 is the
    coefficient of ``wp3[k]`` (super1 in the penta system), band 1 of
    ``wp3[k-1]`` (sub1). Boundaries zero.
    """
    fac = invrs_rho_ds_zm[:, 1:-1] * gr.invrs_dzm[:, 1:-1]
    lhs = jnp.zeros((2,) + invrs_rho_ds_zm.shape, dtype=invrs_rho_ds_zm.dtype)
    lhs = lhs.at[0, :, 1:-1].set(fac * rho_ds_zt[:, 1:])
    lhs = lhs.at[1, :, 1:-1].set(-fac * rho_ds_zt[:, :-1])
    return lhs


def wp3_term_ta_ADG1_lhs(wp2, a1_coef_zt, a3_coef_zt, wp3_on_wp2,
                         rho_ds_zm, invrs_rho_ds_zt, gr: CLUBBGrid):
    """5-band ADG1 turbulent-advection LHS for wp3 (``wp3_term_ta_ADG1_lhs``).

    The non-standard TA branch (CAM ``l_standard_term_ta = .false.``): wp3 (zt)
    couples to neighbouring wp3 (a1 coefficient) and wp2 (a3 coefficient) levels.
    Bands ``[super2(wp3[k+1]), super1(wp2[k+1]), main(wp3[k]), sub1(wp2[k]),
    sub2(wp3[k-1])]`` → ``(5, ncol, nzt)``; boundaries zero. Uses the inline
    zt->zm weights.
    """
    w_zt2zm = weights_zt2zm(gr)
    lhs = jnp.zeros((5,) + invrs_rho_ds_zt.shape, dtype=invrs_rho_ds_zt.dtype)

    inv = invrs_rho_ds_zt[:, 1:-1]
    a1 = a1_coef_zt[:, 1:-1]
    a3 = a3_coef_zt[:, 1:-1]
    idzt = gr.invrs_dzt[:, 1:-1]

    rho_up, rho_lo = rho_ds_zm[:, 2:-1], rho_ds_zm[:, 1:-2]
    wp2_up, wp2_lo = wp2[:, 2:-1], wp2[:, 1:-2]
    w3w2_up, w3w2_lo = wp3_on_wp2[:, 2:-1], wp3_on_wp2[:, 1:-2]
    wt_up_tab = w_zt2zm[:, 2:-1, 0]
    wt_up_tbe = w_zt2zm[:, 2:-1, 1]
    wt_lo_tab = w_zt2zm[:, 1:-2, 0]
    wt_lo_tbe = w_zt2zm[:, 1:-2, 1]

    lhs = lhs.at[0, :, 1:-1].set(inv * a1 * idzt * rho_up * w3w2_up * wt_up_tab)
    lhs = lhs.at[1, :, 1:-1].set(inv * a3 * idzt * rho_up * wp2_up)
    lhs = lhs.at[2, :, 1:-1].set(
        inv * a1 * idzt * (rho_up * w3w2_up * wt_up_tbe - rho_lo * w3w2_lo * wt_lo_tab))
    lhs = lhs.at[3, :, 1:-1].set(-inv * a3 * idzt * rho_lo * wp2_lo)
    lhs = lhs.at[4, :, 1:-1].set(-inv * a1 * idzt * rho_lo * w3w2_lo * wt_lo_tbe)
    return lhs


def wp3_term_tp_lhs(coef, wp2, rho_ds_zm, invrs_rho_ds_zt, gr: CLUBBGrid):
    """Turbulent-production LHS for wp3 (``wp3_term_tp_lhs``), ``(2, ncol, nzt)``.

    ``coef`` is ``(ncol,)`` per-column. Band 0 is the coefficient of
    ``wp2[k+1]``, band 1 of ``wp2[k]``. The main calls this twice (advection
    ``coef = 1`` and pressure ``coef = -C_wp3_pr_tp``). Boundaries zero.
    """
    lhs = jnp.zeros((2,) + invrs_rho_ds_zt.shape, dtype=invrs_rho_ds_zt.dtype)
    c = coef[:, None]
    inv = invrs_rho_ds_zt[:, 1:-1]
    idzt = gr.invrs_dzt[:, 1:-1]
    rho_up, wp2_up = rho_ds_zm[:, 2:-1], wp2[:, 2:-1]
    rho_lo, wp2_lo = rho_ds_zm[:, 1:-2], wp2[:, 1:-2]
    lhs = lhs.at[0, :, 1:-1].set(c * (-3.0 * inv * idzt * rho_up * wp2_up + 1.5 * idzt * wp2_up))
    lhs = lhs.at[1, :, 1:-1].set(c * (3.0 * inv * idzt * rho_lo * wp2_lo - 1.5 * idzt * wp2_lo))
    return lhs


def wp3_terms_ac_pr2_lhs(C11_Skw_fnc, wm_zm, gr: CLUBBGrid):
    """Accumulation + pressure-2 LHS for wp3 (``wp3_terms_ac_pr2_lhs``), ``(ncol, nzt)``.

    ``(1 - C11_Skw_fnc)·3·d(wm_zm)/dz`` at interior zt levels; boundaries zero.
    """
    lhs = jnp.zeros(C11_Skw_fnc.shape, dtype=C11_Skw_fnc.dtype)
    d_wm = wm_zm[:, 2:-1] - wm_zm[:, 1:-2]
    lhs = lhs.at[:, 1:-1].set(
        (1.0 - C11_Skw_fnc[:, 1:-1]) * 3.0 * gr.invrs_dzt[:, 1:-1] * d_wm)
    return lhs


def wp2_terms_ac_pr2_lhs(C_uu_shr, wm_zt, gr: CLUBBGrid):
    """Accumulation + pressure-2 LHS for wp2 (``wp2_terms_ac_pr2_lhs``), ``(ncol, nzm)``.

    ``(1 - C_uu_shr)·2·d(wm_zt)/dz`` at interior zm levels; ``C_uu_shr`` is
    ``(ncol,)``. Boundaries zero.
    """
    lhs = jnp.zeros((C_uu_shr.shape[0], gr.invrs_dzm.shape[1]), dtype=gr.invrs_dzm.dtype)
    d_wm = wm_zt[:, 1:] - wm_zt[:, :-1]
    lhs = lhs.at[:, 1:-1].set(
        (1.0 - C_uu_shr[:, None]) * 2.0 * gr.invrs_dzm[:, 1:-1] * d_wm)
    return lhs


def wp2_term_dp1_lhs(C1_Skw_fnc, invrs_tau_C1_zm):
    """Dissipation-1 LHS for wp2 (``wp2_term_dp1_lhs``), ``(ncol, nzm)``.

    ``C1_Skw_fnc·invrs_tau_C1_zm`` at interior zm levels; boundaries zero.
    """
    lhs = jnp.zeros_like(C1_Skw_fnc)
    lhs = lhs.at[:, 1:-1].set(C1_Skw_fnc[:, 1:-1] * invrs_tau_C1_zm[:, 1:-1])
    return lhs


def wp2_term_pr1_lhs(C4, invrs_tau_C4_zm):
    """Pressure-1 LHS for wp2 (``wp2_term_pr1_lhs``, CAM ``l_tke_aniso = True``).

    ``(2·C4·invrs_tau_C4_zm)/3`` at interior zm levels; ``C4`` is ``(ncol,)``.
    Boundaries zero. ``(ncol, nzm)``.
    """
    lhs = jnp.zeros_like(invrs_tau_C4_zm)
    lhs = lhs.at[:, 1:-1].set((2.0 * C4[:, None] * invrs_tau_C4_zm[:, 1:-1]) / 3.0)
    return lhs


def wp3_term_pr1_lhs(C8, C8b, invrs_tau_wp3_zt, Skw_zt):
    """Pressure-1 LHS for wp3 (``wp3_term_pr1_lhs``), ``(ncol, nzt)``.

    ``C8·invrs_tau_wp3_zt·(3·C8b·Skw_zt^2 + 1)`` at interior zt levels; ``C8``,
    ``C8b`` are ``(ncol,)``. In the CAM-default tree ``l_damp_wp3_Skw_squared =
    .false.`` so the orchestration passes ``C8b = 0`` and the factor reduces to
    ``C8·invrs_tau_wp3_zt``. Boundaries zero.
    """
    lhs = jnp.zeros_like(invrs_tau_wp3_zt)
    lhs = lhs.at[:, 1:-1].set(
        C8[:, None] * invrs_tau_wp3_zt[:, 1:-1]
        * (3.0 * C8b[:, None] * Skw_zt[:, 1:-1] ** 2 + 1.0))
    return lhs


# ---------------------------------------------------------------------------
# RHS term builders
# ---------------------------------------------------------------------------
# Most match CLUBB-JAX (CAM == ARM for these). Two diverge: the CLUBB-JAX port
# hardcodes the ARM defaults l_damp_wp2_using_em=True and
# l_use_tke_in_wp3_pr_turb_term=True, but CAM sets BOTH False — so for those the
# CAM-branch formula is taken from the CLUBB Fortran
# (CESM .../CLUBB_core/advance_wp2_wp3_module.F90), not from CLUBB-JAX (which
# only implements the True branch). Those two carry no JAX bit-exact oracle and
# are validated structurally + golden-pinned + against the Fortran by hand.


def wp2_term_pr_dfsn_rhs(C_wp2_pr_dfsn, rho_ds_zt, invrs_rho_ds_zm,
                         wpup2, wpvp2, wp3, gr: CLUBBGrid):
    """Pressure-diffusion RHS for wp2 (``wp2_term_pr_dfsn_rhs``), ``(ncol, nzm)``.

    ``C_wp2_pr_dfsn·d/dz[rho_ds·(w'u'^2 + w'v'^2 + w'^3)]/rho_ds`` at interior zm
    levels; the lower boundary copies the first interior value (per the
    reference). ``wpup2``/``wpvp2``/``wp3`` are zt-level; ``C_wp2_pr_dfsn`` is
    ``(ncol,)``.
    """
    wpuip2 = wpup2 + wpvp2 + wp3
    rhs = jnp.zeros_like(invrs_rho_ds_zm)
    fac = C_wp2_pr_dfsn[:, None] * invrs_rho_ds_zm[:, 1:-1] * gr.invrs_dzm[:, 1:-1]
    interior = fac * (rho_ds_zt[:, 1:] * wpuip2[:, 1:] - rho_ds_zt[:, :-1] * wpuip2[:, :-1])
    rhs = rhs.at[:, 1:-1].set(interior)
    rhs = rhs.at[:, 0].set(rhs[:, 1])
    return rhs


def wp3_term_pr_dfsn_rhs(C_wp3_pr_dfsn, rho_ds_zm, invrs_rho_ds_zt,
                         wp2up2, wp2vp2, wp4, up2, vp2, wp2, gr: CLUBBGrid):
    """Pressure-diffusion RHS for wp3 (``wp3_term_pr_dfsn_rhs``), ``(ncol, nzt)``.

    ``C_wp3_pr_dfsn·d/dz[rho_ds·net]/rho_ds`` where ``net = (w'^2 u'^2 + w'^2 v'^2
    + w'^4) - w'^2(u'^2 + v'^2 + w'^2)`` (all zm-level), at interior zt levels;
    boundaries zero. ``C_wp3_pr_dfsn`` is ``(ncol,)``.
    """
    net = (wp2up2 + wp2vp2 + wp4) - wp2 * (up2 + vp2 + wp2)
    rhs = jnp.zeros_like(invrs_rho_ds_zt)
    fac = C_wp3_pr_dfsn[:, None] * invrs_rho_ds_zt[:, 1:-1] * gr.invrs_dzt[:, 1:-1]
    val = rho_ds_zm[:, 2:-1] * net[:, 2:-1] - rho_ds_zm[:, 1:-2] * net[:, 1:-2]
    rhs = rhs.at[:, 1:-1].set(fac * val)
    return rhs


def wp2_terms_bp_pr2_rhs(C_uu_buoy, thv_ds_zm, wpthvp):
    """Buoyancy + pressure-2 RHS for wp2 (``wp2_terms_bp_pr2_rhs``), ``(ncol, nzm)``.

    ``(1 - C_uu_buoy)·2·(g/thv_ds)·w'thv'`` at interior zm levels; boundaries
    zero. ``C_uu_buoy`` is ``(ncol,)``; uses ``constants.g`` (CLUBB ``grav``).
    """
    rhs = jnp.zeros_like(thv_ds_zm)
    rhs = rhs.at[:, 1:-1].set(
        (1.0 - C_uu_buoy[:, None]) * 2.0
        * buoyancy_coefficient(thv_ds_zm[:, 1:-1]) * wpthvp[:, 1:-1])
    return rhs


def wp2_term_pr3_rhs(C_uu_shr, C_uu_buoy, thv_ds_zm, wpthvp, upwp, um, vpwp, vm,
                     gr: CLUBBGrid):
    """Pressure-3 RHS for wp2 (``wp2_term_pr3_rhs``), ``(ncol, nzm)``.

    ``(2/3)·[C_uu_buoy·(g/thv_ds)·w'thv' + C_uu_shr·(-u'w'·d(um)/dz
    - v'w'·d(vm)/dz)]`` clamped to ``>= 0`` at interior zm levels; boundaries
    zero. ``um``/``vm`` are zt-level; ``upwp``/``vpwp`` zm-level;
    ``C_uu_shr``/``C_uu_buoy`` are ``(ncol,)``; uses ``constants.g``.
    """
    rhs = jnp.zeros_like(thv_ds_zm)
    buoy = C_uu_buoy[:, None] * buoyancy_coefficient(thv_ds_zm[:, 1:-1]) * wpthvp[:, 1:-1]
    shear = C_uu_shr[:, None] * (
        -upwp[:, 1:-1] * gr.invrs_dzm[:, 1:-1] * (um[:, 1:] - um[:, :-1])
        - vpwp[:, 1:-1] * gr.invrs_dzm[:, 1:-1] * (vm[:, 1:] - vm[:, :-1]))
    val = jnp.maximum(_TWO_THIRDS * (buoy + shear), 0.0)
    rhs = rhs.at[:, 1:-1].set(val)
    return rhs


def wp2_term_pr1_rhs(C4, up2, vp2, invrs_tau_C4_zm):
    """Pressure-1 RHS for wp2 (``wp2_term_pr1_rhs``, CAM ``l_tke_aniso = True``).

    ``C4·(u'^2 + v'^2)·invrs_tau_C4_zm / 3`` at interior zm levels; boundaries
    zero. ``C4`` is ``(ncol,)``. ``(ncol, nzm)``.
    """
    rhs = jnp.zeros_like(invrs_tau_C4_zm)
    rhs = rhs.at[:, 1:-1].set(
        (C4[:, None] * (up2[:, 1:-1] + vp2[:, 1:-1]) * invrs_tau_C4_zm[:, 1:-1]) / 3.0)
    return rhs


def wp3_terms_bp1_pr2_rhs(C11_Skw_fnc, thv_ds_zt, wp2thvp):
    """Buoyancy + pressure-2 RHS for wp3 (``wp3_terms_bp1_pr2_rhs``), ``(ncol, nzt)``.

    ``(1 - C11_Skw_fnc)·3·(g/thv_ds)·w'^2 thv'`` at interior zt levels;
    boundaries zero. Uses ``constants.g``.
    """
    rhs = jnp.zeros_like(thv_ds_zt)
    rhs = rhs.at[:, 1:-1].set(
        (1.0 - C11_Skw_fnc[:, 1:-1]) * 3.0
        * buoyancy_coefficient(thv_ds_zt[:, 1:-1]) * wp2thvp[:, 1:-1])
    return rhs


def wp3_term_pr1_rhs(C8, C8b, invrs_tau_wp3_zt, Skw_zt, wp3):
    """Pressure-1 RHS for wp3 (``wp3_term_pr1_rhs``), ``(ncol, nzt)``.

    ``C8·invrs_tau_wp3_zt·(2·C8b·Skw_zt^2)·wp3`` at interior zt levels;
    boundaries zero. In the CAM-default tree ``l_damp_wp3_Skw_squared = .false.``
    so the orchestration passes ``C8b = 0`` and this term vanishes. ``C8``,
    ``C8b`` are ``(ncol,)``.
    """
    rhs = jnp.zeros_like(wp3)
    rhs = rhs.at[:, 1:-1].set(
        C8[:, None] * invrs_tau_wp3_zt[:, 1:-1]
        * (2.0 * C8b[:, None] * Skw_zt[:, 1:-1] ** 2) * wp3[:, 1:-1])
    return rhs


# --- CAM-branch builders taken from the CLUBB Fortran (CLUBB-JAX has only the
#     ARM/True branch of each) ------------------------------------------------

def wp2_term_dp1_rhs(C1_Skw_fnc, invrs_tau_C1_zm, threshold):
    """Dissipation-1 RHS for wp2 — CAM ``l_damp_wp2_using_em = .false.`` branch.

    From ``advance_wp2_wp3_module.F90:wp2_term_dp1_rhs`` (the ``.false.`` path):
    the wp2 dissipation damps ``w'^2`` only toward its floor ``threshold``
    (``w_tol^2``), so the explicit RHS is ``+(C1_Skw_fnc·invrs_tau)·threshold``
    at interior zm levels (boundaries zero). Note: in this branch
    ``C1_Skw_fnc`` carries NO ``1/3`` factor (the ``1/3`` is the ARM/True path).
    CLUBB-JAX implements only the True path (``-(C1_Skw_fnc·invrs_tau)·(u'^2 +
    v'^2)``), so the pin is a direct independent Fortran transcription
    (``tests/unit/test_clubb_cam_branch_fortran_pins.py``, rel 1e-12).
    """
    rhs = jnp.zeros_like(C1_Skw_fnc)
    rhs = rhs.at[:, 1:-1].set(
        (C1_Skw_fnc[:, 1:-1] * invrs_tau_C1_zm[:, 1:-1]) * threshold)
    return rhs


def wp3_term_pr_turb_rhs(C_wp3_pr_turb, Kh_zt, wpthvp, dum_dz, dvm_dz,
                         upwp, vpwp, thv_ds_zt, gr: CLUBBGrid):
    """Pressure-turbulence RHS for wp3 — CAM ``l_use_tke_in_wp3_pr_turb_term =
    .false.`` branch (the experimental shear term, CLUBB TRAC #411).

    From ``advance_wp2_wp3_module.F90:wp3_term_pr_turb_rhs`` (the ``.false.``
    path):

      ``-C_wp3_pr_turb·Kh_zt·d/dz{ (g/thv_ds)·Δw'thv'
          - Δ(u'w'·du/dz) - Δ(v'w'·dv/dz) }``

    at interior zt levels, where ``Δ`` is the zm-level difference bracketing the
    zt level. ``wpthvp``/``upwp``/``vpwp``/``dum_dz``/``dvm_dz`` are zm-level
    (``dum_dz = ddzt(um)``); ``Kh_zt``/``thv_ds_zt`` are zt-level;
    ``C_wp3_pr_turb`` is ``(ncol,)``; uses ``constants.g``. Boundaries zero.
    CLUBB-JAX implements only the TKE (True) path, so the pin is a direct
    independent Fortran transcription
    (``tests/unit/test_clubb_cam_branch_fortran_pins.py``, rel 1e-12).
    """
    rhs = jnp.zeros_like(Kh_zt)
    C = C_wp3_pr_turb[:, None]
    buoy = buoyancy_coefficient(thv_ds_zt[:, 1:-1]) * (wpthvp[:, 2:-1] - wpthvp[:, 1:-2])
    shr_u = upwp[:, 2:-1] * dum_dz[:, 2:-1] - upwp[:, 1:-2] * dum_dz[:, 1:-2]
    shr_v = vpwp[:, 2:-1] * dvm_dz[:, 2:-1] - vpwp[:, 1:-2] * dvm_dz[:, 1:-2]
    rhs = rhs.at[:, 1:-1].set(
        -C * Kh_zt[:, 1:-1] * gr.invrs_dzt[:, 1:-1] * (buoy - shr_u - shr_v))
    return rhs


def compute_a1_a3_coef(sigma_sqd_w, a3_coef_min, gr: CLUBBGrid):
    """ADG1 ``a1``/``a3`` coefficients on zm and zt levels (``advance_wp2_wp3`` pre-compute).

    ``a1 = 1/(1 - sigma_sqd_w)``; ``a3 = max(-2·(1 - sigma_sqd_w)^2 + 3,
    a3_coef_min)`` (zm-level), then interpolated to zt. ``sigma_sqd_w`` is
    ``(ncol, nzm)`` (``< 1``); ``a3_coef_min`` is ``(ncol,)``. Returns
    ``(a1_coef, a3_coef, a1_coef_zt, a3_coef_zt)``.
    """
    one_minus = 1.0 - sigma_sqd_w
    a1_coef = 1.0 / one_minus
    a3_coef = jnp.maximum(-2.0 * one_minus ** 2 + 3.0, a3_coef_min[:, None])
    return a1_coef, a3_coef, zm2zt(a1_coef, gr), zm2zt(a3_coef, gr)


def compute_skw_fnc(C, Cb, Cc, Skw):
    """Skewness-dependent CLUBB coefficient (``C1_Skw_fnc`` / ``C11_Skw_fnc``).

    ``Cb + (C - Cb)·exp(-½·(Skw/Cc)^2)`` where ``|C - Cb|`` exceeds the
    floor ``|C + Cb|·eps/2`` (else just ``Cb``). ``C``/``Cb``/``Cc`` are
    ``(ncol,)`` params; ``Skw`` is ``(ncol, nz)`` on the matching grid (zm for
    C1, zt for C11). NOTE: the CAM-default ``l_damp_wp2_using_em = .false.`` path
    does NOT apply the extra ``1/3`` factor to ``C1_Skw_fnc`` (that is the True
    path); the caller decides.
    """
    diff = jnp.abs(C - Cb)[:, None]
    thresh = (jnp.abs(C + Cb) * _EPS / 2.0)[:, None]
    smooth = Cb[:, None] + (C[:, None] - Cb[:, None]) * jnp.exp(-0.5 * (Skw / Cc[:, None]) ** 2)
    return jnp.where(diff > thresh, smooth, Cb[:, None] * jnp.ones_like(Skw))


# ---------------------------------------------------------------------------
# Pentadiagonal assembly + solve (interleaved wp2[2k] / wp3[2k+1])
# ---------------------------------------------------------------------------

def wp23_rhs(*, nzm, invrs_dt, rhs_pr_turb_wp3, rhs_pr_dfsn_wp3, rhs_pr_dfsn_wp2,
             rhs_pr1_wp2, rhs_bp1_pr2_wp3, rhs_pr1_wp3, rhs_bp_pr2_wp2,
             rhs_pr3_wp2, rhs_dp1_wp2, lhs_pr1_wp2, lhs_tp_wp3, lhs_pr1_wp3,
             lhs_dp1_wp2, lhs_ta_wp3, wp2, wp3, w_tol_sqd,
             l_ho_nontrad_coriolis=False, fcor_y=None, wp2up=None, upwp=None):
    """Assemble the coupled wp2/wp3 explicit RHS vector (``wp23_rhs``).

    Faithful port of ``advance_wp2_wp3_module.F90:wp23_rhs``. Interleaving: wp2[k]
    at global index 2k, wp3[k] at 2k+1 (ascending grid). Over-implicit terms use
    the ``(1 - gamma)·(-lhs·field)`` contributions; the four corner rows are the
    BCs (lower wp2 = current value, lower/upper wp3 = 0, upper wp2 = ``w_tol_sqd``).
    ``l_ho_nontrad_coriolis`` (CAM ``.false.``) is a static feature gate. Returns
    ``(ncol, 2*nzm-1)``.
    """
    ngrdcol = wp2.shape[0]
    ndim = 2 * nzm - 1
    rhs = jnp.zeros((ngrdcol, ndim), dtype=wp2.dtype)

    # wp3 interior (global 3,5,..): pressure-turb + pressure-dfsn
    rhs = rhs.at[:, 3:-2:2].set(rhs_pr_turb_wp3[:, 1:-1] + rhs_pr_dfsn_wp3[:, 1:-1])
    # wp2 interior (global 2,4,..): pressure-dfsn
    rhs = rhs.at[:, 2:-1:2].set(rhs_pr_dfsn_wp2[:, 1:-1])

    # l_tke_aniso=True: wp2 pr1 + over-implicit pr1
    rhs = rhs.at[:, 2:-1:2].add(rhs_pr1_wp2[:, 1:-1])
    rhs = rhs.at[:, 2:-1:2].add((1.0 - _GAMMA) * (-lhs_pr1_wp2[:, 1:-1] * wp2[:, 1:-1]))

    # wp3 time tendency + turbulent production (over-implicit) + buoyancy/pr2/pr1
    rhs = rhs.at[:, 3:-2:2].add(invrs_dt * wp3[:, 1:-1])
    rhs = rhs.at[:, 3:-2:2].add(
        (1.0 - _GAMMA) * (-lhs_tp_wp3[0, :, 1:-1] * wp2[:, 2:-1]
                          - lhs_tp_wp3[1, :, 1:-1] * wp2[:, 1:-2]))
    rhs = rhs.at[:, 3:-2:2].add(rhs_bp1_pr2_wp3[:, 1:-1])
    rhs = rhs.at[:, 3:-2:2].add(rhs_pr1_wp3[:, 1:-1])
    rhs = rhs.at[:, 3:-2:2].add((1.0 - _GAMMA) * (-lhs_pr1_wp3[:, 1:-1] * wp3[:, 1:-1]))

    # wp2 time tendency + buoyancy/pr2 + pr3 + dp1 (over-implicit)
    rhs = rhs.at[:, 2:-1:2].add(invrs_dt * wp2[:, 1:-1])
    rhs = rhs.at[:, 2:-1:2].add(rhs_bp_pr2_wp2[:, 1:-1])
    rhs = rhs.at[:, 2:-1:2].add(rhs_pr3_wp2[:, 1:-1])
    rhs = rhs.at[:, 2:-1:2].add(rhs_dp1_wp2[:, 1:-1])
    rhs = rhs.at[:, 2:-1:2].add((1.0 - _GAMMA) * (-lhs_dp1_wp2[:, 1:-1] * wp2[:, 1:-1]))

    # ADG1 TA for wp3 (over-implicit, 5 bands)
    rhs = rhs.at[:, 3:-2:2].add(
        (1.0 - _GAMMA) * (
            -lhs_ta_wp3[0, :, 1:-1] * wp3[:, 2:]
            - lhs_ta_wp3[1, :, 1:-1] * wp2[:, 2:-1]
            - lhs_ta_wp3[2, :, 1:-1] * wp3[:, 1:-1]
            - lhs_ta_wp3[3, :, 1:-1] * wp2[:, 1:-2]
            - lhs_ta_wp3[4, :, 1:-1] * wp3[:, :-2]))

    if l_ho_nontrad_coriolis:   # CAM default False (static gate)
        fy = jnp.asarray(fcor_y)
        fy = fy[:, None] if fy.ndim == 1 else fy
        rhs = rhs.at[:, 2:-1:2].add(2.0 * fy * upwp[:, 1:-1])
        rhs = rhs.at[:, 3:-2:2].add(3.0 * fy * jnp.asarray(wp2up)[:, 1:-1])

    # Boundary conditions (k_lb_zm = 0 on the ascending grid)
    rhs = rhs.at[:, 0].set(wp2[:, 0])
    rhs = rhs.at[:, 1].set(0.0)
    rhs = rhs.at[:, 2 * nzm - 3].set(0.0)
    rhs = rhs.at[:, 2 * nzm - 2].set(w_tol_sqd)
    return rhs


def wp23_lhs(*, nzm, ndim, invrs_dt, lhs_ma_zm, lhs_diff_zm, lhs_ta_wp2,
             lhs_ac_pr2_wp2, lhs_dp1_wp2, lhs_pr1_wp2, lhs_splat_wp2,
             lhs_ma_zt, lhs_diff_zt, lhs_tp_wp3, lhs_ac_pr2_wp3, lhs_pr1_wp3,
             lhs_splat_wp3, lhs_ta_wp3):
    """Assemble the coupled wp2/wp3 pentadiagonal LHS matrix (``wp23_lhs``).

    Faithful port of ``advance_wp2_wp3_module.F90:wp23_lhs``. Bands
    ``[super2, super1, main, sub1, sub2]``; wp2[k] at 2k, wp3[k] at 2k+1.
    Over-implicit implicit terms are scaled by gamma; identity rows pin the four
    BC corners. ``l_tke_aniso=True`` (wp2 pr1 on the main diagonal) and the splat
    terms (CAM ``C_wp2_splat=0`` → zero) are included. Returns ``(5, ncol, 2*nzm-1)``.
    """
    ngrdcol = lhs_dp1_wp2.shape[0]
    lhs = jnp.zeros((5, ngrdcol, ndim), dtype=lhs_dp1_wp2.dtype)

    # Lower BC identity (wp2 global 0, wp3 global 1)
    lhs = lhs.at[2, :, 0].set(1.0)
    lhs = lhs.at[2, :, 1].set(1.0)

    # wp2 interior rows
    lhs = lhs.at[0, :, 2:-1:2].set(lhs_ma_zm[0, :, 1:-1] + lhs_diff_zm[0, :, 1:-1])
    lhs = lhs.at[1, :, 2:-1:2].set(lhs_ta_wp2[0, :, 1:-1])
    lhs = lhs.at[2, :, 2:-1:2].set(
        lhs_ma_zm[1, :, 1:-1] + lhs_diff_zm[1, :, 1:-1] + lhs_ac_pr2_wp2[:, 1:-1]
        + _GAMMA * lhs_dp1_wp2[:, 1:-1] + invrs_dt)
    lhs = lhs.at[3, :, 2:-1:2].set(lhs_ta_wp2[1, :, 1:-1])
    lhs = lhs.at[4, :, 2:-1:2].set(lhs_ma_zm[2, :, 1:-1] + lhs_diff_zm[2, :, 1:-1])
    lhs = lhs.at[2, :, 2:-1:2].add(_GAMMA * lhs_pr1_wp2[:, 1:-1])   # l_tke_aniso=True
    lhs = lhs.at[2, :, 2:-1:2].add(lhs_splat_wp2[:, 1:-1])

    # wp3 interior rows
    lhs = lhs.at[0, :, 3:-2:2].set(lhs_ma_zt[0, :, 1:-1] + lhs_diff_zt[0, :, 1:-1])
    lhs = lhs.at[1, :, 3:-2:2].set(_GAMMA * lhs_tp_wp3[0, :, 1:-1])
    lhs = lhs.at[2, :, 3:-2:2].set(
        lhs_ma_zt[1, :, 1:-1] + lhs_diff_zt[1, :, 1:-1] + lhs_ac_pr2_wp3[:, 1:-1]
        + _GAMMA * lhs_pr1_wp3[:, 1:-1] + lhs_splat_wp3[:, 1:-1] + invrs_dt)
    lhs = lhs.at[3, :, 3:-2:2].set(_GAMMA * lhs_tp_wp3[1, :, 1:-1])
    lhs = lhs.at[4, :, 3:-2:2].set(lhs_ma_zt[2, :, 1:-1] + lhs_diff_zt[2, :, 1:-1])

    # ADG1 TA for wp3: add gamma*lhs_ta_wp3 to all 5 bands of the wp3 rows
    for b in range(5):
        lhs = lhs.at[b, :, 3:-2:2].add(_GAMMA * lhs_ta_wp3[b, :, 1:-1])

    # Upper BC identity (wp3 global 2*nzm-3, wp2 global 2*nzm-2)
    lhs = lhs.at[2, :, 2 * nzm - 3].set(1.0)
    lhs = lhs.at[2, :, 2 * nzm - 2].set(1.0)
    return lhs


def wp23_solve(lhs, rhs):
    """Pentadiagonal solve + de-interleave (``wp23_solve``).

    Solves the coupled system with the CLUBB-band penta LU
    (:func:`penta_solve`) and splits the solution: wp2 on even
    slots, wp3 on odd. Returns ``(wp2_new, wp3_new)``.
    """
    solution = penta_solve(lhs, rhs)
    return solution[:, 0::2], solution[:, 1::2]


def advance_wp2_wp3(wp2, wp3, up2, vp2, sigma_sqd_w, wp3_on_wp2,
                    wpup2, wpvp2, wp2up2, wp2vp2, wp4, wpthvp, wp2thvp,
                    um, vm, upwp, vpwp, wm_zm, wm_zt, Kh_zm, Kh_zt,
                    invrs_tau_C4_zm, invrs_tau_wp3_zt, invrs_tau_C1_zm,
                    Skw_zm, Skw_zt, rho_ds_zm, rho_ds_zt,
                    invrs_rho_ds_zm, invrs_rho_ds_zt, thv_ds_zm, thv_ds_zt,
                    sfc_elevation, dt, gr: CLUBBGrid, config):
    """Advance the coupled wp2 (zm) / wp3 (zt) second/third moments one step.

    Faithful port of the core (non-budget) path of
    ``advance_wp2_wp3_module.F90:advance_wp2_wp3`` for the CAM-default tree.
    Assembles the RHS/LHS term builders (section 3), the diffusion LHS
    (section 2) and the centered mean-advection operators, solves the
    interleaved pentadiagonal system (:func:`wp23_solve`), then applies the
    post-solve chain: ``fill_holes_vertical`` → ``fill_holes_wp2_from_horz_tke``
    (``l_wp2_fill_holes_tke``) → ``clip_variance`` (``l_min_wp2_from_corr_wx =
    False`` → the simple ``w_tol^2`` floor) → ``clip_skewness``.

    CAM gating (static / param): ``l_damp_wp2_using_em = False`` (the
    ``wp2_term_dp1_rhs`` floor form; ``C1_Skw_fnc`` carries no ``1/3``),
    ``l_damp_wp3_Skw_squared = False`` (``C8b = 0``),
    ``l_use_tke_in_wp3_pr_turb_term = False`` (the shear ``wp3_term_pr_turb_rhs``
    using ``dum/dvm_dz = ddzt(um/vm)``), ``l_tke_aniso = True``,
    ``C_wp2/wp3_splat = 0``, ``l_ho_nontrad_coriolis = False``,
    ``l_use_wp3_lim_with_smth_Heaviside = False``. Means on zt; moments/fluxes on
    the noted grids. Budget/stats (``l_sample``) are not part of the live path.

    Returns ``(wp2, wp3, wp2_zt)`` — the clipped wp2 (zm) and wp3 (zt) and the
    positive-definite wp2 interpolated to zt.
    """
    params = config.params
    ng, nzm = wp2.shape
    nzt = nzm - 1
    ndim = 2 * nzm - 1
    invrs_dt = 1.0 / dt
    w_tol_sqd = config.w_tol ** 2

    def col(v):
        return jnp.full((ng,), v, dtype=wp2.dtype)

    C4, C8, C8b = col(params.C4), col(params.C8), col(params.C8b)
    C_uu_shr, C_uu_buoy = col(params.C_uu_shr), col(params.C_uu_buoy)
    C_wp2_pr_dfsn = col(params.C_wp2_pr_dfsn)
    C_wp3_pr_tp = col(params.C_wp3_pr_tp)
    C_wp3_pr_turb = col(params.C_wp3_pr_turb)
    C_wp3_pr_dfsn = col(params.C_wp3_pr_dfsn)
    c_K1, c_K8 = col(params.c_K1), col(params.c_K8)
    nu1, nu8 = col(params.nu1), col(params.nu8)
    C12 = col(params.C12)
    a3_min, skw_max = col(params.a3_coef_min), col(params.Skw_max_mag)

    # Skewness-dependent coefficients (CAM l_damp_wp2_using_em=False → no 1/3 on C1).
    C1_Skw_fnc = compute_skw_fnc(col(params.C1), col(params.C1b), col(params.C1c), Skw_zm)
    C11_Skw_fnc = compute_skw_fnc(col(params.C11), col(params.C11b), col(params.C11c), Skw_zt)

    a1_coef, a3_coef, a1_coef_zt, a3_coef_zt = compute_a1_a3_coef(sigma_sqd_w, a3_min, gr)
    Kw1 = c_K1[:, None] * Kh_zt   # zt-level diffusivity for wp2
    Kw8 = c_K8[:, None] * Kh_zm   # zm-level diffusivity for wp3
    dum_dz = ddzt(um, gr)
    dvm_dz = ddzt(vm, gr)

    # ---- explicit RHS terms ----
    rhs_pr_turb_wp3 = wp3_term_pr_turb_rhs(C_wp3_pr_turb, Kh_zt, wpthvp, dum_dz,
                                           dvm_dz, upwp, vpwp, thv_ds_zt, gr)
    rhs_pr_dfsn_wp3 = wp3_term_pr_dfsn_rhs(C_wp3_pr_dfsn, rho_ds_zm, invrs_rho_ds_zt,
                                           wp2up2, wp2vp2, wp4, up2, vp2, wp2, gr)
    rhs_pr_dfsn_wp2 = wp2_term_pr_dfsn_rhs(C_wp2_pr_dfsn, rho_ds_zt, invrs_rho_ds_zm,
                                           wpup2, wpvp2, wp3, gr)
    rhs_bp_pr2_wp2 = wp2_terms_bp_pr2_rhs(C_uu_buoy, thv_ds_zm, wpthvp)
    rhs_dp1_wp2 = wp2_term_dp1_rhs(C1_Skw_fnc, invrs_tau_C1_zm, w_tol_sqd)
    rhs_pr3_wp2 = wp2_term_pr3_rhs(C_uu_shr, C_uu_buoy, thv_ds_zm, wpthvp,
                                   upwp, um, vpwp, vm, gr)
    rhs_pr1_wp2 = wp2_term_pr1_rhs(C4, up2, vp2, invrs_tau_C4_zm)
    rhs_bp1_pr2_wp3 = wp3_terms_bp1_pr2_rhs(C11_Skw_fnc, thv_ds_zt, wp2thvp)
    rhs_pr1_wp3 = wp3_term_pr1_rhs(C8, C8b, invrs_tau_wp3_zt, Skw_zt, wp3)

    # ---- LHS terms needed by wp23_rhs (over-implicit) ----
    lhs_diff_zm = diffusion_zm_lhs(Kw1, nu1, invrs_rho_ds_zm, rho_ds_zt, gr)
    lhs_diff_zt = diffusion_zt_lhs(Kw8, nu8, invrs_rho_ds_zt, rho_ds_zm, gr)
    lhs_tp_wp3 = (wp3_term_tp_lhs(jnp.ones((ng,), dtype=wp2.dtype), wp2, rho_ds_zm,
                                  invrs_rho_ds_zt, gr)
                  + wp3_term_tp_lhs(-C_wp3_pr_tp, wp2, rho_ds_zm, invrs_rho_ds_zt, gr))
    lhs_pr1_wp3 = wp3_term_pr1_lhs(C8, C8b, invrs_tau_wp3_zt, Skw_zt)
    lhs_dp1_wp2 = wp2_term_dp1_lhs(C1_Skw_fnc, invrs_tau_C1_zm)
    lhs_pr1_wp2 = wp2_term_pr1_lhs(C4, invrs_tau_C4_zm)
    lhs_ta_wp3 = wp3_term_ta_ADG1_lhs(wp2, a1_coef_zt, a3_coef_zt, wp3_on_wp2,
                                      rho_ds_zm, invrs_rho_ds_zt, gr)
    lhs_splat_wp2 = jnp.zeros((ng, nzm), dtype=wp2.dtype)   # C_wp2_splat = 0
    lhs_splat_wp3 = jnp.zeros((ng, nzt), dtype=wp2.dtype)   # C_wp3_splat = 0

    rhs = wp23_rhs(
        nzm=nzm, invrs_dt=invrs_dt, rhs_pr_turb_wp3=rhs_pr_turb_wp3,
        rhs_pr_dfsn_wp3=rhs_pr_dfsn_wp3, rhs_pr_dfsn_wp2=rhs_pr_dfsn_wp2,
        rhs_pr1_wp2=rhs_pr1_wp2, rhs_bp1_pr2_wp3=rhs_bp1_pr2_wp3,
        rhs_pr1_wp3=rhs_pr1_wp3, rhs_bp_pr2_wp2=rhs_bp_pr2_wp2,
        rhs_pr3_wp2=rhs_pr3_wp2, rhs_dp1_wp2=rhs_dp1_wp2, lhs_pr1_wp2=lhs_pr1_wp2,
        lhs_tp_wp3=lhs_tp_wp3, lhs_pr1_wp3=lhs_pr1_wp3, lhs_dp1_wp2=lhs_dp1_wp2,
        lhs_ta_wp3=lhs_ta_wp3, wp2=wp2, wp3=wp3, w_tol_sqd=w_tol_sqd)

    # C12 scales the wp3 diffusion LHS (after the RHS, before the LHS assembly).
    lhs_diff_zt = lhs_diff_zt * C12[None, :, None]

    # wp2 mean advection is centered; wp3 uses upwind (CAM l_upwind_xm_ma=True).
    lhs_ma_zm = term_ma_zm_lhs(wm_zm, gr)
    lhs_ma_zt = term_ma_zt_lhs_upwind(wm_zt, gr)
    lhs_ta_wp2 = wp2_term_ta_lhs(invrs_rho_ds_zm, rho_ds_zt, gr)
    lhs_ac_pr2_wp2 = wp2_terms_ac_pr2_lhs(C_uu_shr, wm_zt, gr)
    lhs_ac_pr2_wp3 = wp3_terms_ac_pr2_lhs(C11_Skw_fnc, wm_zm, gr)

    lhs = wp23_lhs(
        nzm=nzm, ndim=ndim, invrs_dt=invrs_dt, lhs_ma_zm=lhs_ma_zm,
        lhs_diff_zm=lhs_diff_zm, lhs_ta_wp2=lhs_ta_wp2, lhs_ac_pr2_wp2=lhs_ac_pr2_wp2,
        lhs_dp1_wp2=lhs_dp1_wp2, lhs_pr1_wp2=lhs_pr1_wp2, lhs_splat_wp2=lhs_splat_wp2,
        lhs_ma_zt=lhs_ma_zt, lhs_diff_zt=lhs_diff_zt, lhs_tp_wp3=lhs_tp_wp3,
        lhs_ac_pr2_wp3=lhs_ac_pr2_wp3, lhs_pr1_wp3=lhs_pr1_wp3,
        lhs_splat_wp3=lhs_splat_wp3, lhs_ta_wp3=lhs_ta_wp3)

    wp2_new, wp3_new = wp23_solve(lhs, rhs)

    # ---- post-solve fill_holes / clip ----
    wp2_c = fill_holes_vertical(wp2_new, rho_ds_zm, gr.dzm, w_tol_sqd,
                                1, nzm - 2, _CAM_FILL_HOLES_TYPE)
    # CAM l_wp2_fill_holes_tke = True (fixed): TKE-conserving wp2 fill.
    wp2_c, _, _ = fill_holes_wp2_from_horz_tke(wp2_c, up2, vp2, w_tol_sqd, 0, nzm - 3)
    # BOTH thresholds, as upstream does. Passing only the floor here was a
    # real defect: `advance_wp2_wp3_module.F90` calls `clip_variance` with the
    # optional `wp2_max` and says why -- "We attempt to clip extreme values of
    # wp2 to prevent a crash ... Chris Golaz found that instability caused by
    # large wp2 in CLUBB led unrealistic results in AM3" (dschanen, 11 Apr
    # 2011). That is exactly the failure measured here: on the two most
    # vigorously convective cases the prognostic path ran away, scoring 15.2
    # against 0.501 for the best closure on Wangara and carrying a parameter
    # gradient of 1.3e16 against ~0.3 on the well-behaved cases.
    #
    # Note this cap was already applied on the DIAGNOSTIC phase-1 path
    # (`clubb_turbulence`), and `clip_variance` already took the optional upper
    # threshold, and `CLUBBConfig.wp2_max` already held upstream's 1000 m^2/s^2
    # -- only the prognostic call site omitted it, which is the path the
    # campaign runs.
    wp2_c = clip_variance(wp2_c, w_tol_sqd, config.wp2_max)
    wp2_zt = jnp.maximum(zm2zt(wp2_c, gr), w_tol_sqd)
    wp3_c = clip_skewness(wp3_new, wp2_zt, gr.zt, sfc_elevation, skw_max)
    return wp2_c, wp3_c, wp2_zt


# ===========================================================================
# 16. Monotonic turbulent-flux limiter (MFL)
# ===========================================================================
# JAX port of CLUBB's monotonic flux limiter (mono_flux_limiter.F90) applied
# inside the xm/wpxp advance (CAM ``l_mono_flux_lim_{thlm,rtm,um,vm} =
# .true.`` + the spike fix): windowed min/max turbulent-advection range
# (masked fori_loop), lax.scan sequential clip, xm re-solve, top spike-fix.
# NOTE: the local ``_safe_sqrt`` here is the intentionally NaN-PROPAGATING
# variant (distinct from the AD-safe ``clubb_helpers.safe_sqrt`` — see the
# de-dup note in docs/dev-notes/clubb_port_history.md; consolidating would
# change behavior).

# Monotonic-flux-limiter field ids + their max-variance caps (constants_clubb).
MFL_RTM, MFL_THLM, MFL_UM, MFL_VM = "rtm", "thlm", "um", "vm"
_MAX_XP2 = {MFL_RTM: 5.0e-6, MFL_THLM: 5.0, MFL_UM: 10.0, MFL_VM: 10.0}

_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_F64_EPS = float(jnp.finfo(jnp.float64).eps)


def _safe_sqrt(x):
    """AD-safe ``sqrt`` with a finite gradient at ``x<=0``, propagating NaN.

    ``sqrt`` is never evaluated at ``<=0`` in the primal (finite VJP), giving 0
    where ``x==0``; a NaN input is passed through (rather than masked to 0) so a
    corrupted variance field surfaces instead of being silently clipped — matches
    the reference ``sqrt(NaN)=NaN`` propagation.
    """
    is_pos = x > 0.0
    safe = jnp.where(is_pos, x, 1.0)
    root = jnp.where(is_pos, jnp.sqrt(safe), 0.0)
    return jnp.where(jnp.isnan(x), x, root)


def calc_mean_w_up_down_component(w_i, varnce_w_i, wm):
    """Mean down/up vertical velocity of one PDF component (``calc_mean_w_up_down_component``).

    Returns ``(mean_w_down, mean_w_up)`` on the zm grid for the assumed-Gaussian
    component ``(w_i, varnce_w_i)``: the truncated-Gaussian means split at
    ``w = 0``, with three saturating branches (the component is too weak vs the
    grid velocity ``wm``, all-down, or all-up). Domain boundaries zeroed.
    Pure-jnp (erf/exp) → differentiable.
    """
    wi = w_i
    # AD-safe sqrt: sqrt is never evaluated at <=0 in the primal, so the VJP
    # stays finite (sqrt(0) has an infinite derivative). Forward-identical to
    # sqrt(max(varnce,0)) since both give 0 where varnce<=0.
    var_pos = varnce_w_i > 0.0
    sig_raw = jnp.sqrt(jnp.where(var_pos, varnce_w_i, 1.0))
    sig = jnp.where(var_pos, sig_raw, 0.0)
    sig_s = jnp.where(var_pos, sig_raw, 1.0)
    z = (0.0 - wi) / (_SQRT_2 * sig_s)
    ev = jnp.exp(-z ** 2)
    ef = jax.scipy.special.erf(z)
    too_weak = jnp.abs(wi) + 3.0 * sig <= wm
    all_dn = (~too_weak) & (wi + 3.0 * sig <= 0.0)
    all_up = (~too_weak) & (~all_dn) & (wi - 3.0 * sig >= 0.0)
    mwd_m = -sig / _SQRT_2PI * ev + wi * 0.5 * (1.0 + ef)
    mwu_m = sig / _SQRT_2PI * ev + wi * 0.5 * (1.0 - ef)
    mwd = jnp.where(too_weak, 0.0, jnp.where(all_dn, wi, jnp.where(all_up, 0.0, mwd_m)))
    mwu = jnp.where(too_weak, 0.0, jnp.where(all_dn, 0.0, jnp.where(all_up, wi, mwu_m)))
    mwd = mwd.at[:, 0].set(0.0).at[:, -1].set(0.0)
    mwu = mwu.at[:, 0].set(0.0).at[:, -1].set(0.0)
    return mwd, mwu


def mean_vert_vel_up_down(w_1, w_2, varnce_w_1, varnce_w_2, mixt_frac, wm):
    """Mixt-frac-weighted mean down/up vertical velocity (``mean_vert_vel_up_down``).

    Combines the two PDF components' :func:`calc_mean_w_up_down_component` results.
    Returns ``(mean_w_down, mean_w_up)`` on the zm grid.
    """
    mwd1, mwu1 = calc_mean_w_up_down_component(w_1, varnce_w_1, wm)
    mwd2, mwu2 = calc_mean_w_up_down_component(w_2, varnce_w_2, wm)
    mean_w_down = mixt_frac * mwd1 + (1.0 - mixt_frac) * mwd2
    mean_w_up = mixt_frac * mwu1 + (1.0 - mixt_frac) * mwu2
    return mean_w_down, mean_w_up


def calc_turb_adv_range(w_1_zm, w_2_zm, varnce_w_1_zm, varnce_w_2_zm,
                        mixt_frac_zm, gr: CLUBBGrid, dt):
    """Range of zt levels reachable by turbulent advection in one step (``calc_turb_adv_range``).

    Pure-JAX (masked ``lax.fori_loop``) reformulation of the host-numpy
    level-range search (``l_constant_thickness=False``, ascending grid): from
    each zt level ``k`` it walks down (using the mean **up** velocity ``vvu``) and
    up (using the mean **down** velocity ``vvd``) accumulating travel time until
    it exceeds ``dt`` or hits a velocity-sign barrier. Returns integer index
    arrays ``(low_lev_effect, high_lev_effect)``, each ``(ncol, nzt)``, that widen
    the limiter's allowable min/max window. The integer outputs are used only as
    masks downstream (no gradient flows through them). JIT-safe; the early-stops
    are replaced by a ``done`` mask over the fixed ``nzt`` loop bound.
    """
    ng, nzm = w_1_zm.shape
    nzt = nzm - 1
    gd = 1.0   # grid_dir, ascending
    dzm = gr.dzm
    wm = gd * dzm / dt
    vvd, vvu = mean_vert_vel_up_down(w_1_zm, w_2_zm, varnce_w_1_zm, varnce_w_2_zm,
                                     mixt_frac_zm, wm)

    def low_col(vvu_c, dzm_c):
        def low_at_k(k):
            def body(i, carry):
                da, done, low = carry
                j = k - 1 - i
                active = (j >= 0) & jnp.logical_not(done)
                ja = jnp.clip(j + 1, 1, nzm - 1)
                vu = vvu_c[ja]
                barrier = vu <= 0.0
                contrib = gd * dzm_c[ja] / jnp.where(vu > 0.0, vu, 1.0)
                new_da = jnp.where(active & jnp.logical_not(barrier), da + contrib, da)
                time_stop = active & jnp.logical_not(barrier) & (new_da >= dt)
                low_j = jnp.where(active, j, low)
                new_low = jnp.where(active & barrier, jnp.minimum(j + 1, nzt - 1), low_j)
                return (new_da, done | (active & (barrier | time_stop)), new_low)
            _, _, low = jax.lax.fori_loop(0, nzt, body, (0.0, False, 0))
            return low
        return jax.vmap(low_at_k)(jnp.arange(nzt))

    def high_col(vvd_c, dzm_c):
        def high_at_k(k):
            def body(i, carry):
                da, done, high = carry
                j = k + 1 + i
                active = (j <= nzt - 1) & jnp.logical_not(done)
                ja = jnp.clip(j, 0, nzm - 1)
                vd = vvd_c[ja]
                barrier = vd >= 0.0
                contrib = -gd * dzm_c[ja] / jnp.where(vd < 0.0, vd, -1.0)
                new_da = jnp.where(active & jnp.logical_not(barrier), da + contrib, da)
                time_stop = active & jnp.logical_not(barrier) & (new_da >= dt)
                high_j = jnp.where(active, j, high)
                new_high = jnp.where(active & barrier, jnp.maximum(j - 1, 0), high_j)
                return (new_da, done | (active & (barrier | time_stop)), new_high)
            _, _, high = jax.lax.fori_loop(0, nzt, body, (0.0, False, k))
            return high
        return jax.vmap(high_at_k)(jnp.arange(nzt))

    low = jax.vmap(low_col)(vvu, dzm)
    high = jax.vmap(high_col)(vvd, dzm)

    # Boundary levels are set explicitly (the reference does not search them).
    k = jnp.arange(nzt)[None, :]
    low = jnp.where(k == 0, 0, low)
    low = jnp.where(k == nzt - 2, nzt - 2, low)
    low = jnp.where(k == nzt - 1, nzt - 1, low)
    high = jnp.where(k == 0, 0, high)
    high = jnp.where(k == nzt - 2, nzt - 1, high)
    high = jnp.where(k == nzt - 1, nzt - 1, high)
    return low, high


def mfl_xm_lhs(wm_zt, invrs_dt, gr: CLUBBGrid):
    """LHS of the MFL xm re-solve tridiagonal system (``mfl_xm_lhs``).

    The re-solve advances ``xm`` alone (the flux ``w'x'`` is now known), so it is
    a plain tridiagonal: the upwind mean-advection operator (CAM
    ``l_upwind_xm_ma = True``) plus ``1/dt`` on the main diagonal.
    ``(3, ncol, nzt)``.
    """
    return term_ma_zt_lhs_upwind(wm_zt, gr).at[1, :, :].add(invrs_dt)


def mfl_xm_rhs(xm_old, wpxp, xm_forcing, invrs_dt, invrs_rho_ds_zt, invrs_dzt, rho_ds_zm):
    """RHS of the MFL xm re-solve (``mfl_xm_rhs``), ``(ncol, nzt)``.

    ``xm_old/dt + xm_forcing - (1/rho_ds_zt)·d(rho_ds_zm·w'x')/dz`` with the
    limited flux ``wpxp``.
    """
    return (xm_old * invrs_dt + xm_forcing
            - invrs_rho_ds_zt * invrs_dzt
            * (rho_ds_zm[:, 1:] * wpxp[:, 1:] - rho_ds_zm[:, :-1] * wpxp[:, :-1]))


def mfl_xm_solve(lhs, rhs):
    """Solve the MFL xm re-solve tridiagonal system (``mfl_xm_solve``).

    Implicit tridiagonal re-solve (``l_mfl_xm_imp_adj = True``) via the CLUBB-band
    Thomas solver (:func:`tridiag_solve`).
    """
    return tridiag_solve(lhs, rhs)


def monotonic_turbulent_flux_limit(
    solve_type, xm, wpxp, xm_old, xp2, wm_zt, xm_forcing,
    rho_ds_zm, rho_ds_zt, invrs_rho_ds_zm, invrs_rho_ds_zt,
    xp2_threshold, xm_tol, low_lev_effect, high_lev_effect,
    gr: CLUBBGrid, dt, l_mono_flux_lim_spikefix=True,
):
    """Monotonic turbulent-flux limiter core (``monotonic_turbulent_flux_limit``).

    Pure-JAX reformulation of the host-numpy ``_monotonic_turbulent_flux_limit``:
    bounds the turbulent flux ``wpxp`` so the mean field ``xm`` stays within
    ``±max(2·sqrt(x'^2), xm_tol)`` of its no-flux value over the
    turbulent-advection-reachable level window ``[low_lev_effect,
    high_lev_effect]`` (the masked min/max), then re-advances ``xm`` implicitly
    with the limited flux. The sequential per-level flux clip (each level's bound
    uses the already-clipped level below) is a :func:`lax.scan`; the level window
    is a mask; the top spike-fix conserves the column. ``solve_type`` (static MFL
    id) selects the variance cap, the ``rtm`` spike-fix, and the wind (uv)
    non-negativity skip. Returns ``(xm, wpxp)``. Differentiable in the field
    inputs (the integer level bounds are stop-gradient masks).
    """
    ng, nzt = xm.shape
    nzm = nzt + 1
    is_uv = solve_type in (MFL_UM, MFL_VM)
    max_xp2 = _MAX_XP2[solve_type]
    spikefix_rtm = bool(l_mono_flux_lim_spikefix) and solve_type == MFL_RTM
    invrs_dt = 1.0 / dt
    gd = 1.0   # grid_dir, ascending
    dzt = gr.dzt
    invrs_dzt = gr.invrs_dzt

    xm_enter = xm

    xp2_zt = jnp.clip(zm2zt(xp2, gr), xp2_threshold, max_xp2)
    max_dev = jnp.maximum(2.0 * _safe_sqrt(xp2_zt), xm_tol)
    xm_without_ta = xm_old + dt * xm_forcing
    min_x_lev = xm_without_ta - max_dev
    if not is_uv:
        min_x_lev = jnp.maximum(min_x_lev, 0.0)
    max_x_lev = xm_without_ta + max_dev

    # Windowed min/max over the reachable level window [low, high] (masked).
    j = jnp.arange(nzt)[None, None, :]
    in_win = (j >= low_lev_effect[:, :, None]) & (j <= high_lev_effect[:, :, None])
    min_x_allowable = jnp.min(jnp.where(in_win, min_x_lev[:, None, :], jnp.inf), axis=2)
    max_x_allowable = jnp.max(jnp.where(in_win, max_x_lev[:, None, :], -jnp.inf), axis=2)

    thr_term_zt = invrs_dt * gd * dzt * (xm_without_ta - min_x_allowable)
    mfl_max_term_zt = rho_ds_zt * thr_term_zt
    mfl_min_term_zt = rho_ds_zt * invrs_dt * gd * dzt * (xm_without_ta - max_x_allowable)
    thr_term_zm = zt2zm(thr_term_zt, gr)   # (ng, nzm)

    # Sequential clip of wpxp over interior zm levels k=1..nzm-2 (scan over levels).
    # step s -> k=s+1, k_zt=s, k-1=s. Each step's bound uses the clipped k-1 flux.
    def step(wp_prev, xs):
        max_term, min_term, thr_km1, irho_k, rho_km1, wp_k = xs
        spikefix_cond = (spikefix_rtm & (jnp.abs(wp_prev) > thr_km1) & (wp_prev < 0.0))
        mfl_max = jnp.where(spikefix_cond, 0.0,
                            irho_k * (max_term + rho_km1 * wp_prev))
        mfl_min = irho_k * (min_term + rho_km1 * wp_prev)
        clipped = jnp.where(wp_k > mfl_max, mfl_max,
                            jnp.where(wp_k < mfl_min, mfl_min, wp_k))
        needed = jnp.abs(clipped - wp_k) > _F64_EPS
        return clipped, (clipped, needed)

    xs = (mfl_max_term_zt[:, :nzm - 2].T, mfl_min_term_zt[:, :nzm - 2].T,
          thr_term_zm[:, :nzm - 2].T, invrs_rho_ds_zm[:, 1:nzm - 1].T,
          rho_ds_zm[:, :nzm - 2].T, wpxp[:, 1:nzm - 1].T)
    _, (clipped_T, needed_T) = jax.lax.scan(step, wpxp[:, 0], xs)
    clipped_interior = clipped_T.T          # (ng, nzm-2)
    wpxp_new = jnp.concatenate([wpxp[:, :1], clipped_interior, wpxp[:, -1:]], axis=1)
    adj_needed = jnp.any(needed_T.T, axis=1)   # (ng,)

    # Re-solve xm implicitly with the limited flux (applied where adjustment fired).
    lhs = mfl_xm_lhs(wm_zt, invrs_dt, gr)
    rhs = mfl_xm_rhs(xm_old, wpxp_new, xm_forcing, invrs_dt, invrs_rho_ds_zt,
                     invrs_dzt, rho_ds_zm)
    xm_mfl = mfl_xm_solve(lhs, rhs)
    xm = jnp.where(adj_needed[:, None], xm_mfl, xm)

    # Top spike-fix: conserve column xm if the top level moved a lot.
    dz_top = gr.zm[:, -1] - gr.zm[:, -2]
    moved = jnp.abs(xm[:, -1] - xm_enter[:, -1]) > 10.0 * xm_tol
    xm_dw = rho_ds_zt[:, -1] * (xm[:, -1] - xm_enter[:, -1]) * dz_top
    k_idx = jnp.arange(nzt)[None, :]
    below_top = k_idx < (nzt - 1)
    xm_vint = jnp.sum(jnp.where(below_top, rho_ds_zt * xm * gd * dzt, 0.0), axis=1)
    small_vint = jnp.abs(xm_vint) < _F64_EPS
    coef = jnp.maximum(xm_dw / jnp.where(small_vint, 1.0, xm_vint), -0.99)
    xm_scaled = (xm * (1.0 + coef[:, None])).at[:, -1].set(xm_enter[:, -1])
    xm_smallv = xm.at[:, -1].set(xm_enter[:, -1])
    xm_fixed = jnp.where(small_vint[:, None], xm_smallv, xm_scaled)
    xm = jnp.where(moved[:, None], xm_fixed, xm)
    return xm, wpxp_new


# ===========================================================================
# 17. Coupled xm/wpxp advance (means + scalar fluxes)
# ===========================================================================
# The coupled xm (zt) / wpxp (zm) advance for rtm/wprtp and thlm/wpthlp
# (CAM ``l_predict_upwp_vpwp = .false.`` → winds go through
# advance_windm_edsclrm instead). Semi-implicit single-LHS solve, CENTERED
# turbulent advection for wpxp, monotonic turbulent-flux limiter (section 5)
# + hole filling on the means, and the Cauchy-Schwarz clipping family.

# Monotonic-flux-limiter xm tolerances (constants_clubb):
_RT_TOL_MFL = 1.0e-4    # [kg/kg]
_THL_TOL_MFL = 0.2      # [K]
# Relaxed-clipping variance floors (advance_xm_wpxp; only if l_enable_relaxed_clipping):


def _weights_zm2zt(gr: CLUBBGrid):
    """zm->zt interpolation weights ``(ncol, nzt, 2)`` = ``[m_above, m_below]``
    (``calc_zm2zt_weights``, ascending grid), computed inline from the grid."""
    total = (gr.zm[:, 1:] - gr.zm[:, :-1]) + 1.0e-30
    w_above = (gr.zt - gr.zm[:, :-1]) / total
    w_below = (gr.zm[:, 1:] - gr.zt) / total
    return jnp.stack([w_above, w_below], axis=-1)


def xpyp_term_ta_pdf_lhs_centered(coef_zt, rho_ds_zt, invrs_rho_ds_zm, gr: CLUBBGrid):
    """Centered ADG1 turbulent-advection LHS for w'x' (``xpyp_term_ta_pdf_lhs``).

    The CAM-default wpxp turbulent advection is implicit and centered
    (``l_explicit_turbulent_adv_wpxp = .false.``, ``l_godunov_upwind_wpxp_ta =
    .false.``) — distinct from the *upwind* operator used by xp2/xpyp
    (``l_upwind_xpyp_ta = .true.``). Discretizes
    ``(1/rho_ds_zm)·d(rho_ds_zt·coef·var_zm)/dz`` at interior zm levels using the
    inline zm->zt weights. ``coef_zt`` is ``(ncol, nzt)``. ``(3, ncol, nzm)`` =
    ``[super, main, sub]``; boundaries zero.
    """
    w2zt = _weights_zm2zt(gr)
    fac = invrs_rho_ds_zm[:, 1:-1] * gr.invrs_dzm[:, 1:-1]
    rho_coef_k = rho_ds_zt[:, 1:] * coef_zt[:, 1:]
    rho_coef_km1 = rho_ds_zt[:, :-1] * coef_zt[:, :-1]
    super_int = fac * rho_coef_k * w2zt[:, 1:, 0]
    main_int = fac * (rho_coef_k * w2zt[:, 1:, 1] - rho_coef_km1 * w2zt[:, :-1, 0])
    sub_int = -fac * rho_coef_km1 * w2zt[:, :-1, 1]
    zb = jnp.zeros((coef_zt.shape[0], 1), dtype=coef_zt.dtype)
    return jnp.stack([jnp.concatenate([zb, super_int, zb], axis=1),
                      jnp.concatenate([zb, main_int, zb], axis=1),
                      jnp.concatenate([zb, sub_int, zb], axis=1)], axis=0)


def calc_xm_wpxp_ta_terms(sigma_sqd_w, wp3_on_wp2_zt, rho_ds_zt, invrs_rho_ds_zm,
                          gr: CLUBBGrid):
    """ADG1 turbulent-advection LHS for w'x' (``calc_xm_wpxp_ta_terms``), ``(3, ncol, nzm)``.

    ``coef = a1_coef_zt·wp3_on_wp2_zt`` with ``a1_coef = 1/(1 - sigma_sqd_w)``
    regridded zm->zt, fed to :func:`xpyp_term_ta_pdf_lhs_centered`. The same
    operator serves wprtp and wpthlp (shared ADG1 TA LHS).
    """
    a1_coef_zt = zm2zt(1.0 / (1.0 - sigma_sqd_w), gr)
    coef_zt = a1_coef_zt * wp3_on_wp2_zt
    return xpyp_term_ta_pdf_lhs_centered(coef_zt, rho_ds_zt, invrs_rho_ds_zm, gr)


def calc_xm_wpxp_lhs_terms(wm_zm, wm_zt, wp2, Kw6, nu6, C7_Skw_fnc,
                           invrs_rho_ds_zm, rho_ds_zt, rho_ds_zm, invrs_rho_ds_zt,
                           gr: CLUBBGrid):
    """Shared LHS terms for the xm/w'x' system (``calc_xm_wpxp_lhs_terms``).

    Computes once and shares: the w'x' diffusion (``Kw6 = c_K6·Kh_zt``) + mean
    advection (zm centered, zt upwind — CAM ``l_upwind_xm_ma = True``) + the
    xm<->wpxp turbulent-advection / production / accumulation operators. The ADG1
    TA operator (:func:`calc_xm_wpxp_ta_terms`) is computed separately. ``nu6`` is
    a scalar background diffusivity. Returns a dict with keys ``lhs_diff_zm,
    lhs_ma_zm, lhs_ma_zt, lhs_ta_xm, lhs_tp, lhs_ac_pr2``.
    """
    nu6_arr = jnp.full((invrs_rho_ds_zm.shape[0],), nu6, dtype=wp2.dtype)
    return dict(
        lhs_diff_zm=diffusion_zm_lhs(Kw6, nu6_arr, invrs_rho_ds_zm, rho_ds_zt, gr),
        lhs_ma_zm=term_ma_zm_lhs(wm_zm, gr),
        lhs_ma_zt=term_ma_zt_lhs_upwind(wm_zt, gr),
        lhs_ta_xm=xm_term_ta_lhs(invrs_rho_ds_zt, rho_ds_zm, gr),
        lhs_tp=wpxp_term_tp_lhs(wp2, gr),
        lhs_ac_pr2=wpxp_terms_ac_pr2_lhs(C7_Skw_fnc, wm_zt, gr),
    )


def solve_xm_wpxp_with_single_lhs(wpxp, xm, wpxp_forcing, xm_forcing, C6_Skw_fnc,
                                  C7_Skw_fnc, invrs_tau_C6_zm, lhs_ta_wpxp,
                                  lhs_diff_zm, lhs_ma_zm, lhs_ma_zt, lhs_ta_xm,
                                  lhs_tp, lhs_ac_pr2, thv_ds_zm, xpthvp, wm_zt,
                                  dt, gr: CLUBBGrid, wp2=None, xp2_relaxed=None):
    """Solve one xm/w'x' variable pair (``solve_xm_wpxp_with_single_lhs``).

    Builds the field-specific pressure-1 LHS and buoyancy/pr3 RHS, assembles the
    coupled penta system with the shared LHS terms, solves + de-interleaves, and
    (if ``wp2``/``xp2_relaxed`` given) applies the Cauchy-Schwarz flux clip.
    ``rhs_ta = 0`` for ADG1. Returns ``(wpxp_new, xm_new)``.
    """
    lhs_pr1 = wpxp_term_pr1_lhs(C6_Skw_fnc, invrs_tau_C6_zm)
    rhs_bp_pr3 = wpxp_terms_bp_pr3_rhs(C7_Skw_fnc, thv_ds_zm, xpthvp)
    rhs_ta = jnp.zeros_like(wpxp)
    lhs = xm_wpxp_lhs(lhs_diff_zm, lhs_ma_zm, lhs_ma_zt, lhs_ta_wpxp, lhs_ta_xm,
                      lhs_tp, lhs_ac_pr2, lhs_pr1, dt)
    rhs = xm_wpxp_rhs(wpxp, xm, wpxp_forcing, xm_forcing, rhs_bp_pr3, rhs_ta,
                      lhs_ta_wpxp, lhs_pr1, dt, k_lb_zm=0)
    wpxp_new, xm_new = xm_wpxp_solve(lhs, rhs)
    if wp2 is not None and xp2_relaxed is not None:
        wpxp_new = clip_covar(wpxp_new, wp2, xp2_relaxed)
    return wpxp_new, xm_new


def xm_wpxp_clipping_and_stats(solve_type, xm, wpxp_preclip, xm_old, xp2, xp2_clip,
                               wp2, wm_zt, xm_forcing, rho_ds_zm, rho_ds_zt,
                               invrs_rho_ds_zm, invrs_rho_ds_zt, xp2_threshold,
                               xm_tol, low_lev_effect, high_lev_effect, field_tol,
                               fill_holes_type, l_mono_flux_lim, dt, gr: CLUBBGrid):
    """Per-field post-solve clipping for advance_xm_wpxp (``xm_wpxp_clipping_and_stats``).

    Applied once per scalar after its solve: (1) the monotonic turbulent-flux
    limiter (no-op unless ``l_mono_flux_lim``; adjusts both ``xm`` and the flux),
    (2) ``fill_holes_vertical`` on the mean field (gated ``fill_holes_type != 0``
    and not a wind component), (3) the Cauchy-Schwarz flux clip (``clip_covar``,
    bounded by ``wp2``/``xp2_clip``). ``solve_type``/``fill_holes_type``/
    ``l_mono_flux_lim`` are static. Returns ``(xm, wpxp)``.

    CAM gating: ``l_pos_def = .false.`` (the CLUBB default — absent from the CAM
    namelist), so the Fortran's RTM ``pos_definite_adj`` branch (which would
    adjust both ``xm`` and ``wpxp`` when the new mean goes negative) is OUT of the
    CAM-default tree and not ported; CLUBB-JAX omits it identically (verified by
    the bit-exact ``xm_wpxp_clipping_and_stats`` parity test).
    """
    if l_mono_flux_lim:
        xm, wpxp_preclip = monotonic_turbulent_flux_limit(
            solve_type, xm, wpxp_preclip, xm_old, xp2, wm_zt, xm_forcing,
            rho_ds_zm, rho_ds_zt, invrs_rho_ds_zm, invrs_rho_ds_zt,
            xp2_threshold, xm_tol, low_lev_effect, high_lev_effect, gr, dt)
    if fill_holes_type != 0 and solve_type not in (MFL_UM, MFL_VM):
        nzt = xm.shape[1]
        xm = fill_holes_vertical(xm, rho_ds_zt, gr.dzt, field_tol, 0, nzt - 1,
                                 fill_holes_type)
    wpxp = clip_covar(wpxp_preclip, wp2, xp2_clip)
    return xm, wpxp


def diagnose_upxp(ypwp, xm, wpxp, ym, C6x_Skw_fnc, tau_C6_zm, C7_Skw_fnc, gr: CLUBBGrid):
    """Diagnose a horizontal turbulent scalar flux ``y'x'`` (``diagnose_upxp``).

    Andre et al. (1978) eqn. 7 / Bougeault et al. (1981) eqn. 4 (CAM
    ``l_predict_upwp_vpwp = .false.``): ``y'x' = (tau_C6/C6x)·(-y'w'·d(xm)/dz
    - (1-C7)·w'x'·d(ym)/dz)`` at interior zm levels; boundaries zero. ``ym`` is
    the smoothed velocity. ``(ncol, nzm)``.
    """
    ddzt_xm = ddzt(xm, gr)
    ddzt_ym = ddzt(ym, gr)
    interior = (tau_C6_zm[:, 1:-1] / C6x_Skw_fnc[:, 1:-1]) * (
        -ypwp[:, 1:-1] * ddzt_xm[:, 1:-1]
        - (1.0 - C7_Skw_fnc[:, 1:-1]) * wpxp[:, 1:-1] * ddzt_ym[:, 1:-1])
    return jnp.zeros_like(ypwp).at[:, 1:-1].set(interior)


def xm_term_ta_lhs(invrs_rho_ds_zt, rho_ds_zm, gr: CLUBBGrid):
    """Turbulent-advection LHS for xm (``xm_term_ta_lhs``), ``(2, ncol, nzt)``.

    The implicit ``(1/rho_ds_zt)·d(rho_ds_zm·w'x')/dz`` couples xm (zt) to the
    bracketing wpxp (zm) levels: band 0 is the coefficient of ``wpxp[k+1]``, band
    1 of ``wpxp[k]``.
    """
    invrs_dzt = gr.invrs_dzt
    sup = invrs_rho_ds_zt * invrs_dzt * rho_ds_zm[:, 1:]
    sub = -invrs_rho_ds_zt * invrs_dzt * rho_ds_zm[:, :-1]
    return jnp.stack([sup, sub], axis=0)


def wpxp_term_tp_lhs(wp2, gr: CLUBBGrid):
    """Turbulent-production LHS for w'x' (``wpxp_term_tp_lhs``), ``(2, ncol, nzm)``.

    Couples wpxp (zm) to the bracketing xm (zt) levels: band 0 is the coefficient
    of ``xm[k]`` (``+wp2·invrs_dzm``), band 1 of ``xm[k-1]`` (``-wp2·invrs_dzm``).
    Boundaries zero.
    """
    interior = wp2[:, 1:-1] * gr.invrs_dzm[:, 1:-1]
    zeros_col = jnp.zeros((wp2.shape[0], 1), dtype=wp2.dtype)
    sup = jnp.concatenate([zeros_col, interior, zeros_col], axis=1)
    sub = jnp.concatenate([zeros_col, -interior, zeros_col], axis=1)
    return jnp.stack([sup, sub], axis=0)


def wpxp_terms_ac_pr2_lhs(C7_Skw_fnc, wm_zt, gr: CLUBBGrid):
    """Accumulation + pressure-2 LHS for w'x' (``wpxp_terms_ac_pr2_lhs``), ``(ncol, nzm)``.

    ``(1 - C7_Skw_fnc)·d(wm_zt)/dz`` at interior zm levels; boundaries zero.
    ``C7_Skw_fnc`` is ``(ncol, nzm)``; ``wm_zt`` is ``(ncol, nzt)``.
    """
    d_wm = wm_zt[:, 1:] - wm_zt[:, :-1]
    interior = (1.0 - C7_Skw_fnc[:, 1:-1]) * gr.invrs_dzm[:, 1:-1] * d_wm
    zeros_col = jnp.zeros((C7_Skw_fnc.shape[0], 1), dtype=C7_Skw_fnc.dtype)
    return jnp.concatenate([zeros_col, interior, zeros_col], axis=1)


def wpxp_term_pr1_lhs(C6_Skw_fnc, invrs_tau_C6_zm):
    """Pressure-1 LHS for w'x' (``wpxp_term_pr1_lhs``), ``(ncol, nzm)``.

    ``C6_Skw_fnc·invrs_tau_C6_zm·w'x'`` at each zm level; boundaries zeroed.
    """
    result = C6_Skw_fnc * invrs_tau_C6_zm
    result = result.at[:, 0].set(0.0)
    result = result.at[:, -1].set(0.0)
    return result


def wpxp_terms_bp_pr3_rhs(C7_Skw_fnc, thv_ds_zm, xpthvp):
    """Buoyancy-production + pressure-3 RHS for w'x' (``wpxp_terms_bp_pr3_rhs``), ``(ncol, nzm)``.

    ``(1 - C7_Skw_fnc)·(g/thv_ds)·x'thv'`` at each zm level (``xpthvp`` = r'thv'
    or thl'thv'); boundaries zeroed. Uses ``constants.g`` (CLUBB ``grav``).
    """
    result = buoyancy_coefficient(thv_ds_zm) * (1.0 - C7_Skw_fnc) * xpthvp
    result = result.at[:, 0].set(0.0)
    result = result.at[:, -1].set(0.0)
    return result


# ---------------------------------------------------------------------------
# Pentadiagonal assembly + solve (interleaved wpxp[2k] / xm[2k+1])
# ---------------------------------------------------------------------------

def xm_wpxp_lhs(lhs_diff_zm, lhs_ma_zm, lhs_ma_zt, lhs_ta_wpxp, lhs_ta_xm,
                lhs_tp, lhs_ac_pr2, lhs_pr1, dt):
    """Assemble the coupled xm/wpxp pentadiagonal LHS (``xm_wpxp_lhs``).

    Faithful port of ``advance_xm_wpxp_module.F90:xm_wpxp_lhs`` for the CAM tree
    (``l_implemented = False`` standalone, ``l_diffuse_rtm_and_thlm = False`` so
    the xm rows carry NO diffusion, ``l_iter = True``). Interleaving: wpxp[k] at
    global index 2k, xm[k] at 2k+1. Bands ``[super2, super1, main, sub1, sub2]``;
    the wpxp turbulent advection and pressure-1 are over-implicit (scaled by
    gamma). wpxp lower/upper rows are identity BCs. Returns ``(5, ncol, 2*nzm-1)``.
    """
    ngrdcol = lhs_diff_zm.shape[1]
    nzm = lhs_diff_zm.shape[2]
    ndim = 2 * nzm - 1
    invrs_dt = 1.0 / dt
    g = _GAMMA
    lhs = jnp.zeros((5, ngrdcol, ndim), dtype=lhs_diff_zm.dtype)

    # xm rows (odd global indices 1,3,..): mean advection (upwind) + xm<->wpxp TA.
    lhs = lhs.at[0, :, 1::2].set(lhs_ma_zt[0])          # super2: xm[k+1]
    lhs = lhs.at[1, :, 1::2].set(lhs_ta_xm[0])          # super1: wpxp[k+1]
    lhs = lhs.at[2, :, 1::2].set(invrs_dt + lhs_ma_zt[1])  # diag
    lhs = lhs.at[3, :, 1::2].set(lhs_ta_xm[1])          # sub1: wpxp[k]
    lhs = lhs.at[4, :, 1::2].set(lhs_ma_zt[2])          # sub2: xm[k-1]

    # wpxp interior rows (even global indices 2..2*(nzm-2)).
    sl = slice(2, 2 * nzm - 2, 2)
    lhs = lhs.at[0, :, sl].set(lhs_ma_zm[0, :, 1:-1] + lhs_diff_zm[0, :, 1:-1]
                              + g * lhs_ta_wpxp[0, :, 1:-1])
    lhs = lhs.at[1, :, sl].set(lhs_tp[0, :, 1:-1])      # super1: xm[k]
    lhs = lhs.at[2, :, sl].set(
        lhs_ma_zm[1, :, 1:-1] + lhs_diff_zm[1, :, 1:-1] + lhs_ac_pr2[:, 1:-1]
        + g * (lhs_ta_wpxp[1, :, 1:-1] + lhs_pr1[:, 1:-1]) + invrs_dt)
    lhs = lhs.at[3, :, sl].set(lhs_tp[1, :, 1:-1])      # sub1: xm[k-1]
    lhs = lhs.at[4, :, sl].set(lhs_ma_zm[2, :, 1:-1] + lhs_diff_zm[2, :, 1:-1]
                              + g * lhs_ta_wpxp[2, :, 1:-1])

    # wpxp lower (j=0) and upper (j=ndim-1) identity BC rows.
    lhs = lhs.at[2, :, 0].set(1.0)
    lhs = lhs.at[2, :, -1].set(1.0)
    return lhs


def xm_wpxp_rhs(wpxp, xm, wpxp_forcing, xm_forcing, rhs_bp_pr3, rhs_ta,
                lhs_ta_wpxp, lhs_pr1, dt, k_lb_zm=0):
    """Assemble the coupled xm/wpxp explicit RHS (``xm_wpxp_rhs``).

    Faithful port of ``advance_xm_wpxp_module.F90:xm_wpxp_rhs`` (``l_iter = True``
    so ``wpxp/dt`` is added). xm rows: ``xm/dt + xm_forcing``; wpxp interior:
    buoyancy/pr3 + forcing + ``rhs_ta`` (0 for ADG1) + the over-implicit TA/pr1
    contributions + ``wpxp/dt``. wpxp lower BC carries the current value, upper BC
    is 0. Returns ``(ncol, 2*nzm-1)``.
    """
    ngrdcol, nzm = wpxp.shape
    ndim = 2 * nzm - 1
    invrs_dt = 1.0 / dt
    g = _GAMMA
    rhs = jnp.zeros((ngrdcol, ndim), dtype=wpxp.dtype)

    rhs = rhs.at[:, 0].set(wpxp[:, k_lb_zm])
    rhs = rhs.at[:, 1::2].set(xm * invrs_dt + xm_forcing)

    ta = lhs_ta_wpxp[:, :, 1:-1]
    pr1 = lhs_pr1[:, 1:-1]
    rhs_int = (
        rhs_bp_pr3[:, 1:-1] + wpxp_forcing[:, 1:-1] + rhs_ta[:, 1:-1]
        + (1.0 - g) * (-ta[0] * wpxp[:, 2:] - ta[1] * wpxp[:, 1:-1]
                       - ta[2] * wpxp[:, :-2] - pr1 * wpxp[:, 1:-1])
        + wpxp[:, 1:-1] * invrs_dt)
    rhs = rhs.at[:, 2:-1:2].set(rhs_int)
    rhs = rhs.at[:, -1].set(0.0)
    return rhs


def xm_wpxp_solve(lhs, rhs):
    """Pentadiagonal solve + de-interleave (``xm_wpxp_solve``).

    Solves with the CLUBB-band penta LU (:func:`penta_solve`) and
    splits: wpxp on even slots, xm on odd. Returns ``(wpxp_new, xm_new)``.
    """
    soln = penta_solve(lhs, rhs)
    return soln[:, 0::2], soln[:, 1::2]


def advance_xm_wpxp(rtm, thlm, wprtp, wpthlp, rtm_forcing, thlm_forcing,
                    wprtp_forcing, wpthlp_forcing, C6rt_Skw_fnc, C6thl_Skw_fnc,
                    C7_Skw_fnc, invrs_tau_C6_zm, sigma_sqd_w, wp3_on_wp2_zt,
                    wp2, Kh_zt, rtp2, thlp2, rtpthvp, thlpthvp, thv_ds_zm,
                    wm_zm, wm_zt, rho_ds_zm, rho_ds_zt, invrs_rho_ds_zm,
                    invrs_rho_ds_zt, w_1_zm, w_2_zm, varnce_w_1_zm,
                    varnce_w_2_zm, mixt_frac_zm, dt, gr: CLUBBGrid, config):
    """Advance the coupled xm/w'x' scalar pairs one step (``advance_xm_wpxp``).

    Faithful port of the core (CAM-default) path of
    ``advance_xm_wpxp_module.F90:advance_xm_wpxp``: advances rtm/wprtp and
    thlm/wpthlp. CAM ``l_predict_upwp_vpwp = .false.`` so the wind fluxes
    upwp/vpwp are NOT advanced here (they are diagnosed; um/vm go through
    ``advance_windm_edsclrm``). Builds the shared ADG1 TA + LHS operators once
    (:func:`calc_xm_wpxp_ta_terms` / :func:`calc_xm_wpxp_lhs_terms`), then for
    each scalar: solve (:func:`solve_xm_wpxp_with_single_lhs`) → per-field
    clipping (:func:`xm_wpxp_clipping_and_stats`: MFL → fill_holes → clip_covar).

    The skewness-dependent coefficients ``C6rt_Skw_fnc``/``C6thl_Skw_fnc``/
    ``C7_Skw_fnc`` and ``invrs_tau_C6_zm`` come from the orchestration (CAM
    ``l_use_C7_Richardson = .false.`` / ``l_diag_Lscale_from_tau = .false.`` make
    these skewness functions, NOT the ARM Richardson/constant the hard-wired
    reference uses — so they are explicit inputs here). The MFL turbulent-
    advection range is field-independent → computed once. Returns
    ``(wprtp, rtm, wpthlp, thlm)``.
    """
    params = config.params
    Kw6 = params.c_K6 * Kh_zt
    nu6 = params.nu6
    rt_tol, thl_tol, fht = config.rt_tol, config.thl_tol, _CAM_FILL_HOLES_TYPE

    lhs_ta = calc_xm_wpxp_ta_terms(sigma_sqd_w, wp3_on_wp2_zt, rho_ds_zt,
                                   invrs_rho_ds_zm, gr)
    sh = calc_xm_wpxp_lhs_terms(wm_zm, wm_zt, wp2, Kw6, nu6, C7_Skw_fnc,
                                invrs_rho_ds_zm, rho_ds_zt, rho_ds_zm,
                                invrs_rho_ds_zt, gr)

    # CAM l_enable_relaxed_clipping = False (fixed): clip on the raw variances.
    rtp2_clip, thlp2_clip = rtp2, thlp2

    # Field-independent MFL reachable-level range (shared by rt/thl).
    lo, hi = calc_turb_adv_range(w_1_zm, w_2_zm, varnce_w_1_zm, varnce_w_2_zm,
                                 mixt_frac_zm, gr, dt)

    def _advance(wpxp, xm, wpxp_forcing, xm_forcing, C6, xpthvp, xp2, xp2_clip,
                 mfl_id, xm_tol, tol_mfl, field_tol, l_mfl):
        wpxp_pre, xm_new = solve_xm_wpxp_with_single_lhs(
            wpxp, xm, wpxp_forcing, xm_forcing, C6, C7_Skw_fnc, invrs_tau_C6_zm,
            lhs_ta, sh["lhs_diff_zm"], sh["lhs_ma_zm"], sh["lhs_ma_zt"],
            sh["lhs_ta_xm"], sh["lhs_tp"], sh["lhs_ac_pr2"], thv_ds_zm, xpthvp,
            wm_zt, dt, gr)
        return xm_wpxp_clipping_and_stats(
            mfl_id, xm_new, wpxp_pre, xm, xp2, xp2_clip, wp2, wm_zt, xm_forcing,
            rho_ds_zm, rho_ds_zt, invrs_rho_ds_zm, invrs_rho_ds_zt,
            xm_tol ** 2, tol_mfl, lo, hi, field_tol, fht, l_mfl, dt, gr)

    rtm_new, wprtp_new = _advance(
        wprtp, rtm, wprtp_forcing, rtm_forcing, C6rt_Skw_fnc, rtpthvp, rtp2,
        rtp2_clip, MFL_RTM, rt_tol, _RT_TOL_MFL, rt_tol, True)  # CAM l_mono_flux_lim_rtm
    thlm_new, wpthlp_new = _advance(
        wpthlp, thlm, wpthlp_forcing, thlm_forcing, C6thl_Skw_fnc, thlpthvp, thlp2,
        thlp2_clip, MFL_THLM, thl_tol, _THL_TOL_MFL, thl_tol, True)  # CAM l_mono_flux_lim_thlm

    return wprtp_new, rtm_new, wpthlp_new, thlm_new


# ===========================================================================
# 18. Core orchestration
# ===========================================================================
# Assembles the parity-tested building blocks into the per-step closure: the
# diagnostics (``compute_clubb_diagnostics`` — skewness, ``sigma_sqd_w``, TKE,
# the dissipation-time-scale family, and the C6/C7 coefficients) that feed the
# ADG1 PDF closure and the four prognostic moment advances, which run in the
# CAM order ``xm_wpxp(1) -> xp2_xpyp(2) -> wp2_wp3(3) -> windm(4)`` with
# ``clip_covars_denom`` between. Each constituent is independently
# bit-exact/round-off parity-tested vs CLUBB-JAX; this section is the thin
# (pure / JIT-safe / differentiable) wiring.


def compute_clubb_diagnostics(wp2, wp3, up2, vp2, thlp2, rtp2, wpthlp, wprtp,
                              Lscale, brunt_vaisala_freq_sqd, gr: CLUBBGrid, config):
    """Per-step CLUBB closure diagnostics (CAM-default tree).

    Composes the parity-tested helpers into one bundle of the inputs the PDF
    closure and the moment advances need:

      * skewness (``Skw_zm``/``Skw_zt``) and the smoothed ``wp3_on_wp2`` ratio
        (:func:`compute_skewness_diagnostics`);
      * the skewness-dependent ``gamma_Skw`` (CAM ``l_gamma_Skw = .true.``) and
        ``sigma_sqd_w`` (:func:`compute_sigma_sqd_w`);
      * TKE ``em``/``sqrt_em_zt`` (:func:`compute_tke`);
      * the ``invrs_tau_*`` family (:func:`compute_tau_family`);
      * the xm/wpxp coefficients ``C6rt/C6thl/C7_Skw_fnc``
        (:func:`compute_C6_C7_Skw_fnc`).

    All moments/fluxes are zm-level (``wp3`` zt); ``Lscale`` is the parcel
    buoyant-sorting length (zt); ``brunt_vaisala_freq_sqd`` is zm. Returns a dict
    with the above (the tau-family keys + ``Skw_*``/``wp3_on_wp2*``/``wp2_zt``/
    ``wp3_zm``/``em``/``sqrt_em_zt``/``sigma_sqd_w``/``gamma_Skw``/``Lscale_zm``/
    ``C6rt_Skw_fnc``/``C6thl_Skw_fnc``/``C7_Skw_fnc``).
    """
    p = config.params
    skw = compute_skewness_diagnostics(wp2, wp3, config.w_tol, p.Skw_denom_coef, gr)
    # CAM l_gamma_Skw = .true. (CLUBB model_flags default).
    gamma_Skw = compute_gamma_Skw(skw["Skw_zm"], p.gamma_coef, p.gamma_coefb,
                                  p.gamma_coefc, True)
    sigma_sqd_w = compute_sigma_sqd_w(
        gamma_Skw, wp2, thlp2, rtp2, wpthlp, wprtp, gr,
        w_tol=config.w_tol, thl_tol=config.thl_tol, rt_tol=config.rt_tol)
    em, sqrt_em_zt = compute_tke(wp2, up2, vp2, gr, config)
    tau = compute_tau_family(Lscale, em, sqrt_em_zt, brunt_vaisala_freq_sqd, gr, config)
    Lscale_zm = jnp.maximum(zt2zm(Lscale, gr), 0.0)
    C6rt, C6thl, C7 = compute_C6_C7_Skw_fnc(skw["Skw_zm"], Lscale_zm, config, gr)

    # Eddy diffusivities Kh = c_K·Lscale·sqrt(em) (zt and zm levels).
    em_min = 1.5 * config.w_tol ** 2
    Kh_zt = config.params.c_K * Lscale * sqrt_em_zt
    Kh_zm = config.params.c_K * Lscale_zm * jnp.sqrt(jnp.maximum(em, em_min))
    return dict(
        **skw, gamma_Skw=gamma_Skw, sigma_sqd_w=sigma_sqd_w, em=em,
        sqrt_em_zt=sqrt_em_zt, Lscale_zm=Lscale_zm, Kh_zt=Kh_zt, Kh_zm=Kh_zm,
        C6rt_Skw_fnc=C6rt, C6thl_Skw_fnc=C6thl, C7_Skw_fnc=C7, **tau,
    )


def compute_pdf_closure(diag, wp2, wp3, rtp2, thlp2, rtpthlp, up2, vp2,
                        wprtp, wpthlp, upwp, vpwp,
                        wm_zt, rtm, thlm, um, vm, exner_zt, p_in_Pa_zt, thv_ds_zt,
                        gr: CLUBBGrid, config):
    """ADG1 assumed-PDF closure (CAM-default tree, post-advance placement).

    The CAM-default subset of ``pdf_closure_module.F90:pdf_closure_driver`` (and
    its ``adg1_pdf_driver_zt_jax`` helper): given the post-advance moment state it
    invokes the ADG1 double-Gaussian PDF and returns the **buoyancy fluxes**
    (``wpthvp``/``wp2thvp``/``rtpthvp``/``thlpthvp``), the **higher-order velocity
    moments** (``wp4``/``wp2up2``/``wp2vp2``/``wpup2``/``wpvp2``/``wp2rtp``/
    ``wp2thlp``/``wp2up``/``wprtp2``/``wpthlp2``/``wprtpthlp``), the **cloud-water
    turbulent fluxes** (``wprcp``/``rtprcp``/``thlprcp``/``uprcp``/``vprcp``), and
    the **cloud diagnostics** (``cloud_frac``/``rcm``/``rc_coef_zm``) that the four
    moment advances and the host model consume. The CAM-irrelevant pieces (the
    stats writer, the non-ADG1 PDF branches, ice-supersat) are omitted; the
    stats-only intermediates are not returned.

    Inputs: ``diag`` is the :func:`compute_clubb_diagnostics` bundle for the SAME
    (post-advance) state — its ``Skw_zt``/``wp2_zt``/``sigma_sqd_w`` are reused
    rather than re-derived. Prognostic moments ``wp2``/``rtp2``/``thlp2``/
    ``rtpthlp``/``up2``/``vp2`` and fluxes ``wprtp``/``wpthlp``/``upwp``/``vpwp``
    are on zm; ``wp3`` is on zt. Means ``wm_zt``/``rtm``/``thlm``/``um``/``vm`` and
    the thermo fields ``exner_zt``/``p_in_Pa_zt``/``thv_ds_zt`` are on zt.

    Returns a dict (buoyancy/HOM on the levels the advances want: ``wpthvp_zm``
    etc. on zm, ``wp2thvp_zt`` on zt).
    """
    p = config.params
    w_tol_sqd = config.w_tol ** 2

    # --- zt-level fields for the ADG1 driver (adg1_pdf_driver_zt_jax) ---
    Skw_zt = diag["Skw_zt"]
    wp2_zt = diag["wp2_zt"]
    sigma_sqd_w_zt = jnp.maximum(zm2zt(diag["sigma_sqd_w"], gr), 0.0)
    # Raw zt regrids feed the buoyancy assembly (calc_xpthvp_terms); the
    # tolerance-floored ``*_adg`` versions feed ONLY the ADG1 driver. CLUBB-JAX
    # pdf_closure_driver keeps these paths separate (the floor must not leak a
    # tolerance-level variance into rtpthvp/thlpthvp in low-variance columns).
    rtp2_zt = zm2zt(rtp2, gr)
    thlp2_zt = zm2zt(thlp2, gr)
    rtp2_zt_adg = jnp.maximum(rtp2_zt, config.rt_tol ** 2)
    thlp2_zt_adg = jnp.maximum(thlp2_zt, config.thl_tol ** 2)
    up2_zt = jnp.maximum(zm2zt(up2, gr), w_tol_sqd)
    vp2_zt = jnp.maximum(zm2zt(vp2, gr), w_tol_sqd)
    wprtp_zt = zm2zt(wprtp, gr)
    wpthlp_zt = zm2zt(wpthlp, gr)
    upwp_zt = zm2zt(upwp, gr)
    vpwp_zt = zm2zt(vpwp, gr)
    rtpthlp_zt = zm2zt(rtpthlp, gr)

    mixt_frac_max_mag = derive_mixt_frac_max_mag(p.Skw_max_mag)
    adg1 = ADG1_pdf_driver(
        wm_zt, rtm, thlm, um, vm, wp2_zt, rtp2_zt_adg, thlp2_zt_adg, up2_zt, vp2_zt,
        Skw_zt, wprtp_zt, wpthlp_zt, upwp_zt, vpwp_zt, jnp.sqrt(wp2_zt),
        sigma_sqd_w_zt, p.beta, mixt_frac_max_mag)

    # --- per-component rt-thl correlation + liquid cloud-fraction closure ---
    corr_rt_thl_1, corr_rt_thl_2 = calc_comp_corrs_binormal(
        rtpthlp_zt, rtm, thlm, adg1["rt_1"], adg1["rt_2"], adg1["thl_1"],
        adg1["thl_2"], adg1["varnce_rt_1"], adg1["varnce_rt_2"],
        adg1["varnce_thl_1"], adg1["varnce_thl_2"], adg1["mixt_frac"])
    comp = calc_pdf_liquid_cloud_frac_components(
        adg1, rtpthlp_zt, rtm, thlm, exner_zt, p_in_Pa_zt)
    rcm_zt = comp["rcm"]

    # --- cloud-water fluxes, higher-order velocity moments, buoyancy fluxes ---
    xprcp = calc_pdf_xprcp_fluxes(adg1, comp, wm_zt, rtm, thlm, um, vm, rcm_zt, gr)
    hom = calc_pdf_higher_order_moments(
        adg1, wm_zt, rtm, thlm, um, vm, corr_rt_thl_1, corr_rt_thl_2, gr)
    (wpthvp_zm, wp2thvp_zt, rtpthvp_zm, thlpthvp_zm,
     _rc_coef_zt, rc_coef_zm) = calc_xpthvp_terms(
        exner_zt, thv_ds_zt, xprcp["wprcp_zt"], xprcp["wp2rcp_zt"],
        xprcp["rtprcp_zt"], xprcp["thlprcp_zt"], wpthlp_zt, wprtp_zt,
        hom["wp2thlp"], hom["wp2rtp"], rtpthlp_zt, rtp2_zt, thlp2_zt, gr)

    # ADG1 w-component PDF params regridded zt->zm (the MFL turbulent-advection
    # range in advance_xm_wpxp reads these on zm; advance_clubb_core_module.F90).
    w_1_zm = zt2zm(adg1["w_1"], gr)
    w_2_zm = zt2zm(adg1["w_2"], gr)
    varnce_w_1_zm = zt2zm(adg1["varnce_w_1"], gr)
    varnce_w_2_zm = zt2zm(adg1["varnce_w_2"], gr)
    mixt_frac_zm = zt2zm(adg1["mixt_frac"], gr)

    return dict(
        wpthvp=wpthvp_zm, wp2thvp=wp2thvp_zt, rtpthvp=rtpthvp_zm,
        thlpthvp=thlpthvp_zm, rc_coef_zm=rc_coef_zm,
        cloud_frac=comp["cloud_frac"], rcm=comp["rcm"],
        wprcp=xprcp["wprcp_zm"], rtprcp=xprcp["rtprcp_zm"],
        thlprcp=xprcp["thlprcp_zm"], uprcp=xprcp["uprcp_zm"],
        vprcp=xprcp["vprcp_zm"], w_1_zm=w_1_zm, w_2_zm=w_2_zm,
        varnce_w_1_zm=varnce_w_1_zm, varnce_w_2_zm=varnce_w_2_zm,
        mixt_frac_zm=mixt_frac_zm, **hom,
    )


class CLUBBMomentState(NamedTuple):
    """The carried CLUBB prognostic higher-order moment state (CAM-default tree).

    The 15 fields advanced each step by :func:`advance_clubb_core`. Means are on
    thermodynamic (zt) levels; second/third moments and fluxes on momentum (zm)
    levels except ``wp3`` (zt). All ``(ngrdcol, nz*)``. CAM
    ``l_predict_upwp_vpwp = .false.`` so ``upwp``/``vpwp`` are diagnosed inside
    ``advance_windm_edsclrm`` rather than prognosed by ``advance_xm_wpxp``.
    """
    # Means (zt)
    rtm: jax.Array
    thlm: jax.Array
    um: jax.Array
    vm: jax.Array
    # Velocity moments
    wp2: jax.Array      # zm
    wp3: jax.Array      # zt
    up2: jax.Array      # zm
    vp2: jax.Array      # zm
    # Fluxes (zm)
    wprtp: jax.Array
    wpthlp: jax.Array
    upwp: jax.Array
    vpwp: jax.Array
    # Scalar second moments (zm)
    rtp2: jax.Array
    thlp2: jax.Array
    rtpthlp: jax.Array


class CLUBBForcing(NamedTuple):
    """Large-scale forcings (tendency sources) for the moment advances [units/s].

    All on the same level as the advanced field: ``rtm``/``thlm``/``um``/``vm``
    on zt; ``wprtp``/``wpthlp``/``rtp2``/``thlp2``/``rtpthlp`` on zm. Zero by
    default for an isolated SCM driver (the host supplies them when coupled).
    """
    rtm: jax.Array
    thlm: jax.Array
    um: jax.Array
    vm: jax.Array
    wprtp: jax.Array
    wpthlp: jax.Array
    rtp2: jax.Array
    thlp2: jax.Array
    rtpthlp: jax.Array


def advance_clubb_core(state: CLUBBMomentState, forcing: CLUBBForcing, *,
                       Lscale, brunt_vaisala_freq_sqd, exner_zt, p_in_Pa_zt,
                       thv_ds_zt, thv_ds_zm, rho_ds_zm, rho_ds_zt,
                       invrs_rho_ds_zm, invrs_rho_ds_zt, wm_zt, wm_zm,
                       sfc_elevation, fcor, ug, vg, dt, gr: CLUBBGrid, config):
    """One prognostic CLUBB step (the CAM-default ``advance_clubb_core`` core).

    Runs, in the CAM order, ``compute_clubb_diagnostics`` -> the pre-advance ADG1
    PDF closure (``l_call_pdf_closure_twice = .true.``) -> the four moment
    advances ``advance_xm_wpxp -> advance_xp2_xpyp -> advance_wp2_wp3 ->
    advance_windm_edsclrm`` with ``clip_covars_denom`` between the variance/flux
    solves -> the post-advance PDF closure for the cloud/buoyancy diagnostics. The
    advances mutate the shared moment set in sequence (each sees the previous
    advance's update), exactly as the Fortran driver does.

    Parameters
    ----------
    state : CLUBBMomentState
        Start-of-step prognostic moments.
    forcing : CLUBBForcing
        Large-scale tendency sources.
    Lscale : jax.Array
        Parcel buoyant-sorting mixing length (zt) from ``compute_mixing_length``.
    brunt_vaisala_freq_sqd : jax.Array
        ``N^2`` (zm).
    exner_zt, p_in_Pa_zt, thv_ds_zt, thv_ds_zm, rho_ds_zm, rho_ds_zt,
    invrs_rho_ds_zm, invrs_rho_ds_zt, wm_zt, wm_zm, sfc_elevation, fcor, ug, vg :
        Host thermodynamic / dry-static-density / geometry / Coriolis-geostrophic
        fields on their noted grids (``fcor``/``sfc_elevation`` are ``(ngrdcol,)``).
    dt : float
        Time step [s].

    Returns
    -------
    CLUBBMomentState
        End-of-step prognostic moments.
    dict
        Diagnostics — ``cloud_frac``/``rcm`` (post-advance PDF) plus the
        diffusivities ``Kh_zt``/``Kh_zm`` and ``wpthvp`` for the host.
    """
    p = config.params
    ng = state.wp2.shape[0]
    # Background eddy diffusivities are per-column arrays (vertical-resolution
    # scaling is identity on a fixed grid); advance_xp2_xpyp indexes nu[:, None].
    # (The per-moment C2 dissipation coefficients C2rt/C2thl/C2rtthl are owned by
    # advance_xp2_xpyp itself, read from config — CAM uses 3 distinct values.)
    nu2 = jnp.full((ng,), p.nu2)
    nu9 = jnp.full((ng,), p.nu9)

    # ---- (1) closure diagnostics on the start-of-step state ----
    diag = compute_clubb_diagnostics(
        state.wp2, state.wp3, state.up2, state.vp2, state.thlp2, state.rtp2,
        state.wpthlp, state.wprtp, Lscale, brunt_vaisala_freq_sqd, gr, config)

    # ---- (2) pre-advance ADG1 PDF closure -> buoyancy + higher-order moments ----
    pdf = compute_pdf_closure(
        diag, state.wp2, state.wp3, state.rtp2, state.thlp2, state.rtpthlp,
        state.up2, state.vp2, state.wprtp, state.wpthlp, state.upwp, state.vpwp,
        wm_zt, state.rtm, state.thlm, state.um, state.vm, exner_zt, p_in_Pa_zt,
        thv_ds_zt, gr, config)

    # ---- (3) advance_xm_wpxp: rtm/wprtp + thlm/wpthlp ----
    wprtp, rtm, wpthlp, thlm = advance_xm_wpxp(
        state.rtm, state.thlm, state.wprtp, state.wpthlp, forcing.rtm,
        forcing.thlm, forcing.wprtp, forcing.wpthlp, diag["C6rt_Skw_fnc"],
        diag["C6thl_Skw_fnc"], diag["C7_Skw_fnc"], diag["invrs_tau_C6_zm"],
        diag["sigma_sqd_w"], diag["wp3_on_wp2_zt"], state.wp2, diag["Kh_zt"],
        state.rtp2, state.thlp2, pdf["rtpthvp"], pdf["thlpthvp"], thv_ds_zm,
        wm_zm, wm_zt, rho_ds_zm, rho_ds_zt, invrs_rho_ds_zm, invrs_rho_ds_zt,
        pdf["w_1_zm"], pdf["w_2_zm"], pdf["varnce_w_1_zm"], pdf["varnce_w_2_zm"],
        pdf["mixt_frac_zm"], dt, gr, config)

    # ---- (4) advance_xp2_xpyp: rtp2/thlp2/rtpthlp/up2/vp2 (uses pre-wp2_wp3 wp2) ----
    rtp2, thlp2, rtpthlp, up2, vp2 = advance_xp2_xpyp(
        rtm, thlm, state.um, state.vm, state.rtp2, state.thlp2, state.rtpthlp,
        state.up2, state.vp2, wprtp, wpthlp, pdf["wpthvp"], state.upwp,
        state.vpwp, state.wp2, diag["wp2_zt"], diag["wp3_on_wp2"],
        diag["wp3_on_wp2_zt"], diag["sigma_sqd_w"], thv_ds_zm, diag["Kh_zt"],
        diag["invrs_tau_xp2_zm"], diag["invrs_tau_C4_zm"],
        diag["invrs_tau_C14_zm"], rho_ds_zm, rho_ds_zt, invrs_rho_ds_zm, wm_zm,
        forcing.rtp2, forcing.thlp2, forcing.rtpthlp, nu2, nu9, dt, gr, config)

    # ---- (5) Cauchy-Schwarz clip of the fluxes (pre-wp2_wp3 wp2) ----
    wprtp, wpthlp, upwp, vpwp = clip_covars_denom(
        wprtp, wpthlp, state.upwp, state.vpwp, state.wp2, rtp2, thlp2, up2, vp2,
        l_tke_aniso=True)  # CAM l_tke_aniso = True (fixed)

    # ---- (6) advance_wp2_wp3: wp2/wp3 (uses post-xp2 up2/vp2 + post-clip fluxes) ----
    wp2, wp3, _wp2_zt = advance_wp2_wp3(
        state.wp2, state.wp3, up2, vp2, diag["sigma_sqd_w"], diag["wp3_on_wp2"],
        pdf["wpup2"], pdf["wpvp2"], pdf["wp2up2_zm"], pdf["wp2vp2_zm"],
        pdf["wp4_zm"], pdf["wpthvp"], pdf["wp2thvp"], state.um, state.vm, upwp,
        vpwp, wm_zm, wm_zt, diag["Kh_zm"], diag["Kh_zt"], diag["invrs_tau_C4_zm"],
        diag["invrs_tau_wp3_zt"], diag["invrs_tau_C1_zm"], diag["Skw_zm"],
        diag["Skw_zt"], rho_ds_zm, rho_ds_zt, invrs_rho_ds_zm, invrs_rho_ds_zt,
        thv_ds_zm, thv_ds_zt, sfc_elevation, dt, gr, config)

    # ---- (7) Cauchy-Schwarz clip with the new wp2 ----
    wprtp, wpthlp, upwp, vpwp = clip_covars_denom(
        wprtp, wpthlp, upwp, vpwp, wp2, rtp2, thlp2, up2, vp2,
        l_tke_aniso=True)  # CAM l_tke_aniso = True (fixed)

    # ---- (8) advance_windm_edsclrm: um/vm + diagnostic upwp/vpwp ----
    # Kh_zm is the START-OF-STEP eddy diffusivity (advance_clubb_core_module.F90
    # "Block M", computed once before the advance loop) and is intentionally NOT
    # recomputed after wp2/wp3: the reference passes this same Kh_zm to BOTH
    # advance_wp2_wp3 and advance_windm_edsclrm. The new (post-advance) wp2/up2/vp2
    # are used only for the Cauchy-Schwarz flux clip inside windm, matching Fortran.
    um, vm, upwp, vpwp = advance_windm_edsclrm(
        state.um, state.vm, upwp, vpwp, wp2, up2, vp2, wm_zt, diag["Kh_zm"],
        ug, vg, forcing.um, forcing.vm, rho_ds_zm, rho_ds_zt, invrs_rho_ds_zt,
        fcor, p.c_K10, p.nu10, dt, gr, l_tke_aniso=True)  # CAM l_tke_aniso

    new_state = CLUBBMomentState(
        rtm=rtm, thlm=thlm, um=um, vm=vm, wp2=wp2, wp3=wp3, up2=up2, vp2=vp2,
        wprtp=wprtp, wpthlp=wpthlp, upwp=upwp, vpwp=vpwp, rtp2=rtp2,
        thlp2=thlp2, rtpthlp=rtpthlp)

    # ---- (9) post-advance PDF closure for the cloud/buoyancy diagnostics ----
    diag_post = compute_clubb_diagnostics(
        wp2, wp3, up2, vp2, thlp2, rtp2, wpthlp, wprtp, Lscale,
        brunt_vaisala_freq_sqd, gr, config)
    pdf_post = compute_pdf_closure(
        diag_post, wp2, wp3, rtp2, thlp2, rtpthlp, up2, vp2, wprtp, wpthlp,
        upwp, vpwp, wm_zt, rtm, thlm, um, vm, exner_zt, p_in_Pa_zt, thv_ds_zt,
        gr, config)

    diagnostics = dict(
        cloud_frac=pdf_post["cloud_frac"], rcm=pdf_post["rcm"],
        wpthvp=pdf_post["wpthvp"], Kh_zt=diag["Kh_zt"], Kh_zm=diag["Kh_zm"])
    return new_state, diagnostics


def init_clubb_moments(ncol: int, nlev: int, config, dtype=jnp.float64) -> CLUBBMomentState:
    """Seed a fresh :class:`CLUBBMomentState` at rest (CAM-default floors).

    Means are zero (the bridge resets them from the live column each step);
    velocity variances start at ``tke_min``, scalar variances at their
    tolerance-squared floors, all fluxes and ``wp3`` zero.

    .. note::

       ``wp2`` is seeded at ``tke_min`` (1e-6), which is 400x below the floor
       the prognostic core itself enforces (``advance_wp2_wp3`` floors ``wp2``
       at ``w_tol**2`` = 4e-4 from its first advance), and the scalar variance
       solve runs BEFORE that advance.  That let the maximum-correlation floor
       ``thlp2 >= wpthlp**2 / (wp2 * 0.99**2)`` divide by 1e-6 on step 1 and
       write an unphysical surface ``thlp2`` that nothing subsequently
       lowered: with a prescribed surface flux actually delivered to the
       closure, the dry convective column got ``thlp2 = 3685 K^2`` (a 61 K RMS
       fluctuation) and drifted linearly by 4.5 K per step (#1508).

       The repair is at the DIVIDE, in :func:`advance_xp2_xpyp`, which floors
       the denominator at ``w_tol**2``.  The seed is deliberately left alone:
       raising it here changes the initial state of every prognostic CLUBB run
       and was measured to break four CLUBB regression tests, one of them by
       turning a finite column into NaN.  Flooring the denominator is
       sufficient, because it makes the seed's value irrelevant to that ratio.
    ``nlev`` thermo (zt) levels → ``nzm = nlev + 1`` momentum levels.
    """
    nzm = nlev + 1
    zt = jnp.zeros((ncol, nlev), dtype=dtype)
    zm0 = jnp.zeros((ncol, nzm), dtype=dtype)
    wtol2 = jnp.full((ncol, nzm), config.tke_min, dtype=dtype)
    return CLUBBMomentState(
        rtm=zt, thlm=zt, um=zt, vm=zt,
        wp2=wtol2, wp3=zt, up2=wtol2, vp2=wtol2,
        wprtp=zm0, wpthlp=zm0, upwp=zm0, vpwp=zm0,
        rtp2=jnp.full((ncol, nzm), config.rt_tol ** 2, dtype=dtype),
        thlp2=jnp.full((ncol, nzm), config.thl_tol ** 2, dtype=dtype),
        rtpthlp=zm0)


# Field layout for packing CLUBBMomentState into a single (ncol, NFIELDS, nzm)
# array carried in PhysicsState (like gwd_spectrum). zt-level fields (nlev) use
# the first nlev slots of the nzm axis with the trailing slot zero-padded.
N_MOMENT_FIELDS = len(CLUBBMomentState._fields)   # 15
_ZT_FIELD_NAMES = frozenset(("rtm", "thlm", "um", "vm", "wp3"))  # rest are zm


def pack_clubb_moments(state: CLUBBMomentState) -> jax.Array:
    """Pack a :class:`CLUBBMomentState` into one ``(ncol, 15, nzm)`` array.

    zm-level fields (``wp2``/variances/fluxes) fill the full ``nzm`` axis; zt-level
    fields (``rtm``/``thlm``/``um``/``vm``/``wp3``, length ``nlev = nzm-1``) fill
    ``[:, :nlev]`` with a zero in the trailing slot. Inverse of
    :func:`unpack_clubb_moments`. Used to carry the moment state in
    ``PhysicsState`` as a single regular array.
    """
    cols = []
    for name in CLUBBMomentState._fields:
        f = getattr(state, name)
        if name in _ZT_FIELD_NAMES:                       # (ncol, nlev) -> (ncol, nzm)
            f = jnp.concatenate([f, jnp.zeros_like(f[:, :1])], axis=1)
        cols.append(f[:, None, :])                        # (ncol, 1, nzm)
    return jnp.concatenate(cols, axis=1)                  # (ncol, 15, nzm)


def unpack_clubb_moments(arr: jax.Array) -> CLUBBMomentState:
    """Unpack a ``(ncol, 15, nzm)`` array into a :class:`CLUBBMomentState`.

    Inverse of :func:`pack_clubb_moments`: zt-level fields are sliced back to
    ``nlev = nzm-1`` (dropping the zero pad).
    """
    nlev = arr.shape[2] - 1
    fields = {}
    for i, name in enumerate(CLUBBMomentState._fields):
        col = arr[:, i, :]
        fields[name] = col[:, :nlev] if name in _ZT_FIELD_NAMES else col
    return CLUBBMomentState(**fields)


# ===========================================================================
# 19. Scheme entries
# ===========================================================================


def clubb_turbulence(
    u: jax.Array,
    v: jax.Array,
    T: jax.Array,
    q_v: jax.Array,
    tke: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    z_full: jax.Array,
    z_half: jax.Array,
    T_sfc: jax.Array,
    q_sfc: jax.Array,
    rho: jax.Array,
    dt: float,
    config: CLUBBConfig,
    surface_flux: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
    | None = None,
) -> tuple[TurbulenceOutput, jax.Array]:
    """CLUBB turbulence tendencies (phase 1: CLUBB ``Lscale`` eddy diffusion).

    Parameters mirror :func:`clubb_lite.clubb_lite_turbulence` (the ``tke`` slot
    carries ``wp2`` [m^2/s^2]); all column fields are TOP-DOWN ``(ncol, nlev)``,
    half-level fields ``(ncol, nlev+1)``.

    ``surface_flux`` (optional ``(tau_x, tau_y, shflx, lhflx, ustar)``, each
    ``(ncol,)``) is the driver-supplied tiled (mosaic) surface flux used as the
    BL bottom boundary condition in place of the single-surface
    ``compute_surface_fluxes`` call — the same contract Louis honours.  Same
    units/sign convention (``tau`` [Pa], ``shflx``/``lhflx`` [W/m^2]).  ``None``
    (default) keeps the legacy single-surface flux (identical behaviour).

    Returns
    -------
    TurbulenceOutput
        Tendencies + diffusivities + surface diagnostics.
    wp2_new : jax.Array
        Updated ``w'^2`` [m^2/s^2], ``(ncol, nlev)``, carried to the next step.
    """
    ncol, nlev = T.shape
    params = config.params

    wp2 = jnp.maximum(tke, config.tke_min)
    sqrt_wp2 = jnp.sqrt(wp2)

    # ---- Thermodynamics (top-down) ----
    exner = exner_function(p_full)                            # (ncol, nlev)
    theta = T / exner
    theta_v = virtual_temperature(T, q_v) / exner             # = theta * (1 + 0.61 q_v)
    # Dry phase-1 mapping to CLUBB variables (rcm = 0): thl ~ theta, rt ~ q_v.
    thlm_td = theta
    rtm_td = q_v
    thvm_td = theta_v
    thv_ds_td = theta_v

    # ---- CLUBB parcel buoyant-sorting mixing length (ascending grid) ----
    gr = make_clubb_grid_from_levels(z_full, z_half)
    thvm = flip_vertical(thvm_td)
    thlm = flip_vertical(thlm_td)
    rtm = flip_vertical(rtm_td)
    exner_a = flip_vertical(exner)
    p_a = flip_vertical(p_full)
    thv_ds = flip_vertical(thv_ds_td)
    # TKE on momentum (zm) levels from the carried wp2 (ascending zt -> zm).
    em_zm = jnp.maximum(zt2zm(flip_vertical(wp2), gr), config.tke_min)

    mu = jnp.full((ncol,), params.mu)
    lmin = derive_lmin(params.lmin_coef)
    Lscale_max = set_Lscale_max(False, None, None, ncol)
    Lscale_a, _, _ = compute_mixing_length(
        thvm, thlm, rtm, em_zm, Lscale_max, p_a, exner_a, thv_ds,
        mu, lmin, False, gr,
    )
    Lscale = flip_vertical(Lscale_a)                          # back to top-down (ncol, nlev)
    Lscale = jnp.clip(Lscale, 1.0, None)

    # ---- Eddy diffusivities from the CLUBB length scale ----
    Km_full = params.c_K * Lscale * sqrt_wp2                  # (ncol, nlev)
    Kh_full = Km_full / _PR_T

    # ---- ADG1 double-Gaussian PDF: cloud fraction + moist buoyancy flux ----
    # (the distinctive CLUBB closure; ascending grid). The buoyancy production of
    # wp2 uses the PDF flux ``w'thv'`` (with its cloud-water latent-heat term),
    # not a plain down-gradient ``-Kh N2``.
    wp2_a = jnp.maximum(flip_vertical(wp2), config.tke_min)   # ascending zt
    Kh_a = (params.c_K / _PR_T) * Lscale_a * jnp.sqrt(wp2_a)
    cloud_frac_a, rcm_a, wpthvp_a = diagnose_cloud_and_buoyancy(
        thlm, rtm, wp2_a, exner_a, p_a, thv_ds, Kh_a, Lscale_a, gr, config)
    buoy_prod = flip_vertical(buoyancy_coefficient(jnp.clip(thvm, 1.0, None)) * wpthvp_a)

    # ---- Geometry + shear (top-down) ----
    dz_half = jnp.clip(jnp.abs(z_full[:, :-1] - z_full[:, 1:]), 1.0, None)
    dz_layer = jnp.clip(jnp.abs(z_half[:, :-1] - z_half[:, 1:]), 1.0, None)
    Km_half = 0.5 * (Km_full[:, :-1] + Km_full[:, 1:])
    Kh_half = 0.5 * (Kh_full[:, :-1] + Kh_full[:, 1:])

    du_dz = (u[:, :-1] - u[:, 1:]) / dz_half
    dv_dz = (v[:, :-1] - v[:, 1:]) / dz_half
    S2_half = du_dz ** 2 + dv_dz ** 2

    def _half_to_full(field_half):
        mid = 0.5 * (field_half[:, :-1] + field_half[:, 1:])
        return jnp.concatenate([field_half[:, :1], mid, field_half[:, -1:]], axis=1)

    S2 = _half_to_full(S2_half)

    # ---- wp2 budget (production - dissipation + diffusion); tau = Lscale/sqrt(wp2) ----
    shear_prod = Km_full * S2
    diss_wp2 = sqrt_wp2 / Lscale                              # 1/tau
    wp2_diffused = implicit_vertical_diffusion(
        wp2, Km_half, rho, dz_layer, dz_half, dt,
        surface_flux=jnp.zeros(ncol, dtype=wp2.dtype),
    )
    wp2_new = (wp2_diffused + dt * (shear_prod + buoy_prod)) / (1.0 + dt * diss_wp2)
    wp2_new = jnp.clip(wp2_new, config.tke_min, config.wp2_max)

    # ---- Surface fluxes ----
    # Either the driver-supplied tiled (mosaic) flux or the legacy single-surface
    # bulk flux from the blended T_sfc (mirrors louis_turbulence).
    if surface_flux is not None:
        tau_x, tau_y, shflx, lhflx, ustar = surface_flux
    else:
        tau_x, tau_y, shflx, lhflx, ustar = compute_surface_fluxes(
            u[:, -1], v[:, -1], T[:, -1], q_v[:, -1],
            T_sfc, q_sfc, rho[:, -1], config.surface,
        )
    sflx_u, sflx_v = tau_x, tau_y
    sflx_T = shflx / constants.c_pd
    sflx_q = lhflx / constants.L_v

    # ---- Implicit vertical diffusion of the mean state ----
    u_new = implicit_vertical_diffusion(u, Km_half, rho, dz_layer, dz_half, dt, sflx_u)
    v_new = implicit_vertical_diffusion(v, Km_half, rho, dz_layer, dz_half, dt, sflx_v)
    T_new = implicit_vertical_diffusion_theta(
        T, Kh_half, rho, dz_layer, dz_half, p_full, dt, sflx_T)
    q_new = implicit_vertical_diffusion(q_v, Kh_half, rho, dz_layer, dz_half, dt, sflx_q)

    h_pbl = diagnose_pbl_height(T, q_v, u, v, p_full, z_full)

    output = TurbulenceOutput(
        du_dt=(u_new - u) / dt,
        dv_dt=(v_new - v) / dt,
        dT_dt=(T_new - T) / dt,
        dq_v_dt=(q_new - q_v) / dt,
        Km=Km_full,
        Kh=Kh_full,
        shflx=shflx,
        lhflx=lhflx,
        ustar=ustar,
        h_pbl=h_pbl,
        # Expose the CLUBB ADG1-PDF liquid cloud fraction so radiation can use
        # it (cloud_scheme="clubb") instead of the RH-diagnosed grid-scale one.
        cloud_fraction=cloud_frac_a,
    )
    return output, wp2_new


def clubb_step(
    u: jax.Array,
    v: jax.Array,
    T: jax.Array,
    q_v: jax.Array,
    moments: CLUBBMomentState,
    p_full: jax.Array,
    p_half: jax.Array,
    z_full: jax.Array,
    z_half: jax.Array,
    T_sfc: jax.Array,
    q_sfc: jax.Array,
    rho: jax.Array,
    dt: float,
    config: CLUBBConfig,
    sfc_wpthlp: jax.Array | None = None,
    sfc_wprtp: jax.Array | None = None,
    sfc_upwp: jax.Array | None = None,
    sfc_vpwp: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, CLUBBMomentState, dict]:
    """Bridge one prognostic CLUBB step from legoESM top-down column inputs.

    This is the *full prognostic* path (phase 2): it builds the ascending-grid
    host environment CLUBB needs, sets the surface turbulent-flux lower boundary
    conditions, advances the complete higher-order moment set with
    :func:`advance_clubb_core`, and maps the advanced means back to
    top-down ``(ncol, nlev)`` tendencies. The 15-field :class:`CLUBBMomentState`
    (means on zt, moments/fluxes on zm, ``wp3`` zt) is carried in and out.

    Modelling choices (documented; refined as the scheme matures):

      * Dry phase mapping ``thl ~ theta``, ``rt ~ q_v`` (``rcm`` enters only via
        the PDF closure inside ``advance_clubb_core``).
      * Mean vertical velocity ``wm = 0`` (grid-scale subsidence is the dycore's
        job, not the column closure).
      * Geostrophic wind ``ug = um``, ``vg = vm`` and ``fcor = 0`` → the
        Coriolis/geostrophic term in ``advance_windm_edsclrm`` vanishes (rotation
        is handled by the dycore; the turbulence scheme only diffuses).
      * Dry-static reference profiles ``thv_ds``/``rho_ds`` taken as the current
        ``thv``/``rho`` (a per-column Boussinesq-style reference).
      * Large-scale ``CLUBBForcing`` = 0 (other physics supply those tendencies).

    **Prescribed surface fluxes** (``sfc_wpthlp``/``sfc_wprtp``/``sfc_upwp``/
    ``sfc_vpwp``, each shape ``(ncol,)``): CLUBB's standard LES/SCM-intercomparison
    interface. When a component is given it OVERRIDES the bulk lower-BC for that
    moment with the prescribed **kinematic** surface flux (``w'thl'`` [K m/s],
    ``w'rt'`` [kg/kg m/s], ``u'w'``/``v'w'`` [m^2/s^2] — same units/sign as the
    internally-computed BCs); components left ``None`` fall back to CLUBB's own
    bulk formula from the air-surface contrast (``T_sfc``/``q_sfc``). The ``None``
    test is a compile-time static choice (CLAUDE.md feature-gating exception), not
    a traced selection. Cases that prescribe all four (BOMEX/DYCOMS/ARM) never
    touch the bulk formula, so the result is independent of ``T_sfc``/``q_sfc``.
    The reported ``shflx``/``lhflx``/``ustar`` diagnostics are made consistent
    with whichever BC was actually used.

    **Heat/moisture vs momentum semantics differ (important):** ``sfc_wpthlp``/
    ``sfc_wprtp`` enter ``advance_xm_wpxp`` directly as the scalar surface-flux
    lower-BC — applied EXACTLY and directionally (a prescribed ``w'thl'_sfc``
    closes the column θl budget to round-off). The momentum components are NOT
    applied as an independent ``(u'w', v'w')`` vector: CAM's
    ``l_imp_sfc_momentum_flux = .true.`` path (``advance_windm_edsclrm``) consumes
    ONLY the surface-stress-vector MAGNITUDE
    ``u_*^2 = sqrt(u'w'_sfc^2 + v'w'_sfc^2)`` (so ``u_* = (u'w'_sfc^2 +
    v'w'_sfc^2)^{1/4}``) and re-applies it implicitly as a drag ANTIPARALLEL to the
    near-surface wind (``-rho u_*^2 u/|V|``). So ``sfc_upwp``/``sfc_vpwp`` set only
    the stress magnitude (equivalently a prescribed ``u_*``); their azimuth is discarded —
    prescribing ``(u'w', 0)`` and ``(0, u'w')`` give identical wind tendencies.
    This is the correct contract for prescribed-``u_*`` LES forcing and is exact
    for the bulk drag (which is already wind-antiparallel by construction), but a
    cross-wind momentum-flux vector cannot be imposed through this interface.

    Returns ``(du_dt, dv_dt, dT_dt, dq_v_dt, new_moments, diagnostics)`` — the
    four mean tendencies (top-down ``(ncol, nlev)``), the advanced moment state,
    and the ``cloud_frac``/``rcm``/``wpthvp``/``Kh_*`` diagnostics dict.
    """
    ncol, nlev = T.shape
    params = config.params

    # ---- Thermodynamics (top-down) ----
    exner = exner_function(p_full)
    theta = T / exner
    thv = virtual_temperature(T, q_v) / exner

    # ---- Ascending CLUBB grid + means on zt ----
    gr = make_clubb_grid_from_levels(z_full, z_half)
    thlm = flip_vertical(theta)          # thl ~ theta (zt)
    rtm = flip_vertical(q_v)             # rt ~ q_v   (zt)
    um = flip_vertical(u)
    vm = flip_vertical(v)
    exner_zt = flip_vertical(exner)
    p_zt = flip_vertical(p_full)
    thv_zt = flip_vertical(thv)

    # ---- Dry-static reference + density profiles (zt and zm) ----
    thv_ds_zt = thv_zt
    thv_ds_zm = zt2zm(thv_zt, gr)
    rho_ds_zt = flip_vertical(rho)
    rho_ds_zm = zt2zm(rho_ds_zt, gr)
    invrs_rho_ds_zt = 1.0 / rho_ds_zt
    invrs_rho_ds_zm = 1.0 / rho_ds_zm

    # ---- Brunt-Vaisala N^2 (dry CAM-default form; rcm/ice unused there) ----
    zeros_zt = jnp.zeros((ncol, nlev), dtype=T.dtype)
    brunt = calc_brunt_vaisala_freq_sqd(
        thlm, exner_zt, rtm, zeros_zt, p_zt, zeros_zt,
        params.bv_efold, config.T0, gr)[0]

    # ---- Parcel buoyant-sorting mixing length ----
    em_zm = jnp.maximum(
        0.5 * (moments.wp2 + moments.up2 + moments.vp2), config.tke_min)  # l_tke_aniso
    mu = jnp.full((ncol,), params.mu)
    lmin = derive_lmin(params.lmin_coef)
    Lscale_max = set_Lscale_max(False, None, None, ncol)
    Lscale, _, _ = compute_mixing_length(
        thv_zt, thlm, rtm, em_zm, Lscale_max, p_zt, exner_zt, thv_ds_zt,
        mu, lmin, False, gr)
    # Enforce the physical minimum mixing length lmin (compute_mixing_length can
    # return < lmin; CLUBB floors it). A strictly positive Lscale keeps the
    # dissipation time tau = Lscale/sqrt(em) finite (invrs_tau = 1/tau).
    Lscale = jnp.maximum(Lscale, lmin)

    # ---- Surface turbulent-flux lower boundary conditions (kinematic) ----
    rho_sfc = rho[:, -1]
    exner_sfc = exner[:, -1]
    # Each BC component is either prescribed (LES/SCM intercomparison cases) or
    # computed by CLUBB's own bulk formula from the air-surface contrast. The
    # bulk formula is evaluated only if at least one component still needs it
    # (static Python branch on None-ness — never a traced selection).
    need_bulk = (sfc_wpthlp is None or sfc_wprtp is None
                 or sfc_upwp is None or sfc_vpwp is None)
    if need_bulk:
        tau_x, tau_y, shflx_b, lhflx_b, ustar_b = compute_surface_fluxes(
            u[:, -1], v[:, -1], T[:, -1], q_v[:, -1],
            T_sfc, q_sfc, rho_sfc, config.surface)
        wpthlp_b = shflx_b / (rho_sfc * constants.c_pd * exner_sfc)  # w'thl' [K m/s]
        wprtp_b = lhflx_b / (rho_sfc * constants.L_v)                # w'rt'  [kg/kg m/s]
        # Surface stress convention is tau = -rho*Cd*|V|*u (compute_surface_fluxes),
        # so the kinematic momentum flux is u'w'_sfc = tau_x/rho (NEGATIVE for u>0 —
        # momentum transported downward / drag), NOT -tau_x/rho.
        upwp_b = tau_x / rho_sfc                                     # u'w'   [m^2/s^2]
        vpwp_b = tau_y / rho_sfc
    # Resolve each BC: prescribed kinematic flux when given, else the bulk value.
    wpthlp_sfc = wpthlp_b if sfc_wpthlp is None else sfc_wpthlp
    wprtp_sfc = wprtp_b if sfc_wprtp is None else sfc_wprtp
    upwp_sfc = upwp_b if sfc_upwp is None else sfc_upwp
    vpwp_sfc = vpwp_b if sfc_vpwp is None else sfc_vpwp
    # W/m^2 + ustar diagnostics consistent with the BC actually used: the bulk
    # values pass through unchanged (exact back-compat); prescribed kinematic
    # fluxes are converted back to W/m^2, and ustar from the prescribed stress
    # (|tau|/rho = sqrt(u'w'^2 + v'w'^2), so ustar = (u'w'^2 + v'w'^2)^(1/4)).
    # The 1e-30 inside the fourth root is a pure AD safety floor: at the valid
    # zero-stress prescribed BC (sfc_upwp=sfc_vpwp=0) the bare (.)**0.25 has an
    # +inf slope, so jax.grad of an objective through ustar would be non-finite;
    # the floor pins the gradient to 0 there while leaving any physical stress
    # (|u'w'| >> 1e-8) bit-unchanged.
    shflx = shflx_b if sfc_wpthlp is None else (
        sfc_wpthlp * (rho_sfc * constants.c_pd * exner_sfc))
    lhflx = lhflx_b if sfc_wprtp is None else sfc_wprtp * (rho_sfc * constants.L_v)
    ustar = ustar_b if (sfc_upwp is None and sfc_vpwp is None) else (
        jnp.maximum(upwp_sfc ** 2 + vpwp_sfc ** 2, 1e-30) ** 0.25)
    # The mean state (rtm/thlm/um/vm) is owned by the model and re-read from the
    # live column each step; only the higher-order moments/fluxes persist in
    # ``moments``. Reset the means here, then set the surface flux BCs.
    state = moments._replace(
        rtm=rtm, thlm=thlm, um=um, vm=vm,
        wprtp=moments.wprtp.at[:, 0].set(wprtp_sfc),
        wpthlp=moments.wpthlp.at[:, 0].set(wpthlp_sfc),
        upwp=moments.upwp.at[:, 0].set(upwp_sfc),
        vpwp=moments.vpwp.at[:, 0].set(vpwp_sfc))

    # ---- Advance the full prognostic moment set ----
    zeros_zm = jnp.zeros((ncol, nlev + 1), dtype=T.dtype)
    forcing = CLUBBForcing(
        rtm=zeros_zt, thlm=zeros_zt, um=zeros_zt, vm=zeros_zt, wprtp=zeros_zm,
        wpthlp=zeros_zm, rtp2=zeros_zm, thlp2=zeros_zm, rtpthlp=zeros_zm)
    sfc_elevation = flip_vertical(z_half)[:, 0]
    new_state, diags = advance_clubb_core(
        state, forcing, Lscale=Lscale, brunt_vaisala_freq_sqd=brunt,
        exner_zt=exner_zt, p_in_Pa_zt=p_zt, thv_ds_zt=thv_ds_zt,
        thv_ds_zm=thv_ds_zm, rho_ds_zm=rho_ds_zm, rho_ds_zt=rho_ds_zt,
        invrs_rho_ds_zm=invrs_rho_ds_zm, invrs_rho_ds_zt=invrs_rho_ds_zt,
        wm_zt=zeros_zt, wm_zm=zeros_zm, sfc_elevation=sfc_elevation,
        fcor=jnp.zeros((ncol,), dtype=T.dtype), ug=um, vg=vm, dt=dt, gr=gr,
        config=config)

    # ---- Map advanced means back to top-down tendencies ----
    u_new = flip_vertical(new_state.um)
    v_new = flip_vertical(new_state.vm)
    T_new = flip_vertical(new_state.thlm) * exner    # thl ~ theta -> T = theta*exner
    q_new = flip_vertical(new_state.rtm)
    du_dt = (u_new - u) / dt
    dv_dt = (v_new - v) / dt
    dT_dt = (T_new - T) / dt
    dq_v_dt = (q_new - q_v) / dt
    diags = dict(diags, ustar=ustar, shflx=shflx, lhflx=lhflx)
    return du_dt, dv_dt, dT_dt, dq_v_dt, new_state, diags


def clubb_turbulence_prognostic(
    u: jax.Array,
    v: jax.Array,
    T: jax.Array,
    q_v: jax.Array,
    clubb_moments: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    z_full: jax.Array,
    z_half: jax.Array,
    T_sfc: jax.Array,
    q_sfc: jax.Array,
    rho: jax.Array,
    dt: float,
    config: CLUBBConfig,
    sfc_wpthlp: jax.Array | None = None,
    sfc_wprtp: jax.Array | None = None,
    sfc_upwp: jax.Array | None = None,
    sfc_vpwp: jax.Array | None = None,
    surface_flux: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
    | None = None,
) -> tuple[TurbulenceOutput, jax.Array]:
    """Prognostic CLUBB scheme entry (``scheme="clubb"``, ``prognostic=True``).

    Drop-in for :func:`clubb_turbulence` with the SAME carry-slot interface — the
    carried state is the packed :class:`CLUBBMomentState` ``(ncol, 15, nlev+1)``
    (``PhysicsState.clubb_moments``) instead of the single ``wp2`` slot. Unpacks
    it, advances the full higher-order moment closure, and repacks the new moments
    as the carry. No host numerical diffusion is added here: in a coupled run the
    dynamical core supplies it (the bare-SCM stand-in lives in
    :func:`integrate_clubb_column`).

    **CLUBB sub-cycling (CAM fidelity):** CAM runs CLUBB at its own
    ``clubb_timestep`` (``config.clubb_dt``, ~300 s) and sub-cycles it within the
    larger host physics ``dt``. This advances a LOCAL copy of the mean state +
    moments for ``n_sub = ceil(dt / clubb_dt)`` sub-steps of ``dt_sub = dt/n_sub``
    and returns the NET (RAW, unclipped) mean tendency over ``dt`` plus the
    sub-cycled final moments — the same coupling contract as the single-step path
    (positivity limiting stays the host/moisture-fixer's job, not folded into the
    physics tendency). For ``dt <= clubb_dt`` (``n_sub = 1``) it is the single
    step, bit-identical to the un-sub-cycled path. Surface-flux diagnostics are
    the sub-cycle mean; ``Kh`` the final sub-step's.

    **Prescribed surface fluxes** (``sfc_wpthlp``/``sfc_wprtp``/``sfc_upwp``/
    ``sfc_vpwp``, each ``(ncol,)`` or ``None``) are forwarded to :func:`clubb_step`
    unchanged — see its docstring. They are held constant across the sub-cycle
    (steady surface forcing, as in BOMEX/DYCOMS/ARM), the natural contract for a
    prescribed-flux case run within one host ``dt``.

    **Driver-injected tiled surface flux** (``surface_flux`` =
    ``(tau_x, tau_y, shflx, lhflx, ustar)`` in the DYNAMIC convention, identical
    to ``compute_surface_fluxes`` / Louis) is the coupled-run path: each
    :func:`clubb_step` (sub-)step converts it to the kinematic ``sfc_*`` lower BC
    at THAT step's near-surface density, so the prescribed DYNAMIC W/m^2 / Pa flux
    is conserved exactly across the sub-cycle (the coupler energy-budget contract),
    and the reported ``shflx``/``lhflx``/``ustar`` are the injected values
    directly (Louis-equivalent).  So ``clubb`` runs in tiled (mosaic-surface)
    production exactly like Louis.  Mutually exclusive with the explicit ``sfc_*``
    prescriptions (raises if both are given).

    Returns ``(TurbulenceOutput, clubb_moments_new)``; the second element flows
    back into ``PhysicsState.clubb_moments`` via the carry machinery.
    """
    moments = unpack_clubb_moments(clubb_moments)
    n_sub = max(1, math.ceil(dt / config.clubb_dt))

    # When the driver injects the tiled (mosaic) surface flux — the DYNAMIC
    # convention identical to compute_surface_fluxes / louis: (tau_x, tau_y [Pa],
    # shflx, lhflx [W/m^2], ustar [m/s]) — forward it as the KINEMATIC
    # prescribed-flux lower BC that clubb_step consumes, converting PER (SUB)STEP
    # with that step's near-surface density (``_sfc_bcs`` below).  Converting per
    # step (not once with the host rho) keeps the prescribed DYNAMIC W/m^2 / Pa
    # flux CONSERVED exactly as the column density drifts across the sub-cycle —
    # the coupler energy-budget contract (the atmosphere must receive exactly the
    # flux the surface/ocean tile exchanged).  Mutually exclusive with the
    # explicit SCM sfc_* prescriptions (a static None-ness check, not traced).
    if surface_flux is not None and any(
            s is not None for s in (sfc_wpthlp, sfc_wprtp, sfc_upwp, sfc_vpwp)):
        raise ValueError(
            "clubb_turbulence_prognostic: pass EITHER surface_flux (the "
            "driver's tiled bottom BC) OR the explicit sfc_* kinematic "
            "prescriptions, not both."
        )
    exner_sfc = exner_function(p_full[:, -1])

    def _sfc_bcs(rho_local):
        """Kinematic lower-BC ``(sfc_wpthlp, sfc_wprtp, sfc_upwp, sfc_vpwp)`` for
        the given near-surface density.  With an injected tiled DYNAMIC flux we
        convert it HERE — per (sub)step density ``rho_local[:, -1]`` — so the
        prescribed W/m^2 / Pa flux is conserved exactly across the sub-cycle (the
        EXACT inverse of clubb_step's internal bulk->kinematic map: shflx/lhflx
        positive UP; tau = -rho*Cd*|V|*u so u'w'_sfc = tau_x/rho is NEGATIVE for
        u>0 / downward drag, matching clubb_step's upwp_b = tau_x/rho).  Without
        an injected flux, pass the explicit SCM kinematic prescriptions through."""
        if surface_flux is None:
            return sfc_wpthlp, sfc_wprtp, sfc_upwp, sfc_vpwp
        tau_x_sf, tau_y_sf, shflx_sf, lhflx_sf, _ = surface_flux
        rho_s = rho_local[:, -1]
        return (
            shflx_sf / (rho_s * constants.c_pd * exner_sfc),   # w'thl' [K m/s]
            lhflx_sf / (rho_s * constants.L_v),                # w'rt'  [kg/kg m/s]
            tau_x_sf / rho_s,                                  # u'w'   [m^2/s^2]
            tau_y_sf / rho_s,                                  # v'w'   [m^2/s^2]
        )

    if n_sub == 1:
        du_dt, dv_dt, dT_dt, dq_v_dt, new_moments, diags = clubb_step(
            u, v, T, q_v, moments, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho, dt, config, *_sfc_bcs(rho))
        shflx, lhflx, ustar = diags["shflx"], diags["lhflx"], diags["ustar"]
        Kh_full = flip_vertical(diags["Kh_zt"])
    else:
        dt_sub = dt / n_sub
        tv_floor = config.T0 * 0.5

        def _sub(carry, _):
            u_c, v_c, T_c, q_c, m_c = carry
            # Density floor (only) guards a strictly-positive rho if q_c dips
            # slightly negative mid-cycle; q itself is NOT clipped — the host
            # applies the RAW integrated CLUBB tendency, identical to the n_sub=1
            # contract (positivity limiting is the host/moisture-fixer's job, not
            # folded into the physics tendency). clubb_step floors rt internally
            # (rt_tol), so a slightly-negative mean rtm is robust (cloud → 0).
            tv = jnp.maximum(virtual_temperature(T_c, q_c), tv_floor)
            rho_c = p_full / (constants.R_d * tv)
            # Re-derive the kinematic BC from the (constant) injected DYNAMIC flux
            # at THIS sub-step's density so the applied W/m^2 / Pa flux is exact.
            du, dv, dT, dq, m_new, diag = clubb_step(
                u_c, v_c, T_c, q_c, m_c, p_full, p_half, z_full, z_half,
                T_sfc, q_sfc, rho_c, dt_sub, config, *_sfc_bcs(rho_c))
            carry = (u_c + dt_sub * du, v_c + dt_sub * dv, T_c + dt_sub * dT,
                     q_c + dt_sub * dq, m_new)
            return carry, diag

        (u_f, v_f, T_f, q_f, new_moments), diag_stk = jax.lax.scan(
            _sub, (u, v, T, q_v, moments), xs=None, length=n_sub)
        du_dt = (u_f - u) / dt
        dv_dt = (v_f - v) / dt
        dT_dt = (T_f - T) / dt
        dq_v_dt = (q_f - q_v) / dt
        # Net surface exchange = sub-cycle-mean flux; Kh from the final sub-step.
        shflx = jnp.mean(diag_stk["shflx"], axis=0)
        lhflx = jnp.mean(diag_stk["lhflx"], axis=0)
        ustar = jnp.mean(diag_stk["ustar"], axis=0)
        Kh_full = flip_vertical(diag_stk["Kh_zt"][-1])

    # With an injected tiled flux, report the surface diagnostics as the INJECTED
    # dynamic values directly (Louis-equivalent), rather than clubb_step's
    # reconstruction-from-stress: the per-sub-step conversion already makes the
    # applied flux equal these, and it avoids clubb_step's ustar fourth-root floor
    # (a dead gradient at zero stress) and any sub-cycle-mean drift in the report.
    if surface_flux is not None:
        _, _, shflx, lhflx, ustar = surface_flux
    h_pbl = diagnose_pbl_height(T, q_v, u, v, p_full, z_full)
    output = TurbulenceOutput(
        du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, dq_v_dt=dq_v_dt,
        Km=Kh_full, Kh=Kh_full, shflx=shflx, lhflx=lhflx, ustar=ustar, h_pbl=h_pbl)
    return output, pack_clubb_moments(new_moments)


def integrate_clubb_column(
    u: jax.Array,
    v: jax.Array,
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    z_full: jax.Array,
    z_half: jax.Array,
    T_sfc: jax.Array,
    q_sfc: jax.Array,
    dt: float,
    nsteps: int,
    config: CLUBBConfig,
    moments: CLUBBMomentState | None = None,
    host_numerical_diffusion: float = 0.05,
    sfc_wpthlp: jax.Array | None = None,
    sfc_wprtp: jax.Array | None = None,
    sfc_upwp: jax.Array | None = None,
    sfc_vpwp: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, CLUBBMomentState, dict]:
    """Integrate a single-column prognostic CLUBB run for ``nsteps`` steps.

    A self-contained SCM-style driver: ``lax.scan`` over :func:`clubb_step`,
    carrying the full :class:`CLUBBMomentState` *and* the mean state
    ``(u, v, T, q_v)`` (forward-Euler updated by the CLUBB tendencies each step;
    ``rho`` is recomputed hydrostatically from the evolving ``T``/``q_v`` on the
    fixed pressure grid). This exercises the genuinely-prognostic higher-order
    moment closure end-to-end — the moments persist and evolve across steps,
    unlike the diagnostic phase-1 entry — and is the multi-step stability/AD
    test bed for the scheme. All column fields are top-down ``(ncol, nlev)``.

    **Host numerical diffusion.** In a coupled model CLUBB returns *tendencies*
    and the dynamical core advances + numerically diffuses the means; that
    diffusion damps grid-scale (2Δz) vertical noise. A *bare* single-column
    driver advances the means with CLUBB alone, so it must supply that stand-in
    itself — without it, a long near-dry weakly-stratified column grows
    grid-scale ``T`` noise and ``wp2`` (the iter-48 instability; root-caused iter
    49-51 to absent host diffusion, NOT a closure/conservation/port error — a
    tiny ``host_numerical_diffusion`` removes it entirely, ``wp2max`` 10.7→0.06).
    ``host_numerical_diffusion`` is a dimensionless 2nd-order vertical-diffusion
    coefficient (0 ⇒ bare CLUBB, exposes the noise; default 0.05 ⇒ a coupled-
    model-like stand-in). Applied in **flux form** so it conserves the
    column-summed means exactly (zero-flux top/bottom).

    **Prescribed surface fluxes** (``sfc_wpthlp``/``sfc_wprtp``/``sfc_upwp``/
    ``sfc_vpwp``, each ``(ncol,)`` or ``None``) are forwarded to :func:`clubb_step`
    unchanged (held constant across the run) — the standard way to drive a
    prescribed-flux LES/SCM case (BOMEX/DYCOMS/ARM). ``None`` ⇒ CLUBB's bulk
    surface formula from ``T_sfc``/``q_sfc`` (the default, back-compatible).

    ``moments`` defaults to a rest state (:func:`init_clubb_moments`).
    ``nsteps`` is a **static** Python int (the ``lax.scan`` length, fixed at trace
    time); jit callers close over it (it is not a traced argument).
    Returns the final ``(u, v, T, q_v, moments)`` and a dict of per-step stacked
    diagnostics (``cloud_frac``/``rcm``/``wpthvp``/``ustar``/...), shape
    ``(nsteps, ...)``. The returned ``moments`` means (rtm/thlm/um/vm) are kept
    consistent with the returned ``(u, v, T, q_v)``.
    """
    ncol, nlev = T.shape
    if moments is None:
        moments = init_clubb_moments(ncol, nlev, config, dtype=T.dtype)
    exner_td = exner_function(p_full)
    nu = host_numerical_diffusion
    # Safety floor for the recomputed virtual temperature so the prognostic
    # density stays strictly positive even if a long/dry SCM run drifts T low
    # (mirrors the shared compute_rho floor; finite-gradient via max).
    tv_floor = config.T0 * 0.5

    def _diffuse(f):
        """Flux-form 2nd-order vertical diffusion (zero-flux BCs → conserves
        sum(f) exactly): f[k] += flux[k] - flux[k-1], flux[k+1/2]=nu*(f[k+1]-f[k]).
        For ``nu <= 0.5`` the update is a convex combination of {f[k-1],f[k],f[k+1]}
        → monotone (stays within neighbour min/max), so it preserves positivity of
        a non-negative input AND the column sum."""
        flux = nu * (f[:, 1:] - f[:, :-1])           # (ncol, nlev-1) interfaces
        return f.at[:, :-1].add(flux).at[:, 1:].add(-flux)

    def _step(carry, _):
        u_c, v_c, T_c, q_c, m_c = carry
        tv = jnp.maximum(virtual_temperature(T_c, q_c), tv_floor)
        rho_c = p_full / (constants.R_d * tv)
        du, dv, dT, dq, m_new, diag = clubb_step(
            u_c, v_c, T_c, q_c, m_c, p_full, p_half, z_full, z_half,
            T_sfc, q_sfc, rho_c, dt, config,
            sfc_wpthlp, sfc_wprtp, sfc_upwp, sfc_vpwp)
        # Moisture positivity is enforced on the CLUBB-tendency update FIRST (the
        # physical moisture-fixer; a non-conservative source, as in any model),
        # THEN the conservative host-stand-in diffusion is applied LAST. Because
        # _diffuse is monotone for nu<=0.5, diffusing a non-negative field keeps it
        # non-negative — so q stays >= 0 AND its (clipped) column sum is conserved
        # by the diffusion (no spurious water creation by the diffusion itself).
        u_n = _diffuse(u_c + dt * du)
        v_n = _diffuse(v_c + dt * dv)
        T_n = _diffuse(T_c + dt * dT)
        q_n = _diffuse(jnp.maximum(q_c + dt * dq, 0.0))
        # Keep the carried CLUBBMomentState means consistent with the updated mean
        # state (they are reset from the column inside clubb_step each step, but a
        # consistent returned state matters for callers/restart inspection).
        m_new = m_new._replace(
            um=flip_vertical(u_n), vm=flip_vertical(v_n),
            thlm=flip_vertical(T_n / exner_td), rtm=flip_vertical(q_n))
        carry = (u_n, v_n, T_n, q_n, m_new)
        return carry, diag

    (u_f, v_f, T_f, q_f, m_f), diags = jax.lax.scan(
        _step, (u, v, T, q_v, moments), xs=None, length=nsteps)
    return u_f, v_f, T_f, q_f, m_f, diags


# ---------------------------------------------------------------------------
# CAM-default CLUBB model-flag values (reference table)
# ---------------------------------------------------------------------------
# This scheme implements ONLY the call tree selected by the CAM-default CLUBB
# model flags (namelist_defaults_cam.xml ``clubb_*`` base values, falling back
# to the CLUBB library defaults in model_flags.F90 where the namelist is
# silent — marked "absent->lib"). The flags are therefore NOT defined as
# runtime configuration: no other tree is implemented, and every former
# ``flags.l_*`` branch in the code is hardcoded to the value below (with a
# "CAM <flag> = <value>" comment at the site). Where the CAM namelist
# OVERRIDES the library default the entry is marked "lib X -> CAM Y".
# Integer flags use the enum codes from ``model_flags.F90``.
# ``tests/unit/test_clubb_config.py`` parses this table and checks it against
# the CAM namelist source when the CESM tree is present.
#
#
# ── Integer / enum flags ─────────────────────────────────────────────
#   iiPDF_type = 1   # ADG1                 (absent->lib)
#   ipdf_call_placement = 2   # ipdf_post_advance_fields
#   saturation_formula = 3   # flatau               (absent->lib)
#   penta_solve_method = 1   # lib 2 -> CAM 1
#   tridiag_solve_method = 1   # lib 2 -> CAM 1
#   grid_remap_method = 1   # lib 2(ppm) -> CAM 1
#   grid_adapt_in_time_method = 0   # no grid adaptation
#   fill_holes_type = 2   # sliding_window
#
# ── Logical flags — CAM namelist base values (⚠ several differ from lib) ──
#   l_use_precip_frac = True
#   l_predict_upwp_vpwp = False   # lib True -> CAM False
#   l_min_wp2_from_corr_wx = False
#   l_min_xp2_from_corr_wx = True
#   l_C2_cloud_frac = False
#   l_diffuse_rtm_and_thlm = False
#   l_stability_correct_Kh_N2_zm = False
#   l_calc_thlp2_rad = True
#   l_upwind_xpyp_ta = True
#   l_upwind_xm_ma = True
#   l_uv_nudge = False   # absent->lib
#   l_rtm_nudge = False
#   l_tke_aniso = True
#   l_vert_avg_closure = True   # lib False -> CAM True
#   l_trapezoidal_rule_zt = True   # lib False -> CAM True
#   l_trapezoidal_rule_zm = True   # lib False -> CAM True
#   l_call_pdf_closure_twice = True   # lib False -> CAM True
#   l_standard_term_ta = False
#   l_partial_upwind_wp3 = False
#   l_godunov_upwind_wpxp_ta = False
#   l_godunov_upwind_xpyp_ta = False
#   l_use_cloud_cover = True   # lib False -> CAM True
#   l_diagnose_correlations = False
#   l_calc_w_corr = False
#   l_const_Nc_in_cloud = False
#   l_fix_w_chi_eta_correlations = True
#   l_stability_correct_tau_zm = True   # lib False -> CAM True
#   l_damp_wp2_using_em = False   # lib True -> CAM False
#   l_do_expldiff_rtm_thlm = False
#   l_Lscale_plume_centered = False
#   l_diag_Lscale_from_tau = False   # lib True -> CAM False
#   l_use_C7_Richardson = False   # lib True -> CAM False
#   l_use_C11_Richardson = False
#   l_use_shear_Richardson = False
#   l_brunt_vaisala_freq_moist = False
#   l_use_thvm_in_bv_freq = False
#   l_rcm_supersat_adj = False   # lib True -> CAM False
#   l_damp_wp3_Skw_squared = False   # lib True -> CAM False
#   l_prescribed_avg_deltaz = False
#   l_lmm_stepping = False
#   l_e3sm_config = False
#   l_vary_convect_depth = False
#   l_use_tke_in_wp3_pr_turb_term = False   # lib True -> CAM False
#   l_use_tke_in_wp2_wp3_K_dfsn = False
#   l_smooth_Heaviside_tau_wpxp = False
#   l_enable_relaxed_clipping = False
#   l_mono_flux_lim_thlm = True
#   l_mono_flux_lim_rtm = True
#   l_mono_flux_lim_um = True
#   l_mono_flux_lim_vm = True
#   l_mono_flux_lim_spikefix = True
#   l_host_applies_sfc_fluxes = False   # absent->lib
#   l_wp2_fill_holes_tke = True   # absent->lib
#   l_add_dycore_grid = False
#   l_ascending_grid = False   # CAM namelist
#   l_c14_ml = False   # CAM namelist (neural C14, OFF)
#   l_intr_sfc_flux_smooth = False   # CAM namelist (clubb_intr smoothing)
