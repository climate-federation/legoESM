"""Default LESCase catalog for the LES-truth suite.

Two parts (LES_SUITE.md §5/§6):

* :data:`ANCHOR_CASES` — the four *published* reference setups the plane-LES
  drivers already implement, parameters taken verbatim from the driver defaults
  (``scripts/run/run_spectral_cbl.py``, ``run_spectral_sbl.py``,
  ``run_bomex_les.py``, ``run_dycoms_les.py``). These are real (``provisional_axis
  = False``) and anchor the dry/moist regimes against their intercomparison.
* :func:`dry_shear_buoyancy_grid` — the ``(w'θ'_s × |U_g|)`` sweep of the dry
  regime (the Q1/Q3 core, D8 = 4 fluxes × 3 winds). The axis *values* are
  physically-anchored PLACEHOLDERS (``provisional_axis = True``) until the gate-0
  Nieuwstadt-CBL run fixes the achievable resolution / span (§8/§9); the grid
  *structure* is real. Sheared-convective points (flux > 0, wind > 0) currently
  have no single driver — ``run_spectral_cbl`` is free-convection only — so wiring
  a unified dry driver (or adding ``--Ug`` to the CBL driver) is a follow-up of
  the harness step, tracked in the case ``description``.

:func:`register_default_catalog` populates the process-global registry.
"""
from __future__ import annotations

import dataclasses
import math

from legoesm import constants

from .registry import (
    LESCase,
    LESGrid,
    LESSetup,
    Regime,
    SgsClosure,
    register_case,
)

# SGS spread used for the D7 LES error bar (a 5-case subset runs all three;
# every other case runs the dynamic scale-dependent default only).
_SGS_DEFAULT: tuple[SgsClosure, ...] = ("lasd",)
_SGS_SPREAD: tuple[SgsClosure, ...] = ("lasd", "smagorinsky", "vreman")

# Canonical dry-regime production grid (96^3 over a 3.2 x 3.2 x 1.6 km box,
# dt 0.5 s) — the run_spectral_cbl default, D8's production resolution.
_DRY_GRID = LESGrid(nx=96, ny=96, nz=96, Lx_m=3200.0, Ly_m=3200.0, Lz_m=1600.0, dt_s=0.5)


# --- anchor cases (real published setups; parameters == driver defaults) -----
ANCHOR_CASES: tuple[LESCase, ...] = (
    LESCase(
        name="cbl_nieuwstadt",
        regime="dry_convective",
        core="spectral",
        grid=_DRY_GRID,
        duration_hours=2.0,              # gate-0 validation run length (2 h)
        surface_theta_flux_K_m_s=0.06,   # run_spectral_cbl --Q0 default
        geostrophic_wind_m_s=None,       # free convection (u_geo = 0)
        sgs_variants=_SGS_SPREAD,        # dry-convective anchor is in the D7 spread subset
        driver="run_spectral_cbl.py",
        reference="Nieuwstadt et al. (1993) convective BL LES intercomparison (CBL_N91)",
        ci_marker="nightly",
        description="Free-convective dry CBL, prescribed surface heat flux + capping "
                    "inversion; the gate-0 truth-core validation case.",
    ),
    LESCase(
        name="sbl_gabls1",
        regime="dry_stable",
        core="spectral",
        grid=LESGrid(nx=64, ny=64, nz=96, Lx_m=400.0, Ly_m=400.0, Lz_m=400.0, dt_s=0.1),
        duration_hours=9.0,              # Beare et al. (2006): 9 h, last hour analysed
        surface_theta_flux_K_m_s=None,    # flux comes from MOST under the cooling below
        geostrophic_wind_m_s=8.0,         # --Ug default
        surface_cooling_rate_K_per_h=0.25,  # Beare et al. (2006): T_s = 265 K − 0.25 K/h·t
        sgs_variants=_SGS_SPREAD,         # stable anchor is in the D7 spread subset
        driver="run_spectral_sbl.py",
        reference="Beare et al. (2006) GABLS1 stable BL LES intercomparison",
        ci_marker="nightly",
        description="GABLS1 stably-stratified sheared Ekman SBL (Ug=8, f=1.39e-4); "
                    "prescribed surface cooling 0.25 K/h with a Monin-Obukhov "
                    "surface layer drives a low-level jet at the SBL top.",
    ),
    LESCase(
        name="bomex_cu",
        regime="shallow_cumulus",
        core="spectral",
        grid=LESGrid(nx=64, ny=64, nz=75, Lx_m=6400.0, Ly_m=6400.0, Lz_m=3000.0, dt_s=1.0),
        duration_hours=6.0,
        surface_theta_flux_K_m_s=None,   # prescribed moist surface fluxes (driver-internal)
        geostrophic_wind_m_s=8.75,       # trade-wind u_g ~ -8.75 m/s
        sgs_variants=_SGS_DEFAULT,
        driver="run_bomex_les.py",
        reference="Siebesma et al. (2003) BOMEX shallow-cumulus LES intercomparison",
        ci_marker="manual_only",
        description="Non-precipitating shallow trade cumulus; cloud fraction from the "
                    "moist path. Moist confound handled per LES_SUITE.md D9.",
    ),
    LESCase(
        name="dycoms_rf01_sc",
        regime="stratocumulus",
        core="spectral",
        grid=LESGrid(nx=96, ny=96, nz=192, Lx_m=3360.0, Ly_m=3360.0, Lz_m=1500.0, dt_s=0.5),
        duration_hours=4.0,
        surface_theta_flux_K_m_s=None,   # interactive/prescribed moist surface (driver-internal)
        geostrophic_wind_m_s=7.0,        # DYCOMS-II RF01 U_g=(7, -5.5); dominant u-component
        sgs_variants=_SGS_DEFAULT,
        driver="run_dycoms_les.py",
        reference="Stevens et al. (2005) DYCOMS-II RF01 stratocumulus LES intercomparison",
        ci_marker="manual_only",
        description="Nocturnal marine stratocumulus with a sharp cloud-top inversion; "
                    "cloud response is the main moist closure differentiator.",
    ),
)


def _classify_dry_regime(flux_K_m_s: float) -> Regime:
    """Regime of a dry grid point from its surface heat flux sign (D2 axes)."""
    if flux_K_m_s > 1.0e-4:
        return "dry_convective"
    if flux_K_m_s < -1.0e-4:
        return "dry_stable"
    return "dry_neutral"


# PROVISIONAL dry-grid axes (pending gate-0). Fluxes span stable -> free-convective;
# winds span free-convection -> shear-dominated (D8: 4 x 3 = 12 points).
_DEFAULT_FLUXES_K_M_S: tuple[float, ...] = (-0.01, 0.01, 0.03, 0.06)
_DEFAULT_WINDS_M_S: tuple[float, ...] = (0.0, 5.0, 10.0)


def dry_shear_buoyancy_grid(
    fluxes_K_m_s: tuple[float, ...] = _DEFAULT_FLUXES_K_M_S,
    winds_m_s: tuple[float, ...] = _DEFAULT_WINDS_M_S,
    *,
    grid: LESGrid = _DRY_GRID,
    duration_hours: float = 4.0,
    ci_marker: str = "manual_only",
) -> list[LESCase]:
    """Generate the dry ``(w'θ'_s × |U_g|)`` grid as provisional LESCases.

    One case per (flux, wind) pair. Regime is classified from the flux sign; the
    driver is the nearest existing one (stable -> SBL, else CBL). Axis values are
    placeholders (``provisional_axis=True``) until gate-0 calibration. Sheared-
    convective points (flux>0, wind>0) note the missing unified driver.
    """
    if not fluxes_K_m_s or not winds_m_s:
        raise ValueError("fluxes_K_m_s and winds_m_s must both be non-empty")
    cases: list[LESCase] = []
    for i, flux in enumerate(fluxes_K_m_s):
        for j, wind in enumerate(winds_m_s):
            regime = _classify_dry_regime(flux)
            stable = regime == "dry_stable"
            driver = "run_spectral_sbl.py" if stable else "run_spectral_cbl.py"
            needs_wiring = (not stable) and wind > 0.0  # CBL driver is free-convection only
            note = (
                " NOTE: sheared-convective point — run_spectral_cbl is free-convection "
                "only; needs a unified dry driver / --Ug (harness follow-up)."
                if needs_wiring else ""
            )
            cases.append(
                LESCase(
                    name=f"dry_grid_b{i}s{j}",
                    regime=regime,
                    core="spectral",
                    grid=grid,
                    duration_hours=duration_hours,
                    surface_theta_flux_K_m_s=float(flux),
                    geostrophic_wind_m_s=(float(wind) if wind > 0.0 else None),
                    sgs_variants=_SGS_DEFAULT,
                    driver=driver,
                    reference="LES_SUITE.md D2 dry (buoyancy x shear) grid "
                              "(anchors: Nieuwstadt CBL, GABLS1 SBL)",
                    ci_marker=ci_marker,  # type: ignore[arg-type]  (validated in register_case)
                    description=(
                        f"Dry-regime grid point: w'θ'_s={flux:+.3f} K m/s, "
                        f"|U_g|={wind:.1f} m/s.{note}"
                    ),
                    provisional_axis=True,
                )
            )
    return cases


# --- published dry cases with a full LESSetup (2026-10-15 campaign) ----------
# Every physical parameter is the paper's. Values the paper does not fix (or that
# the spectral core cannot reproduce as published: a radiation top boundary, an
# adaptive-CFL dt) are PROPOSED and listed in ``setup.pending`` — the driver
# refuses a production artifact until the user approves them (--smoke runs only).
def _lat_for_f(f_cor: float) -> float:
    """Latitude [deg] whose ``2Ω sin φ`` is the paper's Coriolis parameter."""
    return math.degrees(math.asin(f_cor / (2.0 * constants.Omega)))


_GABLS1_F = 1.39e-4                 # Beare et al. (2006) / Sullivan 2016 / Huang 2013
_SUITE_SPONGE_TAU_S = 50.0          # the suite's SBL sponge timescale (proposal only)
_PENDING_DT = "dt: PROPOSED {dt} s fixed (paper: adaptive CFL / not stated)"
_PENDING_SEED = ("seed noise: PROPOSED {amp} K θ below {top} m (paper: {paper})")

_CF_REF = ("Conzemius & Fedorovich (2006, JAS 63, 1151); matrix and setup from "
           "Conzemius & Fedorovich, AMS 16th BLT symposium preprint 5.6 (24 runs, "
           "flux x lapse rate x shear)")
_CF_GRID = LESGrid(nx=256, ny=256, nz=80, Lx_m=5120.0, Ly_m=5120.0, Lz_m=1600.0,
                   dt_s=0.5)
_CF_GEO = {
    "ns": ((0.0, 0.0, 0.0),),                         # no geostrophic wind
    "gc": ((0.0, 20.0, 0.0),),                        # 20 m/s, constant with height
    "gs": ((0.0, 0.0, 0.0), (1600.0, 20.0, 0.0)),     # 0 at surface -> 20 m/s at top
}


def _cf_case(q_K_m_s, gamma_K_m, shear, hours, extra_pending=()):
    """One Conzemius–Fedorovich point: θ = 300 K + γz from the surface (initial CBL
    depth 0), geostrophic initial wind, 40°N, z0 = 0.01 m, constant surface flux,
    MO-similarity surface layer, sponge in the top 20 %."""
    tag = f"cf06_q{round(q_K_m_s * 100):02d}_g{round(gamma_K_m * 1000):02d}_{shear}"
    return LESCase(
        name=tag, regime="dry_convective", core="spectral", grid=_CF_GRID,
        duration_hours=hours, surface_theta_flux_K_m_s=q_K_m_s,
        geostrophic_wind_m_s=(None if shear == "ns" else 20.0),
        sgs_variants=_SGS_DEFAULT, driver="run_les_suite.py", reference=_CF_REF,
        ci_marker="manual_only",
        description=f"CBL growth into dθ/dz={gamma_K_m} K/m, w'θ'_s={q_K_m_s} K m/s, "
                    f"shear={shear} (ns none / gc constant 20 m/s / gs 0->20 m/s).",
        setup=LESSetup(
            lat_deg=40.0, z0_m=0.01, theta_ref_K=300.0, perturb_K=0.1,
            perturb_top_m=200.0,
            theta_knots=((0.0, 300.0), (1600.0, 300.0 + gamma_K_m * 1600.0)),
            geostrophic_knots=_CF_GEO[shear], surface_drag="most",
            sponge_bottom_m=1280.0, sponge_tau_s=_SUITE_SPONGE_TAU_S,
            pending=(
                _PENDING_DT.format(dt=0.5),
                f"sponge: top 20 % per paper; timescale PROPOSED {_SUITE_SPONGE_TAU_S}"
                " s (paper: not stated)",
                _PENDING_SEED.format(amp=0.1, top=200, paper="'2 K' in preprint 3.1, "
                                     "form not stated"),
                f"duration: PROPOSED {hours} h = zero-order-jump time for z_i to reach "
                "1000 m, z_i^2 = 2.8 Q t / γ (paper: run until z_i ≈ 1000 m)",
                "domain: 5.12 x 5.12 x 1.6 km at 20 m (preprint 5.6); preprint 3.1 "
                "says 'minimum 6.4 x 6.4 x 1.6 km'",
            ) + tuple(extra_pending)))


def _gabls1_family_setup(Lz_m, sponge, pending):
    """GABLS1 physics as Sullivan et al. (2016) / Huang & Bou-Zeid (2013) state it:
    θ = 265 K to 100 m then +0.01 K/m (sharp), U_g = (8, 0) m/s, f = 1.39e-4 s⁻¹,
    z0m = z0h = 0.1 m, still-air θ0 = 265 K."""
    return LESSetup(
        lat_deg=_lat_for_f(_GABLS1_F), z0_m=0.1, theta_ref_K=265.0, perturb_K=0.1,
        perturb_top_m=50.0,
        theta_knots=((0.0, 265.0), (100.0, 265.0), (Lz_m, 265.0 + 0.01 * (Lz_m - 100.0))),
        geostrophic_knots=((0.0, 8.0, 0.0),), surface_drag="most",
        sponge_bottom_m=None if sponge is None else sponge[0],
        sponge_tau_s=None if sponge is None else sponge[1],
        pending=pending)


_SULLIVAN16_REF = ("Sullivan, McWilliams, Weil, Patton & Fernando (2016, JAS 73, "
                   "1815), doi:10.1175/JAS-D-15-0339.1")
_HUANG13_REF = ("Huang & Bou-Zeid (2013, JAS 70, 1513), "
                "doi:10.1175/JAS-D-12-0167.1")


def _sullivan16_case(rate, grid, extra_pending=()):
    sponge = (300.0, _SUITE_SPONGE_TAU_S)
    return LESCase(
        name=f"sullivan16_cr{round(rate * 100):03d}", regime="dry_stable",
        core="spectral", grid=grid, duration_hours=9.0, surface_theta_flux_K_m_s=None,
        geostrophic_wind_m_s=8.0, sgs_variants=_SGS_DEFAULT, driver="run_les_suite.py",
        reference=_SULLIVAN16_REF, ci_marker="manual_only",
        surface_cooling_rate_K_per_h=rate,
        description=f"GABLS1 with surface cooling {rate} K/h, 400^3 m, 9 h.",
        setup=_gabls1_family_setup(400.0, sponge, (
            _PENDING_DT.format(dt=grid.dt_s),
            f"top boundary: PROPOSED Rayleigh sponge above 300 m, "
            f"{_SUITE_SPONGE_TAU_S} s (paper: radiation condition, not available in "
            "the spectral core)",
            _PENDING_SEED.format(amp=0.1, top=50, paper="'small random perturbations "
                                 "in temperature near the surface' + an SGS-energy "
                                 "profile this core has no equivalent of"),
        ) + tuple(extra_pending)))


def _huang13_case(rate, extra_pending=()):
    return LESCase(
        name=f"huang13_cr{round(rate * 100):03d}", regime="dry_stable",
        core="spectral",
        grid=LESGrid(nx=162, ny=162, nz=160, Lx_m=800.0, Ly_m=800.0, Lz_m=400.0,
                     dt_s=0.05),
        duration_hours=10.0, surface_theta_flux_K_m_s=None, geostrophic_wind_m_s=8.0,
        sgs_variants=_SGS_DEFAULT, driver="run_les_suite.py", reference=_HUANG13_REF,
        ci_marker="manual_only", surface_cooling_rate_K_per_h=rate,
        description=f"GABLS1 with surface cooling {rate} K/h, 800x800x400 m, 10 h "
                    "(statistics over the last 3 h).",
        setup=_gabls1_family_setup(400.0, (300.0, _SUITE_SPONGE_TAU_S), (
            _PENDING_DT.format(dt=0.05),
            "grid: PROPOSED 162x162x160 (5 x 5 x 2.5 m) for all 10 h; the paper ran "
            "80^3 for 0-6 h then interpolated to 162x162x160 for 6-10 h, a regrid "
            "this driver does not do",
            f"top boundary: Rayleigh sponge above 300 m, {_SUITE_SPONGE_TAU_S} s, as "
            "the suite's GABLS1 (user 2026-10-15; paper: none, stress-free — the "
            "suite's GABLS1 runs blew up at ~0.9 h without one)",
            _PENDING_SEED.format(amp=0.1, top=50, paper="not stated"),
        ) + tuple(extra_pending)))


def _dephy_case(name, regime, dephy, grid, hours, lat, z0, z0h, theta_ref, reference,
                description, pending, sponge=None):
    return LESCase(
        name=name, regime=regime, core="spectral", grid=grid, duration_hours=hours,
        surface_theta_flux_K_m_s=None, geostrophic_wind_m_s=None,
        sgs_variants=_SGS_DEFAULT, driver="run_les_suite.py", reference=reference,
        ci_marker="manual_only", description=description, held_out=True,
        setup=LESSetup(
            lat_deg=lat, z0_m=z0, z0h_m=z0h, theta_ref_K=theta_ref, perturb_K=0.1,
            perturb_top_m=100.0, dephy_case=dephy,
            sponge_bottom_m=None if sponge is None else sponge[0],
            sponge_tau_s=None if sponge is None else sponge[1],
            pending=pending))


_AYOTTE_THETA0 = {"00sc": 300.5, "00wc": 300.3, "03sc": 300.3, "05sc": 300.5,
                  "05wc": 300.2, "24sc": 301.1}   # θ at z=0 in each DEPHY file
_AYOTTE_FLUX = {"00sc": 0.0, "00wc": 0.0, "03sc": 0.03, "05sc": 0.05, "05wc": 0.05,
                "24sc": 0.24}                     # the original kinematic fluxes


def _ayotte_case(sub):
    grid = LESGrid(nx=128, ny=128, nz=120, Lx_m=5120.0, Ly_m=5120.0, Lz_m=2400.0,
                   dt_s=0.5)
    return _dephy_case(
        name=f"ayotte96_{sub}", regime="dry_neutral" if sub.startswith("00")
        else "dry_convective", dephy=f"ayotte_{sub}", grid=grid, hours=7.0,
        lat=45.0, z0=0.16, z0h=None, theta_ref=_AYOTTE_THETA0[sub],
        reference="Ayotte et al. (1996, BLM 79, 131); DEPHY-SCM AYOTTE/"
                  f"{sub.upper()}",
        description=f"Ayotte {sub.upper()}: U_g = 15 m/s, w'θ'_s = "
                    f"{_AYOTTE_FLUX[sub]} K m/s, "
                    f"{'weak' if sub.endswith('wc') else 'strong'} capping inversion.",
        sponge=(1920.0, _SUITE_SPONGE_TAU_S),
        pending=(
            "grid: PROPOSED 128x128x120 over 5.12 x 5.12 x 2.4 km (40 x 40 x 20 m); "
            "the DEPHY file gives no LES grid and Ayotte et al. (1996) was not "
            "available to check",
            _PENDING_DT.format(dt=0.5),
            f"sponge: PROPOSED top 20 % (1920 m), {_SUITE_SPONGE_TAU_S} s (not stated)",
            f"θ_ref: PROPOSED {_AYOTTE_THETA0[sub]} K = the file's surface θ",
            f"surface flux: the DEPHY loader turns hfss into "
            f"{_AYOTTE_FLUX[sub] * 1125.4 / 1164.5:.4f} K m/s (ρ from the file's "
            f"300 K, 1000 hPa surface); DEPHY made hfss from the ORIGINAL "
            f"{_AYOTTE_FLUX[sub]} K m/s with ρc_p = 1125.4 — 3.5 % apart",
            _PENDING_SEED.format(amp=0.1, top=100, paper="not stated"),
        ))


PUBLISHED_CASES: tuple[LESCase, ...] = (
    _cf_case(0.10, 0.003, "ns", 2.98),
    _cf_case(0.30, 0.001, "ns", 0.33),
    _cf_case(0.03, 0.010, "ns", 33.0, extra_pending=(
        "case: NOT RUN in the paper (preprint 5.6: the 0.03 K m/s x 0.010 K/m "
        "combination was skipped as too slow); keep, replace, or drop?",)),
    _cf_case(0.10, 0.003, "gc", 2.98),
    _cf_case(0.10, 0.003, "gs", 2.98),
    LESCase(
        name="sullivan_patton11_d", regime="dry_convective", core="spectral",
        grid=LESGrid(nx=256, ny=256, nz=256, Lx_m=5120.0, Ly_m=5120.0, Lz_m=2048.0,
                     dt_s=0.5),
        duration_hours=3.70, surface_theta_flux_K_m_s=0.24, geostrophic_wind_m_s=1.0,
        sgs_variants=_SGS_DEFAULT, driver="run_les_suite.py",
        reference="Sullivan & Patton (2011, JAS 68, 2395), "
                  "doi:10.1175/JAS-D-10-05010.1, run D",
        ci_marker="manual_only", held_out=True,
        description="Weakly sheared CBL under an 8 K inversion: run D (20 x 20 x 8 m), "
                    "the coarsest of the paper's meshes at least as fine as the CBL "
                    "anchor in every direction.",
        setup=LESSetup(
            lat_deg=_lat_for_f(1.0e-4), z0_m=0.1, theta_ref_K=300.0, perturb_K=0.1,
            perturb_top_m=100.0,
            theta_knots=((0.0, 300.0), (974.0, 300.0), (1074.0, 308.0),
                         (2048.0, 308.0 + 0.003 * (2048.0 - 1074.0))),
            geostrophic_knots=((0.0, 1.0, 0.0),), surface_drag="most",
            sponge_bottom_m=1638.4, sponge_tau_s=_SUITE_SPONGE_TAU_S,
            pending=(
                _PENDING_DT.format(dt=0.5),
                "duration: PROPOSED 3.70 h = 25 z_i/w_* with z_i = 1092 m, w_* = 2.05 "
                "m/s (paper Table 2, run D; 'about 25 large-eddy turnover times')",
                f"top boundary: PROPOSED Rayleigh sponge in the top 20 %, "
                f"{_SUITE_SPONGE_TAU_S} s (paper: radiation condition)",
                "θ_ref: PROPOSED 300 K (paper writes β = g/θ0 without the value)",
                _PENDING_SEED.format(amp=0.1, top=100, paper="'small random seed "
                                     "perturbations in temperature near the "
                                     "surface'"),
            ))),
    _sullivan16_case(0.25, LESGrid(nx=200, ny=200, nz=200, Lx_m=400.0, Ly_m=400.0,
                                   Lz_m=400.0, dt_s=0.05)),
    _sullivan16_case(1.0, LESGrid(nx=200, ny=200, nz=200, Lx_m=400.0, Ly_m=400.0,
                                  Lz_m=400.0, dt_s=0.05), extra_pending=(
        "grid: the paper ran 1.0 K/h ONLY on 1024^3 (0.39 m, ~900k steps); PROPOSED "
        "200^3 (2 m), the paper's grid for 0.25 K/h — not a published configuration",)),
    _huang13_case(1.0),
    _huang13_case(2.5),
    _dephy_case(
        name="gabls4_e2", regime="dry_stable", dephy="gabls4_stage3",
        grid=LESGrid(nx=200, ny=200, nz=500, Lx_m=1000.0, Ly_m=1000.0, Lz_m=1000.0,
                     dt_s=0.3),
        hours=24.0, lat=-75.1, z0=0.001, z0h=0.0001, theta_ref=271.3,
        reference="Couvreux et al. (2020, BLM 176, 369), "
                  "doi:10.1007/s10546-020-00539-4, experiment 2",
        description="GABLS4 stage 3, experiment 2: Dome C diurnal cycle, prescribed "
                    "surface temperature, z0 = 1e-3 m, z0h = 1e-4 m, 24 h from 00 UTC.",
        sponge=(700.0, 1000.0),
        pending=(
            "grid: PROPOSED 200x200x500 (5 x 5 x 2 m, the protocol's reference); the "
            "coarsest participant ran 10 x 10 x 5 m",
            "dt: 0.3 s ≈ 0.8 of the CFL limit (CFL 0.26 at 0.1 s at t0; user "
            "2026-10-15: raise from 0.1 s)",
            "sponge: above 700 m per paper; timescale PROPOSED 1000 s (Meso-NH's "
            "K = 1e-3 s^-1 in the paper's Table 2)",
            "θ_ref: PROPOSED 271.3 K = the file's surface θ at t0",
            "experiment: PROPOSED 2 (the DEPHY file's roughness); experiment 3 is "
            "the 10 h run from 10 UTC on a 250-500 m domain",
            _PENDING_SEED.format(amp=0.1, top=100, paper="not stated"),
        )),
    *(_ayotte_case(s) for s in ("00sc", "00wc", "03sc", "05sc", "05wc", "24sc")),
)
# The driver's own SGS closure is the suite's convention, not any paper's.
_PENDING_CLOSURE = ("SGS closure + numerics: PROPOSED the suite's (run_les_suite): "
                    "turbulent Prandtl 1.0; stable cases add background viscosity "
                    "0.01 m2/s and static Smagorinsky C_s 0.18; RK3, 3/2-rule "
                    "zero-padding dealiasing, no SGS buoyancy term (paper: own "
                    "closure/numerics)")
# User 2026-10-15: every proposed value above is APPROVED (Huang 2013 with the
# GABLS1 sponge instead of none). They move to ``setup.approved`` (artifact
# provenance); only what is still open stays pending.
_STILL_PENDING = {
    "gabls4_e2": ("dt 0.3 s: validation against 0.1 s over the first 1 h (mean "
                  "profiles + fluxes vs σ_LES) not yet done",),
}
PUBLISHED_CASES = tuple(
    dataclasses.replace(c, setup=dataclasses.replace(
        c.setup, approved=tuple(f"APPROVED 2026-10-15 (was: {p})"
                                for p in c.setup.pending + (_PENDING_CLOSURE,)),
        pending=_STILL_PENDING.get(c.name, ())))
    for c in PUBLISHED_CASES)

_PEDERSEN14_REF = ("Pedersen, Gryning & Kelly (2014, BLM 153, 43), "
                   "doi:10.1007/s10546-014-9937-z")
_PEDERSEN14_GAMMA = {"n01": 0.001, "n02": 0.003, "n03": 0.006, "n04": 0.010}


def _pedersen14_case(run):
    """Pedersen et al. (2014) neutral inversion-capped Ekman layer (their "neutral
    cases" n01-n04, Table 1): θ = 290 K + γz from the surface, u = G = (10, 0) m/s
    everywhere at t0, f = 1e-4 s⁻¹, z0 = 0.01 m, zero surface heat flux, 3 x 3 x
    2 km at 256³, 24 h (n02: 32 h)."""
    gamma = _PEDERSEN14_GAMMA[run]
    hours = 32.0 if run == "n02" else 24.0
    grid = LESGrid(nx=256, ny=256, nz=256, Lx_m=3000.0, Ly_m=3000.0, Lz_m=2000.0,
                   dt_s=0.5)
    return LESCase(
        name=f"ekman_pedersen14_{run}", regime="dry_neutral", core="spectral",
        grid=grid, duration_hours=hours, surface_theta_flux_K_m_s=0.0,
        geostrophic_wind_m_s=10.0, sgs_variants=_SGS_DEFAULT,
        driver="run_les_suite.py", reference=f"{_PEDERSEN14_REF}, run {run}",
        ci_marker="manual_only", held_out=True,
        description=f"Neutral Ekman layer capped by dθ/dz = {gamma} K/m, G = 10 m/s, "
                    f"{hours:g} h (Pedersen et al. 2014 {run}).",
        setup=LESSetup(
            lat_deg=_lat_for_f(1.0e-4), z0_m=0.01, theta_ref_K=290.0, perturb_K=0.1,
            perturb_top_m=50.0,
            theta_knots=((0.0, 290.0), (2000.0, 290.0 + gamma * 2000.0)),
            geostrophic_knots=((0.0, 10.0, 0.0),),
            wind_init_knots=((0.0, 10.0, 0.0),), surface_drag="most",
            sponge_bottom_m=1600.0, sponge_tau_s=_SUITE_SPONGE_TAU_S,
            pending=(
                "dt: PROPOSED 0.5 s fixed (paper: ~0.92 s mean, 500 steps ~ 460 s; "
                "at 0.92 s the advective CFL bound would exceed 0.8 at |u| ~ 11 m/s)",
                f"top boundary: PROPOSED Rayleigh sponge in the top 20 % (1600 m), "
                f"{_SUITE_SPONGE_TAU_S} s (paper: Klemp & Durran radiation condition)",
                "θ_ref: PROPOSED 290 K = the surface θ (paper: θ0 not stated)",
                _PENDING_SEED.format(amp=0.1, top=50, paper="small divergence-free "
                                     "u, v, θ perturbations below 50 m, amplitude "
                                     "not stated"),
                _PENDING_CLOSURE,
            )))


# Added 2026-10-15 after the approval above: their own values are still pending.
PUBLISHED_CASES = PUBLISHED_CASES + tuple(
    _pedersen14_case(r) for r in ("n01", "n02", "n03", "n04"))


def register_default_catalog(*, include_dry_grid: bool = True,
                             include_published: bool = True) -> list[LESCase]:
    """Register the anchor cases, by default the provisional dry grid, and by
    default the published full-setup cases (:data:`PUBLISHED_CASES`).

    Returns the registered cases. Idempotency is the caller's responsibility —
    :func:`~legoesm.atmosphere.les_suite.registry.register_case` raises on a
    duplicate name, so call :func:`~...registry.clear_registry` first if
    re-registering (tests do)."""
    registered: list[LESCase] = []
    for case in ANCHOR_CASES:
        registered.append(register_case(case))
    if include_dry_grid:
        for case in dry_shear_buoyancy_grid():
            registered.append(register_case(case))
    if include_published:
        for case in PUBLISHED_CASES:
            registered.append(register_case(case))
    return registered
