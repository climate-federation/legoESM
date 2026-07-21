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

from .registry import (
    LESCase,
    LESGrid,
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
        duration_hours=1.0,
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
        duration_hours=4.0,
        surface_theta_flux_K_m_s=-0.005,  # run_spectral_sbl --Q0 (GABLS1 0.25 K/hr cooling)
        geostrophic_wind_m_s=8.0,         # --Ug default
        sgs_variants=_SGS_SPREAD,         # stable anchor is in the D7 spread subset
        driver="run_spectral_sbl.py",
        reference="Beare et al. (2006) GABLS1 stable BL LES intercomparison",
        ci_marker="nightly",
        description="GABLS1 stably-stratified sheared Ekman SBL (Ug=8, f=1.39e-4); "
                    "surface cooling drives a low-level jet at the SBL top.",
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


def register_default_catalog(*, include_dry_grid: bool = True) -> list[LESCase]:
    """Register the anchor cases (and, by default, the provisional dry grid).

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
    return registered
