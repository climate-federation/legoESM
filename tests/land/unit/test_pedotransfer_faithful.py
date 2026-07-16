"""Cosby (1984) pedotransfer ORACLE-FAITHFULNESS tests.

``land/pedotransfer.py`` maps soil texture (% sand / clay) to Clapp-Hornberger
(1978) hydraulic parameters via the Cosby et al. (1984) regressions as used by
CLM4 (Tech Note eqs. 7.82, 7.84, 7.87, 7.90).  The existing
``tests/land/test_pedotransfer.py`` is behavioral — it recomputes theta_sat and b
inline and only range-/sign-checks psi_sat and K_sat.

These pin ALL FOUR closed forms to round-off (rel 1e-9) against an independent
scalar reimplementation and canary the regression coefficients against
``CosbyPedotransferConfig``.  The oracle literals are transcribed from the on-disk
gSAM SLM reference (``SLM/slm_vars.f90:1227-1236``, the identical Cosby fit); the
two departures are documented:

  * theta_sat = 0.489 - 0.00126 %sand                              (CLM4 7.82)
  * b_ch      = 2.91  + 0.159   %clay                              (CLM4 7.84)
  * psi_sat   = -(10 * 10^(1.88 - 0.0131 %sand)) mm -> m           (CLM4 7.87)
  * K_sat     = (0.0070556 * 10^(-0.884 + 0.0153 %sand)) mm/s -> m/s (CLM4 7.90)

Departures (documented, not bugs): legoesm implements the RAW Cosby/CLM4 psi_sat
(no cap), whereas gSAM floors its MAGNITUDE at 150 mm (``min(-150, raw)``, which
binds for HIGH sand); and legoesm uses CLM4's rounded K_sat prefactor 0.0070556
where gSAM uses the exact 25.4/3600 (inch/hr -> mm/s), differing by ~6.3e-6.

Independence: the regression coefficients are local ``_O_*`` literals (transcribed
from gSAM slm_vars.f90) canaried against ``CosbyPedotransferConfig``
(config == ``_O_*`` == value).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

_ENTRY_X64 = jax.config.read("jax_enable_x64")


@pytest.fixture(autouse=True)
def _force_x64():
    """Per-test float64 for the rel-1e-9 pins; restore the process-entry state in
    finally so selecting a single test never leaks x64 into another module."""
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", _ENTRY_X64)


from legoesm.land.pedotransfer import (                              # noqa: E402
    cosby_hydraulic_params, soil_hydraulics_config_from_texture,
    CosbyPedotransferConfig,
)

# Independent Cosby (1984) / CLM4 oracle literals (canaried below).
_O_THETA_INT = 0.489      # 7.82 intercept
_O_THETA_SAND = 0.00126   # 7.82 sand slope
_O_B_INT = 2.91           # 7.84 intercept
_O_B_CLAY = 0.159         # 7.84 clay slope
_O_PSI_INT = 1.88         # 7.87 base-10 exponent intercept
_O_PSI_SAND = 0.0131      # 7.87 base-10 exponent sand slope
_O_PSI_COEFF_MM = 10.0    # 7.87 |psi_sat| prefactor [mm]
_O_KSAT_INT = -0.884      # 7.90 base-10 exponent intercept
_O_KSAT_SAND = 0.0153     # 7.90 base-10 exponent sand slope
_O_KSAT_COEFF_MM_S = 0.0070556   # 7.90 K_sat prefactor [mm/s] (CLM4 rounding of 25.4/3600)
_MM_TO_M = 1.0e-3


def _cosby_oracle(sand, clay):
    theta_sat = _O_THETA_INT - _O_THETA_SAND * sand
    b_ch = _O_B_INT + _O_B_CLAY * clay
    psi_sat = -(_O_PSI_COEFF_MM * 10.0 ** (_O_PSI_INT - _O_PSI_SAND * sand)) * _MM_TO_M
    K_sat = (_O_KSAT_COEFF_MM_S * 10.0 ** (_O_KSAT_INT + _O_KSAT_SAND * sand)) * _MM_TO_M
    return theta_sat, psi_sat, b_ch, K_sat


def _a(x):
    return jnp.array(float(x))


_TEXTURES = [(40.0, 20.0), (90.0, 5.0), (10.0, 60.0), (0.0, 0.0), (100.0, 0.0)]


# --- form pins -----------------------------------------------------------------

@pytest.mark.parametrize("sand,clay", _TEXTURES)
def test_cosby_matches_clm4_forms(sand, clay):
    """All four Cosby/CLM4 forms match the independent oracle to round-off."""
    p = cosby_hydraulic_params(_a(sand), _a(clay))
    ts, psi, b, ks = _cosby_oracle(sand, clay)
    assert float(p.theta_sat) == pytest.approx(ts, rel=1e-9, abs=0.0)
    assert float(p.psi_sat) == pytest.approx(psi, rel=1e-9, abs=0.0)
    assert float(p.b_ch) == pytest.approx(b, rel=1e-9, abs=0.0)
    assert float(p.K_sat) == pytest.approx(ks, rel=1e-9, abs=0.0)


def test_cosby_constants_match_config_and_transcribed_literals():
    """The config defaults equal the independent oracle literals (transcribed from
    gSAM slm_vars.f90:1227-1236).  This canaries config == _O_* == value; it does
    NOT parse the Fortran — the _O_* values are the manually-transcribed oracle.
    The K prefactor is additionally checked against gSAM's exact 25.4/3600."""
    c = CosbyPedotransferConfig()
    assert c.theta_sat_intercept == _O_THETA_INT == 0.489
    assert c.theta_sat_sand == _O_THETA_SAND == 0.00126
    assert c.b_intercept == _O_B_INT == 2.91
    assert c.b_clay == _O_B_CLAY == 0.159
    assert c.psi_sat_intercept == _O_PSI_INT == 1.88
    assert c.psi_sat_sand == _O_PSI_SAND == 0.0131
    assert c.psi_sat_coeff_mm == _O_PSI_COEFF_MM == 10.0
    assert c.ksat_intercept == _O_KSAT_INT == -0.884
    assert c.ksat_sand == _O_KSAT_SAND == 0.0153
    assert c.ksat_coeff_mm_s == _O_KSAT_COEFF_MM_S == 0.0070556
    # gSAM slm_vars.f90 uses the EXACT 25.4/3600 [mm/s per inch/hr]; CLM4/legoesm
    # round it to 0.0070556 — a ~6.3e-6 relative departure (documented).
    assert c.ksat_coeff_mm_s == pytest.approx(25.4 / 3600.0, rel=1e-5)
    assert c.ksat_coeff_mm_s != 25.4 / 3600.0                  # rounded, not exact
    assert abs(c.ksat_coeff_mm_s - 25.4 / 3600.0) / (25.4 / 3600.0) == pytest.approx(6.3e-6, rel=0.05)


# --- units + non-vacuity -------------------------------------------------------

def test_cosby_unit_conversions_mm_to_m():
    """psi_sat and K_sat carry the mm->m / (mm/s)->(m/s) 1e-3 factor: the returned
    values are in metres and m/s, i.e. 1e-3 * the raw-mm form."""
    sand, clay = 40.0, 20.0
    p = cosby_hydraulic_params(_a(sand), _a(clay))
    psi_mm = -(_O_PSI_COEFF_MM * 10.0 ** (_O_PSI_INT - _O_PSI_SAND * sand))
    ksat_mm_s = _O_KSAT_COEFF_MM_S * 10.0 ** (_O_KSAT_INT + _O_KSAT_SAND * sand)
    assert float(p.psi_sat) == pytest.approx(psi_mm * _MM_TO_M, rel=1e-9, abs=0.0)
    assert float(p.K_sat) == pytest.approx(ksat_mm_s * _MM_TO_M, rel=1e-9, abs=0.0)


def test_cosby_texture_monotonicity():
    """Non-vacuity: theta_sat and K_sat decrease/increase with sand; b and |psi_sat|
    respond to clay/sand as the regressions dictate."""
    sandy = cosby_hydraulic_params(_a(90.0), _a(5.0))
    clayey = cosby_hydraulic_params(_a(10.0), _a(60.0))
    assert float(sandy.theta_sat) < float(clayey.theta_sat)    # more sand -> less porosity
    assert float(sandy.K_sat) > float(clayey.K_sat)            # more sand -> higher K
    assert float(clayey.b_ch) > float(sandy.b_ch)              # more clay -> higher b
    assert abs(float(sandy.psi_sat)) < abs(float(clayey.psi_sat))  # more sand -> smaller |psi|


def test_cosby_no_gsam_psi_cap_departure():
    """DEPARTURE canary: legoesm implements the RAW Cosby/CLM4 psi_sat (eq. 7.87)
    with NO cap.  gSAM's ``min(-150, raw)`` floors the MAGNITUDE at 150 mm, which
    binds for HIGH sand (small |raw|): at 90% sand raw = -50.2 mm, so legoesm
    returns -0.0502 m (uncapped) while gSAM would return -150 mm = -0.150 m."""
    sand = 90.0
    p = cosby_hydraulic_params(_a(sand), _a(20.0))
    raw_mm = -(_O_PSI_COEFF_MM * 10.0 ** (_O_PSI_INT - _O_PSI_SAND * sand))
    gsam_m = min(-150.0, raw_mm) * _MM_TO_M                     # gSAM's capped value
    assert float(p.psi_sat) == pytest.approx(raw_mm * _MM_TO_M, rel=1e-9, abs=0.0)  # raw
    assert abs(float(p.psi_sat)) < 0.150                       # uncapped (|psi| < 0.15 m)
    assert abs(float(p.psi_sat) - gsam_m) > 1e-3               # differs from gSAM's -0.150 m


# --- config builder ------------------------------------------------------------

def test_config_builder_from_texture():
    """soil_hydraulics_config_from_texture writes the four Cosby params into a
    Clapp-Hornberger SoilHydraulicsConfig with theta_r = 0."""
    sand, clay = 40.0, 20.0
    cfg = soil_hydraulics_config_from_texture(sand, clay)
    ts, psi, b, ks = _cosby_oracle(sand, clay)
    assert cfg.retention_curve == "clapp_hornberger"
    assert cfg.theta_r == 0.0
    assert cfg.theta_sat == pytest.approx(ts, rel=1e-9, abs=0.0)
    assert cfg.psi_sat == pytest.approx(psi, rel=1e-9, abs=0.0)
    assert cfg.b_ch == pytest.approx(b, rel=1e-9, abs=0.0)
    assert cfg.K_sat == pytest.approx(ks, rel=1e-9, abs=0.0)


# --- AD-safety -----------------------------------------------------------------

def test_cosby_analytic_derivatives_x64_and_float32():
    """The regressions are differentiable in (sand, clay) with the exact analytic
    derivatives: d theta/d sand = -0.00126, d b/d clay = 0.159, d K_sat/d sand =
    K_sat ln10 * 0.0153, d psi_sat/d sand = psi_sat ln10 * (-0.0131).  Finite in
    x64 and float32."""
    def _check(rel):
        sand0, clay0 = _a(40.0), _a(20.0)
        gts = jax.grad(lambda s: cosby_hydraulic_params(s, clay0).theta_sat)(sand0)
        gb = jax.grad(lambda c: cosby_hydraulic_params(sand0, c).b_ch)(clay0)
        gks = jax.grad(lambda s: cosby_hydraulic_params(s, clay0).K_sat)(sand0)
        gpsi = jax.grad(lambda s: cosby_hydraulic_params(s, clay0).psi_sat)(sand0)
        # expected amplitudes from the INDEPENDENT oracle (not production output).
        _ts, psi_o, _b, ks_o = _cosby_oracle(40.0, 20.0)
        assert float(gts) == pytest.approx(-_O_THETA_SAND, rel=rel)
        assert float(gb) == pytest.approx(_O_B_CLAY, rel=rel)
        assert float(gks) == pytest.approx(ks_o * math.log(10.0) * _O_KSAT_SAND, rel=rel)
        assert float(gpsi) == pytest.approx(psi_o * math.log(10.0) * (-_O_PSI_SAND), rel=rel)
        assert all(bool(jnp.isfinite(g)) for g in (gts, gb, gks, gpsi))

    _check(rel=1e-9)
    jax.config.update("jax_enable_x64", False)
    _check(rel=1e-4)   # float32; autouse fixture restores the entry state afterwards
