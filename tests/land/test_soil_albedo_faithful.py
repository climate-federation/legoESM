"""Oracle-faithfulness pins for CLM background soil albedo (eq. 3.52 + Table 3.3).

Oracle: CLM4 Technical Note (Oleson et al. 2010) eq. 3.52 + the 20-class soil-colour
albedo table, typed HERE from CTSM ``src/biogeophys/SurfaceAlbedoMod.F90``
(``SurfaceAlbedoInitTimeConst``) — an INDEPENDENT source, NOT read back from the
module under test (non-circular).  The CTSM closed form is::

    inc      = max(0.11 - 0.40 * theta_top, 0)          # theta_top = top-layer vol. water
    alb_band = min(albsat_band + inc, albdry_band)        # per band, band in {vis, nir}

``legoesm.land.soil_albedo.soil_albedo`` is pinned to round-off (rel 1e-12) against
an independent numpy reimplementation of that form with the coefficients (0.11, 0.40)
and the four 20-class ``albsat``/``albdry`` arrays typed from CTSM.  Structure canaries
(each diverges on a wrong implementation): wrong 0.11 intercept, wrong 0.40 slope, the
``min`` cap swapped to ``max``, a vis<->nir table swap, and the 1-based class index.

``test_module_table_matches_clm`` pins the module's own ``SOIL_COLOR_ALBEDO`` against
the independent CTSM arrays, so a corrupted table value fails LOUDLY.

DEPARTURE / SURROGATE:
  ``soil_albedo_broadband`` collapses (vis, nir) with ``vis_fraction=0.5``.  CLM keeps
  the two bands SEPARATE for the two-stream canopy solver and has NO single broadband
  soil albedo, so the 0.5 blend is a legoESM CONVENIENCE, not a CLM quantity — pinned
  only as the arithmetic ``0.5*vis + 0.5*nir`` (``test_broadband_is_band_average``),
  not against any CLM reference.

Complements ``tests/land/test_soil_albedo.py`` (behavioral: dry/wet limits, monotone,
clipping) — which reads dry/sat FROM the module table (circular on the table) — with
the round-off pin + INDEPENDENT arrays + coefficient/structure canaries.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.land.soil_albedo import (
    SOIL_COLOR_ALBEDO,
    soil_albedo,
    soil_albedo_broadband,
)

jax.config.update("jax_enable_x64", True)

# --- CLM4 eq. 3.52 soil-wetness coefficients (typed from the CLM tech note) ---
_O_DELTA_INTERCEPT = 0.11
_O_DELTA_SLOPE = 0.40

# --- CTSM SurfaceAlbedoMod.F90 (SurfaceAlbedoInitTimeConst): 20 soil-colour classes ---
_O_ALBSAT_VIS = [0.25, 0.23, 0.21, 0.20, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14,
                 0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04]
_O_ALBSAT_NIR = [0.50, 0.46, 0.42, 0.40, 0.38, 0.36, 0.34, 0.32, 0.30, 0.28,
                 0.26, 0.24, 0.22, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08]
_O_ALBDRY_VIS = [0.36, 0.34, 0.32, 0.31, 0.30, 0.29, 0.28, 0.27, 0.26, 0.25,
                 0.24, 0.23, 0.22, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08]
_O_ALBDRY_NIR = [0.61, 0.57, 0.53, 0.51, 0.49, 0.48, 0.45, 0.43, 0.41, 0.39,
                 0.37, 0.35, 0.33, 0.31, 0.29, 0.27, 0.25, 0.23, 0.21, 0.16]
_N = 20


def _soil_albedo_oracle(cls, theta, *, intercept=_O_DELTA_INTERCEPT,
                        slope=_O_DELTA_SLOPE, cap=min, band_swap=False):
    """Independent CLM eq. 3.52 soil albedo (vis, nir). ``cap``/``intercept``/``slope``/
    ``band_swap`` overridable for the structure canaries."""
    i = min(max(int(cls) - 1, 0), _N - 1)                 # 1-based class, clipped
    sat_vis, sat_nir = _O_ALBSAT_VIS[i], _O_ALBSAT_NIR[i]
    dry_vis, dry_nir = _O_ALBDRY_VIS[i], _O_ALBDRY_NIR[i]
    if band_swap:
        sat_vis, sat_nir = sat_nir, sat_vis
        dry_vis, dry_nir = dry_nir, dry_vis
    inc = max(intercept - slope * float(theta), 0.0)
    return cap(sat_vis + inc, dry_vis), cap(sat_nir + inc, dry_nir)


def _call(cls, theta):
    v, n = soil_albedo(jnp.asarray(cls), jnp.asarray(float(theta)))
    return float(v), float(n)


_THETAS = [0.0, 0.05, 0.15, 0.25, 0.275, 0.30, 0.5]     # dry -> saturated (inc=0 at 0.275)


# ===================== independent-table pin =====================

def test_module_table_matches_clm():
    # The module's SOIL_COLOR_ALBEDO (dry_vis, dry_nir, sat_vis, sat_nir) must equal
    # the CTSM albsat/albdry arrays typed above — pins the 20x4 table NON-circularly.
    assert len(SOIL_COLOR_ALBEDO) == _N
    for i, row in enumerate(SOIL_COLOR_ALBEDO):
        dv, dn, sv, sn = row
        assert dv == pytest.approx(_O_ALBDRY_VIS[i], abs=1e-12)
        assert dn == pytest.approx(_O_ALBDRY_NIR[i], abs=1e-12)
        assert sv == pytest.approx(_O_ALBSAT_VIS[i], abs=1e-12)
        assert sn == pytest.approx(_O_ALBSAT_NIR[i], abs=1e-12)


# ===================== full round-off pin =====================

@pytest.mark.parametrize("cls", range(1, _N + 1))
@pytest.mark.parametrize("theta", _THETAS)
def test_soil_albedo_matches_clm(cls, theta):
    got = _call(cls, theta)
    ref = _soil_albedo_oracle(cls, theta)
    assert got[0] == pytest.approx(ref[0], rel=1e-12, abs=1e-14)
    assert got[1] == pytest.approx(ref[1], rel=1e-12, abs=1e-14)


# ===================== coefficient + structure canaries =====================
# theta=0.15 => inc=0.05 keeps sat+inc strictly below dry for these classes (uncapped),
# so a wrong intercept/slope/cap actually changes the answer (non-vacuous).

def test_intercept_canary():
    cls, theta = 1, 0.15
    ref = _soil_albedo_oracle(cls, theta)
    assert _call(cls, theta)[0] == pytest.approx(ref[0], rel=1e-12)
    wrong = _soil_albedo_oracle(cls, theta, intercept=0.12)
    assert abs(ref[0] - wrong[0]) > 1e-6


def test_slope_canary():
    cls, theta = 1, 0.15
    ref = _soil_albedo_oracle(cls, theta)
    wrong = _soil_albedo_oracle(cls, theta, slope=0.35)
    assert abs(ref[0] - wrong[0]) > 1e-6
    assert _call(cls, theta)[0] == pytest.approx(ref[0], rel=1e-12)


def test_cap_is_min_not_max_canary():
    cls, theta = 1, 0.15                                  # sat+inc=0.30 < dry=0.36
    ref = _soil_albedo_oracle(cls, theta, cap=min)
    wrong = _soil_albedo_oracle(cls, theta, cap=max)      # would return dry
    assert abs(ref[0] - wrong[0]) > 1e-6
    assert _call(cls, theta)[0] == pytest.approx(ref[0], rel=1e-12)


def test_band_not_swapped_canary():
    cls, theta = 1, 0.15
    ref = _soil_albedo_oracle(cls, theta)
    swapped = _soil_albedo_oracle(cls, theta, band_swap=True)
    assert abs(ref[0] - swapped[0]) > 1e-6
    got = _call(cls, theta)
    assert got[0] == pytest.approx(ref[0], rel=1e-12)     # vis is the vis band, not nir


def test_class_index_is_one_based():
    # class 1 -> row 0 (not row 1); class 2 differs from class 1.
    assert _call(1, 0.0)[0] == pytest.approx(_O_ALBDRY_VIS[0], abs=1e-12)   # capped at dry
    assert abs(_call(2, 0.15)[0] - _call(1, 0.15)[0]) > 1e-6


# ===================== broadband surrogate + differentiability =====================

def test_broadband_is_band_average():
    # SURROGATE: 0.5*vis + 0.5*nir (legoESM convenience; CLM has no single broadband).
    cls, theta = 7, 0.1
    v, n = _call(cls, theta)
    bb = float(soil_albedo_broadband(jnp.asarray(cls), jnp.asarray(theta)))
    assert bb == pytest.approx(0.5 * v + 0.5 * n, rel=1e-12)


def test_differentiable_in_moisture():
    # Uncapped regime (inc active): d alb / d theta = -slope < 0 (wetter -> darker).
    g = float(jax.grad(lambda th: soil_albedo(jnp.asarray(1), th)[0])(jnp.asarray(0.15)))
    assert g == pytest.approx(-_O_DELTA_SLOPE, rel=1e-9)
    # Saturated regime (inc clamped at 0): gradient is 0.
    g_sat = float(jax.grad(lambda th: soil_albedo(jnp.asarray(1), th)[0])(jnp.asarray(0.4)))
    assert g_sat == pytest.approx(0.0, abs=1e-12)
