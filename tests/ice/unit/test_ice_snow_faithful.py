"""Snow-on-ice conduction + flooding ORACLE-FAITHFULNESS tests.

``ice/snow`` implements the Semtner (1976) / Maykut-Untersteiner (1971) SERIES
snow+ice thermal resistance and the Leppäranta (1983) / Notz (2002) Archimedes
snow-ice flooding.  The existing ice tests are behavioral (conductive flux
decreases with snow; flooding conserves mass and gates on negative freeboard) —
they never pin the closed FORMS against an independent reimplementation.

These pin the snow ALGEBRA to round-off (rel 1e-9) against an independent scalar
oracle:
  * combined_conductance      K = 1/(h_snow/k_snow + h_ice/k_ice)   (series R)
  * combined_conductive_flux  F = K (T_base - T_sfc)
  * snow_ice_flooding         fb, d = -fb rho_oc/(rho_oc-rho_ice+rho_snow),
                              h_ice' = h_ice + d, h_snow' = h_snow - d

plus the DEFINING flotation identity (flooding drives the new freeboard to
EXACTLY zero), the (rho_ice-rho_snow) d seawater mass draw, and the no-flood
gate at fb >= 0.

Independence: the rho/k coefficients are local _O_* literals canaried against
legoesm.constants (constants == _O_* == value).  Departures (the max(h,h_min)
resistance floors, the max(-fb,0) flooding gate) are reproduced by the oracle.
The d <= h_snow cap is an UNREACHABLE safety for physical densities
(rho_snow < rho_ice < rho_ocean, so an active flood always has d < h_snow), so
it is not claimed as an exercised guard.
"""

from __future__ import annotations

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


from legoesm import constants                                        # noqa: E402
from legoesm.ice.snow import (                                       # noqa: E402
    combined_conductance, combined_conductive_flux, snow_ice_flooding,
)

# Independent oracle literals (canaried in test_snow_constants_match_cice).
_O_RHO_ICE = 917.0       # [kg/m^3]
_O_RHO_SNOW = 330.0      # [kg/m^3] CICE default dry snow
_O_RHO_OCEAN = 1025.0    # [kg/m^3] reference seawater
_O_K_ICE = 2.04          # [W/m/K] pure ice
_O_K_SNOW = 0.31         # [W/m/K] CICE default dry snow
_H_ICE_MIN = 0.01        # [m] resistance-floor (test value)
_H_SNOW_MIN = 0.001      # [m] resistance-floor (test value)


def _conductance_oracle(h_ice, h_snow, k_ice, k_snow, hi_min, hs_min):
    h_i = max(h_ice, hi_min)
    h_s = max(h_snow, hs_min)
    return 1.0 / (h_s / k_snow + h_i / k_ice)


def _flooding_oracle(h_ice, h_snow, rho_ice, rho_snow, rho_ocean):
    fb = ((rho_ocean - rho_ice) * h_ice - rho_snow * h_snow) / rho_ocean
    denom = max(rho_ocean - rho_ice + rho_snow, 1.0e-30)
    d = max(-fb, 0.0) * rho_ocean / denom
    d = min(d, h_snow)
    return h_ice + d, max(h_snow - d, 0.0), d


def _a(x):
    return jnp.array(float(x))


# --- combined_conductance / flux (series thermal resistance) -------------------

@pytest.mark.parametrize("h_ice,h_snow", [
    (1.5, 0.3),     # typical snow-covered ice
    (2.0, 0.0),     # bare ice (snow floor engages)
    (0.005, 0.2),   # thin ice (ice floor engages), deep snow
    (3.0, 0.5),     # thick both
])
def test_combined_conductance_matches_series_oracle(h_ice, h_snow):
    """combined_conductance matches K = 1/(h_snow/k_snow + h_ice/k_ice) with the
    max(h,h_min) floors, to round-off."""
    got = float(combined_conductance(_a(h_ice), _a(h_snow), _O_K_ICE, _O_K_SNOW,
                                     _H_ICE_MIN, _H_SNOW_MIN))
    exp = _conductance_oracle(h_ice, h_snow, _O_K_ICE, _O_K_SNOW,
                              _H_ICE_MIN, _H_SNOW_MIN)
    assert got == pytest.approx(exp, rel=1e-9, abs=0.0)


def test_combined_conductive_flux_matches_oracle():
    """combined_conductive_flux = K (T_base - T_sfc); pinned incl the sign
    (positive = upward, warm base to cold surface)."""
    T_base, T_sfc, h_ice, h_snow = 271.35, 253.15, 1.5, 0.3
    K = _conductance_oracle(h_ice, h_snow, _O_K_ICE, _O_K_SNOW, _H_ICE_MIN, _H_SNOW_MIN)
    got = float(combined_conductive_flux(_a(T_base), _a(T_sfc), _a(h_ice), _a(h_snow),
                                         _O_K_ICE, _O_K_SNOW, _H_ICE_MIN, _H_SNOW_MIN))
    assert got == pytest.approx(K * (T_base - T_sfc), rel=1e-9, abs=0.0)
    assert got > 0.0                                # warm base, cold surface -> upward


def test_conductance_series_resistance_adds():
    """Non-vacuity: adding snow to bare ice REDUCES the conductance (series
    resistances add) — and the bare-ice limit approaches k_ice/h_ice."""
    bare = float(combined_conductance(_a(1.5), _a(0.0), _O_K_ICE, _O_K_SNOW,
                                      _H_ICE_MIN, _H_SNOW_MIN))
    snowy = float(combined_conductance(_a(1.5), _a(0.3), _O_K_ICE, _O_K_SNOW,
                                       _H_ICE_MIN, _H_SNOW_MIN))
    assert snowy < bare                             # snow insulates
    # bare-ice (snow at floor 0.001 m, k_snow 0.31 -> negligible R) ~ k_ice/h_ice.
    assert bare == pytest.approx(_O_K_ICE / 1.5, rel=5e-3)


# --- snow_ice_flooding (Archimedes flotation) ----------------------------------

@pytest.mark.parametrize("h_ice,h_snow", [
    (0.3, 0.4),     # heavy snow on thin ice -> negative freeboard, floods
    (1.0, 0.05),    # normal -> positive freeboard, no flooding (d=0)
    (0.2, 0.6),     # deep snow on very thin ice -> floods (d=0.40 m < h_snow)
    (2.0, 0.0),     # no snow -> no flooding
])
def test_snow_ice_flooding_matches_oracle(h_ice, h_snow):
    """snow_ice_flooding (h_ice', h_snow', d) matches the independent Archimedes
    reimplementation to round-off, across flood / no-flood / no-snow states."""
    hi, hs, d = (float(x) for x in snow_ice_flooding(
        _a(h_ice), _a(h_snow), _O_RHO_ICE, _O_RHO_SNOW, _O_RHO_OCEAN))
    ehi, ehs, ed = _flooding_oracle(h_ice, h_snow, _O_RHO_ICE, _O_RHO_SNOW, _O_RHO_OCEAN)
    assert hi == pytest.approx(ehi, rel=1e-9, abs=0.0)
    assert hs == pytest.approx(ehs, rel=1e-9, abs=0.0)
    assert d == pytest.approx(ed, rel=1e-9, abs=0.0)


def test_flooding_raises_freeboard_to_exactly_zero():
    """The DEFINING flotation property: un-capped flooding (d < h_snow) drives
    the NEW freeboard to EXACTLY zero, i.e. d = -fb rho_oc/(rho_oc-rho_ice+
    rho_snow) is the flotation solve, not an arbitrary rate."""
    h_ice, h_snow = 0.3, 0.4                         # negative freeboard, not snow-limited
    hi, hs, d = (float(x) for x in snow_ice_flooding(
        _a(h_ice), _a(h_snow), _O_RHO_ICE, _O_RHO_SNOW, _O_RHO_OCEAN))
    assert 0.0 < d < h_snow                          # flooded, not snow-capped
    fb_new = ((_O_RHO_OCEAN - _O_RHO_ICE) * hi - _O_RHO_SNOW * hs) / _O_RHO_OCEAN
    assert fb_new == pytest.approx(0.0, abs=1e-12)


def test_flooding_seawater_mass_draw_and_no_flood_when_positive():
    """The column mass GAIN equals the (rho_ice-rho_snow) d seawater draw
    (Leppäranta/Notz mass balance), and a positive freeboard floods nothing."""
    h_ice, h_snow = 0.3, 0.4
    hi, hs, d = (float(x) for x in snow_ice_flooding(
        _a(h_ice), _a(h_snow), _O_RHO_ICE, _O_RHO_SNOW, _O_RHO_OCEAN))
    m_before = _O_RHO_ICE * h_ice + _O_RHO_SNOW * h_snow
    m_after = _O_RHO_ICE * hi + _O_RHO_SNOW * hs
    assert (m_after - m_before) == pytest.approx((_O_RHO_ICE - _O_RHO_SNOW) * d,
                                                 rel=1e-9, abs=0.0)
    # Positive freeboard (thick ice, little snow) -> no flooding.
    hi2, hs2, d2 = (float(x) for x in snow_ice_flooding(
        _a(1.0), _a(0.05), _O_RHO_ICE, _O_RHO_SNOW, _O_RHO_OCEAN))
    assert d2 == 0.0 and hi2 == 1.0 and hs2 == 0.05


# --- constant canaries ---------------------------------------------------------

def test_snow_constants_match_cice():
    """The ice densities/conductivities equal both the independent oracle
    literals and their CICE-default legoesm.constants values."""
    assert constants.rho_ice == _O_RHO_ICE == 917.0
    assert constants.rho_snow == _O_RHO_SNOW == 330.0
    assert constants.rho_ocean == _O_RHO_OCEAN == 1025.0
    assert constants.k_ice_default == _O_K_ICE == 2.04
    assert constants.k_snow == _O_K_SNOW == 0.31


# --- AD-safety -----------------------------------------------------------------

def test_snow_grad_finite_x64_and_float32():
    """grad of the conductance (through the max(h,h_min) floors + 1/R) and the
    flooded thickness d (through the max(-fb,0) flooding gate) is finite in x64
    and float32; the guards must not NaN the VJP.  (The d<=h_snow cap is
    unreachable in-domain, so it is not exercised here.)"""
    def _check():
        gk = jax.grad(lambda h: combined_conductance(_a(1.5), h, _O_K_ICE, _O_K_SNOW,
                                                     _H_ICE_MIN, _H_SNOW_MIN))(_a(0.3))
        assert bool(jnp.isfinite(gk)) and float(gk) < 0.0   # more snow -> lower K
        # flooded d wrt h_snow at a flooding state: interior of max(-fb,0), so
        # dd/dh_snow = rho_snow/(rho_oc-rho_ice+rho_snow) > 0 (non-vacuous).
        gd = jax.grad(lambda hs: snow_ice_flooding(_a(0.3), hs, _O_RHO_ICE,
                                                   _O_RHO_SNOW, _O_RHO_OCEAN)[2])(_a(0.4))
        assert bool(jnp.isfinite(gd)) and float(gd) > 0.0
        # POSITIVE-freeboard state: d floored at 0 -> grad exactly 0 and finite.
        gd0 = jax.grad(lambda hs: snow_ice_flooding(_a(1.0), hs, _O_RHO_ICE,
                                                    _O_RHO_SNOW, _O_RHO_OCEAN)[2])(_a(0.02))
        assert bool(jnp.isfinite(gd0)) and float(gd0) == 0.0

    _check()
    jax.config.update("jax_enable_x64", False)
    _check()   # autouse fixture restores the entry state afterwards
