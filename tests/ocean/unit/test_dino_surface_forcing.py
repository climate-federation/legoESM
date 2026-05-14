"""Unit tests for DINO surface forcing (Phase 2C, Appendix B).

Cross-checks against:
- Paper Fig 2: wind/heat/T*/S*/ρ* profiles vs latitude.
- Paper eqs B1-B5.
- Zenodo source (vopikamm/DINO@v0.2.0 MY_SRC/usrdef_sbc.F90):
  wind interpolation is cubic Hermite smooth-step (NOT PCHIP); T*
  uses sin-form algebraically equivalent to paper cos-form; S* has
  equatorial Gaussian dip; Q_sr seasonal eq B5 averages to the
  annual-mean profile.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.experiments.dino import (
    DINOConfig,
    _Q_sr_instantaneous,
    dino_Q_sr_annual_mean,
    dino_S_star,
    dino_T_star_annual_mean,
    dino_wind_stress,
)


# -------------------- wind stress -----------------------------------

def test_wind_zero_at_outer_knots():
    cfg = DINOConfig()
    tau = dino_wind_stress(jnp.array([-70.0, 70.0]), cfg)
    assert float(tau[0]) == pytest.approx(0.0, abs=1e-9)
    assert float(tau[1]) == pytest.approx(0.0, abs=1e-9)


def test_wind_passes_through_all_knots():
    """At each knot latitude, the interpolant should equal the knot value."""
    cfg = DINOConfig()
    lats = jnp.asarray(cfg.wind_tau_lats_deg)
    taus_expected = jnp.asarray(cfg.wind_tau_values)
    taus = dino_wind_stress(lats, cfg)
    assert bool(jnp.allclose(taus, taus_expected, atol=1e-9))


def test_wind_positive_westerlies_southern_band():
    """Southern westerlies between -70 and -45° should be positive
    (peak 0.2 N/m² at -45°). At -50°, the value should be positive."""
    cfg = DINOConfig()
    tau = dino_wind_stress(jnp.array([-50.0]), cfg)
    assert float(tau[0]) > 0.0


def test_wind_easterlies_in_tropics():
    """Around -15° and +15° (NE/SE trades) the wind is -0.1 N/m²
    (easterlies). At -20° we should still be roughly easterly."""
    cfg = DINOConfig()
    tau = dino_wind_stress(jnp.array([-20.0, 20.0]), cfg)
    assert float(tau[0]) < 0.0
    assert float(tau[1]) < 0.0


def test_wind_smooth_step_has_zero_derivative_at_knots():
    """Cubic smooth-step (3-2s)s² gives df/ds=0 at s=0 and s=1, so
    df/dlat = 0 at every knot. Verify numerically."""
    cfg = DINOConfig()
    eps = 1e-4
    knot_lat = -45.0  # interior knot (peak of southern westerlies)
    tau_lo = float(dino_wind_stress(jnp.array([knot_lat - eps]), cfg)[0])
    tau_hi = float(dino_wind_stress(jnp.array([knot_lat + eps]), cfg)[0])
    tau_at = float(dino_wind_stress(jnp.array([knot_lat]), cfg)[0])
    # Both one-sided derivatives should be near zero at the knot
    deriv_left = (tau_at - tau_lo) / eps
    deriv_right = (tau_hi - tau_at) / eps
    assert abs(deriv_left) < 1e-3
    assert abs(deriv_right) < 1e-3


def test_wind_out_of_range_clamps():
    """Outside the knot range (|lat| > 70°), tau should clamp to 0
    (the outermost knot value)."""
    cfg = DINOConfig()
    tau = dino_wind_stress(jnp.array([-85.0, 85.0]), cfg)
    assert float(tau[0]) == pytest.approx(0.0, abs=1e-9)
    assert float(tau[1]) == pytest.approx(0.0, abs=1e-9)


# -------------------- temperature restoring -------------------------

def test_T_star_at_equator_is_eq_value():
    cfg = DINOConfig()
    T = float(dino_T_star_annual_mean(jnp.array(0.0), cfg))
    assert T == pytest.approx(cfg.T_star_eq, abs=1e-9)


def test_T_star_at_north_pole_is_north_value():
    cfg = DINOConfig()
    T = float(dino_T_star_annual_mean(jnp.array(cfg.lat_max_deg), cfg))
    assert T == pytest.approx(cfg.T_star_n_mean, abs=1e-6)


def test_T_star_at_south_pole_is_south_value():
    cfg = DINOConfig()
    T = float(dino_T_star_annual_mean(jnp.array(-cfg.lat_max_deg), cfg))
    assert T == pytest.approx(cfg.T_star_s_mean, abs=1e-6)


def test_T_star_monotone_decrease_with_latitude_in_NH():
    """In NH, T* decreases from 27°C at equator to 5°C at +70°."""
    cfg = DINOConfig()
    lats = jnp.linspace(0.0, 70.0, 20)
    T = dino_T_star_annual_mean(lats, cfg)
    diffs = T[1:] - T[:-1]
    assert bool(jnp.all(diffs <= 0))


# -------------------- salinity restoring ----------------------------

def test_S_star_at_north_pole():
    cfg = DINOConfig()
    S = float(dino_S_star(jnp.array(cfg.lat_max_deg), cfg))
    # At φ=70: cos(2π·70/140)=cos(π)=-1, so (1+cos)/2 = 0 → S = S*_n - dip
    # Dip at φ=70: 1.25·exp(-70²/7.5²) ≈ 0 (huge negative exponent)
    assert S == pytest.approx(cfg.S_star_n, abs=1e-6)


def test_S_star_at_south_pole():
    cfg = DINOConfig()
    S = float(dino_S_star(jnp.array(-cfg.lat_max_deg), cfg))
    assert S == pytest.approx(cfg.S_star_s, abs=1e-6)


def test_S_star_dip_at_equator():
    """At φ=0: cos(0)=1, so (1+cos)/2=1 → S = S*_n (or S*_s) + (S*_eq - S*_n/s)
    BUT minus the 1.25 g/kg Gaussian dip. So S_eq < S*_eq."""
    cfg = DINOConfig()
    S_eq = float(dino_S_star(jnp.array(0.0), cfg))
    # At φ=0, n-side path used (lat <= 0 is False).
    # Without dip: S would be S*_n + (S*_eq - S*_n) = S*_eq = 37.25
    # With dip:   S = 37.25 - 1.25 = 36.0
    expected = cfg.S_star_eq - cfg.S_star_eq_dip_amp
    assert S_eq == pytest.approx(expected, abs=1e-9)


def test_S_star_dip_decays_away_from_equator():
    """The Gaussian dip should be ~0 outside ±25° (5σ from 7.5°)."""
    cfg = DINOConfig()
    # At φ=±30° the dip should be very small
    S_at_30 = float(dino_S_star(jnp.array(30.0), cfg))
    # The "no dip" value at φ=30: S*_n + (S*_eq - S*_n)·(1+cos(60π/140))/2
    cos_factor = (1.0 + np.cos(2 * np.pi * 30.0 / 140.0)) / 2.0
    no_dip = cfg.S_star_n + (cfg.S_star_eq - cfg.S_star_n) * cos_factor
    diff = abs(S_at_30 - no_dip)
    assert diff < 0.05  # dip is small at |φ|=30°


def test_S_star_is_finite_everywhere():
    cfg = DINOConfig()
    lats = jnp.linspace(-cfg.lat_max_deg, cfg.lat_max_deg, 141)
    S = dino_S_star(lats, cfg)
    assert bool(jnp.all(jnp.isfinite(S)))


# -------------------- Q_sr -------------------------------------------

def test_Q_sr_instantaneous_peaks_at_equator_at_equinox():
    """At equinox (declination = 0), Q_sr peaks at φ=0 with Q_sr_amp."""
    cfg = DINOConfig()
    # Day 81 ≈ first equinox (where cos(π(d-171)/180)=cos(π(81-171)/180)=cos(-π/2)=0)
    q = float(_Q_sr_instantaneous(jnp.array(0.0), jnp.array(81.0), cfg))
    assert q == pytest.approx(cfg.Q_sr_amp, abs=1e-6)


def test_Q_sr_instantaneous_zero_at_winter_pole():
    """At the southern summer solstice (Dec ~Day 351), the southern
    pole sees daylight and the northern pole is dark. The opposite
    holds at Day 171 (NH summer solstice → SH winter pole is dark)."""
    cfg = DINOConfig()
    # Day 171: northern summer solstice, declination is max (+23.5)
    # At φ = -70°, Q_sr = max(230·cos(π·(-70 - 23.5)/180), 0)
    # cos(π·(-93.5)/180) = cos(-1.633) ≈ -0.0610  → q would be negative → max=0
    q = float(_Q_sr_instantaneous(jnp.array(-70.0), jnp.array(171.0), cfg))
    assert q == 0.0


def test_Q_sr_annual_mean_is_positive_in_tropics():
    cfg = DINOConfig()
    Q = float(dino_Q_sr_annual_mean(jnp.array(0.0), cfg))
    # At the equator, the seasonal average is significantly less than
    # 230 W/m² because the sun spends only part of the year directly
    # overhead. But it must be substantial — well over 100 W/m².
    assert 150.0 < Q < 230.0


def test_Q_sr_annual_mean_decays_to_zero_at_polar_night():
    """Polar regions get less average insolation."""
    cfg = DINOConfig()
    Q_eq = float(dino_Q_sr_annual_mean(jnp.array(0.0), cfg))
    Q_pole = float(dino_Q_sr_annual_mean(jnp.array(70.0), cfg))
    assert Q_pole < Q_eq
    # At φ=70°, the sun is below the horizon for much of the year
    # (since declination max is ±23.5°). Annual mean is well below 100 W/m².
    assert Q_pole < 100.0


def test_Q_sr_annual_mean_symmetric_about_equator():
    """The annual mean (over the full year) should be N/S symmetric."""
    cfg = DINOConfig()
    Q_n = float(dino_Q_sr_annual_mean(jnp.array(40.0), cfg))
    Q_s = float(dino_Q_sr_annual_mean(jnp.array(-40.0), cfg))
    assert Q_n == pytest.approx(Q_s, abs=0.1)


def test_Q_sr_annual_mean_polar_clipping_decreases_high_lat_value():
    """At φ=70° (polar night ~30 days/yr in our 360-day calendar) the
    quadrature should give a value less than or equal to 230·cos(φ) —
    a tighter test than just monotone decrease, since polar-night
    clipping never adds energy. (The difference is small at φ=70°
    because the sun still rises ~330 days/year; the test merely
    verifies the sign of the bias.)"""
    cfg = DINOConfig()
    Q_quad = float(dino_Q_sr_annual_mean(jnp.array(70.0), cfg))
    lazy = 230.0 * np.cos(np.radians(70.0))
    assert Q_quad <= lazy + 0.1  # never exceeds the unclipped formula


def test_Q_sr_annual_mean_shape_broadcasts():
    cfg = DINOConfig()
    lats = jnp.linspace(-70.0, 70.0, 31)
    Q = dino_Q_sr_annual_mean(lats, cfg)
    assert Q.shape == lats.shape
    assert bool(jnp.all(jnp.isfinite(Q)))
