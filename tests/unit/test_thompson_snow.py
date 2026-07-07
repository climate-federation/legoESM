"""Direct unit tests for the faithful Thompson-2008 snow module."""
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.physics.microphysics import _thompson_snow as ts


def _cols():
    q_s = jnp.array([1e-3, 1e-4, 1e-5, 1e-6, 0.0])
    rho = jnp.array([0.6, 0.4, 0.3, 0.2, 0.4])
    T = jnp.array([260.0, 250.0, 240.0, 230.0, 240.0])
    p = jnp.array([5e4, 4e4, 3e4, 2e4, 3e4])
    return q_s, rho, T, p


def test_snow_fall_speed_physical_and_monotone():
    q_s, rho, T, p = _cols()
    V = ts.snow_fall_speed(q_s, rho, T)
    assert bool(jnp.all(jnp.isfinite(V)))
    # zero snow -> zero fall speed
    assert float(V[-1]) == 0.0
    # physical aggregate snow fall speeds (mass-weighted): O(0.1-3 m/s)
    assert bool(jnp.all(V[:-1] > 0.0))
    assert bool(jnp.all(V <= 5.0))
    assert float(jnp.max(V[:-1])) < 3.0


def test_snow_fall_speed_density_correction():
    # Same q_s, lower density -> faster fall (rho0/rho)^0.5 correction.
    q_s = jnp.array([1e-4, 1e-4])
    T = jnp.array([245.0, 245.0])
    rho_hi = jnp.array([1.0, 1.0])
    rho_lo = jnp.array([0.2, 0.2])
    V_hi = ts.snow_fall_speed(q_s, rho_hi, T)
    V_lo = ts.snow_fall_speed(q_s, rho_lo, T)
    assert float(V_lo[0]) > float(V_hi[0])


def test_snow_deposition_sign_and_bounds():
    q_s, rho, T, p = _cols()
    q_sat_i = jnp.array([1.5e-3, 1.2e-3, 8e-4, 4e-4, 8e-4])
    # supersaturated -> deposition > 0
    q_v_super = q_sat_i * 1.3
    dep = ts.snow_deposition(q_v_super, q_s, q_sat_i, T, p, rho, 6.0)
    assert bool(jnp.all(jnp.isfinite(dep)))
    assert bool(jnp.all(dep[:-1] >= 0.0))
    # deposition cannot exceed available supersaturation per step
    assert bool(jnp.all(dep <= (q_v_super - q_sat_i) / 6.0 + 1e-12))
    # subsaturated -> sublimation < 0, bounded by available q_s
    q_v_sub = q_sat_i * 0.5
    subl = ts.snow_deposition(q_v_sub, q_s, q_sat_i, T, p, rho, 6.0)
    assert bool(jnp.all(subl[:-1] <= 0.0))
    assert bool(jnp.all(subl >= -jnp.clip(q_s, 0.0) / 6.0 - 1e-12))
    # zero snow -> zero rate
    assert float(dep[-1]) == 0.0 and float(subl[-1]) == 0.0


def test_snow_deposition_supersaturation_ratio_magnitude():
    """The A+B growth law must be driven by the DIMENSIONLESS supersaturation
    ratio S_i - 1 = q_v/q_sat_i - 1 (WRF ``ssati``), closed to [kg/kg/s] with
    the 1/rho per-volume-moment conversion.

    Pre-fix, the numerator was the mixing-ratio excess q_v - q_sat_i =
    q_sat_i*(S_i-1), making deposition weaker by a factor ~q_sat_i (1e3-1e4 at
    cold upper-tropospheric temperatures) — the supersaturation relaxation
    timescale came out at ~1e7 s (months).  Post-fix, a moderately snowy
    240 K layer at 30 % ice supersaturation relaxes on minutes-to-tens-of-
    minutes: tau = (q_v - q_sat_i)/prds in [10 s, 2 h].  Sublimation flips
    sign for q_v < q_sat_i with the same magnitude (linear in S_i - 1).
    """
    from legoesm.thermo import saturation_mixing_ratio_ice
    from legoesm import constants

    q_s = jnp.array([1e-3, 1e-4])
    T = jnp.array([240.0, 240.0])
    p = jnp.array([3e4, 3e4])
    rho = p / (constants.R_d * T)
    q_sat_i = saturation_mixing_ratio_ice(T, p)
    dt = 1.0  # cap = excess/dt, so tau >= 1 s can't be cap-limited

    q_v_super = 1.3 * q_sat_i
    dep = ts.snow_deposition(q_v_super, q_s, q_sat_i, T, p, rho, dt)
    assert bool(jnp.all(dep > 0.0))
    tau = (q_v_super - q_sat_i) / dep
    assert bool(jnp.all(tau > 10.0)) and bool(jnp.all(tau < 7200.0)), (
        f"snow-deposition supersaturation relaxation timescale {tau} s "
        "outside the physical minutes-to-tens-of-minutes range — the "
        "(q_v/q_sat_i - 1)/(A+B)/rho closure regressed."
    )
    # More snow -> faster deposition (larger PSD moments).
    assert float(dep[0]) > float(dep[1])
    # Sublimation flips sign symmetrically (rate linear in S_i - 1).
    q_v_sub = 0.7 * q_sat_i
    subl = ts.snow_deposition(q_v_sub, q_s, q_sat_i, T, p, rho, dt)
    assert bool(jnp.all(subl < 0.0))
    assert bool(jnp.allclose(subl, -dep, rtol=1e-6))


def test_moments_reproduce_M2_M3():
    # The bimodal-PSD normalization must reproduce the 2nd and 3rd moments.
    q_s = jnp.array([1e-4]); rho = jnp.array([0.4]); T = jnp.array([245.0])
    M2, M3, ratio = ts._snow_moments(q_s, rho, T)
    I2 = ts._psd_integral(2.0, M2, M3, ratio)
    I3 = ts._psd_integral(3.0, M2, M3, ratio)
    assert float(jnp.abs(I2 / M2 - 1.0)[0]) < 1e-3
    assert float(jnp.abs(I3 / M3 - 1.0)[0]) < 1e-3
