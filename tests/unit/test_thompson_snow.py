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


def test_snow_capacitance_oracle_ramp():
    """Pin the gSAM/WRF Thompson snow-capacitance temperature ramp
    (module_mp_thompson.f90:108-109, :2032-2033)::

        C_snow = MAX(C_sqrd, MIN(C_sqrd + (tempc+15)*(C_cube-C_sqrd)/(-30+15),
                                 C_cube)),   C_sqrd=0.3, C_cube=0.5
    """
    import numpy as np

    from legoesm import constants

    # Endpoints, midpoint, and both clamp plateaus.
    for tc, expect in [(-5.0, 0.3), (-15.0, 0.3), (-22.5, 0.4),
                       (-30.0, 0.5), (-45.0, 0.5)]:
        got = float(ts.snow_capacitance(jnp.array([constants.T_freeze + tc]))[0])
        assert got == pytest.approx(expect, rel=1e-6), (tc, got, expect)
    # Random sweep vs an independent transcription of the Fortran line.
    rng = np.random.default_rng(7)
    tc = rng.uniform(-60.0, 5.0, size=64)
    oracle = np.clip(0.3 + (tc + 15.0) * (0.5 - 0.3) / (-30.0 + 15.0), 0.3, 0.5)
    got = np.asarray(ts.snow_capacitance(jnp.asarray(constants.T_freeze + tc)))
    assert np.allclose(got, oracle, rtol=1e-6, atol=0.0)


def test_snow_deposition_prds_coefficient_pin():
    """Coefficient-level pin of the full PRDS prefactor (uncapped regime)
    against an independent scalar transcription of the documented closed form

        PRDS = 4*pi*C_snow(T)*(S_i-1)/(A+B)
               * [t1*I(1) + t2*rhof2*vsc2*I(c_vent)] / rho

    with the Thompson-2008 snow constants (am_s=0.069, av_s=40, bv_s=0.55,
    fv_s=100, mu_s=0.6357, Kap0=490.6, Kap1=17.46, Lam0=20.78, Lam1=3.29,
    Sc=0.632, t1=0.86, t2=0.28*Sc^(1/3)*sqrt(av_s)) and the T-ramped
    capacitance.  This became pinnable once the fixed C=0.15 departure was
    replaced by the faithful ramp.
    """
    import math

    from legoesm import constants
    from legoesm.thermo import saturation_mixing_ratio_ice

    x64 = jnp.zeros(1).dtype == jnp.float64
    rtol = 1e-9 if x64 else 3e-4

    def mirror(q_v, q_s, q_sat_i, T, p, rho):
        # PSD moments via the Field-2005 relation (Thompson 2008 Table 1).
        sa = (5.065339, -0.062659, -3.032362, 0.029469, -0.000285,
              0.31255, 0.000204, 0.003199, 0.0, -0.015952)
        sb = (0.476221, -0.015896, 0.165977, 0.007468, -0.000141,
              0.060366, 0.000079, 0.000594, 0.0, -0.003577)
        tc_raw = T - constants.T_freeze
        tc = min(max(tc_raw, -55.0), -0.1)
        pm = 3.0
        loga = (sa[0] + sa[1] * tc + sa[2] * pm + sa[3] * tc * pm
                + sa[4] * tc * tc + sa[5] * pm * pm + sa[6] * tc * tc * pm
                + sa[7] * tc * pm * pm + sa[8] * tc ** 3 + sa[9] * pm ** 3)
        b = (sb[0] + sb[1] * tc + sb[2] * pm + sb[3] * tc * pm
             + sb[4] * tc * tc + sb[5] * pm * pm + sb[6] * tc * tc * pm
             + sb[7] * tc * pm * pm + sb[8] * tc ** 3 + sb[9] * pm ** 3)
        M2 = q_s * rho / 0.069
        M3 = 10.0 ** loga * M2 ** b
        ratio = M2 / M3
        lam0, lam1 = 20.78 * ratio, 3.29 * ratio
        mu_s, kap0, kap1 = 0.6357, 490.6, 17.46
        norm = M2 * ratio ** 3
        I1 = norm * (kap0 * math.gamma(2.0) / lam0 ** 2
                     + kap1 * ratio ** mu_s * math.gamma(2.0 + mu_s)
                     / lam1 ** (2.0 + mu_s))
        c_vent = 1.0 + (1.0 + 0.55) / 2.0
        fv_half = 0.5 * 100.0
        I_vent = norm * (
            kap0 * math.gamma(c_vent + 1.0) / (lam0 + fv_half) ** (c_vent + 1.0)
            + kap1 * ratio ** mu_s * math.gamma(c_vent + mu_s + 1.0)
            / (lam1 + fv_half) ** (c_vent + mu_s + 1.0))
        # Thermodynamic resistances (Pruppacher-Klett / Rogers-Yau ice form).
        dv = 8.794e-5 * T ** 1.81 / p
        ka = 2.3971e-2 + 7.078e-5 * tc_raw
        e_si = q_sat_i * p / constants.epsilon
        A = constants.L_s ** 2 / (ka * constants.R_v * T ** 2)
        B = constants.R_v * T / (e_si * dv)
        # Ventilation factors.
        sc3 = 0.632 ** (1.0 / 3.0)
        t1, t2 = 0.86, 0.28 * sc3 * math.sqrt(40.0)
        rho_not = constants.p_atm_std / (constants.R_d * 298.0)
        rhof = math.sqrt(rho_not / rho)
        rhof2 = math.sqrt(rhof)
        mu_air = 1.458e-6 * T ** 1.5 / (T + 110.4)
        vsc2 = math.sqrt(rho / mu_air)
        c_snow = min(max(0.3 + (tc_raw + 15.0) * 0.2 / (-15.0), 0.3), 0.5)
        s_i = q_v / q_sat_i - 1.0
        vent = t1 * I1 + t2 * rhof2 * vsc2 * I_vent
        return 4.0 * math.pi * c_snow * s_i / (A + B) * vent / rho

    # Three states across the ramp: warm plateau, mid-ramp, cold plateau.
    for tc_case, q_s in [(-10.0, 2e-4), (-22.5, 1e-4), (-35.0, 5e-5)]:
        T = constants.T_freeze + tc_case
        p = 4.0e4
        rho = p / (constants.R_d * T)
        q_sat_i = float(saturation_mixing_ratio_ice(jnp.array(T), jnp.array(p)))
        q_v = 1.2 * q_sat_i
        got = float(ts.snow_deposition(
            jnp.array([q_v]), jnp.array([q_s]), jnp.array([q_sat_i]),
            jnp.array([T]), jnp.array([p]), jnp.array([rho]), 1.0)[0])
        want = mirror(q_v, q_s, q_sat_i, T, p, rho)
        # Uncapped regime: rate well below the availability cap excess/dt.
        assert got < (q_v - q_sat_i) / 1.0
        assert got == pytest.approx(want, rel=rtol), (T, got, want)


def test_moments_reproduce_M2_M3():
    # The bimodal-PSD normalization must reproduce the 2nd and 3rd moments.
    q_s = jnp.array([1e-4]); rho = jnp.array([0.4]); T = jnp.array([245.0])
    M2, M3, ratio = ts._snow_moments(q_s, rho, T)
    I2 = ts._psd_integral(2.0, M2, M3, ratio)
    I3 = ts._psd_integral(3.0, M2, M3, ratio)
    assert float(jnp.abs(I2 / M2 - 1.0)[0]) < 1e-3
    assert float(jnp.abs(I3 / M3 - 1.0)[0]) < 1e-3
