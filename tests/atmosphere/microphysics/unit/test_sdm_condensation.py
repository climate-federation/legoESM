"""Unit tests for the Super-Droplet Method diffusional-growth core.

Covers the particle SoA, the config, and the condensation growth law against
the analytic Maxwell-Mason solution (the canonical SDM growth check), plus
sign/monotonicity and Köhler-equilibrium behaviour.

Run with x64 for the analytic comparisons::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/atmosphere/microphysics/unit/test_sdm_condensation.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.thermo import saturation_vapor_pressure
from legoesm.atmosphere.physics.microphysics.sdm import (
    SDMConfig,
    SuperDropletState,
    drsq_dt,
    drsq_dt_jac,
    integrate_radius,
    make_monodisperse,
    represented_water_mass,
    water_mass_per_droplet,
)


def _drsq_dt_oracle(R, S, T, N_s, include_curvature, include_solute):
    """Independent re-implementation of d(R²)/dt for cross-checking the module.

    Same physical constants (values are repo policy, not under test) but a
    separate algebraic path, so it catches power/sign/coefficient/d_cf bugs:
    ``α + β/R + γ/R³`` with the full Fukuta-Walter Knudsen correction.
    """
    D = constants.D_vapor
    K = constants.k_air
    Rv = constants.R_v
    rho_l = constants.rho_water
    L = constants.L_v
    e_s = float(saturation_vapor_pressure(jnp.asarray(T)))
    lam = 2.0 * D / np.sqrt(8.0 * T * Rv / np.pi)
    Kn = lam / R
    d_cf = (1.0 + Kn) / (1.0 + 2.0 * Kn * (1.0 + Kn))
    F_k = (L / (Rv * T) - 1.0) * (L * rho_l) / (K * T)
    F_d = (rho_l * Rv * T) / (d_cf * D * e_s)
    denom = F_k + F_d
    out = 2.0 * (S - 1.0) / denom
    if include_curvature:
        a = 2.0 * constants.sigma_water / (Rv * rho_l)
        out -= 2.0 * (a / T) / denom / R
    if include_solute:
        b = (3.0 / (4.0 * np.pi)) * (constants.M_H2O * 1.0e-3 / rho_l)
        out += 2.0 * b * N_s / denom / R**3
    return out


# --------------------------------------------------------------------------
# Particles + config
# --------------------------------------------------------------------------
def test_make_monodisperse_shapes_and_values():
    st = make_monodisperse(n_sd=5, radius=1.0e-5, multiplicity=1.0e8)
    assert st.radius.shape == (5,)
    assert jnp.allclose(st.radius, 1.0e-5)
    assert jnp.allclose(st.multiplicity, 1.0e8)
    assert jnp.allclose(st.active, 1.0)
    assert jnp.allclose(st.solute_mass, 0.0)
    # NamedTuple is a pytree
    leaves = jax.tree_util.tree_leaves(st)
    assert len(leaves) == 4


def test_water_mass_helpers():
    R = 1.0e-5
    xi = 1.0e8
    st = make_monodisperse(n_sd=3, radius=R, multiplicity=xi)
    m_w = 4.0 / 3.0 * np.pi * constants.rho_water * R**3
    assert jnp.allclose(water_mass_per_droplet(st), m_w, rtol=1e-12)
    assert jnp.allclose(represented_water_mass(st), xi * m_w, rtol=1e-12)
    # inactive droplets contribute no represented mass
    st_off = st._replace(active=jnp.zeros_like(st.active))
    assert jnp.allclose(represented_water_mass(st_off), 0.0)


def test_config_defaults_and_unknown_integrator():
    cfg = SDMConfig()
    assert cfg.condensation_integrator == "rk4"
    assert cfg.n_substeps_condensation >= 1
    st = make_monodisperse(n_sd=2, radius=1e-5, multiplicity=1e8)
    with pytest.raises(ValueError, match="Unknown SDM condensation_integrator"):
        integrate_radius(st, S=1.01, T=283.0, dt=1.0,
                         cfg=cfg._replace(condensation_integrator="bogus"))
    with pytest.raises(ValueError, match="n_substeps_condensation must be"):
        integrate_radius(st, S=1.01, T=283.0, dt=1.0,
                         cfg=cfg._replace(n_substeps_condensation=0))


# --------------------------------------------------------------------------
# Growth law vs analytic Maxwell
# --------------------------------------------------------------------------
@pytest.mark.parametrize("include_curvature,include_solute", [
    (False, False), (True, False), (False, True), (True, True),
])
def test_drsq_dt_matches_oracle_exactly(include_curvature, include_solute):
    """drsq_dt reproduces the independent algebraic oracle to ~machine
    precision (abs=0), term by term. Catches wrong powers (/R vs /R³), a wrong
    a/T placement, a missing factor of 2, or a dropped d_cf — none of which the
    earlier sign-only checks would have caught."""
    T = 283.0
    S = 1.005
    R = 2.0e-6          # small enough that curvature/solute/Knudsen all matter
    N_s = 3.0e-15       # solute moles
    e_s = float(saturation_vapor_pressure(jnp.asarray(T)))
    got = float(drsq_dt(jnp.asarray(R) ** 2, S, T, e_s, jnp.asarray(N_s),
                        include_curvature, include_solute))
    expected = _drsq_dt_oracle(R, S, T, N_s, include_curvature, include_solute)
    assert got == pytest.approx(expected, rel=1e-12, abs=0.0)


def test_euler_single_step_is_exact():
    """One forward-Euler sub-step must equal R0² + dt·drsq_dt(R0²) exactly —
    validates the integrator wiring (that integrate_radius actually advances by
    the drsq_dt RHS) independently of the RHS's physical correctness."""
    T = 283.0
    S = 1.01
    R0 = 1.5e-5
    dt = 0.5
    cfg = SDMConfig(include_curvature=False, include_solute=False,
                    n_substeps_condensation=1, condensation_integrator="euler")
    st = make_monodisperse(n_sd=1, radius=R0, multiplicity=1.0)
    e_s = float(saturation_vapor_pressure(jnp.asarray(T)))
    rhs = float(drsq_dt(jnp.asarray(R0) ** 2, S, T, e_s, jnp.asarray(0.0),
                        False, False))
    st2 = integrate_radius(st, S=S, T=T, dt=dt, cfg=cfg)
    assert float(st2.radius[0]) ** 2 == pytest.approx(R0**2 + dt * rhs,
                                                      rel=1e-12, abs=0.0)
    assert float(st2.radius[0]) > R0  # supersaturated -> growth


def test_rk4_more_accurate_than_euler_on_growth():
    """RK4 with 1 sub-step is closer to a fine-Euler reference than 1 Euler
    sub-step (4th vs 1st order), confirming the higher-order integrator wins."""
    T, S, R0, dt = 283.0, 1.02, 5.0e-6, 5.0
    st = make_monodisperse(n_sd=1, radius=R0, multiplicity=1.0)
    base = SDMConfig(include_curvature=True, include_solute=False)
    ref = float(integrate_radius(
        st, S, T, dt,
        base._replace(condensation_integrator="euler",
                      n_substeps_condensation=20000)).radius[0])
    r_eul = float(integrate_radius(
        st, S, T, dt,
        base._replace(condensation_integrator="euler",
                      n_substeps_condensation=1)).radius[0])
    r_rk4 = float(integrate_radius(
        st, S, T, dt,
        base._replace(condensation_integrator="rk4",
                      n_substeps_condensation=1)).radius[0])
    assert abs(r_rk4 - ref) < abs(r_eul - ref)


def test_evaporation_shrinks_droplet():
    cfg = SDMConfig(include_curvature=False, include_solute=False)
    st = make_monodisperse(n_sd=1, radius=2.0e-5, multiplicity=1.0)
    st2 = integrate_radius(st, S=0.9, T=283.0, dt=5.0, cfg=cfg)
    assert float(st2.radius[0]) < 2.0e-5


def test_growth_rate_monotone_in_supersaturation():
    cfg = SDMConfig(include_curvature=False, include_solute=False)
    st = make_monodisperse(n_sd=1, radius=1.5e-5, multiplicity=1.0)
    r_lo = integrate_radius(st, S=1.005, T=283.0, dt=2.0, cfg=cfg).radius[0]
    r_hi = integrate_radius(st, S=1.02, T=283.0, dt=2.0, cfg=cfg).radius[0]
    assert float(r_hi) > float(r_lo) > 1.5e-5


def test_inactive_droplet_does_not_grow():
    cfg = SDMConfig(include_curvature=False, include_solute=False)
    st = make_monodisperse(n_sd=2, radius=1.5e-5, multiplicity=1.0)
    st = st._replace(active=jnp.array([1.0, 0.0], dtype=st.radius.dtype))
    st2 = integrate_radius(st, S=1.05, T=283.0, dt=5.0, cfg=cfg)
    assert float(st2.radius[0]) > 1.5e-5      # active grew
    assert float(st2.radius[1]) == pytest.approx(1.5e-5)  # inactive unchanged


# --------------------------------------------------------------------------
# Kelvin / Köhler behaviour
# --------------------------------------------------------------------------
def test_curvature_suppresses_growth():
    """The Kelvin term raises the equilibrium vapor pressure over a curved
    surface, so a small pure droplet grows slower (smaller dR²/dt) with the
    curvature term than without it, at the same supersaturation."""
    T = 283.0
    S = 1.005
    R = 0.5e-6  # small droplet -> curvature matters
    e_s = float(saturation_vapor_pressure(jnp.asarray(T)))
    rsq = jnp.asarray(R) ** 2
    without = float(drsq_dt(rsq, S, T, e_s, jnp.asarray(0.0),
                            include_curvature=False, include_solute=False))
    with_curv = float(drsq_dt(rsq, S, T, e_s, jnp.asarray(0.0),
                              include_curvature=True, include_solute=False))
    assert with_curv < without


def test_kohler_equilibrium_radius_is_fixed_point():
    """A soluble haze droplet at S<1 has a finite equilibrium radius where
    dR²/dt = 0; below it dR>0, above it dR<0 (stable equilibrium)."""
    T = 283.0
    S = 0.98
    e_s = float(saturation_vapor_pressure(jnp.asarray(T)))
    # NaCl haze particle, dry mass ~ 1e-16 kg
    m_s = 1.0e-16
    N_s = m_s * 2.0 / 0.05844

    def rate(R):
        return float(drsq_dt(jnp.asarray(R) ** 2, S, T, e_s, jnp.asarray(N_s),
                             include_curvature=True, include_solute=True))

    # Scan radii to bracket the equilibrium (sign change of dR²/dt).
    radii = np.geomspace(5e-8, 5e-6, 200)
    rates = np.array([rate(R) for R in radii])
    sign_changes = np.where(np.diff(np.sign(rates)) != 0)[0]
    assert sign_changes.size >= 1  # at least one equilibrium radius exists
    # Stable equilibrium: rate goes from + (grow) to - (shrink) as R increases.
    i = sign_changes[0]
    assert rates[i] > 0.0 >= rates[i + 1]


# --------------------------------------------------------------------------
# Adaptive (ERF stiffness-based) integrator
# --------------------------------------------------------------------------
def _jac_oracle(R, T, e_s, N_s, include_curvature, include_solute):
    """Independent transcription of the ERF rhs_jac (the oracle's APPROXIMATE
    Jacobian: F_k+F_d incl. d_cf evaluated at R, but its R-dependence
    neglected — so it is NOT jax.grad(drsq_dt), and is 0 for pure Maxwell)."""
    D, K = constants.D_vapor, constants.k_air
    Rv, rho_l, L = constants.R_v, constants.rho_water, constants.L_v
    lam = 2.0 * D / np.sqrt(8.0 * T * Rv / np.pi)
    Kn = lam / R
    dcf = (1.0 + Kn) / (1.0 + 2.0 * Kn * (1.0 + Kn))
    F_k = (L / (Rv * T) - 1.0) * (L * rho_l) / (K * T)
    F_d = (rho_l * Rv * T) / (dcf * D * e_s)
    denom = F_k + F_d
    out = 0.0
    if include_curvature:
        beta = -2.0 * (2.0 * constants.sigma_water / (Rv * rho_l) / T) / denom
        out -= 0.5 * beta / R**3
    if include_solute:
        gamma = 2.0 * (3.0 / (4.0 * np.pi)) * (constants.M_H2O * 1e-3 / rho_l) * N_s / denom
        out -= 1.5 * gamma / R**5
    return out


@pytest.mark.parametrize("include_curvature,include_solute", [
    (True, False), (False, True), (True, True), (False, False),
])
def test_drsq_dt_jac_matches_erf_oracle(include_curvature, include_solute):
    """drsq_dt_jac reproduces the ERF rhs_jac transcription exactly (incl. the
    pure-Maxwell jac == 0 case — the oracle neglects the d_cf(R) sensitivity,
    so jax.grad(drsq_dt) would NOT be the right reference here)."""
    T = 283.0
    e_s = float(saturation_vapor_pressure(jnp.asarray(T)))
    N_s = 3.0e-15
    for R in (5.0e-7, 2.0e-6, 2.0e-5):
        u = jnp.asarray(R) ** 2
        got = float(drsq_dt_jac(u, T, e_s, jnp.asarray(N_s),
                                include_curvature, include_solute))
        expected = _jac_oracle(R, T, e_s, N_s, include_curvature, include_solute)
        assert got == pytest.approx(expected, rel=1e-12, abs=1e-300)


def test_adaptive_equals_single_rk4_for_pure_maxwell():
    """tau = 0 (no curvature/solute) -> the adaptive integrator takes ONE RK4
    step over the whole interval (ERF cfl/0 -> inf, limited to t_final), so it
    must match the fixed single-substep RK4 bitwise-tight."""
    base = SDMConfig(include_curvature=False, include_solute=False)
    st = make_monodisperse(n_sd=4, radius=1.5e-5, multiplicity=1.0)
    r_fix = integrate_radius(st, 1.01, 283.0, 2.0,
                             base._replace(condensation_integrator="rk4",
                                           n_substeps_condensation=1)).radius
    r_ada = integrate_radius(st, 1.01, 283.0, 2.0,
                             base._replace(condensation_integrator="rk4_adaptive")).radius
    assert jnp.allclose(r_fix, r_ada, rtol=1e-14)


def test_adaptive_handles_stiff_kohler_better_than_fixed():
    """A sub-micron solute droplet near its Köhler equilibrium is stiff: a
    single fixed RK4 step is inaccurate/oscillatory, the adaptive integrator
    must land closer to a dense-Euler reference."""
    cfg0 = SDMConfig(include_curvature=True, include_solute=True)
    st = make_monodisperse(n_sd=1, radius=2.0e-7, multiplicity=1.0,
                           solute_mass=1.0e-16)
    S, T, dt = 0.99, 283.0, 5.0
    ref = float(integrate_radius(
        st, S, T, dt, cfg0._replace(condensation_integrator="euler",
                                    n_substeps_condensation=200000)).radius[0])
    r_fix = float(integrate_radius(
        st, S, T, dt, cfg0._replace(condensation_integrator="rk4",
                                    n_substeps_condensation=1)).radius[0])
    r_ada = float(integrate_radius(
        st, S, T, dt, cfg0._replace(condensation_integrator="rk4_adaptive")).radius[0])
    assert abs(r_ada - ref) <= abs(r_fix - ref)
    assert abs(r_ada - ref) / ref < 1e-3      # adaptive is accurate outright


def test_adaptive_steady_state_exit_at_saturation():
    """S=1, no curvature/solute: zero RHS -> first step is a no-op and the
    steady exit fires; radius unchanged exactly."""
    cfg = SDMConfig(include_curvature=False, include_solute=False,
                    condensation_integrator="rk4_adaptive")
    st = make_monodisperse(n_sd=3, radius=8.0e-6, multiplicity=1.0)
    out = integrate_radius(st, 1.0, 283.0, 100.0, cfg)
    assert jnp.allclose(out.radius, st.radius, rtol=0.0, atol=0.0)


def test_adaptive_step_cap_counts_accepted_steps_only():
    """An evaporating solute droplet takes early REJECTED (halved) attempts
    before settling toward its Köhler equilibrium. Halvings must not consume
    the accepted-step budget (ERF n_step counts accepted only): the default
    cap of 100 must give the same converged answer as a huge cap. Before the
    fix, halvings burned the budget and a low cap silently truncated."""
    cfg0 = SDMConfig(include_curvature=True, include_solute=True,
                     condensation_integrator="rk4_adaptive")
    st = make_monodisperse(n_sd=1, radius=1.0e-6, multiplicity=1.0,
                           solute_mass=1.0e-16)
    S, T, dt = 0.97, 283.0, 50.0
    r_100 = float(integrate_radius(
        st, S, T, dt, cfg0._replace(adaptive_max_steps=100)).radius[0])
    r_big = float(integrate_radius(
        st, S, T, dt, cfg0._replace(adaptive_max_steps=10000)).radius[0])
    assert r_100 == pytest.approx(r_big, rel=1e-12, abs=0.0)
    # and it actually moved toward the haze equilibrium (shrank)
    assert r_100 < 1.0e-6


def test_be_matches_reference_on_stiff_kohler():
    """Implicit BE (Newton) must land on the same dense-Euler reference as the
    adaptive explicit integrator for a stiff sub-micron Köhler droplet."""
    cfg0 = SDMConfig(include_curvature=True, include_solute=True)
    st = make_monodisperse(n_sd=1, radius=2.0e-7, multiplicity=1.0,
                           solute_mass=1.0e-16)
    S, T, dt = 0.99, 283.0, 5.0
    ref = float(integrate_radius(
        st, S, T, dt, cfg0._replace(condensation_integrator="euler",
                                    n_substeps_condensation=200000)).radius[0])
    r_be = float(integrate_radius(
        st, S, T, dt, cfg0._replace(condensation_integrator="be")).radius[0])
    assert r_be == pytest.approx(ref, rel=1e-3)


def test_be_exact_for_pure_maxwell():
    """Constant RHS (pure Maxwell): BE's update u + dt·alpha is EXACT, so one
    adaptive-BE integration must match the analytic increment tightly."""
    cfg = SDMConfig(include_curvature=False, include_solute=False,
                    condensation_integrator="be")
    R0, S, T, dt = 1.5e-5, 1.01, 283.0, 2.0
    st = make_monodisperse(n_sd=1, radius=R0, multiplicity=1.0)
    e_s = float(saturation_vapor_pressure(jnp.asarray(T)))
    rhs = float(drsq_dt(jnp.asarray(R0) ** 2, S, T, e_s, jnp.asarray(0.0),
                        False, False))
    out = integrate_radius(st, S, T, dt, cfg)
    # RHS is u-independent at fixed R-arg... it varies via d_cf(R); BE solves
    # with the implicit endpoint, so compare against the dense reference.
    ref = float(integrate_radius(
        st, S, T, dt, SDMConfig(include_curvature=False, include_solute=False,
                                condensation_integrator="euler",
                                n_substeps_condensation=100000)).radius[0])
    assert float(out.radius[0]) == pytest.approx(ref, rel=1e-4)
    assert float(out.radius[0]) ** 2 > R0**2 + 0.5 * dt * rhs  # grew sensibly


@pytest.mark.parametrize("method", ["rk4_adaptive", "be", "cn", "dirk2"])
def test_all_adaptive_integrators_agree_on_stiff_kohler(method):
    """The whole ERF adaptive family (explicit rk4, BE, CN, DIRK2) must land on
    the same dense-Euler reference for the stiff sub-micron Köhler droplet."""
    cfg0 = SDMConfig(include_curvature=True, include_solute=True)
    st = make_monodisperse(n_sd=1, radius=2.0e-7, multiplicity=1.0,
                           solute_mass=1.0e-16)
    S, T, dt = 0.99, 283.0, 5.0
    ref = float(integrate_radius(
        st, S, T, dt, cfg0._replace(condensation_integrator="euler",
                                    n_substeps_condensation=200000)).radius[0])
    got = float(integrate_radius(
        st, S, T, dt, cfg0._replace(condensation_integrator=method)).radius[0])
    assert got == pytest.approx(ref, rel=2e-3)


def test_cn_dirk2_single_attempt_matches_exact_stage_algebra():
    """Single-attempt stage-algebra oracle: solve each implicit stage equation
    EXACTLY (brentq root of mu*u' - F(u') - rhs_const) and rebuild the oracle's
    update. Tight tolerance (rel 1e-6, Newton rtol-limited) — a wrong CN mu
    (1/dt instead of 2/dt) shifts the stage ROOT itself and fails, which the
    adaptive end-to-end tests cannot see (codex probe)."""
    from scipy.optimize import brentq
    from legoesm.atmosphere.physics.microphysics.sdm.condensation import (
        _make_attempt,
    )

    T, S = 283.0, 0.99
    e_s = float(saturation_vapor_pressure(jnp.asarray(T)))
    N_s = 1.0e-16 * 2.0 / 0.05844         # nonlinear: curvature+solute active
    u0 = (3.0e-7) ** 2
    dt = 0.05

    def F(u):
        return float(drsq_dt(jnp.asarray(u), S, T, e_s, jnp.asarray(N_s),
                             True, True))

    def solve(mu, rhs_const, lo=1e-18, hi=1e-10):
        return brentq(lambda u: mu * u - F(u) - rhs_const, lo, hi,
                      xtol=1e-30, rtol=8.9e-16)

    args = (jnp.asarray(S), jnp.asarray(T), jnp.asarray(e_s), jnp.asarray(N_s),
            True, True,
            jnp.asarray(1e-12), jnp.asarray(1e-40), jnp.asarray(1e-14),
            jnp.asarray(60, jnp.int32), jnp.float64)

    # CN oracle: mu = 2/dt; u2 root of mu*u2 - F(u2) = mu*(u0 + dt/2*f1);
    # update u0 + dt/2*(f1 + F(u2)).
    f1 = F(u0)
    mu_cn = 1.0 / (0.5 * dt)
    u2_cn = solve(mu_cn, mu_cn * (u0 + 0.5 * dt * f1))
    expected_cn = u0 + 0.5 * dt * (f1 + F(u2_cn))
    got_cn, ok_cn = _make_attempt("cn", *args)(jnp.asarray(u0), jnp.asarray(dt))
    assert bool(ok_cn)
    assert float(got_cn) == pytest.approx(expected_cn, rel=1e-6, abs=0.0)
    # discrimination: the wrong-mu (1/dt) construction must differ measurably
    u2_wrong = solve(1.0 / dt, (1.0 / dt) * (u0 + 0.5 * dt * f1))
    expected_wrong = u0 + 0.5 * dt * (f1 + F(u2_wrong))
    assert abs(expected_wrong - expected_cn) / abs(expected_cn) > 1e-5

    # DIRK2 oracle: mu = 1/dt; u1 root of mu*u1 - F(u1) = mu*u0; f1d = F(u1);
    # u2 root of mu*u2 - F(u2) = mu*(u0 - dt*f1d); update u0 + dt/2*(f1d+F(u2)).
    mu_d = 1.0 / dt
    u1_d = solve(mu_d, mu_d * u0)
    f1_d = F(u1_d)
    u2_d = solve(mu_d, mu_d * (u0 - dt * f1_d))
    expected_d = u0 + 0.5 * dt * (f1_d + F(u2_d))
    got_d, ok_d = _make_attempt("dirk2", *args)(jnp.asarray(u0), jnp.asarray(dt))
    assert bool(ok_d)
    assert float(got_d) == pytest.approx(expected_d, rel=1e-6, abs=0.0)


@pytest.mark.parametrize("method", ["cn", "dirk2"])
def test_cn_dirk2_steady_noop_and_growth(method):
    cfg = SDMConfig(include_curvature=False, include_solute=False,
                    condensation_integrator=method)
    st = make_monodisperse(n_sd=2, radius=8.0e-6, multiplicity=1.0)
    out = integrate_radius(st, 1.0, 283.0, 100.0, cfg)
    assert jnp.allclose(out.radius, st.radius, rtol=0.0, atol=0.0)  # S=1 no-op
    out2 = integrate_radius(st, 1.02, 283.0, 2.0, cfg)
    assert bool(jnp.all(out2.radius > st.radius))                    # grows
    # matches the dense reference for the smooth case
    ref = float(integrate_radius(
        st, 1.02, 283.0, 2.0,
        SDMConfig(include_curvature=False, include_solute=False,
                  condensation_integrator="euler",
                  n_substeps_condensation=100000)).radius[0])
    assert float(out2.radius[0]) == pytest.approx(ref, rel=1e-4)


def test_be_steady_and_haze_equilibrium():
    """BE at S=1 (no curvature/solute) is an exact no-op; a soluble haze
    droplet at S<1 relaxes toward its stable Köhler equilibrium."""
    cfg = SDMConfig(include_curvature=False, include_solute=False,
                    condensation_integrator="be")
    st = make_monodisperse(n_sd=2, radius=8.0e-6, multiplicity=1.0)
    out = integrate_radius(st, 1.0, 283.0, 100.0, cfg)
    assert jnp.allclose(out.radius, st.radius, rtol=0.0, atol=0.0)

    cfg2 = SDMConfig(include_curvature=True, include_solute=True,
                     condensation_integrator="be")
    hz = make_monodisperse(n_sd=1, radius=1.0e-6, multiplicity=1.0,
                           solute_mass=1.0e-16)
    r1 = float(integrate_radius(hz, 0.97, 283.0, 50.0, cfg2).radius[0])
    r2 = float(integrate_radius(hz._replace(radius=jnp.asarray([r1])),
                                0.97, 283.0, 50.0, cfg2).radius[0])
    assert r1 < 1.0e-6                       # shrinks toward equilibrium
    assert abs(r2 - r1) < abs(1.0e-6 - r1)   # converging (contraction)


def test_adaptive_jit_and_unknown_integrator():
    cfg = SDMConfig(condensation_integrator="rk4_adaptive",
                    include_curvature=True, include_solute=False)
    st = make_monodisperse(n_sd=16, radius=1.0e-5, multiplicity=1.0)
    run = jax.jit(integrate_radius, static_argnames=("cfg",))
    out = run(st, 1.02, 283.0, 2.0, cfg)
    assert bool(jnp.all(jnp.isfinite(out.radius)))
    assert bool(jnp.all(out.radius > st.radius))   # supersaturated -> grew
    with pytest.raises(ValueError, match="Unknown SDM condensation_integrator"):
        integrate_radius(st, 1.0, 283.0, 1.0,
                         cfg._replace(condensation_integrator="rk3bs"))


# --------------------------------------------------------------------------
# JIT + differentiability
# --------------------------------------------------------------------------
def test_integrate_radius_jit_and_grad():
    cfg = SDMConfig(include_curvature=True, include_solute=False,
                    n_substeps_condensation=4)
    st = make_monodisperse(n_sd=8, radius=1.0e-5, multiplicity=1.0e8)

    @jax.jit
    def final_mean_radius(S):
        st2 = integrate_radius(st, S=S, T=283.0, dt=2.0, cfg=cfg)
        return jnp.mean(st2.radius)

    val = final_mean_radius(1.01)
    assert jnp.isfinite(val)
    # Growth increases with S -> positive gradient of final radius wrt S.
    g = jax.grad(final_mean_radius)(1.01)
    assert jnp.isfinite(g)
    assert float(g) > 0.0
