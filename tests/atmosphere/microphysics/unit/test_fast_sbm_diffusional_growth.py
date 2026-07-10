"""Fast-SBM diffusional-growth coefficients (oracle JERRATE/JERTIMESC).

Headline check: with ventilation off (V=0) and matched diffusivity, the bin
growth coefficient must reproduce the SDM ``drsq_dt`` Maxwell core — two
independent implementations of the same physics pinned against each other:

    dm/dt = B s  and  m = (4/3)πρ r³  ⇒  d(r²)/dt = B s / (2πρ_w r)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.fast_sbm import (
    FastSBMConfig,
    drop_growth_coefficient,
    mass_doubling_grid,
    radius_from_mass,
    supersat_relaxation_integral,
    vapor_diffusivity,
    ventilation_factor,
    discretize_lognormal,
    bin_mass_widths,
)
from legoesm.atmosphere.physics.microphysics.sdm.condensation import drsq_dt
from legoesm.thermo import saturation_vapor_pressure

jax.config.update("jax_enable_x64", True)

T0 = 283.0
P0 = 9.0e4


def test_diffusivity_reference_state():
    cfg = FastSBMConfig()
    # At (T_freeze, p_atm_std) the law collapses to the reference value.
    D = float(vapor_diffusivity(jnp.asarray(constants.T_freeze),
                                jnp.asarray(constants.p_atm_std), cfg))
    assert D == pytest.approx(cfg.d_vapor_ref_m2s, rel=1e-12)
    # Lower pressure / warmer air → faster diffusion (signs of the law).
    assert float(vapor_diffusivity(jnp.asarray(T0), jnp.asarray(P0), cfg)) > D


def test_ventilation_limits_and_cap():
    cfg = FastSBMConfig()
    m = mass_doubling_grid()
    # Still air: f = 1 exactly (Re = 0).
    f0 = ventilation_factor(m, jnp.zeros_like(m), T0, P0, config=cfg)
    np.testing.assert_allclose(np.asarray(f0), 1.0, rtol=0, atol=0)
    # Fast-falling large drops: monotone ≥ 1, capped at ventilation_max.
    v = jnp.linspace(0.0, 10.0, m.shape[0])
    f = ventilation_factor(m, v, T0, P0, config=cfg)
    assert np.all(np.asarray(f) >= 1.0 - 1e-12)
    assert np.all(np.asarray(f) <= cfg.ventilation_max + 1e-12)
    big = ventilation_factor(m, 100.0 * jnp.ones_like(m), T0, P0, config=cfg)
    assert float(big[-1]) == pytest.approx(cfg.ventilation_max)


def test_ventilation_grad_finite_at_zero_v_term():
    """Reverse-mode AD guard at Re = 0 (the shipped condensation path).

    ``warm_condensation_step`` calls ``drop_growth_coefficient`` with
    v_term = 0, so reynolds = 0 for every bin.  A shared
    ``x = sqrt(reynolds)`` fed to both jnp.where branches routes a
    0*inf = NaN cotangent through sqrt into every traced input of
    reynolds (masses grid, nu_air_ref_m2s) — the same masked-branch trap
    guarded in nucleation.critical_dry_radius.  The fix keeps the primal
    identical (low branch uses x**2 = Re*Sc^(2/3) directly; high branch
    clamps the sqrt argument) and must yield finite gradients w.r.t. BOTH
    masses and v_term at exactly zero terminal velocity.
    """
    cfg = FastSBMConfig()
    m = mass_doubling_grid()

    def vent_sum_wrt_masses(masses):
        return jnp.sum(ventilation_factor(
            masses, jnp.zeros_like(masses), T0, P0, config=cfg))

    def vent_sum_wrt_v(v_term):
        return jnp.sum(ventilation_factor(m, v_term, T0, P0, config=cfg))

    g_m = jax.grad(vent_sum_wrt_masses)(m)
    g_v = jax.grad(vent_sum_wrt_v)(jnp.zeros_like(m))
    assert np.all(np.isfinite(np.asarray(g_m))), (
        "ventilation_factor grad w.r.t. masses at v_term=0 contains NaN/Inf")
    assert np.all(np.isfinite(np.asarray(g_v))), (
        "ventilation_factor grad w.r.t. v_term at v_term=0 contains NaN/Inf")
    # Primal unchanged: f = 1 exactly in still air.
    f0 = ventilation_factor(m, jnp.zeros_like(m), T0, P0, config=cfg)
    np.testing.assert_allclose(np.asarray(f0), 1.0, rtol=0, atol=0)

    # And the full growth-coefficient path (the shipped caller) is AD-safe.
    def growth_sum(masses):
        return jnp.sum(drop_growth_coefficient(
            masses, jnp.asarray(T0), jnp.asarray(P0),
            jnp.zeros_like(masses), config=cfg))

    g_b = jax.grad(growth_sum)(m)
    assert np.all(np.isfinite(np.asarray(g_b))), (
        "drop_growth_coefficient grad w.r.t. masses at v_term=0 not finite")


def test_growth_coefficient_matches_sdm_maxwell_core():
    # Ventilation off (V=0) + diffusivity matched to the SDM constant at
    # the evaluation state + Knudsen negligible (large drops) → the two
    # implementations must agree to high precision.
    m = mass_doubling_grid()
    r = radius_from_mass(m)
    big = slice(12, 30)               # 30 um .. 1.6 mm: Kn ~ 1e-3, d_cf ≈ 1
    T = jnp.asarray(T0)
    p = jnp.asarray(P0)
    # Choose d_ref so vapor_diffusivity(T0, P0) == constants.D_vapor.
    cfg = FastSBMConfig(
        d_vapor_ref_m2s=float(
            constants.D_vapor
            / ((constants.p_atm_std / P0)
               * (T0 / constants.T_freeze) ** FastSBMConfig().diffusivity_T_exponent)
        )
    )
    B = drop_growth_coefficient(m, T, p, jnp.zeros_like(m), config=cfg)
    s = 0.01
    dm_dt_bin = np.asarray(B)[big] * s

    e_s = saturation_vapor_pressure(T)
    drsq = np.asarray(drsq_dt(r[big] ** 2, jnp.asarray(1.0 + s), T, e_s,
                              jnp.zeros_like(r[big]),
                              include_curvature=False, include_solute=False))
    dm_dt_sdm = drsq * 2.0 * np.pi * constants.rho_water * np.asarray(r)[big]
    # d_cf (Knudsen) is ~0.999 at 30 um — tolerance covers it.
    np.testing.assert_allclose(dm_dt_bin, dm_dt_sdm, rtol=2e-3)


def test_growth_coefficient_positive_and_grows_with_size():
    m = mass_doubling_grid()
    B = drop_growth_coefficient(m, jnp.asarray(T0), jnp.asarray(P0),
                                jnp.zeros_like(m))
    b = np.asarray(B)
    assert np.all(b > 0.0)
    assert np.all(np.diff(b) > 0.0)   # capacitance = r grows with mass


def test_relaxation_integral_linear_and_explicit():
    m = mass_doubling_grid()
    f = discretize_lognormal(m, 1.0e8, 10.0e-6, 1.4)
    B = drop_growth_coefficient(m, jnp.asarray(T0), jnp.asarray(P0),
                                jnp.zeros_like(m))
    rho_air = jnp.asarray(1.1)
    sfn = supersat_relaxation_integral(f, m, B, rho_air)
    explicit = float(jnp.sum(f * B * bin_mass_widths(m)) / rho_air)
    assert float(sfn) == pytest.approx(explicit, rel=1e-14)
    np.testing.assert_allclose(
        float(supersat_relaxation_integral(2.0 * f, m, B, rho_air)),
        2.0 * float(sfn), rtol=1e-14)
    # Physical scale: ~100 cm^-3 of 10 um droplets relax supersaturation on
    # O(seconds) — SFN·(quasi-steady thermo factor ~ O(1..10)) ⇒ SFN in
    # a broad but bounded window.
    assert 1.0e-3 < float(sfn) < 1.0e2


def test_growth_differentiable_in_T_and_p():
    m = mass_doubling_grid()

    def sfn_of(Tp):
        T, p = Tp
        f = discretize_lognormal(m, 1.0e8, 10.0e-6, 1.4)
        B = drop_growth_coefficient(m, T, p, jnp.zeros_like(m))
        return supersat_relaxation_integral(f, m, B, jnp.asarray(1.1))

    g = jax.grad(sfn_of)(jnp.array([T0, P0]))
    assert np.all(np.isfinite(np.asarray(g)))
