"""Validation of the Shima Monte-Carlo collision-coalescence step.

The headline "simple case" is the **Golovin box test**: with the additive
(Golovin) kernel ``K = b(X_i+X_j)`` the mean-field stochastic coalescence
equation has the exact number decay ``N(t) = N0·exp(-(b/ρ_w)·L·t)`` for *any*
initial size distribution (``L = Σξm/V`` is the conserved liquid water
content). An ensemble of super-droplet boxes must reproduce this, and every
step must conserve the represented water mass ``Σξm`` exactly.

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/atmosphere/microphysics/unit/test_sdm_coalescence.py
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax, random

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.sdm import (
    SDMConfig,
    SuperDropletState,
    coalescence_step,
    represented_number,
    represented_water_mass,
)


def _monodisperse_box(n_sd, radius, multiplicity, dtype=jnp.float64):
    ones = jnp.ones((n_sd,), dtype=dtype)
    return SuperDropletState(
        multiplicity=ones * multiplicity,
        radius=ones * radius,
        solute_mass=ones * 0.0,
        active=ones,
    )


# --------------------------------------------------------------------------
# Exact invariants of a single step
# --------------------------------------------------------------------------
def test_mass_conserved_and_number_non_increasing():
    cfg = SDMConfig(collision_kernel="golovin", golovin_b=1.5e3)
    st = _monodisperse_box(n_sd=512, radius=1.4e-5, multiplicity=5.0e5)
    V, rho, p, T, dt = 1.0, 1.0, 9.0e4, 283.0, 1.0
    m0 = float(represented_water_mass(st).sum())
    n0 = float(represented_number(st))
    st2 = coalescence_step(st, V, rho, p, T, dt, random.PRNGKey(0), cfg)
    m1 = float(represented_water_mass(st2).sum())
    n1 = float(represented_number(st2))
    assert m1 == pytest.approx(m0, rel=1e-9, abs=0.0)   # mass exactly conserved
    assert n1 <= n0                                      # number cannot increase
    assert n1 < n0                                       # some coalescence happened


def test_step_is_deterministic_for_fixed_key():
    cfg = SDMConfig(collision_kernel="golovin")
    st = _monodisperse_box(256, 1.4e-5, 5.0e5)
    args = (1.0, 1.0, 9.0e4, 283.0, 1.0)
    a = coalescence_step(st, *args, random.PRNGKey(7), cfg)
    b = coalescence_step(st, *args, random.PRNGKey(7), cfg)
    assert jnp.array_equal(a.multiplicity, b.multiplicity)
    assert jnp.array_equal(a.radius, b.radius)


def test_coalescence_update_direction_and_formula():
    """Deterministic unequal-pair check of the *asymmetric* Shima update:
    the larger-multiplicity droplet loses count and keeps its radius; the
    smaller-multiplicity droplet keeps its count and GROWS to
    cbrt(gamma R_big^3 + R_small^3), absorbing gamma of the big droplet's solute.
    A swapped direction (big grows / small loses) would fail this even though the
    moment tests would not."""
    from legoesm.atmosphere.physics.microphysics.sdm import golovin_kernel

    b = 1.5e3
    cfg = SDMConfig(collision_kernel="golovin", golovin_b=b)
    xi_big0, xi_small0 = 100.0, 10.0
    R_big0, R_small0 = 5.0e-6, 8.0e-6
    s_big0, s_small0 = 1.0e-16, 2.0e-16
    st = SuperDropletState(
        multiplicity=jnp.array([xi_big0, xi_small0]),
        radius=jnp.array([R_big0, R_small0]),
        solute_mass=jnp.array([s_big0, s_small0]),
        active=jnp.array([1.0, 1.0]),
    )
    V = 1.0e-6
    K = float(golovin_kernel(jnp.asarray(R_big0), jnp.asarray(R_small0), b))
    # n=2 -> scaling=1, P = xi_big*(K/V)*dt; pick dt for ~5 collisions (case A, <10).
    dt = 5.0 * V / (xi_big0 * K)
    st2 = coalescence_step(st, V, 1.0, 9.0e4, 283.0, dt, random.PRNGKey(3), cfg)
    xi2 = np.asarray(st2.multiplicity)
    R2 = np.asarray(st2.radius)
    s2 = np.asarray(st2.solute_mass)

    # index 0 is always the larger-xi predator, index 1 the smaller-xi grower.
    gamma = (xi_big0 - xi2[0]) / xi_small0
    assert 1.0 <= gamma < 10.0                          # coalesced, case A (not full)
    assert xi2[0] == pytest.approx(xi_big0 - gamma * xi_small0)  # predator lost count
    assert xi2[1] == pytest.approx(xi_small0)                    # prey kept its count
    assert R2[0] == pytest.approx(R_big0)                        # predator radius unchanged
    assert R2[1] == pytest.approx(
        (gamma * R_big0**3 + R_small0**3) ** (1.0 / 3.0), rel=1e-9)  # prey grew
    assert s2[1] == pytest.approx(s_small0 + gamma * s_big0, rel=1e-9)  # prey absorbed solute
    assert s2[0] == pytest.approx(s_big0)                        # predator solute unchanged
    assert float(represented_water_mass(st2).sum()) == pytest.approx(
        float(represented_water_mass(st).sum()), rel=1e-9)


def test_no_coalescence_when_kernel_zero_radius():
    """Zero-radius droplets have zero Golovin kernel -> no coalescence, state
    unchanged (sanity that the probability gate works)."""
    cfg = SDMConfig(collision_kernel="golovin")
    st = _monodisperse_box(64, 0.0, 5.0e5)
    st2 = coalescence_step(st, 1.0, 1.0, 9.0e4, 283.0, 1.0, random.PRNGKey(1), cfg)
    assert float(represented_number(st2)) == pytest.approx(float(represented_number(st)))


# --------------------------------------------------------------------------
# Golovin analytic number decay (the simple validation case)
# --------------------------------------------------------------------------
def test_golovin_number_decay_matches_analytic():
    cfg = SDMConfig(collision_kernel="golovin", golovin_b=1.5e3)
    n_sd = 2048
    ensemble = 16
    r0 = 1.4e-5
    N0_density = 1.0e9       # real droplets per m^3
    V = 1.0                  # m^3
    dt = 0.5
    n_steps = 116
    xi0 = N0_density * V / n_sd

    init = _monodisperse_box(n_sd, r0, xi0)
    # Replicate into an ensemble of independent boxes.
    states = jax.tree_util.tree_map(
        lambda a: jnp.broadcast_to(a, (ensemble,) + a.shape), init)
    keys = random.split(random.PRNGKey(2024), ensemble)

    @functools.partial(jax.jit, static_argnums=())
    def run_one(state, key):
        step_keys = random.split(key, n_steps)

        def body(st, k):
            return coalescence_step(st, V, 1.0, 9.0e4, 283.0, dt, k, cfg), None

        final, _ = lax.scan(body, state, step_keys)
        return final

    finals = jax.vmap(run_one)(states, keys)

    # Conserved liquid water content L = Σξm / V (per box; identical at init).
    L = float(represented_water_mass(init).sum()) / V
    b_mass = cfg.golovin_b / constants.rho_water
    t = n_steps * dt
    analytic_ratio = np.exp(-b_mass * L * t)

    n_init = represented_number(init)
    n_final = jax.vmap(represented_number)(finals)
    mean_ratio = float(jnp.mean(n_final / n_init))

    assert mean_ratio == pytest.approx(analytic_ratio, rel=0.03)

    # Independent second-moment check: the second mass moment M2 = Σξm² grows as
    # exp(2 b_mass L t) for the additive kernel (twice the number-decay exponent;
    # not derivable from N(t) alone, so it adversarially probes the pair update).
    pref = 4.0 / 3.0 * np.pi * constants.rho_water

    def _M2(state):
        m = pref * state.radius**3
        return jnp.sum(state.active * state.multiplicity * m**2)

    M2_init = float(_M2(init))
    M2_ratio = float(jnp.mean(jax.vmap(_M2)(finals) / M2_init))
    analytic_M2 = np.exp(2.0 * b_mass * L * t)
    assert M2_ratio == pytest.approx(analytic_M2, rel=0.10)

    # Mass conserved across the whole integration, every box.
    m_init = float(represented_water_mass(init).sum())
    m_final = jax.vmap(lambda s: represented_water_mass(s).sum())(finals)
    assert np.allclose(np.asarray(m_final), m_init, rtol=1e-9)
