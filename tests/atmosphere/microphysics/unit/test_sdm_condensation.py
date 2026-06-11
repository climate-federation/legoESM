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
    integrate_radius,
    make_monodisperse,
    represented_water_mass,
    water_mass_per_droplet,
)


def _maxwell_denominator(T: float) -> float:
    """Analytic F_k + F_d at the dilute limit (Kn->0, d_cf->1)."""
    D = constants.D_vapor
    K = constants.k_air
    Rv = constants.R_v
    rho_l = constants.rho_water
    L = constants.L_v
    e_s = float(saturation_vapor_pressure(jnp.asarray(T)))
    F_k = (L / (Rv * T) - 1.0) * (L * rho_l) / (K * T)
    F_d = (rho_l * Rv * T) / (D * e_s)
    return F_k, F_d


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
def test_maxwell_growth_matches_analytic():
    """With curvature/solute off, d(R²)/dt = 2(S-1)/(F_k+F_d) is constant, so a
    single RK4 step reproduces the analytic R² increment to machine precision."""
    T = 283.0
    S = 1.01
    R0 = 1.5e-5
    dt = 2.0
    cfg = SDMConfig(include_curvature=False, include_solute=False,
                    n_substeps_condensation=1, condensation_integrator="rk4")
    st = make_monodisperse(n_sd=1, radius=R0, multiplicity=1.0)
    st2 = integrate_radius(st, S=S, T=T, dt=dt, cfg=cfg)

    F_k, F_d = _maxwell_denominator(T)
    drsq = 2.0 * (S - 1.0) / (F_k + F_d)
    rsq_expected = R0**2 + drsq * dt
    assert float(st2.radius[0]) ** 2 == pytest.approx(rsq_expected, rel=1e-10)
    assert float(st2.radius[0]) > R0  # supersaturated -> growth


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
