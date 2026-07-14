"""Unit tests for :mod:`legoesm.atmosphere.dynamics.les.tke_sgs_plane`.

Gap #4: the 1.5-order TKE SGS closure for the plane LES.  Analytic checks on
ν_t, mixing length, dissipation, Prandtl, the local-equilibrium TKE (production
= dissipation), the **Smagorinsky-equivalence** of the equilibrium limit, the
physical signs of the TKE source terms, and AD-safety at e=0.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.les.tke_sgs_plane import (
    TKESGSConfig,
    smagorinsky_equivalent_cs,
    tke_dissipation_rate,
    tke_eddy_viscosity,
    tke_equilibrium,
    tke_equilibrium_eddy_viscosity,
    tke_mixing_length,
    tke_tendency,
    tke_turbulent_prandtl,
    validate_tke_config,
)

jax.config.update("jax_enable_x64", True)

_CFG = TKESGSConfig()


def test_eddy_viscosity_formula():
    # ν_t = c_k ℓ √e
    nu = tke_eddy_viscosity(jnp.array(4.0), jnp.array(50.0))
    assert float(nu) == pytest.approx(_CFG.c_k * 50.0 * 2.0, rel=1e-12)


def test_mixing_length_neutral_is_delta():
    ell = tke_mixing_length(jnp.array(100.0), jnp.array(1.0), jnp.array(-1e-4))
    assert float(ell) == pytest.approx(100.0)


def test_mixing_length_stable_limited():
    delta, e, n2 = 100.0, 4.0, 1e-2  # N=0.1, √e=2
    ell = tke_mixing_length(jnp.array(delta), jnp.array(e), jnp.array(n2))
    expected = min(delta, _CFG.stable_length_coeff * 2.0 / 0.1)  # 0.76*20=15.2
    assert float(ell) == pytest.approx(expected, rel=1e-10)
    assert float(ell) < delta


def test_turbulent_prandtl_neutral():
    # ℓ = Δ ⇒ Pr_t = 1/3
    pr = tke_turbulent_prandtl(jnp.array(80.0), jnp.array(80.0))
    assert float(pr) == pytest.approx(1.0 / 3.0, rel=1e-12)


def test_dissipation_formula():
    e, ell, delta = 4.0, 80.0, 80.0  # ℓ/Δ=1 ⇒ C_ε=c0+c1
    eps = tke_dissipation_rate(jnp.array(e), jnp.array(ell), jnp.array(delta))
    c_eps = _CFG.c_eps_0 + _CFG.c_eps_1
    expected = c_eps * (e ** 1.5) / ell
    assert float(eps) == pytest.approx(expected, rel=1e-10)


def test_equilibrium_balances_tendency():
    """At e_eq the local TKE tendency is exactly zero (P = ε)."""
    s2, n2, delta = 1e-4, 0.0, 60.0  # neutral ⇒ ℓ=Δ
    ell = float(tke_mixing_length(jnp.array(delta), jnp.array(1.0), jnp.array(n2)))
    e_eq = tke_equilibrium(jnp.array(s2), jnp.array(n2), jnp.array(ell), jnp.array(delta))
    tend = tke_tendency(e_eq, jnp.array(s2), jnp.array(n2), jnp.array(ell), jnp.array(delta))
    assert float(e_eq) > 0.0
    assert float(tend) == pytest.approx(0.0, abs=1e-12)


def test_equilibrium_balances_with_stable_buoyancy():
    s2, n2, delta = 1e-3, 1e-5, 60.0
    ell = 40.0  # fixed ℓ
    e_eq = tke_equilibrium(jnp.array(s2), jnp.array(n2), jnp.array(ell), jnp.array(delta))
    tend = tke_tendency(e_eq, jnp.array(s2), jnp.array(n2), jnp.array(ell), jnp.array(delta))
    assert float(e_eq) > 0.0
    assert float(tend) == pytest.approx(0.0, abs=1e-12)


def test_stable_branch_is_fixed_l_only():
    """Document the fixed-ℓ contract: in the stable branch ℓ depends on e, so
    recomputing ℓ(e_eq) leaves a NONZERO residual (not the coupled fixed point)."""
    s2, n2, delta = 1e-3, 1e-5, 60.0
    ell0 = 40.0
    e_eq = float(tke_equilibrium(jnp.array(s2), jnp.array(n2),
                                 jnp.array(ell0), jnp.array(delta)))
    # At the frozen ℓ0 the tendency is zero (verified elsewhere); but the
    # self-consistent stable length differs, so the coupled residual is nonzero.
    ell_new = float(tke_mixing_length(jnp.array(delta), jnp.array(e_eq), jnp.array(n2)))
    assert ell_new != pytest.approx(ell0, rel=1e-6)
    resid = float(tke_tendency(jnp.array(e_eq), jnp.array(s2), jnp.array(n2),
                               jnp.array(ell_new), jnp.array(delta)))
    assert abs(resid) > 0.0  # coupled fixed point needs iteration (documented)


def test_equilibrium_zero_when_subcritical():
    # |S|² < N²/Pr_t (strongly stable, weak shear) ⇒ no turbulence.
    e_eq = tke_equilibrium(jnp.array(1e-6), jnp.array(1e-2), jnp.array(50.0), jnp.array(50.0))
    assert float(e_eq) == 0.0


def test_equilibrium_reduces_to_smagorinsky():
    """Neutral local-equilibrium ν_t == Smagorinsky (C_s Δ)²√|S|² with the
    derived C_s = (C_k³/C_ε)^{1/4}."""
    s2, delta = 2.5e-4, 70.0
    nu_tke = float(
        tke_equilibrium_eddy_viscosity(jnp.array(s2), jnp.array(0.0), jnp.array(delta))
    )
    c_s = smagorinsky_equivalent_cs()
    nu_smag = (c_s * delta) ** 2 * np.sqrt(s2)
    assert nu_tke == pytest.approx(nu_smag, rel=1e-10)
    # The derived C_s is the canonical LES value (~0.17-0.21).
    assert 0.15 < c_s < 0.22


def test_smagorinsky_equivalence_with_stratification():
    """With N²≠0 the equilibrium ν_t matches Smagorinsky's −Pr·N² form,
    Pr = 1/Pr_t = 3 at ℓ=Δ."""
    s2, n2, delta = 1e-3, 5e-5, 70.0
    nu_tke = float(
        tke_equilibrium_eddy_viscosity(jnp.array(s2), jnp.array(n2), jnp.array(delta))
    )
    c_s = smagorinsky_equivalent_cs()
    pr_smag = 3.0  # 1/Pr_t at ℓ=Δ (Pr_t=1/3)
    nu_smag = (c_s * delta) ** 2 * np.sqrt(s2 - pr_smag * n2)
    assert nu_tke == pytest.approx(nu_smag, rel=1e-10)


def test_tendency_signs():
    e, ell, delta = 1.0, 60.0, 60.0
    # Pure shear (N²=0): production > 0 ⇒ tendency positive for small e.
    t_shear = float(tke_tendency(jnp.array(1e-3), jnp.array(1e-2), jnp.array(0.0),
                                 jnp.array(ell), jnp.array(delta)))
    assert t_shear > 0.0
    # No shear, stable N²>0: buoyancy sink + dissipation ⇒ tendency negative.
    t_stable = float(tke_tendency(jnp.array(e), jnp.array(0.0), jnp.array(1e-3),
                                  jnp.array(ell), jnp.array(delta)))
    assert t_stable < 0.0
    # No shear, unstable N²<0: buoyancy is a SOURCE.
    p_buoy_unstable = float(tke_tendency(jnp.array(1e-6), jnp.array(0.0), jnp.array(-1e-3),
                                         jnp.array(ell), jnp.array(delta)))
    assert p_buoy_unstable > 0.0


def test_drop_in_subcritical_zero_viscosity_finite_grad():
    """Strongly stable, weak shear ⇒ drop-in ν_t = 0, and grad stays finite."""
    delta = 60.0

    def nu_of(s2):
        return tke_equilibrium_eddy_viscosity(s2, jnp.array(1e-2), jnp.array(delta))

    assert float(nu_of(jnp.array(1e-6))) == 0.0  # |S|² ≪ 3N²
    g = jax.grad(lambda s2: nu_of(s2))(jnp.array(1e-6))
    assert jnp.isfinite(g)


def test_ad_safe_at_zero_tke():
    def loss(e):
        ell = tke_mixing_length(jnp.array(50.0), e, jnp.array(1e-4))
        nu = tke_eddy_viscosity(e, ell)
        eps = tke_dissipation_rate(e, ell, jnp.array(50.0))
        return jnp.sum(nu + eps)

    g = jax.grad(loss)(jnp.array(0.0))
    assert jnp.isfinite(g)


def test_ad_safe_tendency_and_equilibrium_edge_cases():
    # grad of tke_tendency at e=0, N²=0.
    g1 = jax.grad(lambda e: tke_tendency(
        e, jnp.array(1e-4), jnp.array(0.0), jnp.array(50.0), jnp.array(50.0)
    ))(jnp.array(0.0))
    assert jnp.isfinite(g1)
    # grad of tke_equilibrium through the subcritical (floored) branch.
    g2 = jax.grad(lambda s2: tke_equilibrium(
        s2, jnp.array(1e-2), jnp.array(50.0), jnp.array(50.0)
    ))(jnp.array(1e-6))
    assert jnp.isfinite(g2)


def test_validate_config():
    validate_tke_config(TKESGSConfig())  # defaults OK
    for bad in (
        TKESGSConfig(c_k=-0.1),
        TKESGSConfig(stable_length_coeff=0.0),
        TKESGSConfig(c_eps_0=0.0),
        TKESGSConfig(c_eps_0=0.5, c_eps_1=-0.6),  # c_eps_0+c_eps_1 < 0
    ):
        with pytest.raises(ValueError):
            validate_tke_config(bad)


def test_jit_and_vmap():
    e = jnp.linspace(0.0, 5.0, 8)
    s2 = jnp.full((8,), 1e-4)
    n2 = jnp.zeros((8,))
    delta = jnp.full((8,), 60.0)
    fn = jax.jit(jax.vmap(
        lambda e_, s_, n_, d_: tke_tendency(
            e_, s_, n_, tke_mixing_length(d_, e_, n_), d_)
    ))
    out = fn(e, s2, n2, delta)
    assert out.shape == (8,)
    assert bool(jnp.all(jnp.isfinite(out)))
