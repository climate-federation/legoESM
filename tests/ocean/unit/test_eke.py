"""Direct unit tests for the prognostic EKE closure (build-spec gate E1).

Tests the pure closure properties: kappa_GM monotone + nonnegative in E, the
production form (kappa_GM * sigma^2), dissipation sign + E^{3/2} scaling, the
mixing-length floor, and finiteness — independent of the state/step coupling.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.physics.lateral_mixing.eke import (
    EKEConfig,
    eke_kappa_gm,
    eke_local_tendency,
    eke_mixing_length,
)


def test_eke_config_defaults_match_veros_acc():
    cfg = EKEConfig()
    assert cfg.c_k == 0.4
    assert cfg.c_eps == 0.5
    assert cfg.l_min == 100.0


def test_mixing_length_floor():
    cfg = EKEConfig(l_min=100.0)
    L_rossby = jnp.array([10.0, 100.0, 5.0e4])
    L = eke_mixing_length(L_rossby, cfg)
    assert float(L[0]) == 100.0      # floored
    assert float(L[1]) == 100.0      # at the floor
    assert float(L[2]) == 5.0e4      # above the floor, unchanged


def test_kappa_gm_nonnegative_and_monotone_in_E():
    cfg = EKEConfig()
    L = jnp.full((20,), 3.0e4)
    E = jnp.linspace(0.0, 1.0, 20)
    kappa = eke_kappa_gm(E, L, cfg)
    assert jnp.all(kappa >= 0.0), "kappa_GM must be non-negative"
    # monotone non-decreasing in E (sqrt).
    assert jnp.all(jnp.diff(kappa) >= -1e-12), "kappa_GM must be monotone in E"
    # exact form away from the regulariser.
    E1 = jnp.array([0.25]); L1 = jnp.array([3.0e4])
    np.testing.assert_allclose(
        np.asarray(eke_kappa_gm(E1, L1, cfg)),
        np.asarray(cfg.c_k * L1 * jnp.sqrt(E1)), rtol=1e-6,
    )


def test_kappa_gm_capped():
    cfg = EKEConfig(kappa_gm_max=2.0e3)
    kappa = eke_kappa_gm(jnp.array([1.0e6]), jnp.array([1.0e5]), cfg)
    assert float(kappa[0]) == 2.0e3


def test_tendency_zero_at_zero_E():
    """At E=0: production (kappa~0) and dissipation (E^{3/2}=0) both vanish."""
    cfg = EKEConfig()
    E = jnp.zeros((5,)); sigma = jnp.full((5,), 1.0e-5); L = jnp.full((5,), 3.0e4)
    t = eke_local_tendency(E, sigma, L, cfg)
    assert float(jnp.max(jnp.abs(t))) < 1e-15


def test_production_form_and_sign():
    """Production = kappa_GM * sigma^2 >= 0; equals the closed form."""
    cfg = EKEConfig()
    E = jnp.array([0.04]); sigma = jnp.array([2.0e-5]); L = jnp.array([3.0e4])
    # With dissipation subtracted; isolate by checking production-only via a
    # tiny E where dissipation (E^{3/2}) is sub-dominant, plus the closed form.
    kappa = eke_kappa_gm(E, L, cfg)
    prod = kappa * sigma ** 2
    diss = cfg.c_eps * E ** 1.5 / L
    np.testing.assert_allclose(
        np.asarray(eke_local_tendency(E, sigma, L, cfg)),
        np.asarray(prod - diss), rtol=1e-10,
    )
    assert float(prod[0]) >= 0.0


def test_dissipation_dominates_at_large_E():
    """With no production (sigma=0) the tendency is pure dissipation: <= 0 and
    scales as E^{3/2}."""
    cfg = EKEConfig()
    L = jnp.array([3.0e4])
    sigma0 = jnp.array([0.0])
    for E in (jnp.array([0.01]), jnp.array([0.1]), jnp.array([1.0])):
        t = eke_local_tendency(E, sigma0, L, cfg)
        assert float(t[0]) <= 0.0, "dissipation-only tendency must be <= 0"
    # E^{3/2} scaling: doubling... 8x E -> 8^{1.5}=~22.6x dissipation magnitude.
    t1 = -float(eke_local_tendency(jnp.array([0.1]), sigma0, L, cfg)[0])
    t8 = -float(eke_local_tendency(jnp.array([0.8]), sigma0, L, cfg)[0])
    np.testing.assert_allclose(t8 / t1, 8.0 ** 1.5, rtol=1e-6)


def test_tendency_finite_on_field():
    cfg = EKEConfig()
    rng = np.random.default_rng(0)
    E = jnp.asarray(np.abs(rng.standard_normal((8, 16))) * 0.05)
    sigma = jnp.asarray(np.abs(rng.standard_normal((8, 16))) * 1e-5)
    L = jnp.asarray(rng.uniform(1e4, 5e4, (8, 16)))
    t = eke_local_tendency(E, sigma, L, cfg)
    assert t.shape == (8, 16)
    assert jnp.all(jnp.isfinite(t))
