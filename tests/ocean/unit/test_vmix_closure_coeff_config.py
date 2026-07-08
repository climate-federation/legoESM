"""Pin the migrated closure coefficients (#518 item 10) to bit-identical defaults.

The Galperin Pr-Ri slope + Bryan-Lewis (1979) background-diffusivity profile
(formerly bare ``_GALPERIN_RI_COEFF`` / ``_BG_DIFF_*`` module literals in
``tke.py``) and the Businger-Dyer MOST constants + u* proxy ratio (formerly
``_BUSINGER_*`` / ``_USTAR_SPEED_RATIO`` in ``kpp.py``) now live in
``TKEConfig`` / ``KPPConfig``.  This guards both the default *values* (so
production stays bit-identical) and that overriding a field actually changes the
scheme output (so the migration is not a dead pass-through).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.ocean.physics.vertical_mixing.config import TKEConfig, KPPConfig
from legoesm.ocean.physics.vertical_mixing.tke import (
    _prandtl_number,
    _bryan_lewis_kappaH_floor,
)
from legoesm.ocean.physics.vertical_mixing.kpp import _kpp_velocity_scales

import pytest

jax.config.update("jax_enable_x64", True)


def test_kpp_vt2_prefactor_includes_neg_beta_t():
    """LMD94 Eq. 23 unresolved-shear variance V_t^2 carries the (-beta_T)^1/2
    factor explicitly (beta_T = -0.2, ``KPPConfig.neg_beta_T``).  The combined
    prefactor Cv*(-beta_T)^1/2 / (Ri_c*kappa^2*sqrt(c_s*eps)) is the canonical
    4.74 — NOT the 2.236x-too-large 10.6 that resulted when (-beta_T)^1/2 was
    (wrongly) assumed folded into Cv.  (Physics review: ocean/kpp, 2026-06-29.)
    """
    cfg = KPPConfig()
    assert cfg.neg_beta_T == 0.2  # = -beta_T (LMD94 App. B), fixed/excluded const
    assert cfg.Cv == 1.6          # standalone LMD94 value (does NOT absorb sqrt(0.2))
    sqrt_factor = cfg.neg_beta_T ** 0.5
    prefactor = (
        cfg.Cv * sqrt_factor
        / (cfg.Ri_crit * cfg.kappa_vk ** 2 * (cfg.c_s * cfg.epsilon_lmd) ** 0.5)
    )
    assert prefactor == pytest.approx(4.74, abs=0.02)
    # Regression guard: dropping the sqrt factor recovers the old buggy 10.6.
    assert prefactor / sqrt_factor == pytest.approx(10.6, abs=0.1)


def test_tke_defaults_match_old_literals():
    cfg = TKEConfig()
    assert cfg.prandtl_ri_coeff == 6.6
    assert cfg.bg_diff_amp == 0.8
    assert cfg.bg_diff_arctan_coeff == 1.05
    assert cfg.bg_diff_depth_m == 2500.0
    assert cfg.bg_diff_width_m == 222.2
    assert cfg.bg_diff_scale == 1.0e-4


def test_kpp_defaults_match_old_literals():
    cfg = KPPConfig()
    assert cfg.businger_unstable_coeff == 16.0
    assert cfg.businger_stable_coeff == 5.0
    assert cfg.ustar_speed_ratio == 0.01


def test_bryan_lewis_floor_bit_identical_to_old_formula():
    cfg = TKEConfig()
    z = jnp.asarray(np.linspace(-50.0, -5000.0, 40))
    got = _bryan_lewis_kappaH_floor(z, cfg)
    depth = -z
    ref = (0.8 + 1.05 / jnp.pi
           * jnp.arctan((depth - 2500.0) / 222.2)) * 1.0e-4
    assert np.array_equal(np.asarray(got), np.asarray(ref))


def test_prandtl_richardson_bit_identical_to_old_formula():
    cfg = TKEConfig(prandtl_mode="richardson")
    rng = np.random.default_rng(0)
    N2 = jnp.asarray(rng.standard_normal((4, 5)) * 1e-4)
    shear = jnp.asarray(np.abs(rng.standard_normal((4, 5))) * 1e-3)
    K_M = jnp.asarray(np.abs(rng.standard_normal((4, 5))) * 1e-2)
    got = _prandtl_number(N2, shear, K_M, cfg)
    Ri = N2 / jnp.maximum(shear, 1e-12)
    ref = jnp.maximum(1.0, jnp.minimum(10.0, 6.6 * Ri))
    assert np.array_equal(np.asarray(got), np.asarray(ref))


def test_overrides_change_output():
    """A non-default coefficient must actually move the result (not dead)."""
    cfg = TKEConfig()
    cfg2 = cfg._replace(bg_diff_scale=5.0e-4)
    z = jnp.asarray([-3000.0])
    a = float(np.asarray(_bryan_lewis_kappaH_floor(z, cfg))[0])
    b = float(np.asarray(_bryan_lewis_kappaH_floor(z, cfg2))[0])
    assert not np.isclose(a, b)


def test_businger_override_changes_velocity_scales():
    cfg = KPPConfig()
    cfg2 = cfg._replace(businger_unstable_coeff=32.0)
    u_star = jnp.asarray([0.02])
    B_f = jnp.asarray([1e-7])  # destabilising
    d = jnp.asarray([[5.0, 20.0]])
    h_bl = jnp.asarray([[50.0]])
    eps = 1e-12
    wm0, _ = _kpp_velocity_scales(u_star, B_f, d, h_bl, cfg, eps)
    wm1, _ = _kpp_velocity_scales(u_star, B_f, d, h_bl, cfg2, eps)
    assert not np.allclose(np.asarray(wm0), np.asarray(wm1))


def test_config_differentiable_in_new_coeffs():
    """The new fields flow gradients (config-pytree trainability)."""
    z = jnp.asarray([-3000.0, -1000.0])

    def loss(scale):
        cfg = TKEConfig(bg_diff_scale=scale)
        return jnp.sum(_bryan_lewis_kappaH_floor(z, cfg))

    g = jax.grad(loss)(1.0e-4)
    assert np.isfinite(float(g)) and g != 0.0
