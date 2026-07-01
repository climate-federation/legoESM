"""Tests for the NEMO simplified EOS (S-EOS / np_seos), Roquet et al. (2015).

The reference implementation is NEMO ``src/OCE/TRA/eosbn2.F90`` (``np_seos``
branch).  Defaults are the Kamm et al. (2025) DINO coefficients — this EOS is
the NEMO oracle for the DINO ACC thermocline comparison, so the tests pin both
the exact polynomial and the DINO coefficient set.

Density anomaly (NEMO ``prd = zn / rho0`` ⇒ ``rho = rho0 + zn``):

    zt = T - T0 ;  zs = S - S0 ;  zh = depth [m]
    zn = - a0 (1 + ½ λ1 zt + μ1 zh) zt
         + b0 (1 - ½ λ2 zs - μ2 zh) zs
         - nu zt zs
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean.eos import (
    NemoSEOSConfig,
    make_eos_fn,
    nemo_seos_eos,
)

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Independent NumPy reference, written directly from eosbn2.F90 (np_seos).
# ---------------------------------------------------------------------------


def _nemo_seos_ref(T, S, depth_m, cfg: NemoSEOSConfig):
    """In-situ density [kg/m³] from the NEMO S-EOS, depth in metres."""
    zt = T - cfg.T0
    zs = S - cfg.S0
    zh = depth_m
    zn = (
        -cfg.a0 * (1.0 + 0.5 * cfg.lambda1 * zt + cfg.mu1 * zh) * zt
        + cfg.b0 * (1.0 - 0.5 * cfg.lambda2 * zs - cfg.mu2 * zh) * zs
        - cfg.nu * zt * zs
    )
    return cfg.rho0 + zn


def _p_for_depth(depth_m, cfg: NemoSEOSConfig):
    """Boussinesq pressure [Pa] whose ``p/(rho0·g)`` equals ``depth_m``."""
    return depth_m * cfg.rho0 * constants.g


# ---------------------------------------------------------------------------
# DINO coefficient set (must not drift from the paper / NEMO namelist)
# ---------------------------------------------------------------------------


def test_default_config_is_dino_coefficients():
    cfg = NemoSEOSConfig()
    assert cfg.a0 == 0.165
    assert cfg.b0 == 0.76554
    assert cfg.lambda1 == 0.06
    assert cfg.lambda2 == 0.0
    assert cfg.mu1 == 1.4970e-4
    assert cfg.mu2 == 0.0
    assert cfg.nu == 0.0
    assert cfg.T0 == 10.0
    assert cfg.S0 == 35.0
    assert cfg.rho0 == 1026.0


# ---------------------------------------------------------------------------
# Exact, hand-computed reference points
# ---------------------------------------------------------------------------


def test_density_at_reference_TS_is_rho0():
    """At (T0, S0, surface) the anomaly is zero ⇒ rho = rho0 = 1026."""
    rho = float(nemo_seos_eos(jnp.asarray(10.0), jnp.asarray(35.0),
                              jnp.asarray(0.0)))
    np.testing.assert_allclose(rho, 1026.0, atol=1e-10)


def test_salinity_offset_is_pure_b0():
    """At (T0, S0+1, surface): zn = b0·1 = 0.76554 ⇒ rho = 1026.76554."""
    rho = float(nemo_seos_eos(jnp.asarray(10.0), jnp.asarray(36.0),
                              jnp.asarray(0.0)))
    np.testing.assert_allclose(rho, 1026.0 + 0.76554, atol=1e-9)


def test_temperature_offset_includes_cabbeling_in_T():
    """At (T0+1, S0, surface): zn = -a0(1+½λ1)·1 = -0.165·1.03 = -0.16995."""
    rho = float(nemo_seos_eos(jnp.asarray(11.0), jnp.asarray(35.0),
                              jnp.asarray(0.0)))
    np.testing.assert_allclose(rho, 1026.0 - 0.16995, atol=1e-9)


def test_cold_anomaly_is_denser():
    """At (T0-5, S0, surface): zn = -0.165·(1-0.15)·(-5) = +0.70125."""
    rho = float(nemo_seos_eos(jnp.asarray(5.0), jnp.asarray(35.0),
                              jnp.asarray(0.0)))
    np.testing.assert_allclose(rho, 1026.0 + 0.70125, atol=1e-9)
    assert rho > 1026.0  # cold water is denser


# ---------------------------------------------------------------------------
# Independent-reimplementation cross-check over a T/S/depth grid
# ---------------------------------------------------------------------------


def test_matches_numpy_reference_over_grid():
    cfg = NemoSEOSConfig()
    Ts = np.linspace(-2.0, 30.0, 7)
    Ss = np.linspace(33.0, 37.5, 5)
    depths = np.array([0.0, 250.0, 1000.0, 2500.0, 4000.0])
    for T in Ts:
        for S in Ss:
            for d in depths:
                p = _p_for_depth(d, cfg)
                got = float(nemo_seos_eos(jnp.asarray(T), jnp.asarray(S),
                                          jnp.asarray(p)))
                want = _nemo_seos_ref(T, S, d, cfg)
                np.testing.assert_allclose(got, want, rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# Sign conventions (thermobaric + cabbeling structure)
# ---------------------------------------------------------------------------


def test_thermobaric_depth_strengthens_thermal_term():
    """μ1>0: a warm anomaly (zt>0) becomes LESS dense with depth (the
    effective thermal contraction grows with pressure)."""
    cfg = NemoSEOSConfig()
    rho_surf = float(nemo_seos_eos(jnp.asarray(11.0), jnp.asarray(35.0),
                                   jnp.asarray(0.0)))
    rho_deep = float(nemo_seos_eos(jnp.asarray(11.0), jnp.asarray(35.0),
                                   jnp.asarray(_p_for_depth(1000.0, cfg))))
    # zn(deep) = -0.165·(1+0.03+0.1497) ; zn(surf) = -0.165·1.03
    assert rho_deep < rho_surf
    np.testing.assert_allclose(rho_deep, 1026.0 - 0.165 * (1.0 + 0.03 + 0.1497),
                               atol=1e-9)


def test_at_reference_T_no_thermobaric_effect():
    """The thermobaric term multiplies zt, so at T=T0 depth has NO effect
    (λ2=μ2=0 ⇒ the salinity term is depth-independent too)."""
    cfg = NemoSEOSConfig()
    rho_surf = float(nemo_seos_eos(jnp.asarray(10.0), jnp.asarray(36.0),
                                   jnp.asarray(0.0)))
    rho_deep = float(nemo_seos_eos(jnp.asarray(10.0), jnp.asarray(36.0),
                                   jnp.asarray(_p_for_depth(3000.0, cfg))))
    np.testing.assert_allclose(rho_deep, rho_surf, atol=1e-10)


# ---------------------------------------------------------------------------
# Dispatcher integration
# ---------------------------------------------------------------------------


def test_make_eos_fn_dispatches_nemo_seos():
    fn = make_eos_fn("nemo_seos")
    rho = float(fn(jnp.asarray(10.0), jnp.asarray(36.0), jnp.asarray(0.0)))
    np.testing.assert_allclose(rho, 1026.0 + 0.76554, atol=1e-9)


def test_make_eos_fn_honours_custom_nemo_config():
    cfg = NemoSEOSConfig(rho0=1025.0, b0=0.8)
    fn = make_eos_fn("nemo_seos", eos_nemo_seos=cfg)
    rho = float(fn(jnp.asarray(10.0), jnp.asarray(36.0), jnp.asarray(0.0)))
    np.testing.assert_allclose(rho, 1025.0 + 0.8, atol=1e-9)


def test_make_eos_fn_default_nemo_config_ignored_for_other_eos():
    """Passing eos_nemo_seos with a non-nemo eos must be a no-op, not an error."""
    fn = make_eos_fn("wright", eos_nemo_seos=NemoSEOSConfig(rho0=999.0))
    # wright at (10, 35, 0) is ~1026.x, nowhere near the bogus rho0=999.
    rho = float(fn(jnp.asarray(10.0), jnp.asarray(35.0), jnp.asarray(0.0)))
    assert rho > 1020.0


def test_nemo_seos_in_valid_schemes():
    from legoesm.ocean.eos import VALID_EOS_SCHEMES
    assert "nemo_seos" in VALID_EOS_SCHEMES


# ---------------------------------------------------------------------------
# compute_dtype adapter (the f32-EOS lever): None must be byte-identical
# ---------------------------------------------------------------------------


def test_compute_dtype_none_is_identical():
    fn = make_eos_fn("nemo_seos")
    a = float(fn(jnp.asarray(12.0), jnp.asarray(34.5), jnp.asarray(1.0e7)))
    b = float(fn(jnp.asarray(12.0), jnp.asarray(34.5), jnp.asarray(1.0e7),
                 compute_dtype=None))
    assert a == b


# ---------------------------------------------------------------------------
# Differentiability (DINO trains through the EOS via thermal-wind ACC)
# ---------------------------------------------------------------------------


def test_differentiable_wrt_T_at_reference():
    """d_rho/d_T = -a0 = -0.165 at the reference point (warmer ⇒ less dense)."""
    fn = lambda T: nemo_seos_eos(T, jnp.asarray(35.0), jnp.asarray(0.0))
    grad = float(jax.grad(fn)(jnp.asarray(10.0)))
    np.testing.assert_allclose(grad, -0.165, atol=1e-9)


def test_differentiable_wrt_S_at_reference():
    """d_rho/d_S = b0 = 0.76554 at the reference point (saltier ⇒ denser)."""
    fn = lambda S: nemo_seos_eos(jnp.asarray(10.0), S, jnp.asarray(0.0))
    grad = float(jax.grad(fn)(jnp.asarray(35.0)))
    np.testing.assert_allclose(grad, 0.76554, atol=1e-9)


def test_vmap_and_jit():
    fn = make_eos_fn("nemo_seos")
    T = jnp.linspace(0.0, 25.0, 6)
    S = jnp.full_like(T, 35.0)
    p = jnp.zeros_like(T)
    out_vmap = jax.vmap(fn)(T, S, p)
    out_jit = jax.jit(jax.vmap(fn))(T, S, p)
    np.testing.assert_allclose(np.asarray(out_vmap), np.asarray(out_jit),
                               atol=1e-12)
    assert out_vmap.shape == T.shape
