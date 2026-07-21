"""FIX D1: the two-stream eigenvalue floor must preserve the k -> 0
conservative-scattering limit.

At ssa = 1 the Zdunkowski SW exchange coefficients give gamma1 == gamma2
exactly (for any asymmetry g), so the true Meador-Weaver eigenvalue
k = sqrt((gamma1+gamma2)*(gamma1-gamma2)) is 0 and the diffuse
reflectance/transmittance reduce to the analytic conservative limits

    r_diff -> gamma1*tau / (1 + gamma1*tau)      (gamma1 == gamma2)
    t_diff -> 1 / (1 + gamma1*tau)               (r + t = 1, no absorption)

Canonical RRTMGP (mo_rte_solver_kernels) floors the sqrt argument at 1e-12
in double precision precisely so this limit is reproduced.  A legacy outer
floor of k >= 1e-2 biased thick-cloud (tau ~ 50) SW transmittance by ~5%
and reflectance by ~1%; these tests pin the limit tight enough that any
floor of that magnitude fails.

Run with JAX_ENABLE_X64=1 (the fixture below enables it explicitly).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.radiation.rrtmgp.rte import (
    monochromatic_two_stream as mts,
)


def _conservative_case(tau_val, g_val, dtype):
  """Uniform conservative-scattering layer fields, shape (1, 1, 3)."""
  shape = (1, 1, 3)
  tau = jnp.full(shape, tau_val, dtype=dtype)
  ssa = jnp.ones(shape, dtype=dtype)
  g = jnp.full(shape, g_val, dtype=dtype)
  return tau, ssa, g


def _analytic_conservative(tau_val, g_val):
  """Conservative-scattering limit of Meador-Weaver diffuse R/T."""
  gamma1 = 0.25 * (8.0 - (5.0 + 3.0 * g_val))  # ssa = 1
  # == 0.25 * 3 * (1 - g_val) == gamma2
  r = gamma1 * tau_val / (1.0 + gamma1 * tau_val)
  t = 1.0 / (1.0 + gamma1 * tau_val)
  return r, t


@pytest.mark.parametrize("tau_val", [1.0, 50.0, 200.0])
@pytest.mark.parametrize("g_val", [0.0, 0.85])
def test_sw_conservative_thick_layer_matches_analytic_f64(tau_val, g_val):
  """float64: R and T match the analytic k->0 limit to ~(k_floor*tau)^2
  (k_floor = 1e-6 -> bias <= 4e-8 even at tau=200) and R + T = 1."""
  tau, ssa, g = _conservative_case(tau_val, g_val, jnp.float64)
  props = mts.sw_cell_properties(0.3, tau, ssa, g)
  r = np.asarray(props["r_diff"])
  t = np.asarray(props["t_diff"])
  r_exact, t_exact = _analytic_conservative(tau_val, g_val)

  assert np.all(np.isfinite(r)) and np.all(np.isfinite(t))
  # Conservative scattering: no absorption, r + t = 1.
  np.testing.assert_allclose(r + t, 1.0, atol=1e-6)
  # The analytic limit itself.  The legacy k >= 1e-2 floor gave ~1e-2
  # relative error at tau=50 -- 5 orders of magnitude above this gate.
  np.testing.assert_allclose(r, r_exact, rtol=1e-6)
  np.testing.assert_allclose(t, t_exact, rtol=1e-6)


def test_sw_conservative_thick_layer_float32_floor():
  """float32: the dtype-aware floor (k >= 1e-3) keeps the limit to ~1e-3
  relative (floor bias O((k*tau)^2)/6 plus fp32 rounding), with no NaN."""
  tau_val, g_val = 50.0, 0.85
  tau, ssa, g = _conservative_case(tau_val, g_val, jnp.float32)
  # Zenith as a float32 array so nothing silently promotes to f64 under
  # the x64-enabled test session.
  props = mts.sw_cell_properties(jnp.asarray(0.3, dtype=jnp.float32),
                                 tau, ssa, g)
  r = np.asarray(props["r_diff"])
  t = np.asarray(props["t_diff"])
  r_exact, t_exact = _analytic_conservative(tau_val, g_val)

  assert r.dtype == np.float32 and t.dtype == np.float32
  assert np.all(np.isfinite(r)) and np.all(np.isfinite(t))
  np.testing.assert_allclose(r + t, 1.0, atol=1e-3)
  np.testing.assert_allclose(r, r_exact, rtol=1e-3)
  np.testing.assert_allclose(t, t_exact, rtol=1e-3)


def test_sw_conservative_limit_gradients_finite():
  """Reverse-mode AD stays finite at the floored k (the sqrt VJP is bounded
  by 1/(2*k_floor) and the maximum() zeroes the cotangent on the floored
  branch); gradients w.r.t. tau, ssa and g at ssa=1, large tau."""
  tau_val, g_val = 50.0, 0.85

  def loss(tau_s, ssa_s, g_s):
    shape = (1, 1, 3)
    props = mts.sw_cell_properties(
        0.3,
        jnp.full(shape, tau_s),
        jnp.full(shape, ssa_s),
        jnp.full(shape, g_s),
    )
    return jnp.sum(props["r_diff"] + props["t_diff"])

  grads = jax.grad(loss, argnums=(0, 1, 2))(
      jnp.asarray(tau_val), jnp.asarray(1.0), jnp.asarray(g_val)
  )
  for name, gr in zip(("tau", "ssa", "g"), grads):
    assert bool(jnp.isfinite(gr)), f"non-finite d/d{name} at the k->0 limit"


def test_lw_diffuse_rt_uses_same_floor():
  """The LW path shares _k_fn: a conservative-scattering layer under the
  Fu (1997) diffusivity secant also reproduces its k->0 limit
  r = gamma1*tau/(1+gamma1*tau) with gamma1 = D*(1 - 0.5*(1+g))."""
  tau_val, g_val = 20.0, 0.5
  d_fac = mts._LW_DIFFUSIVE_FACTOR
  gamma1 = d_fac * (1.0 - 0.5 * 1.0 * (1.0 + g_val))
  # ssa=1: gamma2 = D*0.5*(1-g) == gamma1
  assert abs(gamma1 - d_fac * 0.5 * (1.0 - g_val)) < 1e-15
  shape = (1, 1, 3)
  tau = jnp.full(shape, tau_val, dtype=jnp.float64)
  g1 = jnp.full(shape, gamma1, dtype=jnp.float64)
  r = np.asarray(mts._diffuse_reflectance(g1, g1, tau))
  t = np.asarray(mts._diffuse_transmittance(g1, g1, tau))
  np.testing.assert_allclose(r + t, 1.0, atol=1e-6)
  np.testing.assert_allclose(
      r, gamma1 * tau_val / (1.0 + gamma1 * tau_val), rtol=1e-6
  )
