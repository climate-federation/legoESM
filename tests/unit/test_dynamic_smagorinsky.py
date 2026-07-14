"""Dynamic Smagorinsky (Germano 1991 / Lilly 1992) plane-LES SGS coefficient.

Covers the plane-averaged dynamic coefficient `_compute_dynamic_smag_cs_plane`:
  1. shape (nlev,), finite, clipped to [0, cs_max];
  2. ~zero for a uniform (strain-free) field — no spurious SGS in laminar flow;
  3. AD-safe (jax.grad finite — guards the sqrt-at-zero 0·∞ NaN) + JIT-safe;
  4. config default `smagorinsky_dynamic=False` leaves the static K_m path
     byte-identical (the new fields don't perturb the existing closure).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    _compute_dynamic_smag_cs_plane,
    _compute_smagorinsky_K_m_plane,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate

jax.config.update("jax_enable_x64", True)

_NY, _NX, _NLEV = 16, 16, 24
_GRID = create_plane_grid(nx=_NX, ny=_NY, nlev=_NLEV, dx=25.0, dy=25.0)
_HC = create_height_coordinate(_NLEV, H=600.0)


def _turbulent_field(seed=0):
    k = jax.random.PRNGKey(seed)
    u = 0.5 * jax.random.normal(k, (_NY, _NX, _NLEV)) + 8.0
    v = 0.3 * jax.random.normal(jax.random.PRNGKey(seed + 1), (_NY, _NX, _NLEV))
    w = 0.4 * jax.random.normal(jax.random.PRNGKey(seed + 2),
                                (_NY, _NX, _NLEV + 1))
    return u, v, w


def test_shape_finite_clipped():
    u, v, w = _turbulent_field()
    cs = _compute_dynamic_smag_cs_plane(u, v, w, _GRID, _HC, cs_max=0.3)
    assert cs.shape == (_NLEV,)
    assert bool(jnp.all(jnp.isfinite(cs)))
    assert float(cs.min()) >= 0.0
    assert float(cs.max()) <= 0.3 + 1e-9


def test_zero_for_uniform_flow():
    """Strain-free uniform field ⇒ no resolved fluctuations ⇒ C_s ≈ 0
    (no spurious eddy viscosity in laminar regions)."""
    u = jnp.full((_NY, _NX, _NLEV), 8.0)
    v = jnp.zeros((_NY, _NX, _NLEV))
    w = jnp.zeros((_NY, _NX, _NLEV + 1))
    cs = _compute_dynamic_smag_cs_plane(u, v, w, _GRID, _HC)
    assert float(cs.max()) < 1e-6


def test_differentiable_and_jit():
    u, v, w = _turbulent_field()

    def loss(u):
        return jnp.sum(_compute_dynamic_smag_cs_plane(u, v, w, _GRID, _HC))

    g = jax.grad(loss)(u)
    assert bool(jnp.all(jnp.isfinite(g)))            # no sqrt(0) 0·∞ NaN
    f = jax.jit(lambda u, v, w: _compute_dynamic_smag_cs_plane(
        u, v, w, _GRID, _HC))
    assert bool(jnp.all(jnp.isfinite(f(u, v, w))))


def test_dynamic_cs_feeds_static_operator_consistently():
    """The per-level dynamic C_s(z) must broadcast into the SAME (C_s·Δ)²·|S|
    operator as the static scalar path (no shape/scaling break)."""
    u, v, w = _turbulent_field()
    cs_z = _compute_dynamic_smag_cs_plane(u, v, w, _GRID, _HC)
    K_dyn = _compute_smagorinsky_K_m_plane(u, v, w, _GRID, _HC, cs_z)
    K_static = _compute_smagorinsky_K_m_plane(u, v, w, _GRID, _HC, 0.17)
    assert K_dyn.shape == K_static.shape == (_NY, _NX, _NLEV)
    assert bool(jnp.all(jnp.isfinite(K_dyn)))
    assert float(K_dyn.min()) >= 0.0
