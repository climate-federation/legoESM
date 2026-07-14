"""Vreman (2004) optional SGS closure for the plane CRM.

Covers the shared algebra (`turbulence.vreman.vreman_nu_t`) and the CRM wiring
(`_compute_vreman_K_m_plane`, `turbulence_closure="vreman"`):

1. K_m = 0 on the rest state (no spontaneous turbulence).
2. K_m > 0 for a strain-bearing 3-D field.
3. K_m -> 0 for a well-resolved UNIDIRECTIONAL shear u(z) (the defining Vreman
   property — no spurious dissipation of laminar shear).
4. AD: jax.grad through the closure is finite.
5. Config validation: "vreman" accepted; incompatible with dynamic; one model
   step runs finite.

Vreman is an OPTIONAL closure; the SAM-faithful default stays Smagorinsky.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.gcm.compressible_euler import CompressibleEulerConfig
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    PlaneCompressibleEulerModel,
    _compute_vreman_K_m_plane,
    make_flat_plane_terrain_metric,
    make_rest_state,
)
from legoesm.atmosphere.physics.turbulence.vreman import vreman_nu_t
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate

jax.config.update("jax_enable_x64", True)


def _grid():
    grid = create_plane_grid(nx=8, ny=8, nlev=6, dx=200.0, dy=200.0,
                             dtype=jnp.float64)
    hc = create_height_coordinate(grid.nlev, H=2_000.0)
    return grid, hc


def test_vreman_core_vanishes_for_unidirectional_shear():
    """Shared algebra: only a13=∂u/∂z nonzero ⇒ Bβ=0 ⇒ ν_t=0."""
    shp = (5, 5, 4)
    z = jnp.zeros(shp)
    a13 = jnp.ones(shp) * 3.0
    nu = vreman_nu_t(z, z, a13, z, z, z, z, z, z,
                     dx=5.0, dy=5.0, dz=10.0, c_vreman=0.07)
    assert float(jnp.max(jnp.abs(nu))) < 1e-12


def test_vreman_core_nonnegative_and_active_3d():
    shp = (6, 6, 4)
    g = [jnp.asarray(np.random.default_rng(i).standard_normal(shp))
         for i in range(9)]
    nu = vreman_nu_t(*g, dx=5.0, dy=5.0, dz=10.0, c_vreman=0.07)
    assert float(jnp.min(nu)) >= 0.0
    assert float(jnp.max(nu)) > 0.0


def test_K_m_zero_on_rest_state():
    grid, hc = _grid()
    z = jnp.zeros((grid.ny, grid.nx, grid.nlev))
    zw = jnp.zeros((grid.ny, grid.nx, grid.nlev + 1))
    K = _compute_vreman_K_m_plane(z, z, zw, grid, hc, c_vreman=0.07)
    assert K.shape == (grid.ny, grid.nx, grid.nlev)
    assert float(jnp.max(jnp.abs(K))) == 0.0


def test_K_m_positive_for_3d_strain():
    grid, hc = _grid()
    rng = np.random.default_rng(0)
    u = jnp.asarray(rng.standard_normal((grid.ny, grid.nx, grid.nlev)))
    v = jnp.asarray(rng.standard_normal((grid.ny, grid.nx, grid.nlev)))
    w = jnp.asarray(rng.standard_normal((grid.ny, grid.nx, grid.nlev + 1)))
    K = _compute_vreman_K_m_plane(u, v, w, grid, hc, c_vreman=0.07)
    assert float(jnp.min(K)) >= 0.0
    assert float(jnp.max(K)) > 0.0


def test_K_m_small_for_unidirectional_shear():
    """u = u(z), v=w=0 ⇒ only ∂u/∂z ⇒ Vreman K_m ≈ 0 (≪ Smagorinsky would give)."""
    grid, hc = _grid()
    zc = jnp.asarray(np.linspace(0.0, 1.0, grid.nlev))
    u = jnp.broadcast_to(2.0 * zc, (grid.ny, grid.nx, grid.nlev))
    v = jnp.zeros((grid.ny, grid.nx, grid.nlev))
    w = jnp.zeros((grid.ny, grid.nx, grid.nlev + 1))
    K = _compute_vreman_K_m_plane(u, v, w, grid, hc, c_vreman=0.07)
    assert float(jnp.max(jnp.abs(K))) < 1e-10


def test_K_m_differentiable():
    grid, hc = _grid()
    rng = np.random.default_rng(1)
    u0 = jnp.asarray(rng.standard_normal((grid.ny, grid.nx, grid.nlev)))
    v0 = jnp.asarray(rng.standard_normal((grid.ny, grid.nx, grid.nlev)))
    w0 = jnp.asarray(rng.standard_normal((grid.ny, grid.nx, grid.nlev + 1)))

    def loss(u):
        return jnp.sum(_compute_vreman_K_m_plane(u, v0, w0, grid, hc, 0.07))

    grad = jax.grad(loss)(u0)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.max(jnp.abs(grad))) > 0.0


def test_K_m_grad_finite_at_rest_and_shear():
    """AD must stay finite at the ZERO-invariant points (rest, 1-D shear) where
    √(0) would otherwise NaN the gradient (codex review iter)."""
    grid, hc = _grid()
    zc = jnp.asarray(np.linspace(0.0, 1.0, grid.nlev))
    for u in (jnp.zeros((grid.ny, grid.nx, grid.nlev)),               # rest
              jnp.broadcast_to(2.0 * zc, (grid.ny, grid.nx, grid.nlev))):  # u(z)
        v = jnp.zeros((grid.ny, grid.nx, grid.nlev))
        w = jnp.zeros((grid.ny, grid.nx, grid.nlev + 1))

        def loss(uu, v=v, w=w):
            return jnp.sum(_compute_vreman_K_m_plane(uu, v, w, grid, hc, 0.07))

        grad = jax.grad(loss)(jnp.asarray(u, dtype=jnp.float64))
        assert bool(jnp.all(jnp.isfinite(grad)))


def _model(closure="vreman", **kw):
    grid = create_plane_grid(nx=8, ny=8, nlev=4, dx=200.0, dy=200.0,
                             dtype=jnp.float64)
    hc = create_height_coordinate(grid.nlev, H=2_000.0)
    tm = make_flat_plane_terrain_metric(grid, hc)
    cfg = CompressibleEulerConfig(
        sponge_coeff=0.0, hyperdiff_coeff=0.0, hyperdiff_rho_coeff=0.0,
        hyperdiff_w_coeff=0.0, semi_implicit_acoustic=False,
        use_coriolis=False, fix_mass=False, turbulence_closure=closure, **kw)
    return PlaneCompressibleEulerModel(grid, hc, tm, cfg), grid, hc


def test_config_vreman_validates_and_steps_finite():
    model, grid, hc = _model(closure="vreman", vreman_c=0.07)
    state = make_rest_state(grid, hc)
    # seed a perturbation so the closure is exercised
    key = jax.random.PRNGKey(0)
    u0 = 0.5 * jax.random.normal(key, (grid.ny, grid.nx, grid.nlev))
    state = state._replace(u=state.u.replace(data=u0))
    out = model.step(state, dt=0.2, physics_fn=None)
    assert bool(jnp.all(jnp.isfinite(out.u.data)))


def test_config_vreman_rejects_dynamic():
    with pytest.raises(ValueError):
        _model(closure="vreman", vreman_c=0.07, smagorinsky_dynamic=True)
