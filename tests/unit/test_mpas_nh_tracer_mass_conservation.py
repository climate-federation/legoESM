"""Tracer mass conservation of the MPAS NH split-explicit step (#2641).

``sum(area * J * dz * rho * q)`` must hold to round-off over a resolved
hill (J != 1) with a zonal wind, and a uniform tracer must stay uniform.
Fails on the pre-#2641 advective-form transport.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest
from jax.test_util import check_grads

HILL = {"mountain_height": 2000.0, "mountain_halfwidth": 2000.0e3}


def _setup(n_tracers_q, **cfg):
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_mpas import (
        MPASCompressibleEulerConfig, MPASCompressibleEulerModel)
    from legoesm.grids.voronoi import create_voronoi_mesh
    from tests.atmosphere.nonhydrostatic.test_cases.dcmip2025.test_case_1_mpas import (
        dcmip25_tc1_init_mpas)

    mesh = create_voronoi_mesh(3)
    state, hc, tm = dcmip25_tc1_init_mpas(mesh, n_levels=8, params=HILL)
    cfg = {"n_acoustic_substeps": 6, "fix_mass": False, **cfg}
    model = MPASCompressibleEulerModel(mesh, hc, tm, MPASCompressibleEulerConfig(**cfg))
    lat = mesh.latCell[:, None]
    lon = mesh.lonCell[:, None]
    z = (jnp.arange(8) + 0.5)[None, :] / 8
    theta_pert = 0.5 * jnp.cos(lat) * jnp.sin(lon) * jnp.sin(jnp.pi * z)
    state = state._replace(
        theta_prime=state.theta_prime.replace(data=state.theta_prime.data + theta_pert))
    q = n_tracers_q(lat, lon, z, state.u.data.dtype)
    return model, state._replace(tracers=state.tracers.replace(data=q))


def _varying(lat, lon, z, dtype):
    q1 = 1e-2 * (1.0 + 0.3 * jnp.sin(lat) * jnp.cos(2 * lon) + 0.2 * z)
    q2 = 1e-4 * jnp.exp(-((z - 0.3) ** 2) / 0.01) * (1.0 + jnp.cos(lon))
    return jnp.stack([q1, q2], axis=-1).astype(dtype)


def _tracer_mass(model, state):
    rho = model.height_coord.rho_ref + state.rho_prime.data
    w = (model.mesh.areaCell[:, None] * model.terrain_metric.jacobian[:, None]
         * model.height_coord.dz[None, :] * rho)
    return jnp.sum(w[..., None] * state.tracers.data, axis=(0, 1))


@pytest.mark.parametrize("cfg", [{}, {"outer_integrator": "ssp_rk34"},
                                 {"nu_del2": 1.0e5, "K_h": 1.0e4}])
def test_tracer_mass_conserved_over_hill(cfg):
    model, state = _setup(_varying, **cfg)
    m0 = _tracer_mass(model, state)
    for _ in range(30):
        state = model.step(state, 60.0)
    assert jnp.all(jnp.isfinite(state.tracers.data))
    rel = jnp.abs(_tracer_mass(model, state) - m0) / m0
    assert float(jnp.max(rel)) < 1e-13, rel


def test_uniform_tracer_stays_uniform_over_hill():
    model, state = _setup(lambda lat, lon, z, dt: jnp.full(
        (lat.shape[0], 8, 1), 7e-3, dt))
    for _ in range(20):
        state = model.step(state, 60.0)
    assert float(jnp.max(jnp.abs(state.tracers.data / 7e-3 - 1.0))) < 1e-12


def test_mass_fixer_holds_tracer_mass():
    model, state = _setup(_varying, fix_mass=True, anchor_mass_to_initial=True)
    dry0 = model.compute_dry_mass(state)
    model._target_mass = dry0 * (1.0 + 1e-4)          # force a real correction
    m0 = _tracer_mass(model, state)
    out = model.step(state, 60.0)
    assert abs(float(model.compute_dry_mass(out) / dry0) - 1.0 - 1e-4) < 1e-12
    assert float(jnp.max(jnp.abs(_tracer_mass(model, out) - m0) / m0)) < 1e-13


def test_rk54_with_tracers_refused():
    model, state = _setup(_varying, outer_integrator="ssp_rk54")
    with pytest.raises(ValueError, match="ssp_rk3 and ssp_rk34"):
        model.step(state, 60.0)


def test_tracer_step_gradient():
    model, state = _setup(_varying)

    def loss(q):
        out = model._step_jit(state._replace(tracers=state.tracers.replace(data=q)),
                              60.0, None, None)
        return jnp.sum(out.tracers.data ** 2)

    q0 = state.tracers.data
    assert float(jnp.max(jnp.abs(jax.grad(loss)(q0)))) > 0.0
    check_grads(loss, (q0,), order=1, modes=("rev",), eps=1e-7)
