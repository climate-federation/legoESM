"""Tracer mass conservation of the plane NH split-explicit step (#2641).

Total tracer mass ``sum(rho * q * dV)`` must hold to round-off for every
acoustic stepping method, and a uniform tracer must stay uniform when no
non-flux term touches the density. Fails on the pre-#2641 transport, which
leaked ~1.3e-5 of the tracer mass per step on this case.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax.test_util import check_grads

from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    PlaneCompressibleEulerModel,
    _plane_weighted_mass_sum,
    make_flat_plane_terrain_metric,
    make_rest_state,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate

jax.config.update("jax_enable_x64", True)

METHODS = {
    "explicit": {},
    "semi_implicit": {"semi_implicit_acoustic": True},
    "si_horizontal": {"semi_implicit_acoustic": True,
                      "substep_horizontal_acoustic": True},
}


def _model(n=8, nlev=10, **overrides):
    cfg = dict(sponge_coeff=0.0, hyperdiff_coeff=0.0, hyperdiff_rho_coeff=0.0,
               hyperdiff_w_coeff=0.0, use_coriolis=False, fix_mass=False)
    cfg.update(overrides)
    grid = create_plane_grid(nx=n, ny=n, nlev=nlev, dx=1.0e3, dy=1.0e3,
                             dtype=jnp.float64)
    hc = create_height_coordinate(nlev, H=20.0e3)
    model = PlaneCompressibleEulerModel(
        grid, hc, make_flat_plane_terrain_metric(grid, hc),
        CompressibleEulerConfig(**cfg))
    return model, make_rest_state(grid, hc, dtype=jnp.float64)


def _perturbed(model, state, q):
    ny, nx, nl = state.u.data.shape
    x = jnp.arange(nx) / nx
    y = jnp.arange(ny) / ny
    z = (jnp.arange(nl) + 0.5) / nl
    rho_ref = model.height_coord.rho_ref
    bubble = (1e-3 * jnp.exp(-((x - 0.5) ** 2) / 0.02)[None, :, None]
              * jnp.sin(jnp.pi * z)[None, None, :] * rho_ref)
    u = 2.0 + jnp.cos(2 * jnp.pi * y)[:, None, None] * jnp.ones((1, nx, nl))
    v = -1.5 + jnp.sin(2 * jnp.pi * x)[None, :, None] * jnp.ones((ny, 1, nl))
    w = 0.5 * jnp.cos(2 * jnp.pi * x)[None, :, None] * jnp.ones((ny, 1, nl - 1))
    return state._replace(
        u=state.u.replace(data=u),
        v=state.v.replace(data=v),
        tracers=state.tracers.replace(data=q),
        rho_prime=state.rho_prime.replace(data=jnp.broadcast_to(bubble, (ny, nx, nl))),
        w=state.w.replace(data=state.w.data.at[:, :, 1:-1].set(w)),
    )


def _varying_q(shape):
    ny, nx, nl = shape
    x = jnp.arange(nx) / nx
    z = (jnp.arange(nl) + 0.5) / nl
    y = jnp.arange(ny) / ny
    q1 = (1.0 + 0.3 * jnp.cos(2 * jnp.pi * x)[None, :, None] * jnp.sin(jnp.pi * z)
          + 0.2 * jnp.sin(2 * jnp.pi * y)[:, None, None])
    q2 = 1e-3 * jnp.exp(-((z - 0.3) ** 2) / 0.01) * jnp.ones((ny, nx, 1))
    return jnp.stack([jnp.broadcast_to(q1, shape), q2], axis=-1)


def _tracer_mass(model, state):
    rho = model.height_coord.rho_ref + state.rho_prime.data
    return jnp.stack([
        _plane_weighted_mass_sum(rho * state.tracers.data[..., i], model.grid,
                                 model.height_coord, model.terrain_metric)
        for i in range(state.tracers.data.shape[-1])])


@pytest.mark.parametrize("method", sorted(METHODS))
@pytest.mark.parametrize("schemes", [
    {}, {"horizontal_advection_scheme": "van_leer",
         "vertical_tracer_advection": "van_leer"}])
def test_tracer_mass_conserved_every_acoustic_method(method, schemes):
    model, state = _model(**METHODS[method], **schemes)
    state = _perturbed(model, state, _varying_q(state.u.data.shape))
    m0 = _tracer_mass(model, state)
    for _ in range(50):
        state = model.step(state, dt=2.0)
    rel = jnp.abs(_tracer_mass(model, state) - m0) / m0
    assert jnp.all(jnp.isfinite(state.tracers.data))
    assert float(jnp.max(rel)) < 1e-13, rel


@pytest.mark.parametrize("overrides", [
    {"outer_integrator": "ssp_rk34"},
    {"smagorinsky_cs": 0.17, "sgs_vertical_diffusion": True},
])
def test_tracer_mass_conserved_rk34_and_sgs(overrides):
    model, state = _model(**overrides)
    state = _perturbed(model, state, _varying_q(state.u.data.shape))
    m0 = _tracer_mass(model, state)
    for _ in range(50):
        state = model.step(state, dt=2.0)
    assert float(jnp.max(jnp.abs(_tracer_mass(model, state) - m0) / m0)) < 1e-13


def test_mass_fixer_holds_tracer_mass():
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        compute_dry_mass_plane)
    model, state = _model(fix_mass=True, anchor_mass_to_initial=True)
    state = _perturbed(model, state, _varying_q(state.u.data.shape))
    dry0 = compute_dry_mass_plane(state, model.grid, model.height_coord,
                                  model.terrain_metric)
    model._target_mass = dry0 * (1.0 + 1e-4)     # force a real correction
    m0 = _tracer_mass(model, state)
    out = model.step(state, dt=2.0)
    dry1 = compute_dry_mass_plane(out, model.grid, model.height_coord,
                                  model.terrain_metric)
    assert abs(float(dry1 / dry0) - 1.0 - 1e-4) < 1e-12
    assert float(jnp.max(jnp.abs(_tracer_mass(model, out) - m0) / m0)) < 1e-13


@pytest.mark.parametrize("method", sorted(METHODS))
def test_uniform_tracer_stays_uniform(method):
    model, state = _model(**METHODS[method])
    q = jnp.full(state.u.data.shape + (1,), 0.7)
    state = _perturbed(model, state, q)
    for _ in range(20):
        state = model.step(state, dt=2.0)
    assert float(jnp.max(jnp.abs(state.tracers.data - 0.7))) < 1e-13


def test_rk54_with_tracers_refused():
    model, state = _model(outer_integrator="ssp_rk54")
    state = _perturbed(model, state, _varying_q(state.u.data.shape))
    with pytest.raises(ValueError, match="ssp_rk3 and ssp_rk34"):
        model.step(state, dt=2.0)


def test_tracer_step_gradient():
    model, state = _model(n=4, nlev=6)
    state = _perturbed(model, state, _varying_q(state.u.data.shape))

    def loss(q):
        out = model._step_jit(state._replace(tracers=state.tracers.replace(data=q)),
                              2.0, None, None)
        return jnp.sum(out.tracers.data ** 2)

    q0 = state.tracers.data
    assert float(jnp.max(jnp.abs(jax.grad(loss)(q0)))) > 0.0
    check_grads(loss, (q0,), order=1, modes=("rev",), eps=1e-6)


def test_face_values_take_the_upwind_donor():
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        _tracer_face_value, _tracer_vertical_face_value)
    step = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])          # jump at face 3
    for scheme in ("upwind1", "van_leer"):
        f = step[None, :, None, None]
        assert float(_tracer_face_value(f, jnp.ones_like(f), 1, scheme)[0, 3, 0, 0]) == 0.0
        assert float(_tracer_face_value(f, -jnp.ones_like(f), 1, scheme)[0, 3, 0, 0]) == 1.0
    f = step[None, None, :, None]                              # index 0 = top
    up = jnp.ones((1, 1, 7, 1))
    assert float(_tracer_vertical_face_value(f, up, "van_leer")[0, 0, 3, 0]) == 1.0
    assert float(_tracer_vertical_face_value(f, -up, "van_leer")[0, 0, 3, 0]) == 0.0


def test_tracer_blob_moves_with_the_wind():
    model, state = _model(n=16, nlev=6)
    nx = state.u.data.shape[1]
    x = jnp.arange(nx) / nx
    blob = 1.0 + jnp.exp(-((x - 0.3) ** 2) / 0.005)
    q = jnp.broadcast_to(blob[None, :, None], state.u.data.shape)[..., None]
    state = state._replace(u=state.u.replace(data=jnp.full_like(state.u.data, 10.0)),
                           tracers=state.tracers.replace(data=q))

    def centroid(s):
        a = s.tracers.data[..., 0] - 1.0
        return float(jnp.sum(a * x[None, :, None]) / jnp.sum(a))

    c0 = centroid(state)
    for _ in range(20):
        state = model.step(state, dt=2.0)
    shift = (centroid(state) - c0) * nx * model.grid.dx
    assert abs(shift - 10.0 * 40.0) < 0.1 * 400.0, shift


def test_conservative_sgs_tendency_is_active_and_mass_neutral():
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        plane_compressible_euler_slow_tendencies)
    model, state = _model(smagorinsky_cs=0.17, sgs_vertical_diffusion=True)
    state = _perturbed(model, state, _varying_q(state.u.data.shape))
    tend = plane_compressible_euler_slow_tendencies(
        state, model.grid, model.height_coord, model.terrain_metric,
        model.config, conservative_tracers=True).dtracers_dt.data
    rho = (model.height_coord.rho_ref + state.rho_prime.data)[..., None]
    mass_tend = jnp.sum(rho * tend * model.height_coord.dz[:, None], axis=(0, 1, 2))
    assert float(jnp.max(jnp.abs(tend))) > 0.0
    assert float(jnp.max(jnp.abs(mass_tend))) < 1e-12 * float(jnp.sum(jnp.abs(rho * tend)))
