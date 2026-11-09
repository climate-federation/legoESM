"""RBF-FD shallow water through the shared core and the integrator dispatch.

Williamson TC2: h error vs node count (Voronoi levels 3/4/5, 5 days, rk4);
ssp_rk3 also runs.  TC5: 1-day run at level 4.  Global mass drift is a KNOWN
Rule 0 defect (no flux form) and is printed, not asserted.  fp64.
"""

import math

import jax
import jax.numpy as jnp
import pytest
from jax.test_util import check_grads
from legoesm import constants
from legoesm.atmosphere.dynamics.gcm import shallow_water_rbffd as sw
from legoesm.atmosphere.dynamics.gcm.shallow_water_generic import rbffd_sw_operators
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.core.williamson_sw_analytic import (
    W2_GH0,
    W5_H0_M,
    W5_UBAR_MS,
    solid_body_rotation_speed,
)
from legoesm.grids.neighbour_connectivity import rbffd_stencils
from legoesm.grids.voronoi import create_voronoi_mesh

LEVELS = (3, 4, 5)
TC2_DAYS = 5.0
TC5_DAYS = 1.0
CFL = 0.4
MIN_TC2_ORDER = 2.0
PERT_MS = 5.0
# Voted 2026-11-09 (3/3): measured 1.008 and -0.37 at level 4.
HYPER_RATE_RTOL = 0.1
RANDOM_RATE_TAU = (-1.0, -0.1)
R = constants.R_earth


@pytest.fixture(scope="module", autouse=True)
def _x64():
    prev_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    saved = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(saved)
    jax.config.update("jax_enable_x64", prev_x64)


@pytest.fixture(scope="module")
def nodes(_x64):
    return {lev: sw.voronoi_nodes(create_voronoi_mesh(lev, lloyd_iterations=50))
            for lev in LEVELS}


def _config(integrator):
    return sw.RBFFDShallowWaterConfig(integrator=integrator)


def _dt(xyz, c, u0):
    idx = rbffd_stencils(xyz, 2)
    xu = xyz / jnp.linalg.norm(xyz, axis=-1, keepdims=True)
    h_min = R * float(jnp.min(jnp.linalg.norm(xu[idx[:, 1]] - xu, axis=-1)))
    return CFL * h_min / (c + u0)


def _run(state, mesh, h_s, config, dt, days):
    n_steps = int(math.ceil(days * 86400.0 / dt))
    dt = days * 86400.0 / n_steps

    @jax.jit
    def run(s):
        return jax.lax.scan(lambda c, _: (sw.step(c, mesh, h_s, config, dt), None),
                            s, None, length=n_steps)[0]

    return run(state), n_steps, dt


def _tc2(level_nodes, integrator, days):
    xyz, lon, lat, area = level_nodes
    cfg = _config(integrator)
    mesh = sw.build_mesh(xyz, R, cfg)
    s0, h_s = sw.williamson2_state(lon, lat, R, cfg.g, cfg.omega)
    dt = _dt(xyz, math.sqrt(W2_GH0), solid_body_rotation_speed(R))
    s1, n_steps, dt = _run(s0, mesh, h_s, cfg, dt, days)
    err = s1.h - s0.h
    l2 = float(jnp.sqrt(jnp.sum(area * err**2) / jnp.sum(area * s0.h**2)))
    linf = float(jnp.max(jnp.abs(err)) / jnp.max(jnp.abs(s0.h)))
    m0, m1 = sw.total_mass(s0.h, area), sw.total_mass(s1.h, area)
    return s1, l2, linf, float((m1 - m0) / m0), n_steps, dt


def test_tc2_converges_with_node_count(nodes):
    res = {lev: _tc2(nodes[lev], "rk4", TC2_DAYS) for lev in LEVELS}
    for lev in LEVELS:
        _, l2, linf, dm, n_steps, dt = res[lev]
        print(f"TC2 level {lev} N={nodes[lev][0].shape[0]} dt={dt:.1f}s steps={n_steps} "
              f"l2={l2:.3e} linf={linf:.3e} rel_mass_drift={dm:.3e}")
    l2s = [res[lev][1] for lev in LEVELS]
    linfs = [res[lev][2] for lev in LEVELS]
    assert l2s[0] > l2s[1] > l2s[2] and linfs[0] > linfs[1] > linfs[2]
    n4, n5 = nodes[4][0].shape[0], nodes[5][0].shape[0]
    for e in (l2s, linfs):
        order = math.log(e[1] / e[2]) / (0.5 * math.log(n5 / n4))
        print(f"TC2 order 4->5: {order:.2f}")
        assert order >= MIN_TC2_ORDER


def test_tc2_other_integrator_runs(nodes):
    s1, l2, linf, dm, _, _ = _tc2(nodes[4], "ssp_rk3", TC2_DAYS)
    print(f"TC2 ssp_rk3 level 4 l2={l2:.3e} linf={linf:.3e} rel_mass_drift={dm:.3e}")
    assert all(bool(jnp.all(jnp.isfinite(a))) for a in s1)


def test_unknown_integrator_raises(nodes):
    xyz, lon, lat, _ = nodes[3]
    cfg = _config("no_such_integrator")
    mesh = sw.build_mesh(xyz, R, cfg)
    s0, h_s = sw.williamson2_state(lon, lat, R, cfg.g, cfg.omega)
    with pytest.raises(ValueError):
        sw.step(s0, mesh, h_s, cfg, 100.0)


def test_tc2_initial_state_tangent_and_balanced(nodes):
    """Velocity is tangent; the tendency of the steady state is small and
    shrinks with resolution (discretisation error only)."""
    out = []
    for lev in (4, 5):
        xyz, lon, lat, _ = nodes[lev]
        cfg = _config("rk4")
        mesh = sw.build_mesh(xyz, R, cfg)
        s0, h_s = sw.williamson2_state(lon, lat, R, cfg.g, cfg.omega)
        xu = xyz / jnp.linalg.norm(xyz, axis=-1, keepdims=True)
        radial = s0.ux * xu[:, 0] + s0.uy * xu[:, 1] + s0.uz * xu[:, 2]
        assert float(jnp.max(jnp.abs(radial))) <= 1e-12 * float(jnp.max(jnp.abs(s0.ux)))
        out.append(float(jnp.max(jnp.abs(sw.tendency(s0, mesh, h_s, cfg).h))))
    assert out[1] < out[0]


def test_tendency_check_grads_and_jit(nodes):
    # TC5 plus an unbalanced rotational perturbation (voted 3/3), so every
    # tendency component is O(its own scale), not a balance residual.
    xyz, lon, lat, _ = nodes[3]
    cfg = _config("rk4")
    mesh = sw.build_mesh(xyz, R, cfg)
    s0, h_s = sw.williamson5_state(lon, lat, R, cfg.g, cfg.omega)
    xu = xyz / jnp.linalg.norm(xyz, axis=-1, keepdims=True)
    gpsi = jax.vmap(jax.grad(lambda x: (x[0] ** 3 - 3 * x[0] * x[1] ** 2) * x[2]))(xu)
    gpsi = gpsi - jnp.sum(gpsi * xu, -1, keepdims=True) * xu
    pert = jnp.cross(xu, gpsi)
    pert = PERT_MS * pert / jnp.max(jnp.linalg.norm(pert, axis=-1))
    s0 = s0._replace(ux=s0.ux + pert[:, 0], uy=s0.uy + pert[:, 1], uz=s0.uz + pert[:, 2])

    # Nondimensional tendency (time 1 day, depth W5_H0_M, speed W5_UBAR_MS).
    scale = 86400.0 / jnp.asarray([W5_H0_M, W5_UBAR_MS, W5_UBAR_MS, W5_UBAR_MS])[:, None]

    def f(*fields):
        return scale * jnp.stack(
            sw.tendency(sw.RBFFDShallowWaterState(*fields), mesh, h_s, cfg))

    check_grads(f, tuple(s0), order=2, modes=("fwd", "rev"))
    eager, jitted = f(*s0), jax.jit(f)(*s0)
    for k in range(4):
        assert float(jnp.max(jnp.abs(jitted[k] - eager[k]))) <= \
            1e-12 * float(jnp.max(jnp.abs(eager[k])))


def test_tc5_short_run(nodes):
    xyz, lon, lat, area = nodes[4]
    cfg = _config("rk4")
    mesh = sw.build_mesh(xyz, R, cfg)
    s0, h_s = sw.williamson5_state(lon, lat, R, cfg.g, cfg.omega)
    dt = _dt(xyz, math.sqrt(cfg.g * W5_H0_M), W5_UBAR_MS)
    s1, n_steps, dt = _run(s0, mesh, h_s, cfg, dt, TC5_DAYS)
    assert all(bool(jnp.all(jnp.isfinite(a))) for a in s1)
    m0, m1 = sw.total_mass(s0.h, area), sw.total_mass(s1.h, area)
    e0, e1 = (sw.total_energy(s, h_s, area, cfg.g) for s in (s0, s1))
    print(f"TC5 level 4 1 day dt={dt:.1f}s steps={n_steps} rel_mass_drift="
          f"{float((m1 - m0) / m0):.3e} rel_energy_change={float((e1 - e0) / e0):.3e} "
          f"h range {float(jnp.min(s1.h + h_s)):.1f}..{float(jnp.max(s1.h + h_s)):.1f} m")


def test_east_north_to_cartesian_roundtrip():
    lon = jnp.linspace(-3.0, 3.0, 7)
    lat = jnp.linspace(-1.4, 1.4, 7)
    ux, uy, uz = sw.east_north_to_cartesian(jnp.ones(7), jnp.zeros(7), lon, lat)
    assert jnp.allclose(ux, -jnp.sin(lon)) and jnp.allclose(uy, jnp.cos(lon))
    ux, uy, uz = sw.east_north_to_cartesian(jnp.zeros(7), jnp.ones(7), lon, lat)
    assert jnp.allclose(uz, jnp.cos(lat))


def test_hyperviscosity_damps_with_expected_rate(nodes):
    """(-1)^(k+1) gamma Lap^k on depth and on every velocity component (before
    projection): the area-weighted rate on an l=4 harmonic matches
    -gamma (l(l+1)/a^2)^k, and a random field is damped on the order of tau.
    Pointwise checks are invalid: stacked Laplacians amplify the grid-scale
    part of the single-Laplacian error (41x the tiny smooth-mode rate here)."""
    xyz, _, _, area = nodes[4]
    cfg = _config("rk4")
    mesh = sw.build_mesh(xyz, R, cfg)
    ops = rbffd_sw_operators(mesh, cfg)
    xu = xyz / jnp.linalg.norm(xyz, axis=-1, keepdims=True)
    y = (xu[:, 0] ** 3 - 3 * xu[:, 0] * xu[:, 1] ** 2) * xu[:, 2]
    r = jax.random.normal(jax.random.PRNGKey(0), y.shape)
    k, n = cfg.hyperviscosity_order, xyz.shape[0]
    gamma = (R * math.sqrt(4 * math.pi / n) / math.pi) ** (2 * k) / cfg.hyperviscosity_tau_s
    expected_rate = (-1.0) ** (k + 1) * gamma * (-20.0 / R**2) ** k
    assert expected_rate < 0.0
    zero = jnp.zeros_like(y)

    def applied(f):
        return [ops.mass_tendency(f, (zero, zero, zero))] + list(
            ops.dissipation((f, f, f), (zero, zero, zero)))

    for f, check in ((y, "mode"), (r, "random")):
        for out in applied(f):
            rate = float(jnp.sum(area * f * out) / jnp.sum(area * f * f))
            if check == "mode":
                assert abs(rate / expected_rate - 1.0) <= HYPER_RATE_RTOL, rate / expected_rate
            else:
                assert RANDOM_RATE_TAU[0] <= rate * cfg.hyperviscosity_tau_s <= RANDOM_RATE_TAU[1]
