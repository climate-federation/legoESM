"""Generic shallow water (neighbour layer) vs the existing lat-lon and MPAS cores.

Controlled comparison: the ONLY difference between arms is the tendency
function.  Trajectories run the existing model's own jitted step (integrator,
polar filter, mass fixer) with its tendency function swapped for the generic
drop-in, so stepping, fixers and forcing are byte-identical between arms.

Tolerances (fp64; decided by vote, see PR): lat-lon tendencies 1e-13 relative
to the field's largest reference magnitude over the three test states;
10/100-step states and 100-step gradients 1e-11 on both grids (XLA fuses the
whole jitted step differently, measured 1e-15..2.4e-14 on MPAS); MPAS
operators and tendencies bit-identical.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.dynamics.gcm import shallow_water_latlon_cgrid as ll
from legoesm.atmosphere.dynamics.gcm import shallow_water_mpas as mp
from legoesm.atmosphere.dynamics.gcm.shallow_water_generic import (
    latlon_sw_tendencies_generic,
    mpas_sw_tendencies_generic,
)
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.voronoi import create_voronoi_mesh

from tests.atmosphere.shallow_water.test_cases.williamson_mpas import (
    williamson_test2_mpas,
    williamson_test5_mpas,
)

TEND_RTOL = 1e-13
TRAJ_RTOL = 1e-11

LATLON_CONFIGS = {
    "ppm": dict(),
    "centered": dict(use_ppm_transport=False),
    "A_h": dict(A_h=1e5),
    "del4": dict(nu_del4=1e15),
    "polar_filter_rk4": dict(use_polar_filter=True, time_integrator="rk4"),
    "no_fix": dict(fix_mass=False),
    "no_fix_centered": dict(fix_mass=False, use_ppm_transport=False),
}
MPAS_CONFIGS = {
    "energy": dict(),
    "enstrophy": dict(pv_scheme="enstrophy"),
    "order3": dict(thickness_order=3),
    "apvm": dict(apvm_scale=0.5),
    "del2_del4": dict(nu_del2=1e4, nu_del4=1e14),
    "ssp_rk3_energy_fix": dict(time_integrator="ssp_rk3", fix_energy=True),
    "no_fix": dict(fix_mass=False),
}
# APVM is tendency-tested only: the existing model's jitted step traces dt and
# the shared ``dt > 0`` Python check then raises in BOTH arms (pre-existing).
MPAS_TRAJ_CONFIGS = [c for c in MPAS_CONFIGS if c != "apvm"]
LATLON_GRIDS = {"16x32": (16, 32, 300.0), "46x90": (46, 90, 60.0)}
MPAS_DT = 600.0


@pytest.fixture(scope="module", autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    saved = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(saved)
    jax.config.update("jax_enable_x64", prev)


@pytest.fixture(scope="module")
def mesh(_x64):
    return create_voronoi_mesh(3, lloyd_iterations=50)


def _rel(new, old, scale=None):
    """max|new-old| / scale (default max|old|); a zero scale demands exact zero."""
    assert jnp.all(jnp.isfinite(new)) and jnp.all(jnp.isfinite(old))
    err = float(jnp.max(jnp.abs(new - old)))
    scale = float(jnp.max(jnp.abs(old))) if scale is None else scale
    return err / scale if scale > 0 else (0.0 if err == 0.0 else float("inf"))


# ------------------------------------------------------------- lat-lon states

def _perturbed(state, grid):
    """Add smooth h, u and (interior) v so every operator branch is active."""
    lat = grid.lat[:, None]
    lon = grid.lon[None, :]
    h = state.h + 40.0 * jnp.cos(lat) ** 2 * jnp.cos(2 * lon)
    du = 3.0 * jnp.cos(lat) * jnp.sin(lon - 0.5 * grid.dlon)
    u = state.u + jnp.concatenate([du, du[:, :1]], axis=1)
    lat_v = grid.lat_v[:, None]
    v = (state.v + 2.0 * jnp.cos(lat_v) * jnp.sin(3 * lon)).at[0].set(0.0).at[-1].set(0.0)
    return state._replace(h=h, u=u, v=v)


def _latlon_states(grid):
    tc2 = ll.williamson_test2_cgrid(grid)
    return {"tc2": tc2, "tc5": ll.williamson_test5_cgrid(grid),
            "tc2_perturbed": _perturbed(tc2, grid)}


def _assert_admissible(state):
    np.testing.assert_array_equal(np.asarray(state.u[:, 0]), np.asarray(state.u[:, -1]))
    assert jnp.all(state.v[0] == 0.0) and jnp.all(state.v[-1] == 0.0)


@pytest.mark.parametrize("gname", list(LATLON_GRIDS))
@pytest.mark.parametrize("cname", list(LATLON_CONFIGS))
def test_latlon_tendencies_match(gname, cname):
    n_lat, n_lon, dt = LATLON_GRIDS[gname]
    grid = create_latlon_grid(n_lat, n_lon)
    cfg = ll.CGridLatLonShallowWaterConfig(**LATLON_CONFIGS[cname])
    states = _latlon_states(grid)
    olds = {k: ll.cgrid_latlon_sw_tendencies(s, grid, cfg, dt) for k, s in states.items()}
    # Per-field physical scale: a steady state's exact-zero tendency is judged
    # against the flow's tendency magnitude, not against its own round-off.
    scales = [max(float(jnp.max(jnp.abs(o[i]))) for o in olds.values()) for i in range(3)]
    for sname, state in states.items():
        _assert_admissible(state)
        new = latlon_sw_tendencies_generic(state, grid, cfg, dt)
        for name, a, b, sc in zip(("dh", "du", "dv"), new, olds[sname], scales):
            assert _rel(a, b, sc) <= TEND_RTOL, (sname, name, _rel(a, b, sc))
        # Compiled vs compiled: XLA may fuse/contract differently from eager.
        jit_new = jax.jit(lambda s: latlon_sw_tendencies_generic(s, grid, cfg, dt))(state)
        jit_old = jax.jit(lambda s: ll.cgrid_latlon_sw_tendencies(s, grid, cfg, dt))(state)
        for name, a, b, sc in zip(("dh", "du", "dv"), jit_new, jit_old, scales):
            assert _rel(a, b, sc) <= TEND_RTOL, (sname, "jit", name, _rel(a, b, sc))


def _run_latlon(grid, cfg, dt, state, nsteps, generic, monkeypatch):
    with monkeypatch.context() as m:
        if generic:
            m.setattr(ll, "cgrid_latlon_sw_tendencies", latlon_sw_tendencies_generic)
        model = ll.CGridLatLonShallowWaterModel(grid, cfg, dt=dt)
        for _ in range(nsteps):
            state = model.step(state, dt)
    return state


@pytest.mark.parametrize("gname", list(LATLON_GRIDS))
@pytest.mark.parametrize("cname", list(LATLON_CONFIGS))
def test_latlon_trajectories_match(gname, cname, monkeypatch):
    n_lat, n_lon, dt = LATLON_GRIDS[gname]
    grid = create_latlon_grid(n_lat, n_lon)
    cfg = ll.CGridLatLonShallowWaterConfig(**LATLON_CONFIGS[cname])
    state0 = _latlon_states(grid)["tc2_perturbed"]
    old = new = state0
    for nsteps in (10, 90):  # cumulative: 10 and 100 steps
        old = _run_latlon(grid, cfg, dt, old, nsteps, False, monkeypatch)
        new = _run_latlon(grid, cfg, dt, new, nsteps, True, monkeypatch)
        for name in ("h", "u", "v"):
            r = _rel(getattr(new, name), getattr(old, name))
            assert r <= TRAJ_RTOL, (nsteps, name, r)
    # Mass: the fixer restores it; without the fixer the generic (centered)
    # divergence alone must conserve it.  PPM without the fixer is not asserted:
    # the existing PPM path leaks ~2.9e-9 relative per 10 steps identically in
    # both arms (pre-existing, reported separately).
    if cfg.fix_mass or not cfg.use_ppm_transport:
        mass = lambda s: jnp.sum(s.h * grid.area)  # noqa: E731
        assert abs(float(mass(new) - mass(state0))) <= 1e-13 * float(mass(state0))


@pytest.mark.parametrize("use_ppm", [True, False])
def test_latlon_gradient_matches(use_ppm, monkeypatch):
    grid = create_latlon_grid(16, 32)
    cfg = ll.CGridLatLonShallowWaterConfig(use_ppm_transport=use_ppm)
    state0 = _latlon_states(grid)["tc2_perturbed"]
    w = jnp.cos(grid.lat)[:, None] * jnp.sin(grid.lon)[None, :]

    def loss_grad(generic):
        with monkeypatch.context() as m:
            if generic:
                m.setattr(ll, "cgrid_latlon_sw_tendencies", latlon_sw_tendencies_generic)
            model = ll.CGridLatLonShallowWaterModel(grid, cfg, dt=300.0)

            def loss(h0):
                s = state0._replace(h=h0)
                for _ in range(100):
                    s = model.step(s, 300.0)
                return jnp.sum(s.h * w)

            return jax.grad(loss)(state0.h)

    g_old, g_new = loss_grad(False), loss_grad(True)
    assert jnp.all(jnp.isfinite(g_new)) and float(jnp.max(jnp.abs(g_new))) > 0.0
    assert _rel(g_new, g_old) <= TRAJ_RTOL


def test_latlon_generic_check_grads():
    from jax.test_util import check_grads
    grid = create_latlon_grid(16, 32)
    cfg = ll.CGridLatLonShallowWaterConfig(use_ppm_transport=False, A_h=1e5)
    s = _latlon_states(grid)["tc2_perturbed"]
    check_grads(lambda h, u, v: latlon_sw_tendencies_generic(
        s._replace(h=h, u=u, v=v), grid, cfg, 300.0), (s.h, s.u, s.v), order=2)


# ---------------------------------------------------------------- MPAS

def _mpas_states(mesh):
    tc2 = williamson_test2_mpas(mesh)
    h = tc2.h.data + 40.0 * jnp.cos(mesh.latCell) ** 2 * jnp.cos(2 * mesh.lonCell)
    u = tc2.u.data + 3.0 * jnp.cos(mesh.latEdge) * jnp.sin(mesh.lonEdge)
    pert = tc2._replace(h=tc2.h.replace(data=h), u=tc2.u.replace(data=u))
    return {"tc2": tc2, "tc5": williamson_test5_mpas(mesh), "tc2_perturbed": pert}


@pytest.mark.parametrize("cname", list(MPAS_CONFIGS))
def test_mpas_tendencies_bit_identical(cname, mesh):
    cfg = mp.MPASShallowWaterConfig(**MPAS_CONFIGS[cname])
    for sname, state in _mpas_states(mesh).items():
        old = mp.mpas_shallow_water_tendencies(state, mesh, cfg, dt=MPAS_DT)
        new = mpas_sw_tendencies_generic(state, mesh, cfg, dt=MPAS_DT)
        for a, b in ((new.dh_dt, old.dh_dt), (new.du_dt, old.du_dt)):
            assert jnp.all(jnp.isfinite(a.data))
            np.testing.assert_array_equal(np.asarray(a.data), np.asarray(b.data),
                                          err_msg=sname)


def _run_mpas(mesh, cfg, state, nsteps, generic, monkeypatch):
    with monkeypatch.context() as m:
        if generic:
            m.setattr(mp, "mpas_shallow_water_tendencies", mpas_sw_tendencies_generic)
        model = mp.MPASShallowWaterModel(mesh, cfg)
        for _ in range(nsteps):
            state = model.step(state, MPAS_DT)
    return state


@pytest.mark.parametrize("cname", MPAS_TRAJ_CONFIGS)
def test_mpas_trajectories_match(cname, mesh, monkeypatch):
    cfg = mp.MPASShallowWaterConfig(**MPAS_CONFIGS[cname])
    old = new = state0 = _mpas_states(mesh)["tc2_perturbed"]
    mass = lambda s: jnp.sum(s.h.data * mesh.areaCell)  # noqa: E731
    for nsteps in (10, 90):
        old = _run_mpas(mesh, cfg, old, nsteps, False, monkeypatch)
        new = _run_mpas(mesh, cfg, new, nsteps, True, monkeypatch)
        assert abs(float(mass(new) - mass(state0))) <= 1e-13 * float(mass(state0))
        for name in ("h", "u"):
            r = _rel(getattr(new, name).data, getattr(old, name).data)
            assert r <= TRAJ_RTOL, (nsteps, name, r)


def test_mpas_gradient_matches(mesh, monkeypatch):
    cfg = mp.MPASShallowWaterConfig()
    state0 = _mpas_states(mesh)["tc2_perturbed"]
    w = jnp.cos(mesh.latCell) * jnp.sin(mesh.lonCell)

    def loss_grad(generic):
        with monkeypatch.context() as m:
            if generic:
                m.setattr(mp, "mpas_shallow_water_tendencies", mpas_sw_tendencies_generic)
            model = mp.MPASShallowWaterModel(mesh, cfg)

            def loss(h0):
                s = state0._replace(h=state0.h.replace(data=h0))
                for _ in range(100):
                    s = model.step(s, MPAS_DT)
                return jnp.sum(s.h.data * w)

            return jax.grad(loss)(state0.h.data)

    g_old, g_new = loss_grad(False), loss_grad(True)
    assert float(jnp.max(jnp.abs(g_new))) > 0.0
    assert _rel(g_new, g_old) <= TRAJ_RTOL


def test_mpas_generic_check_grads(mesh):
    from jax.test_util import check_grads
    cfg = mp.MPASShallowWaterConfig()
    s = _mpas_states(mesh)["tc2_perturbed"]

    def f(h, u):
        st = s._replace(h=s.h.replace(data=h), u=s.u.replace(data=u))
        t = mpas_sw_tendencies_generic(st, mesh, cfg, dt=MPAS_DT)
        return t.dh_dt.data, t.du_dt.data

    check_grads(f, (s.h.data, s.u.data), order=2)


def test_mpas_unknown_pv_scheme_raises(mesh):
    from legoesm.atmosphere.dynamics.gcm.shallow_water_generic import mpas_sw_operators
    with pytest.raises(ValueError, match="Unknown pv_scheme"):
        mpas_sw_operators(mesh, mp.MPASShallowWaterConfig(pv_scheme="bogus"))


# ------------------------------------------------------- planted controls
# Prove the swapped tendency reaches the jitted step and that the comparisons
# can fail: a 1e-6 relative error in the mass tendency must be detected.

def test_latlon_trajectory_detects_planted_error(monkeypatch):
    grid = create_latlon_grid(16, 32)
    cfg = ll.CGridLatLonShallowWaterConfig()
    state0 = _latlon_states(grid)["tc2_perturbed"]

    def planted(state, g, c, dt=None):
        dh, du, dv = latlon_sw_tendencies_generic(state, g, c, dt)
        return dh * (1.0 + 1e-6), du, dv

    old = _run_latlon(grid, cfg, 300.0, state0, 10, False, monkeypatch)
    with monkeypatch.context() as m:
        m.setattr(ll, "cgrid_latlon_sw_tendencies", planted)
        model = ll.CGridLatLonShallowWaterModel(grid, cfg, dt=300.0)
        bad = state0
        for _ in range(10):
            bad = model.step(bad, 300.0)
    assert _rel(bad.h, old.h) > TRAJ_RTOL


def test_mpas_trajectory_detects_planted_error(mesh, monkeypatch):
    cfg = mp.MPASShallowWaterConfig()
    state0 = _mpas_states(mesh)["tc2_perturbed"]

    def planted(state, m_, c, dt=0.0):
        t = mpas_sw_tendencies_generic(state, m_, c, dt)
        return t._replace(dh_dt=t.dh_dt.replace(data=t.dh_dt.data * (1.0 + 1e-6)))

    old = _run_mpas(mesh, cfg, state0, 10, False, monkeypatch)
    with monkeypatch.context() as m:
        m.setattr(mp, "mpas_shallow_water_tendencies", planted)
        model = mp.MPASShallowWaterModel(mesh, cfg)
        bad = state0
        for _ in range(10):
            bad = model.step(bad, MPAS_DT)
    assert _rel(bad.h.data, old.h.data) > TRAJ_RTOL
