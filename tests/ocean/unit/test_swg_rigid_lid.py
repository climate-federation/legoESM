"""Rigid-lid projection and stepping mode of the SWG SWE RK3 program."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse.linalg as spl
from jax.test_util import check_grads
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.grids.operators_latlon_cgrid import compute_vertex_mask
from legoesm.ocean.dynamics import swe_rk3 as rk3
from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    SWG_DG50_NAMELIST,
    build_swg_card,
    swg_surface_wind,
)
from legoesm.ocean.state import OceanSurfaceForcing
from legoesm.training.swg_online import closed_wall_divergent_part

DIV_TOL = 1.0e-13      # of max|u| / dx, the divergence scale of the input
ROUNDOFF_TOL = 1.0e-13  # relative, projection identities
BUDGET_TOL = 1.0e-9     # of the summed |work| terms over one step
LAYER_M = 500.0


@pytest.fixture(scope="module", autouse=True)
def _fp64():
    old = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    jax.config.update("jax_enable_x64", True)
    yield
    set_policy(old)


@pytest.fixture(scope="module")
def cards():
    return (build_swg_card(SWG_DG50_NAMELIST),
            build_swg_card(SWG_DG50_NAMELIST, surface="rigid_lid"))


@pytest.fixture(scope="module")
def forcing():
    tau_u, tau_v = swg_surface_wind(SWG_DG50_NAMELIST)
    return OceanSurfaceForcing(
        tau_i_native=jnp.asarray(tau_u), tau_j_native=jnp.asarray(tau_v))


def _face_masks(card):
    u_full, v_full = compute_face_masks(card.recipe.land_mask, card.recipe.grid)
    return (jnp.asarray(u_full[:, 1:], dtype=jnp.float64),
            jnp.asarray(v_full[1:, :], dtype=jnp.float64))


def _random_faces(card, seed):
    rng = np.random.default_rng(seed)
    shape = card.recipe.initial_state.eta.data.shape
    umask, vmask = _face_masks(card)
    return (jnp.asarray(rng.standard_normal(shape)) * umask,
            jnp.asarray(rng.standard_normal(shape)) * vmask)


def _divergence(u, v, dx):
    return (u - jnp.roll(u, 1, axis=1) + v - jnp.roll(v, 1, axis=0)) / dx


def _dot(a, b):
    """Energy inner product per unit density: constant thickness and area."""
    return float(LAYER_M * 50000.0 ** 2 * sum(
        jnp.sum(x * y) for x, y in zip(a, b)))


def _run(card, forcing, dt, steps, state=None):
    state = card.recipe.initial_state if state is None else state
    state = state._replace(
        tau_x_prev=forcing.tau_i_native, tau_y_prev=forcing.tau_j_native)
    grid, z_coord = card.recipe.grid, card.recipe.z_coord
    config = card.recipe.model_config

    def body(carry, _):
        return rk3.swe_rk3_step(carry, dt, forcing, grid, z_coord, config), None

    return jax.jit(
        lambda s: jax.lax.scan(body, s, None, length=steps)[0])(state)


def _native(state):
    return state.u.data[:, 1:, 0], state.v.data[1:, :, 0]


def test_projection_is_the_exact_neumann_solve(cards):
    _, lid = cards
    geo = rk3.swe_rigid_lid_geometry(lid.recipe.grid, lid.recipe.z_coord)
    u, v = _random_faces(lid, 0)
    umask, vmask = _face_masks(lid)
    u_nd, v_nd, phi = rk3.swe_rigid_lid_projection(u, v, geo)
    dx = geo.spacing_m
    j0, j1, i0, i1 = geo.wet_bounds
    scale = float(jnp.abs(u).max() / dx)
    assert float(jnp.abs(_divergence(u_nd, v_nd, dx)).max()) <= DIV_TOL * scale
    # Zero normal flow: wall faces are never written.
    assert np.array_equal(np.asarray(u_nd * (1 - umask)), 0 * np.asarray(u))
    assert np.array_equal(np.asarray(v_nd * (1 - vmask)), 0 * np.asarray(v))
    # phi: zero wet-area mean, zero on land, and the Poisson residual.
    wet = np.asarray(lid.recipe.land_mask) > 0
    assert abs(float(phi[wet].mean())) <= ROUNDOFF_TOL * float(jnp.abs(phi).max())
    assert np.all(np.asarray(phi)[~wet] == 0.0)
    rhs = _divergence(u, v, dx)[j0:j1 + 1, i0:i1 + 1]
    rhs = rhs - rhs.mean()
    lap = _divergence(u - u_nd, v - v_nd, dx)[j0:j1 + 1, i0:i1 + 1]
    assert float(jnp.abs(lap - rhs).max()) <= ROUNDOFF_TOL * float(jnp.abs(rhs).max())
    # The removed part is the gradient of phi.
    grad_u = jnp.zeros_like(u).at[j0:j1 + 1, i0:i1].set(
        (phi[j0:j1 + 1, i0 + 1:i1 + 1] - phi[j0:j1 + 1, i0:i1]) / dx)
    grad_v = jnp.zeros_like(v).at[j0:j1, i0:i1 + 1].set(
        (phi[j0 + 1:j1 + 1, i0:i1 + 1] - phi[j0:j1, i0:i1 + 1]) / dx)
    assert float(jnp.abs(u - u_nd - grad_u).max()) <= ROUNDOFF_TOL * float(jnp.abs(u).max())
    assert float(jnp.abs(v - v_nd - grad_v).max()) <= ROUNDOFF_TOL * float(jnp.abs(v).max())


def test_projection_is_an_orthogonal_idempotent_projector(cards):
    _, lid = cards
    geo = rk3.swe_rigid_lid_geometry(lid.recipe.grid, lid.recipe.z_coord)
    a = _random_faces(lid, 1)
    b = _random_faces(lid, 2)
    pa = rk3.swe_rigid_lid_projection(*a, geo)[:2]
    pb = rk3.swe_rigid_lid_projection(*b, geo)[:2]
    ppa = rk3.swe_rigid_lid_projection(*pa, geo)[:2]
    norm = lambda x: np.sqrt(_dot(x, x))  # noqa: E731
    assert max(float(jnp.abs(x - y).max()) for x, y in zip(ppa, pa)) \
        <= ROUNDOFF_TOL * float(jnp.abs(pa[0]).max())
    removed = tuple(x - y for x, y in zip(a, pa))
    assert abs(_dot(pa, removed)) <= ROUNDOFF_TOL * norm(pa) * norm(removed)
    assert abs(_dot(pa, b) - _dot(a, pb)) <= ROUNDOFF_TOL * norm(a) * norm(b)
    jitted = jax.jit(lambda u, v: rk3.swe_rigid_lid_projection(u, v, geo))(*a)
    assert all(np.allclose(np.asarray(x), np.asarray(y), rtol=0,
                           atol=ROUNDOFF_TOL * float(jnp.abs(y).max()))
               for x, y in zip(jitted, rk3.swe_rigid_lid_projection(*a, geo)))


def test_projection_leaves_a_rotated_gradient_unchanged(cards):
    _, lid = cards
    geo = rk3.swe_rigid_lid_geometry(lid.recipe.grid, lid.recipe.z_coord)
    rng = np.random.default_rng(3)
    shape = lid.recipe.initial_state.eta.data.shape
    vertex = compute_vertex_mask(lid.recipe.land_mask, grid=lid.recipe.grid)
    umask, vmask = _face_masks(lid)
    metric = jnp.full(shape, geo.spacing_m)
    thickness = jnp.full(shape, LAYER_M)
    force = rk3.swe_rotated_potential_forcing(
        jnp.asarray(rng.standard_normal(shape)), metric, metric, thickness,
        thickness, umask, vmask, vertex[1:, 1:])
    projected = rk3.swe_rigid_lid_projection(*force, geo)
    scale = float(jnp.abs(force[0]).max())
    assert max(float(jnp.abs(x - y).max()) for x, y in zip(projected, force)) \
        <= ROUNDOFF_TOL * scale
    assert float(jnp.abs(projected[2]).max()) <= ROUNDOFF_TOL * scale * geo.spacing_m


def test_projection_gradients_match_finite_differences(cards):
    _, lid = cards
    geo = rk3.swe_rigid_lid_geometry(lid.recipe.grid, lid.recipe.z_coord)
    u, v = _random_faces(lid, 4)
    weights = _random_faces(lid, 5)
    phi_weight = jnp.asarray(np.random.default_rng(6).standard_normal(u.shape))

    def scalar(u, v):
        u_nd, v_nd, phi = rk3.swe_rigid_lid_projection(u, v, geo)
        return jnp.sum(weights[0] * u_nd ** 2 + weights[1] * v_nd ** 2
                       + phi_weight * phi ** 2)

    check_grads(scalar, (u, v), order=2, modes=("fwd", "rev"),
                atol=1e-6, rtol=1e-6)


def test_offline_numpy_projection_is_the_same_operator(cards):
    _, lid = cards
    geo = rk3.swe_rigid_lid_geometry(lid.recipe.grid, lid.recipe.z_coord)
    u, v = _random_faces(lid, 7)
    phi_np, u_div, v_div = closed_wall_divergent_part(
        np.asarray(u), np.asarray(v), np.asarray(lid.recipe.land_mask),
        geo.spacing_m)
    u_nd, v_nd, phi = rk3.swe_rigid_lid_projection(u, v, geo)
    for ours, theirs in ((u - u_nd, u_div), (v - v_nd, v_div), (phi, phi_np)):
        assert np.allclose(np.asarray(ours), theirs, rtol=0,
                           atol=ROUNDOFF_TOL * np.abs(theirs).max())


def test_surface_selection_is_required_and_closed(cards):
    explicit, lid = cards
    assert explicit.recipe.model_config.swe_surface == "explicit"
    assert lid.recipe.model_config.swe_surface == "rigid_lid"
    assert explicit.recipe.initial_state.rigid_lid_phi is None
    assert lid.recipe.initial_state.rigid_lid_phi.shape == (84, 84)
    with pytest.raises(ValueError, match="surface must be one of"):
        build_swg_card(SWG_DG50_NAMELIST, surface="free")
    unstated = explicit.recipe.model_config._replace(swe_surface=None)
    entry = rk3._entry_from_state(
        explicit.recipe.initial_state, (0.0, 0.0), (0.0, 0.0),
        explicit.recipe.grid, explicit.recipe.z_coord, unstated)
    with pytest.raises(ValueError, match="swe_surface stated"):
        rk3.swe_rk3_stage(
            entry, entry, explicit.dt_s, 1, explicit.recipe.grid,
            explicit.recipe.z_coord, explicit.recipe.land_mask, unstated)
    z_coord = lid.recipe.z_coord
    notched = np.asarray(z_coord.is_active).copy()
    notched[40, 40, 0] = False
    with pytest.raises(ValueError, match="not one rectangle"):
        rk3.swe_rigid_lid_geometry(
            lid.recipe.grid, z_coord._replace(is_active=jnp.asarray(notched)))
    sloped = np.asarray(z_coord.nemo_e3t_0).copy()
    sloped[40, 40, 0] += 1.0
    with pytest.raises(ValueError, match="constant layer thickness"):
        rk3.swe_rigid_lid_geometry(
            lid.recipe.grid, z_coord._replace(nemo_e3t_0=jnp.asarray(sloped)))


def test_explicit_mode_is_the_unchanged_stage_chain(cards, forcing):
    explicit, _ = cards
    grid, z_coord = explicit.recipe.grid, explicit.recipe.z_coord
    config = explicit.recipe.model_config
    dt = explicit.dt_s
    state = explicit.recipe.initial_state._replace(
        tau_x_prev=forcing.tau_i_native, tau_y_prev=forcing.tau_j_native)
    now = (forcing.tau_i_native, forcing.tau_j_native)
    chained = state
    for _ in range(10):
        boundary = rk3._entry_from_state(chained, now, now, grid, z_coord, config)
        s1 = rk3.swe_rk3_stage(boundary, boundary, dt, 1, grid, z_coord,
                               chained.land_mask.data, config)
        s2 = rk3.swe_rk3_stage(s1, boundary, dt, 2, grid, z_coord,
                               chained.land_mask.data, config)
        s3 = rk3.swe_rk3_stage(s2, boundary, dt, 3, grid, z_coord,
                               chained.land_mask.data, config)
        chained = chained._replace(
            eta=chained.eta.replace(data=s3.ssh_k),
            u=chained.u.replace(data=rk3._native_to_u(s3.u_k)),
            v=chained.v.replace(data=rk3._native_to_v(s3.v_k)))
    stepped = state
    for _ in range(10):
        stepped = rk3.swe_rk3_step(stepped, dt, forcing, grid, z_coord, config)
    assert np.array_equal(np.asarray(stepped.u.data), np.asarray(chained.u.data))
    assert np.array_equal(np.asarray(stepped.v.data), np.asarray(chained.v.data))
    assert np.array_equal(np.asarray(stepped.eta.data), np.asarray(chained.eta.data))
    assert float(jnp.abs(stepped.eta.data).max()) > 0.0
    assert stepped.rigid_lid_phi is None


@pytest.fixture(scope="module")
def lid_hundred_steps(cards, forcing):
    return _run(cards[1], forcing, 3600.0, 100)


def test_rigid_lid_hundred_hourly_steps(cards, lid_hundred_steps):
    _, lid = cards
    geo = rk3.swe_rigid_lid_geometry(lid.recipe.grid, lid.recipe.z_coord)
    state = lid_hundred_steps
    u, v = _native(state)
    assert np.all(np.isfinite(np.asarray(u))) and np.all(np.isfinite(np.asarray(v)))
    assert np.array_equal(np.asarray(state.eta.data), np.zeros((84, 84)))
    scale = float(jnp.abs(u).max() / geo.spacing_m)
    assert float(jnp.abs(_divergence(u, v, geo.spacing_m)).max()) <= DIV_TOL * scale
    ke = 0.5 * _dot((u, v), (u, v))
    assert 0.0 < ke and float(jnp.abs(u).max()) < 1.0
    wet = np.asarray(lid.recipe.land_mask) > 0
    phi = np.asarray(state.rigid_lid_phi)
    assert np.all(np.isfinite(phi)) and abs(phi[wet].mean()) <= ROUNDOFF_TOL * np.abs(phi).max()


def test_rigid_lid_energy_budget_pressure_does_no_work(
        cards, forcing, lid_hundred_steps):
    """KE(n+1) - KE(n) = wind + viscous + drag + advective works; lid work 0.

    Every work term is the certified stage program evaluated on the same
    entries: wind and viscosity by switching one input off, advection plus
    Coriolis from the stage-1 program on the stage-2 entry, drag from the
    whole-step before velocity.  Signs: wind positive into the fluid, drag
    and viscosity negative.
    """
    _, lid = cards
    grid, z_coord, mask = lid.recipe.grid, lid.recipe.z_coord, lid.recipe.land_mask
    config = lid.recipe.model_config
    dt = 3600.0
    geo = rk3.swe_rigid_lid_geometry(grid, z_coord)
    before_state = lid_hundred_steps
    after_state = rk3.swe_rk3_step(before_state, dt, forcing, grid, z_coord, config)
    u_b = _native(before_state)
    u_a = _native(after_state)
    now = (forcing.tau_i_native, forcing.tau_j_native)
    boundary = rk3._entry_from_state(before_state, now, now, grid, z_coord, config)
    certified = config._replace(swe_surface="explicit")

    def stage(entry, stage_number, cfg=certified, **changes):
        return rk3.swe_rk3_stage(
            entry._replace(**changes), boundary, dt, stage_number, grid,
            z_coord, mask, cfg)

    s1, _ = rk3.swe_rigid_lid_stage_output(
        stage(boundary, 1), boundary, 1, dt / 3.0, geo)
    s2, _ = rk3.swe_rigid_lid_stage_output(stage(s1, 2), s1, 2, dt / 2.0, geo)
    for projected in (s1, s2):
        scale = float(jnp.abs(projected.u_k).max() / geo.spacing_m)
        assert float(jnp.abs(_divergence(
            projected.u_k, projected.v_k, geo.spacing_m)).max()) <= DIV_TOL * scale
    raw3 = stage(s2, 3)
    s3, _ = rk3.swe_rigid_lid_stage_output(raw3, s2, 3, dt, geo)
    assert np.array_equal(np.asarray(s3.u_k), np.asarray(u_a[0]))
    assert np.array_equal(np.asarray(s3.u_before), np.asarray(s3.u_k))
    assert np.array_equal(np.asarray(s3.v_before), np.asarray(s3.v_k))
    raw1_on_s2 = stage(s2, 1)
    zero = jnp.zeros_like(boundary.utau_now)
    no_wind = stage(s2, 3, utau_before=zero, utau_now=zero,
                    vtau_before=zero, vtau_now=zero)
    inviscid = stage(s2, 3, cfg=certified._replace(
        lateral_viscosity=certified.lateral_viscosity._replace(A_h=0.0)))
    increment = (raw3.u_k - u_b[0], raw3.v_k - u_b[1])
    wind = (raw3.u_k - no_wind.u_k, raw3.v_k - no_wind.v_k)
    viscous = (raw3.u_k - inviscid.u_k, raw3.v_k - inviscid.v_k)
    advective = (3.0 * (raw1_on_s2.u_k - u_b[0]), 3.0 * (raw1_on_s2.v_k - u_b[1]))
    drag_r = config.bottom_drag.bottom_drag_r
    drag = (-dt * drag_r * u_b[0], -dt * drag_r * u_b[1])
    pressure = (u_b[0] + increment[0] - u_a[0], u_b[1] + increment[1] - u_a[1])
    mean = ((u_b[0] + u_a[0]) / 2, (u_b[1] + u_a[1]) / 2)
    works = {name: _dot(mean, term) for name, term in (
        ("wind", wind), ("viscous", viscous), ("drag", drag),
        ("advective", advective), ("pressure", pressure))}
    delta_ke = 0.5 * (_dot(u_a, u_a) - _dot(u_b, u_b))
    budget = sum(abs(w) for w in works.values())
    print(f"\nSTEP101 dKE {delta_ke:.6e}  " + "  ".join(
        f"{name} {work:.6e}" for name, work in works.items()))
    assert works["wind"] > 0 and works["viscous"] < 0 and works["drag"] < 0
    assert abs(works["pressure"]) <= BUDGET_TOL * budget
    assert abs(delta_ke - sum(works[k] for k in ("wind", "viscous", "drag", "advective"))) \
        <= BUDGET_TOL * budget
    # The pressure potential kept on the state is the removed gradient / dt.
    j0, j1, i0, i1 = geo.wet_bounds
    phi = after_state.rigid_lid_phi
    grad_u = (phi[j0:j1 + 1, i0 + 1:i1 + 1] - phi[j0:j1 + 1, i0:i1]) / geo.spacing_m
    assert float(jnp.abs(dt * grad_u - pressure[0][j0:j1 + 1, i0:i1]).max()) \
        <= ROUNDOFF_TOL * float(jnp.abs(pressure[0]).max())


def test_rigid_lid_linear_operator_is_the_gravest_rossby_basin_mode(cards):
    """At rest the projected stage-1 operator's spectral radius is the
    gravest Rossby basin frequency beta L / (2 pi sqrt 2), not f."""
    _, lid = cards
    nl = SWG_DG50_NAMELIST
    grid, z_coord, mask = lid.recipe.grid, lid.recipe.z_coord, lid.recipe.land_mask
    config = lid.recipe.model_config
    geo = rk3.swe_rigid_lid_geometry(grid, z_coord)
    j0, j1, i0, i1 = geo.wet_bounds
    dx = geo.spacing_m
    umask, vmask = _face_masks(lid)
    rest = rk3._entry_from_state(
        lid.recipe.initial_state, (0.0, 0.0), (0.0, 0.0), grid, z_coord, config)
    dt = lid.dt_s
    shape = rest.u_k.shape

    def psi_to_uv(psi_int):
        psi = jnp.zeros(shape).at[j0:j1, i0:i1].set(psi_int)
        south = jnp.concatenate([jnp.zeros_like(psi[:1]), psi[:-1]], 0)
        west = jnp.concatenate([jnp.zeros_like(psi[:, :1]), psi[:, :-1]], 1)
        return -(psi - south) / dx * umask, (psi - west) / dx * vmask

    def uv_to_psi(u, v):
        return jnp.cumsum(v[j0:j1, i0:i1], axis=1) * dx

    def tendency(uv):
        out, _ = rk3.swe_rigid_lid_stage_output(
            rk3.swe_rk3_stage(rest._replace(u_k=uv[0], v_k=uv[1]), rest, dt,
                              1, grid, z_coord, mask, config),
            rest, 1, dt / 3.0, geo)
        return (out.u_k - rest.u_k) / (dt / 3.0), (out.v_k - rest.v_k) / (dt / 3.0)

    ny, nx = j1 - j0, i1 - i0
    psi0 = jnp.asarray(np.random.default_rng(8).standard_normal((ny, nx)))
    assert float(jnp.abs(uv_to_psi(*psi_to_uv(psi0)) - psi0).max()) \
        <= ROUNDOFF_TOL * float(jnp.abs(psi0).max())
    _, linear = jax.linearize(tendency, (rest.u_k, rest.v_k))
    linear = jax.jit(linear)

    def matvec(flat):
        du, dv = linear(psi_to_uv(jnp.asarray(flat).reshape(ny, nx)))
        return np.asarray(uv_to_psi(du, dv)).ravel()

    operator = spl.LinearOperator((ny * nx, ny * nx), matvec=matvec, dtype=np.float64)
    values = spl.eigs(operator, k=2, which="LM", tol=1e-5, maxiter=5000,
                      return_eigenvectors=False)
    omega = nl.rn_beta_m_s * nl.rn_domsiz_m / (2.0 * np.pi * np.sqrt(2.0))
    assert np.abs(values).max() == pytest.approx(omega, rel=0.01)
    assert np.abs(values).max() < 0.2 * nl.rn_f0_s


def _transport_streamfunction_sv(v_native, geo):
    j0, j1, i0, i1 = geo.wet_bounds
    psi = np.cumsum(np.asarray(v_native)[j0:j1, i0:i1], axis=1) * geo.spacing_m
    return psi * LAYER_M * 1.0e-6


def test_thirty_days_from_rest_spin_a_double_gyre(cards, forcing):
    explicit, lid = cards
    geo = rk3.swe_rigid_lid_geometry(lid.recipe.grid, lid.recipe.z_coord)
    lid_state = _run(lid, forcing, 3600.0, 30 * 24)
    explicit_state = _run(explicit, forcing, 400.0, 30 * 216)
    report = {}
    for name, state in (("rigid_lid dt=3600", lid_state),
                        ("explicit dt=400", explicit_state)):
        u, v = _native(state)
        assert np.all(np.isfinite(np.asarray(u))) and np.all(np.isfinite(np.asarray(v)))
        psi = _transport_streamfunction_sv(v, geo)
        ke = 0.5 * _dot((u, v), (u, v))
        report[name] = (float(psi.max()), float(psi.min()), ke,
                        float(np.abs(np.asarray(state.eta.data)).max()))
        # Double gyre: a lobe of each sign, neither negligible.
        assert psi.max() > 0.2 * abs(psi.min()) and -psi.min() > 0.2 * psi.max()
        assert ke > 0.0
    assert report["rigid_lid dt=3600"][3] == 0.0
    pmax, pmin, ke, _ = report["rigid_lid dt=3600"]
    assert pmax == pytest.approx(53.60, abs=0.1)
    assert pmin == pytest.approx(-53.60, abs=0.1)
    assert ke == pytest.approx(6.977e13, rel=0.01)
    print("\nDAY30 " + " | ".join(
        f"{name}: max psi {pmax:.3f} Sv, min psi {pmin:.3f} Sv, "
        f"KE {ke:.4e} m^5 s^-2 (per unit density), max|ssh| {ssh:.3e} m"
        for name, (pmax, pmin, ke, ssh) in report.items()))
