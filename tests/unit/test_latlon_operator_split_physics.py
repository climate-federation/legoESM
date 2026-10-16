"""Lat-lon C-grid lane: OPERATOR-SPLIT physics (user decision 2026-09-28).

Physics is evaluated ONCE per step on the post-dynamics state and applied
as an Euler increment; water tendencies carry their mass (rain leaves the
column through p_s); the mass fixer conserves DRY air.  Before: physics
inside every RK stage, water applied at fixed p_s (rain became dry air),
n_stages + 1 physics evaluations per step.

Gates (GLM/codex-corrected: the dry-mass fixer makes the physics=None step
differ from the old total-mass step at rounding, so the identities below
are between NEW-code paths, and the discriminating test is the budget):
  * physics_fn=None and a zero-tendency physics_fn give the SAME state
    (bitwise: exact-zero increments are exact identities, and the
    no-water-tendency helper path returns the tracers untouched)
  * physics is called exactly once per step and its second return is
    the carry-out
  * a rain-only physics: global dry mass fixed to 1e-13, water down by
    the rain, p_s down by g*rain; a pure phase change moves nothing
  * a wind-tendency physics: the applied increment is dt times the
    face-interpolated tendency away from the poles (filter identity
    there), and damped at the polar rows exactly as the RK filter damps
    a dynamics tendency (same operator)
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (  # noqa: E402
    CGridLatLonHydrostaticState,
    CGridLatLonPrimitiveEquationConfig,
    CGridLatLonPrimitiveEquationModel,
)
from legoesm.grids.operators_latlon_cgrid import (  # noqa: E402
    interp_cell_to_uface,
    interp_cell_to_vface_halo,
)
from legoesm.core.conservation import dry_surface_pressure  # noqa: E402
from legoesm.core.state import HydrostaticTendencies  # noqa: E402
from legoesm.grids.latlon import create_latlon_grid  # noqa: E402
from legoesm.grids.vertical import create_sigma_coordinate  # noqa: E402

DT = 600.0
NLEV = 6


@pytest.fixture(scope="module")
def setup():
    grid = create_latlon_grid(n_lat=16, radius=constants.R_earth, omega=constants.Omega)
    sigma = create_sigma_coordinate(n_levels=NLEV, dtype=jnp.float64)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    rng = np.random.default_rng(0)
    lat = np.asarray(grid.lat)
    T = 280.0 - 30.0 * np.sin(lat)[:, None, None] ** 2 + np.linspace(20.0, -20.0, NLEV)[None, None, :]
    T = jnp.asarray(np.broadcast_to(T, (n_lat, n_lon, NLEV)) + 0.1 * rng.standard_normal((n_lat, n_lon, NLEV)))
    p_s = jnp.asarray(1.0e5 + 500.0 * rng.standard_normal((n_lat, n_lon)))
    q_v = jnp.asarray(0.015 * rng.uniform(0.3, 1.0, (n_lat, n_lon, NLEV)))
    # q_r large enough that the rain-only physics below (2 layers x
    # 1e-6/s x 600 s = 1.2e-3) never drives a column negative: the
    # positivity borrow after the fixer is column-conserving only
    # while the column total stays positive (MEASURED 2026-09-28:
    # 2e-8 dry-mass residual with q_r ~ 1e-3, 3e-14 with 5e-3)
    q_r = jnp.asarray(5e-3 * rng.uniform(0.6, 1.0, (n_lat, n_lon, NLEV)))
    state = CGridLatLonHydrostaticState(
        u=jnp.asarray(5.0 * rng.standard_normal((n_lat, n_lon + 1, NLEV))),
        v=jnp.zeros((n_lat + 1, n_lon, NLEV)),
        T=T, p_s=p_s, phis=jnp.zeros((n_lat, n_lon)),
        tracers={"q_v": q_v, "q_c": jnp.zeros_like(q_v), "q_r": q_r,
                 "trc_o3": jnp.asarray(rng.uniform(0.1, 1.0, (n_lat, n_lon, NLEV)))},
    )
    return grid, sigma, state


def _model(grid, sigma, **cfg):
    return CGridLatLonPrimitiveEquationModel(
        grid, sigma, CGridLatLonPrimitiveEquationConfig(**cfg))


def _zeros(hs):
    return HydrostaticTendencies(
        du_dt=hs.u.replace(data=jnp.zeros_like(hs.u.data)),
        dv_dt=(None if hs.v is None else hs.v.replace(data=jnp.zeros_like(hs.v.data))),
        dT_dt=hs.T.replace(data=jnp.zeros_like(hs.T.data)),
        dp_s_dt=hs.p_s.replace(data=jnp.zeros_like(hs.p_s.data)),
        dphis_dt=hs.p_s.replace(data=jnp.zeros_like(hs.p_s.data)),
    )


def _counting(inner):
    calls = []

    def fn(hs, grid, sigma, phys_state=None):
        calls.append(1)
        out = inner(hs)
        return (out, phys_state) if phys_state is not None else out
    return fn, calls


def _budgets(state, sigma, grid):
    area = np.asarray(grid.area, dtype=np.float64)
    dp = np.asarray(sigma.layer_thickness_dp(state.p_s))
    Q = sum(np.asarray(state.tracers[k]) for k in ("q_v", "q_c", "q_r"))
    water = ((dp * Q).sum(-1) / constants.g * area).sum()
    dry = (np.asarray(dry_surface_pressure(state.p_s, state.tracers, sigma)) / constants.g * area).sum()
    o3 = ((dp * np.asarray(state.tracers["trc_o3"])).sum(-1) / constants.g * area).sum()
    return water, dry, o3, (np.asarray(state.p_s) * area).sum()


def test_zero_physics_equals_no_physics_bitwise_and_is_called_once(setup):
    grid, sigma, state = setup
    model = _model(grid, sigma)
    ref = model.step(state, DT)
    fn, calls = _counting(_zeros)
    got = model.step(state, DT, physics_fn=fn)
    assert len(calls) == 1
    for k in ("u", "v", "T", "p_s"):
        np.testing.assert_array_equal(np.asarray(getattr(got, k)), np.asarray(getattr(ref, k)), err_msg=k)
    for k in state.tracers:
        np.testing.assert_array_equal(np.asarray(got.tracers[k]), np.asarray(ref.tracers[k]), err_msg=k)


def test_carry_out_is_the_single_evaluations_second_return(setup):
    grid, sigma, state = setup
    model = _model(grid, sigma)
    seen = []

    def fn(hs, grid_, sigma_, phys_state=None):
        seen.append(phys_state)
        return _zeros(hs), phys_state + 1.0
    model.step(state, DT, physics_fn=fn, phys_state=jnp.asarray(3.0))
    assert len(seen) == 1                      # ONE evaluation per step (traced)
    assert float(model._phys_state) == 4.0     # its second return is the carry


def _rain_only(rate):
    def inner(hs):
        z = _zeros(hs)
        nl = hs.T.data.shape[-1]
        dq = jnp.zeros_like(hs.T.data).at[..., nl - 2:].set(-rate)
        return z._replace(tracer_tendencies={"q_r": hs.T.replace(data=dq, name="dq_r_dt", units="1/s")})
    return inner


@pytest.mark.parametrize("anchor", [False, True])
def test_rain_leaves_through_ps_and_dry_mass_is_fixed(setup, anchor):
    grid, sigma, state = setup
    model = _model(grid, sigma, fix_mass=True, anchor_mass_to_initial=anchor)
    rate = 1e-6
    fn, _ = _counting(_rain_only(rate))
    w0, d0, o0, tot0 = _budgets(state, sigma, grid)
    out = model.step(state, DT, physics_fn=fn)
    w1, d1, o1, tot1 = _budgets(out, sigma, grid)
    # the passenger's column mass is what DYNAMICS alone leaves (the
    # advection is not column-exact on this lane); physics adds nothing
    _, _, o_dyn, _ = _budgets(model.step(state, DT), sigma, grid)
    # rain as the physics meant it, on the post-dynamics layer masses is
    # not observable from here; the pre-step masses bound it to 1e-3
    dp0 = np.asarray(sigma.layer_thickness_dp(state.p_s))
    area = np.asarray(grid.area, dtype=np.float64)
    rain = (rate * DT * dp0[..., NLEV - 2:].sum(-1) / constants.g * area).sum()
    assert rain > 1e-9 * w0
    np.testing.assert_allclose(d1, d0, rtol=1e-13, atol=0)               # dry mass FIXED
    np.testing.assert_allclose(w1 - w0, -rain, rtol=5e-3, atol=0)        # water left (2.1e-3 MEASURED: pre-step masses)
    np.testing.assert_allclose(tot1 - tot0, -constants.g * rain, rtol=5e-3, atol=0)  # p_s fell
    np.testing.assert_allclose(o1, o_dyn, rtol=1e-12, atol=0)            # passenger untouched by physics
    # the OLD coupling (fixed p_s, total-mass fixer) kept total mass:
    assert abs(tot1 - tot0) > 0.5 * constants.g * rain


def test_phase_change_moves_no_mass(setup):
    grid, sigma, state = setup
    # fixer off: its p_s shift re-weights every tracer by dp/dp' (5e-6
    # MEASURED), which would put the fixer's factor into the increment
    model = _model(grid, sigma, fix_mass=False)

    def inner(hs):
        z = _zeros(hs)
        d = jnp.full_like(hs.T.data, 1e-7)
        return z._replace(tracer_tendencies={
            "q_v": hs.T.replace(data=-d, name="dq_v_dt", units="1/s"),
            "q_c": hs.T.replace(data=d, name="dq_c_dt", units="1/s")})
    fn, _ = _counting(inner)
    ref = model.step(state, DT)
    out = model.step(state, DT, physics_fn=fn)
    np.testing.assert_array_equal(np.asarray(out.p_s), np.asarray(ref.p_s))
    np.testing.assert_allclose(np.asarray(out.tracers["q_v"] + out.tracers["q_c"]),
                               np.asarray(ref.tracers["q_v"] + ref.tracers["q_c"]), rtol=1e-14)
    # each species individually, with its sign (codex: the sum alone passes
    # with both increments dropped or both reversed); p_s unchanged so the
    # mass re-weighting dp/dp' is exactly 1 and the increment is dt*d
    np.testing.assert_allclose(np.asarray(out.tracers["q_v"] - ref.tracers["q_v"]),
                               -DT * 1e-7, rtol=1e-12, atol=0)
    np.testing.assert_allclose(np.asarray(out.tracers["q_c"] - ref.tracers["q_c"]),
                               DT * 1e-7, rtol=1e-12, atol=0)


def test_wind_increment_is_dt_times_face_interpolated_tendency(setup):
    grid, sigma, state = setup
    # the polar filter is OFF by default on this lane: on, so the
    # increment path's filter branch runs
    model = _model(grid, sigma, fix_mass=False, use_polar_filter=True)
    assert model._polar_mask is not None
    n_lat, n_lon = grid.n_lat, grid.n_lon
    rng = np.random.default_rng(3)
    du_c = jnp.asarray(1e-4 * rng.standard_normal((n_lat, n_lon, NLEV)))
    dv_c = jnp.asarray(1e-4 * rng.standard_normal((n_lat, n_lon, NLEV)))

    def inner(hs):
        z = _zeros(hs)
        return z._replace(du_dt=hs.T.replace(data=du_c, name="du_dt", units="m/s^2"),
                          dv_dt=hs.T.replace(data=dv_c, name="dv_dt", units="m/s^2"))
    fn, _ = _counting(inner)
    ref = model.step(state, DT)
    out = model.step(state, DT, physics_fn=fn)
    got_du = np.asarray(out.u - ref.u)
    got_dv = np.asarray(out.v - ref.v)
    exp_du = DT * np.asarray(interp_cell_to_uface(du_c))
    exp_dv = DT * np.asarray(interp_cell_to_vface_halo(dv_c))
    # away from the poles the polar filter is the identity
    mid = slice(n_lat // 4, 3 * n_lat // 4)
    np.testing.assert_allclose(got_du[mid], exp_du[mid], rtol=1e-11, atol=1e-15)
    np.testing.assert_allclose(got_dv[mid], exp_dv[mid], rtol=1e-11, atol=1e-15)
    # at the poles the increment is DAMPED, not dropped, and v is walled
    assert np.abs(got_du[0]).max() < np.abs(exp_du[0]).max()
    assert np.abs(got_du[0]).max() > 0.0
    assert np.abs(got_dv[0]).max() == 0.0 and np.abs(got_dv[-1]).max() == 0.0
    # and the SAME filter that the dynamics tendency gets: filter the
    # expected increment with the model's own operator and compare bitwise
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        fourier_filter_3d, pad_lon_cgrid)   # the module's own names
    f_int = fourier_filter_3d(jnp.asarray(exp_du)[:, :-1, :], grid, model._polar_mask)
    f_du = np.asarray(jnp.concatenate([f_int, pad_lon_cgrid(f_int, halo=1)[:, -1:, :]], axis=1))
    np.testing.assert_allclose(got_du, f_du, rtol=1e-11, atol=1e-15)
    # v: the v-face mask on every INTERIOR face row (codex: equator +
    # walled endpoints alone pass with the wrong mask or no filter)
    f_dv = np.asarray(fourier_filter_3d(jnp.asarray(exp_dv), grid, model._polar_mask_v))
    assert model._polar_mask_v is not None and np.abs(f_dv[1:-1] - exp_dv[1:-1]).max() > 0.0
    np.testing.assert_allclose(got_dv[1:-1], f_dv[1:-1], rtol=1e-11, atol=1e-15)


def test_scalar_increments_are_dt_times_tendency_through_the_polar_filter(setup):
    grid, sigma, state = setup
    model = _model(grid, sigma, fix_mass=False, use_polar_filter=True)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    rng = np.random.default_rng(5)
    dT_c = jnp.asarray(1e-4 * rng.standard_normal((n_lat, n_lon, NLEV)))
    dps_c = jnp.asarray(1e-2 * rng.standard_normal((n_lat, n_lon)))

    def inner(hs):
        z = _zeros(hs)
        return z._replace(dT_dt=hs.T.replace(data=dT_c, name="dT_dt", units="K/s"),
                          dp_s_dt=hs.p_s.replace(data=dps_c, name="dp_s_dt", units="Pa/s"))
    fn, _ = _counting(inner)
    ref = model.step(state, DT)
    out = model.step(state, DT, physics_fn=fn)
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        fourier_filter, fourier_filter_3d)
    exp_dT = np.asarray(fourier_filter_3d(DT * dT_c, grid, model._polar_mask))
    exp_dps = np.asarray(fourier_filter(DT * dps_c, grid, model._polar_mask))
    # non-vacuity: the filter moved the polar rows, and the sign is the field's
    assert np.abs(exp_dT[0] - DT * np.asarray(dT_c)[0]).max() > 0.0
    # atol: (T+dT)-T on a 280 K field rounds at 3e-14 (MEASURED 2.8e-14)
    np.testing.assert_allclose(np.asarray(out.T - ref.T), exp_dT, rtol=1e-11, atol=1e-13)
    # atol: (p_s+dps)-p_s on a 1e5 Pa field rounds at 1e-11 (MEASURED 7e-12)
    np.testing.assert_allclose(np.asarray(out.p_s - ref.p_s), exp_dps, rtol=1e-11, atol=1e-10)
    # tracers untouched by a direct p_s source; the positivity stage's
    # column borrow re-derives dp from the new p_s (1e-18 rounding MEASURED)
    for k in state.tracers:
        np.testing.assert_allclose(np.asarray(out.tracers[k]), np.asarray(ref.tracers[k]), rtol=1e-14, atol=0, err_msg=k)
