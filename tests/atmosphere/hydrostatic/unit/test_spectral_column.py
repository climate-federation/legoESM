"""The spectral PE model as a column model behind the shared column loop's
step contract (composable physics P4): the G-lane ladder of the design
(fv3_duo_gaps/fu/composable_physics_design.md, row P4) at T10 (16 x 32), nlev 5.

(a) rung 1: ``physics_fn=None`` adapter == the native dycore, bitwise, 10 steps;
    a zero physics tendency == the no-physics step, bitwise;
(b) uniform-tendency injection: the post-step increment is the analytic
    per-lane expectation (T through SH, winds through the Bourke operators,
    water mass through p_s per apply_physics_water_mass, exact for a
    uniform increment);
(c) physics identity: the column view the wrapper feeds the physics is the
    synthesis of the post-dynamics state;
(d) restart parity lives in tests/unit/test_column_lane_restart_digest.py;
(e) the acceptance residuals of the design row: column-water projection
    <= 1e-6 relative, pressure-weighted heating <= 0.1 W m-2, the T floor
    count and min synthesised T reported; the dycore's anchored DRY-mass
    fixer (inherited, F13) restores the dry integral under a water sink;
plus the surface-diagnostic slot merge, the no-retrace guarantee under a
changing forcing, the leapfrog refusal (D5), the T write-back as spectral
increments, the driver lane (setup / run / mesh identity / refusals), the
checkpoint's spectral coefficients (planted violation: a file without them
is refused), a pending T edit across the checkpoint, and the production
deck's dt.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.dynamics.gcm.latlon_cgrid_column import (  # noqa: E402
    build_latlon_column_mesh,
)
from legoesm.atmosphere.dynamics.gcm.spectral_column import (  # noqa: E402
    SpectralColumnModel, SpectralColumnState,
)
from legoesm.atmosphere.dynamics.gcm.spectral_pe import (  # noqa: E402
    SpectralPEConfig, SpectralPrimitiveEquationModel,
    isothermal_rest_state_spectral, spectral_dry_mass, spectral_pe_to_grid,
)
from legoesm.core.conservation import (  # noqa: E402
    apply_physics_water_mass, dry_surface_pressure)
from legoesm.core.field import Field  # noqa: E402
from legoesm.core.state import HydrostaticTendencies  # noqa: E402
from legoesm.grids.gaussian import (  # noqa: E402
    create_gaussian_grid, hyperdiff_coeff_for_efold, sh_analysis_3d,
    sh_synthesis, sh_synthesis_3d, vordiv_from_uv_3d, vordiv_from_uv_exact_3d,
)
from legoesm.grids.vertical import create_sigma_coordinate  # noqa: E402

N_MAX, NLEV, DT = 10, 5, 600.0
NAMES = ("q_v", "q_c", "q_r")


@pytest.fixture(autouse=True)
def _drop_compiled_graphs():
    yield
    jax.clear_caches()


@pytest.fixture(scope="module")
def grid():
    return create_gaussian_grid(N_MAX, radius=constants.R_earth, omega=constants.Omega)


@pytest.fixture(scope="module")
def sigma():
    return create_sigma_coordinate(n_levels=NLEV, dtype=jnp.float64)


def _cfg(grid, **over):
    base = dict(hyperdiff_coeff=hyperdiff_coeff_for_efold(grid, 0.5 * 3600.0),
                hyperdiff_order=2, time_integrator="ssp_rk54", p_floor=200.0,
                dealiasing_fraction=0.667, fix_mass=False,
                anchor_mass_to_initial=False)
    base.update(over)
    return SpectralPEConfig(**base)


def _model(grid, sigma, **cfg):
    dyn = SpectralPrimitiveEquationModel(grid, sigma, _cfg(grid, **cfg))
    return dyn, SpectralColumnModel(dyn)


def _native_ic(grid, sigma, seed=0, tracers=True):
    rng = np.random.default_rng(seed)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    lat = np.asarray(grid.lat2d)
    lon = np.asarray(grid.lon2d)
    phis = jnp.asarray(1500.0 * constants.g * np.exp(
        -((lat - 0.5) ** 2 + (lon - 2.5) ** 2) / 0.4))
    tr = None
    if tracers:
        q = 0.01 * rng.uniform(0.3, 1.0, (n_lat, n_lon, NLEV)) * np.asarray(sigma.sigma_full) ** 2
        tr = {"q_v": Field(data=jnp.asarray(q), name="q_v", dims=("n_lat", "n_lon", "nlev"), units="kg/kg"),
              "q_c": Field(data=jnp.zeros((n_lat, n_lon, NLEV)), name="q_c", dims=("n_lat", "n_lon", "nlev"), units="kg/kg"),
              "q_r": Field(data=jnp.asarray(2e-4 * rng.uniform(0.5, 1.0, (n_lat, n_lon, NLEV))), name="q_r",
                           dims=("n_lat", "n_lon", "nlev"), units="kg/kg")}
    st = isothermal_rest_state_spectral(grid, sigma, T_init=280.0, phis=phis,
                                        tracers=tr, seed=seed + 1)
    u = jnp.asarray(8.0 * np.cos(lat)[..., None] * np.ones(NLEV)
                    + rng.normal(0, 0.5, (n_lat, n_lon, NLEV)))
    v = jnp.asarray(rng.normal(0, 0.5, (n_lat, n_lon, NLEV)))
    vor, div = vordiv_from_uv_exact_3d(grid, u, v)
    return st._replace(vor_hat=st.vor_hat.replace(data=vor),
                       div_hat=st.div_hat.replace(data=div))


def _tend(state, du, dv, dT, tracers=None, **extra):
    z = jnp.zeros_like(state.p_s.data)
    f = lambda like, d, nm: like.replace(data=d, name=nm)  # noqa: E731
    return HydrostaticTendencies(
        du_dt=f(state.u, du, "du_dt"), dv_dt=f(state.v, dv, "dv_dt"),
        dT_dt=f(state.T, dT, "dT_dt"), dp_s_dt=f(state.p_s, z, "dp_s"),
        dphis_dt=f(state.phis, z, "dphis"), tracer_tendencies=tracers, **extra)


def _zero_physics(state, mesh, coord, phys_state=None, forcing=None):
    return _tend(state, jnp.zeros_like(state.u.data),
                 jnp.zeros_like(state.v.data), jnp.zeros_like(state.T.data))


def _assert_native_equal(a, b):
    for k in ("vor_hat", "div_hat", "T_hat", "lnps_hat", "phis_hat"):
        np.testing.assert_array_equal(np.asarray(getattr(a, k).data),
                                      np.asarray(getattr(b, k).data), err_msg=k)
    assert (a.tracers is None) == (b.tracers is None)
    if a.tracers is not None:
        assert sorted(a.tracers) == sorted(b.tracers)
        for k in a.tracers:
            np.testing.assert_array_equal(np.asarray(a.tracers[k].data),
                                          np.asarray(b.tracers[k].data), err_msg=k)


def _grid_fields(st, grid, sigma):
    return spectral_pe_to_grid(st, grid, sigma)


def _column_water_load(tracers, p_s, sigma):
    """``W = p_s - dry p_s`` [Pa]: the column water the dry-mass convention
    (``conservation.dry_surface_pressure``) excludes."""
    return p_s - dry_surface_pressure(p_s, tracers, sigma)


# ---------------------------------------------------------------------
# the mesh and the view
# ---------------------------------------------------------------------

def test_mesh_is_the_gaussian_grid_row_major_with_quadrature_areas(grid, sigma):
    _, col = _model(grid, sigma)
    m = col.mesh
    assert m.nCells == grid.n_lat * grid.n_lon and m.grid_shape_2d == (m.nCells,)
    np.testing.assert_array_equal(np.asarray(m.latCell), np.asarray(grid.lat2d).reshape(-1))
    np.testing.assert_array_equal(np.asarray(m.lonCell), np.asarray(grid.lon2d).reshape(-1))
    np.testing.assert_array_equal(np.asarray(m.areaCell), np.asarray(grid.grid_area).reshape(-1))
    # areas are the Gaussian quadrature weights: the sphere to roundoff
    np.testing.assert_allclose(float(np.asarray(m.areaCell).sum()),
                               4 * np.pi * grid.radius ** 2, rtol=1e-12)
    c_lat, c_lon = np.asarray(m.cornerLat), np.asarray(m.cornerLon)
    lat_v = np.asarray(grid.lat_v)
    j = np.repeat(np.arange(grid.n_lat), grid.n_lon)
    np.testing.assert_array_equal(c_lat[:, 0], lat_v[j])
    np.testing.assert_array_equal(c_lat[:, 2], lat_v[j + 1])
    assert np.all(c_lat[:, 0] < np.asarray(m.latCell)) and np.all(np.asarray(m.latCell) < c_lat[:, 2])
    width = (c_lon[:, 1] - c_lon[:, 0]) % (2 * np.pi)
    np.testing.assert_allclose(width, 2 * np.pi / grid.n_lon, rtol=1e-12)
    assert m.radius == grid.radius and m.omega == grid.omega == constants.Omega
    np.testing.assert_allclose(np.asarray(m.grid_coriolis).reshape(grid.n_lat, grid.n_lon),
                               np.asarray(grid.f), rtol=1e-12, atol=1e-18)
    assert col.sigma_coord is sigma and col.grid is grid
    m2 = build_latlon_column_mesh(grid)
    for k in ("latCell", "lonCell", "areaCell", "cornerLat", "cornerLon"):
        np.testing.assert_array_equal(np.asarray(getattr(m2, k)), np.asarray(getattr(m, k)))


def test_column_view_is_the_synthesis_and_T_edits_write_back_as_increments(grid, sigma):
    _, col = _model(grid, sigma)
    nat = _native_ic(grid, sigma)
    st = col.from_bundle(nat)
    assert isinstance(st, SpectralColumnState)
    _assert_native_equal(st.native, nat)
    f = _grid_fields(nat, grid, sigma)
    ncol = col.mesh.nCells
    for k in ("u", "v", "T", "p_s", "phis"):
        exp = np.asarray(f[k]).reshape((ncol,) + np.asarray(f[k]).shape[2:])
        np.testing.assert_array_equal(np.asarray(getattr(st, k).data), exp, err_msg=k)
    assert sorted(st.tracers) == sorted(NAMES)
    np.testing.assert_array_equal(np.asarray(st.tracers["q_v"].data),
                                  np.asarray(nat.tracers["q_v"].data).reshape(ncol, NLEV))
    np.testing.assert_array_equal(np.asarray(st.T_view.data), np.asarray(st.T.data))
    # no edit: the native coefficients come back bitwise (SH(0) == 0)
    back = col.to_bundle(st)
    _assert_native_equal(back, nat)
    # a T edit is written back as the spectral INCREMENT of the edit, tracers
    # replaced on the grid; a re-analysis of the view is NOT what happens
    dT_cols = 0.5 * jnp.sin(st.T.data / 50.0)
    st2 = st._replace(T=st.T.replace(data=st.T.data + dT_cols),
                      tracers={**st.tracers, "q_v": st.tracers["q_v"].replace(
                          data=st.tracers["q_v"].data * 2.0)})
    back2 = col.to_bundle(st2)
    # the increment is the edit over the view, (T + dT) - T in floating point
    edit = (st2.T.data - st.T.data).reshape(grid.n_lat, grid.n_lon, NLEV)
    exp_T_hat = nat.T_hat.data + sh_analysis_3d(grid, edit)
    np.testing.assert_array_equal(np.asarray(back2.T_hat.data), np.asarray(exp_T_hat))
    np.testing.assert_array_equal(np.asarray(back2.tracers["q_v"].data),
                                  2.0 * np.asarray(nat.tracers["q_v"].data))
    np.testing.assert_array_equal(np.asarray(back2.vor_hat.data), np.asarray(nat.vor_hat.data))
    # the state is self-contained: a pytree rebuilt by a JAX transform (no
    # object identity) writes back the same (codex 2026-10-14 r1 #2)
    rebuilt = jax.tree_util.tree_map(lambda x: x, st2)
    _assert_native_equal(SpectralColumnModel(col.dyn).to_bundle(rebuilt), back2)


# ---------------------------------------------------------------------
# G-lane (a): rung 1
# ---------------------------------------------------------------------

def test_rung1_no_physics_is_the_native_dycore_bitwise(grid, sigma):
    dyn, col = _model(grid, sigma, fix_mass=True, anchor_mass_to_initial=True)
    ref_dyn = SpectralPrimitiveEquationModel(
        grid, sigma, _cfg(grid, fix_mass=True, anchor_mass_to_initial=True))
    nat = _native_ic(grid, sigma)
    ref = nat
    st = col.from_bundle(nat)
    for _ in range(10):
        ref = ref_dyn.step(ref, DT)
        st = col.step(st, DT)
    _assert_native_equal(st.native, ref)
    assert np.abs(np.asarray(st.native.T_hat.data) - np.asarray(nat.T_hat.data)).max() > 0.0


def test_zero_tendency_physics_is_the_no_physics_step_bitwise(grid, sigma):
    """Exact-zero increments are exact identities (the operator-split
    test's own invariant), through the wrapper and the water-mass path too
    (the lnps increment is the RATIO form: log(1) == 0)."""
    _, col = _model(grid, sigma, fix_mass=True, anchor_mass_to_initial=True)
    nat = _native_ic(grid, sigma)

    def zero_with_water(state, mesh, coord, phys_state=None, forcing=None):
        return _tend(state, jnp.zeros_like(state.u.data), jnp.zeros_like(state.v.data),
                     jnp.zeros_like(state.T.data),
                     tracers={nm: state.tracers[nm].replace(data=jnp.zeros_like(state.tracers[nm].data))
                              for nm in NAMES})
    a = col.step(col.from_bundle(nat), DT)
    b = col.step(col.from_bundle(nat), DT, physics_fn=_zero_physics)
    c = col.step(col.from_bundle(nat), DT, physics_fn=zero_with_water)
    _assert_native_equal(a.native, b.native)
    _assert_native_equal(a.native, c.native)
    d = col._step_diag
    assert float(d["dry_mass_trunc_rel_max"]) == 0.0 and int(d["T_floor_hits"]) == 0


# ---------------------------------------------------------------------
# G-lane (b): uniform-tendency injection, the analytic per-lane increment
# ---------------------------------------------------------------------

def test_uniform_tendencies_land_as_the_lanes_analytic_increment(grid, sigma):
    # no de-aliasing truncation (a uniform wind's vorticity is not band-limited)
    # and no hyperdiffusion: the increments are the operators' output verbatim
    _, col = _model(grid, sigma, hyperdiff_coeff=0.0, dealiasing_fraction=0.0)
    nat = _native_ic(grid, sigma)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    # dq0 * DT stays above -min(q_v) (~3e-5 at the top level), so the
    # tracer positivity after the increment is inert and the increment
    # analytic
    du0, dv0, dT0, dq0 = 1e-4, -2e-4, 3e-4, -1e-8

    def physics(state, mesh, coord, phys_state=None, forcing=None):
        o3 = jnp.ones_like(state.T.data)
        return _tend(state, du0 * o3, dv0 * o3, dT0 * o3,
                     tracers={"q_v": state.T.replace(data=dq0 * o3, name="dq_v")})
    ref = col.step(col.from_bundle(nat), DT)
    out = col.step(col.from_bundle(nat), DT, physics_fn=physics)
    fr, fo = _grid_fields(ref.native, grid, sigma), _grid_fields(out.native, grid, sigma)
    # T: a uniform increment is the n=0 mode alone, retained exactly
    np.testing.assert_allclose(np.asarray(fo["T"] - fr["T"]), DT * dT0, rtol=1e-11, atol=1e-11)
    # winds: the Bourke operators of the lane's own vordiv_from_uv_3d on the
    # uniform (du, dv): the coefficient increments are that call (eager here,
    # inside the step's jit there: fused-multiply roundoff, 1e-22 absolute)
    dvor, ddiv = vordiv_from_uv_3d(grid, jnp.full((n_lat, n_lon, NLEV), DT * du0),
                                   jnp.full((n_lat, n_lon, NLEV), DT * dv0))
    np.testing.assert_allclose(np.asarray(out.native.vor_hat.data - ref.native.vor_hat.data), np.asarray(dvor),
                               rtol=1e-12, atol=1e-20)
    np.testing.assert_allclose(np.asarray(out.native.div_hat.data - ref.native.div_hat.data), np.asarray(ddiv),
                               rtol=1e-12, atol=1e-20)
    # water: the mass the vapour tendency removes leaves through p_s per the
    # shared routine on the post-dynamics layer masses; with a uniform q
    # tendency the ln p_s increment is uniform, hence exactly representable
    dq = {"q_v": jnp.full((n_lat, n_lon, NLEV), dq0)}
    q_exp, p_s_exp = apply_physics_water_mass(ref.native.tracers, dq, fr["p_s"], sigma, DT)
    np.testing.assert_allclose(np.asarray(fo["p_s"]), np.asarray(p_s_exp), rtol=1e-12, atol=1e-7)
    # inside the step's jit vs eager here: fused roundoff (the p_s line's rtol)
    np.testing.assert_allclose(np.asarray(out.native.tracers["q_v"].data),
                               np.asarray(q_exp["q_v"].data), rtol=1e-12, atol=0.0)
    assert np.abs(np.asarray(fo["p_s"] - fr["p_s"])).max() > 0.0
    assert float(col._step_diag["dry_mass_trunc_rel_max"]) < 1e-12
    # and the columns the driver sees are the view of that native state
    np.testing.assert_array_equal(np.asarray(out.T.data).reshape(n_lat, n_lon, NLEV), np.asarray(fo["T"]))
    assert out.T.data.shape == (n_lat * n_lon, NLEV) and out.p_s.data.shape == (n_lat * n_lon,)


# ---------------------------------------------------------------------
# G-lane (c): the physics identity through the wrapper
# ---------------------------------------------------------------------

def test_physics_sees_the_column_view_of_the_post_dynamics_state(grid, sigma):
    _, col = _model(grid, sigma, hyperdiff_coeff=0.0)
    nat = _native_ic(grid, sigma)
    ref = col.step(col.from_bundle(nat), DT)
    seen = {}
    forcing = {"sst": jnp.full((col.mesh.nCells,), 300.0)}

    def physics(state, mesh, coord, phys_state=None, forcing=None):
        seen["state"], seen["mesh"], seen["coord"], seen["forcing"] = state, mesh, coord, forcing
        return _zero_physics(state, mesh, coord)
    col.step(col.from_bundle(nat), DT, physics_fn=physics, forcing=forcing)
    assert seen["mesh"] is col.mesh and seen["coord"] is col.sigma_coord
    st = seen["state"]
    ncol = col.mesh.nCells
    assert st.T.shape == (ncol, NLEV) and st.p_s.shape == (ncol,) and st.v is not None
    assert sorted(st.tracers) == sorted(NAMES)

    def echo(state, mesh, coord, phys_state=None, forcing=None):
        return _tend(state, jnp.zeros_like(state.u.data), jnp.zeros_like(state.v.data),
                     state.T.data, sw_net_sfc=forcing["sst"])
    out = col.step(col.from_bundle(nat), DT, physics_fn=echo, forcing=forcing)
    fr, fo = _grid_fields(ref.native, grid, sigma), _grid_fields(out.native, grid, sigma)
    # the echoed T tendency lands as dt*T_view: the view is the synthesis of
    # the post-dynamics state (band-limited, so SH then synthesis is identity
    # to roundoff)
    np.testing.assert_allclose(np.asarray(fo["T"] - fr["T"]) / DT, np.asarray(fr["T"]),
                               rtol=1e-10, atol=1e-8)
    np.testing.assert_array_equal(np.asarray(col._sfc_diag[0]), np.asarray(forcing["sst"]))


def test_surface_diagnostics_merge_slot_wise_and_carry_rides(grid, sigma):
    _, col = _model(grid, sigma)
    nat = _native_ic(grid, sigma)
    ncol = col.mesh.nCells

    def _physics(rad, precip):
        def fn(state, mesh, coord, phys_state=None, forcing=None):
            extra = {"precip": jnp.full((ncol,), precip)}
            if rad is not None:
                extra["sw_net_sfc"] = jnp.full((ncol,), rad)
                extra["lw_net_sfc"] = jnp.full((ncol,), -rad)
            return (_tend(state, jnp.zeros_like(state.u.data), jnp.zeros_like(state.v.data),
                          jnp.zeros_like(state.T.data), **extra),
                    None if phys_state is None else phys_state + 1)
        return fn
    st = col.step(col.from_bundle(nat), DT, physics_fn=_physics(100.0, 1.0),
                  phys_state=jnp.asarray(0))
    assert float(col._sfc_diag[0][0]) == 100.0 and float(col._sfc_diag[2][0]) == 1.0
    assert int(col._phys_state) == 1
    col.step(st, DT, physics_fn=_physics(None, 2.0), phys_state=col._phys_state)
    assert float(col._sfc_diag[0][0]) == 100.0 and float(col._sfc_diag[1][0]) == -100.0
    assert float(col._sfc_diag[2][0]) == 2.0 and int(col._phys_state) == 2


def test_no_retrace_after_warm_up_under_a_changing_forcing(grid, sigma):
    _, col = _model(grid, sigma)
    nat = _native_ic(grid, sigma)

    def physics(state, mesh, coord, phys_state=None, forcing=None):
        return _tend(state, jnp.zeros_like(state.u.data), jnp.zeros_like(state.v.data),
                     1e-6 * forcing["sst"][:, None] * jnp.ones_like(state.T.data))
    traces = []
    adv = col.dyn._advance_dynamics

    def counting(*a, **k):
        traces.append(1)
        return adv(*a, **k)
    col.dyn._advance_dynamics = counting
    st = col.from_bundle(nat)
    st = col.step(st, DT, physics_fn=physics, forcing={"sst": jnp.full((col.mesh.nCells,), 1.0)})
    steps = lambda: list(col._fns)  # noqa: E731
    assert len(traces) == 1 and steps() == [(physics, DT)]
    for k in range(2, 6):
        st = col.step(st, DT, physics_fn=physics,
                      forcing={"sst": jnp.full((col.mesh.nCells,), float(k))})
    assert len(traces) == 1 and steps() == [(physics, DT)]
    T1 = np.asarray(col.step(st, DT, physics_fn=physics,
                             forcing={"sst": jnp.full((col.mesh.nCells,), 1.0)}).native.T_hat.data)
    T9 = np.asarray(col.step(st, DT, physics_fn=physics,
                             forcing={"sst": jnp.full((col.mesh.nCells,), 9.0)}).native.T_hat.data)
    assert np.abs(T9 - T1).max() > 0.0 and len(traces) == 1


def test_refusals_leapfrog_and_missing_meridional_tendency(grid, sigma):
    _, col = _model(grid, sigma)
    nat = _native_ic(grid, sigma)

    def no_v(state, mesh, coord, phys_state=None, forcing=None):
        return _zero_physics(state, mesh, coord)._replace(dv_dt=None)
    with pytest.raises(ValueError, match="dv_dt=None"):
        col.step(col.from_bundle(nat), DT, physics_fn=no_v)
    lf = SpectralPrimitiveEquationModel(
        grid, sigma, _cfg(grid, time_integrator="leapfrog_si", semi_implicit=True,
                          implicit_hyperdiff=True))
    col_lf = SpectralColumnModel(lf)
    with pytest.raises(ValueError, match="ssp_rk54 only"):
        col_lf.step(col_lf.from_bundle(nat), DT, physics_fn=_zero_physics)


# ---------------------------------------------------------------------
# G-lane (e): the design row's acceptance -- projection residuals, floors,
# the DRY-mass fixer
# ---------------------------------------------------------------------

def _band_limited(grid, rng, nlev, n_keep=4, amp=1.0):
    """A smooth grid field: a synthesis of random coefficients with n <= n_keep."""
    ls = np.asarray(grid.ls)
    c = (rng.normal(size=(grid.n_sh, nlev)) + 1j * rng.normal(size=(grid.n_sh, nlev)))
    c[np.asarray(grid.ms) == 0] = c[np.asarray(grid.ms) == 0].real
    c = jnp.asarray(c * (ls <= n_keep)[:, None] * amp)
    return sh_synthesis_3d(grid, c)


def test_projection_residuals_meet_the_acceptance(grid, sigma):
    """Column water: the physics' water change lands in the tracers and in
    p_s to <= 1e-6 relative PER COLUMN (L1 over columns) with the
    truncation, the positivity borrow and the dry-mass anchor on; heating:
    the pressure-weighted column heating lands in T to <= 0.1 W m-2 global
    mean; the T floor count and min synthesised T are reported.  The
    dycore's hyperdiffusion is OFF here: it is applied to the tracers too
    (the native lane's tracer filter), which moves water between columns
    (global mass restored, F21) -- the second block measures that config
    and asserts the GLOBAL residual only (measured 2026-10-14: per-column
    L1 8e-4 from the filter, global 5e-11)."""
    _, col = _model(grid, sigma, fix_mass=True, anchor_mass_to_initial=True,
                    hyperdiff_coeff=0.0)
    nat = _native_ic(grid, sigma)
    rng = np.random.default_rng(3)
    # FULL-spectrum (white) tendencies: the truncation of their lnps
    # increment is what the water budget must survive (codex r1 #1: a
    # band-limited tendency passed while a degree-8 one leaked water)
    dT_g = 1e-4 * rng.normal(size=(grid.n_lat, grid.n_lon, NLEV))     # K/s
    dq_g = 1e-8 * rng.normal(size=(grid.n_lat, grid.n_lon, NLEV))     # 1/s
    n_lat, n_lon = grid.n_lat, grid.n_lon
    cols = lambda a: jnp.reshape(a, (n_lat * n_lon,) + a.shape[2:])  # noqa: E731

    def physics(state, mesh, coord, phys_state=None, forcing=None):
        return _tend(state, jnp.zeros_like(state.u.data), jnp.zeros_like(state.v.data),
                     cols(dT_g), tracers={"q_v": state.T.replace(data=cols(dq_g), name="dq_v")})
    ref = col.step(col.from_bundle(nat), DT)
    out = col.step(col.from_bundle(nat), DT, physics_fn=physics)
    fr, fo = _grid_fields(ref.native, grid, sigma), _grid_fields(out.native, grid, sigma)
    area = np.asarray(grid.grid_area, dtype=np.float64)
    # column water [kg m-2]: predicted from the post-dynamics layer masses
    dp = np.asarray(sigma.layer_thickness_dp(fr["p_s"]))
    dW_pred = (DT * np.asarray(dq_g) * dp).sum(-1) / constants.g
    W_ref = np.asarray(_column_water_load(ref.native.tracers, fr["p_s"], sigma)) / constants.g
    W_out = np.asarray(_column_water_load(out.native.tracers, fo["p_s"], sigma)) / constants.g
    water_resid = abs(((W_out - W_ref) - dW_pred) * area).sum() / (W_ref * area).sum()
    # heating [W m-2]: c_p dT dp / g, pressure-weighted on the post-dynamics
    # layer masses; realised from the synthesised T change over the no-physics step
    H_pred = (constants.c_pd * np.asarray(dT_g) * dp).sum(-1) / constants.g
    H_real = (constants.c_pd * np.asarray(fo["T"] - fr["T"]) / DT * dp).sum(-1) / constants.g
    heat_resid = abs(((H_real - H_pred) * area).sum() / area.sum())
    d = {k: float(v) for k, v in col._step_diag.items()}
    print(f"\nP4 acceptance: column-water residual {water_resid:.3e} (<=1e-6), "
          f"heating residual {heat_resid:.3e} W/m2 (<=0.1), min T {d['min_T']:.2f} K, "
          f"T-floor hits {int(d['T_floor_hits'])}, dry-mass truncation rel "
          f"max {d['dry_mass_trunc_rel_max']:.3e} mean {d['dry_mass_trunc_rel_mean']:.3e} (<=1e-6)")
    assert water_resid <= 1e-6
    assert heat_resid <= 0.1
    assert d["dry_mass_trunc_rel_max"] <= 1e-5   # white-noise increment, 2/3 truncation
    assert int(d["T_floor_hits"]) == 0 and d["min_T"] > col.config.T_min
    # hyperdiffusion on (the production configuration): the tracer filter
    # redistributes the increment between columns, the global budget holds
    _, col_h = _model(grid, sigma, fix_mass=True, anchor_mass_to_initial=True)
    ref_h = col_h.step(col_h.from_bundle(nat), DT)
    out_h = col_h.step(col_h.from_bundle(nat), DT, physics_fn=physics)
    frh, foh = _grid_fields(ref_h.native, grid, sigma), _grid_fields(out_h.native, grid, sigma)
    dph = np.asarray(sigma.layer_thickness_dp(frh["p_s"]))
    W_refh = np.asarray(_column_water_load(ref_h.native.tracers, frh["p_s"], sigma)) / constants.g
    W_outh = np.asarray(_column_water_load(out_h.native.tracers, foh["p_s"], sigma)) / constants.g
    dWh = (DT * np.asarray(dq_g) * dph).sum(-1) / constants.g
    global_resid = abs((((W_outh - W_refh) - dWh) * area).sum()) / (W_refh * area).sum()
    print(f"P4 acceptance, hyperdiffusion on: global column-water residual {global_resid:.3e} (<=1e-9)")
    assert global_resid <= 1e-9
    # a LOCALIZED water sink (exact-zero tendency elsewhere, non-uniform
    # background tracers): the projected ln p_s increment spreads beyond the
    # sink column, so every column whose layer masses moved is re-weighted;
    # each tracer's global mass changes by exactly what the physics removed
    # (passengers: not at all).  Filters and the anchor off: the projection
    # alone (codex 2026-10-27 P1).
    _, col_l = _model(grid, sigma, hyperdiff_coeff=0.0)
    sink = np.zeros((n_lat, n_lon), bool)
    sink[n_lat // 2, n_lon // 3] = True

    def local_sink(state, mesh, coord, phys_state=None, forcing=None):
        qv = state.tracers["q_v"].data
        dq = jnp.where(jnp.asarray(sink.reshape(-1))[:, None], -0.3 * qv / DT, 0.0)
        return _tend(state, jnp.zeros_like(state.u.data), jnp.zeros_like(state.v.data),
                     jnp.zeros_like(state.T.data),
                     tracers={"q_v": state.tracers["q_v"].replace(data=dq, name="dq_v")})

    def tracer_masses(st):
        f = _grid_fields(st, grid, sigma)
        dp_ = np.asarray(sigma.layer_thickness_dp(f["p_s"]))
        return {k: float(((np.asarray(t.data) * dp_).sum(-1) * area).sum() / constants.g)
                for k, t in st.tracers.items()}
    ref_l = col_l.step(col_l.from_bundle(nat), DT)
    out_l = col_l.step(col_l.from_bundle(nat), DT, physics_fn=local_sink)
    m_ref, m_out = tracer_masses(ref_l.native), tracer_masses(out_l.native)
    fr_l = _grid_fields(ref_l.native, grid, sigma)
    dp_l = np.asarray(sigma.layer_thickness_dp(fr_l["p_s"]))
    removed = float(((0.3 * np.asarray(ref_l.native.tracers["q_v"].data) * dp_l).sum(-1)
                     * area * sink).sum() / constants.g)
    assert removed > 0.0
    np.testing.assert_allclose(m_out["q_v"], m_ref["q_v"] - removed, rtol=1e-12)
    np.testing.assert_allclose(m_out["q_r"], m_ref["q_r"], rtol=1e-12)
    # the floor count is live: a tendency that plunges T reports hits
    def plunge(state, mesh, coord, phys_state=None, forcing=None):
        return _tend(state, jnp.zeros_like(state.u.data), jnp.zeros_like(state.v.data),
                     jnp.full_like(state.T.data, -300.0 / DT))
    col.step(col.from_bundle(nat), DT, physics_fn=plunge)
    assert int(col._step_diag["T_floor_hits"]) > 0


def test_anchored_fixer_keeps_the_dry_integral_under_a_water_sink(grid, sigma):
    """A physics water sink through the split step: the dycore's anchored
    DRY-mass fixer (F13, inherited by the lane) keeps the global dry mass and
    lets the total p_s mass fall by the water removed -- the removed water
    does not come back as dry air."""
    nat = _native_ic(grid, sigma)

    def sink(state, mesh, coord, phys_state=None, forcing=None):
        return _tend(state, jnp.zeros_like(state.u.data), jnp.zeros_like(state.v.data),
                     jnp.zeros_like(state.T.data),
                     tracers={"q_v": state.tracers["q_v"].replace(
                         data=-0.2 * state.tracers["q_v"].data / DT, name="dq_v")})
    area = np.asarray(grid.grid_area, dtype=np.float64)

    def masses(st):
        f = _grid_fields(st, grid, sigma)
        W = np.asarray(_column_water_load(st.tracers, f["p_s"], sigma))
        p = np.asarray(f["p_s"])
        return (p * area).sum(), ((p - W) * area).sum(), (W * area).sum()
    _, col = _model(grid, sigma, fix_mass=True, anchor_mass_to_initial=True)
    tot0, dry0, W0 = masses(nat)
    out = col.step(col.from_bundle(nat), DT, physics_fn=sink)
    tot1, dry1, W1 = masses(out.native)
    assert W1 < W0 * 0.9
    assert abs(dry1 - dry0) / dry0 < 1e-12, (dry1 - dry0) / dry0
    np.testing.assert_allclose(tot1 - tot0, W1 - W0, rtol=1e-9)
    assert float(col.dyn._target_mass) == pytest.approx(
        float(spectral_dry_mass(grid, sigma, nat)))
    # the anchor is the dycore's, not the lane's: the fixer off, the sink
    # leaves the dry integral to the dynamics' own drift (non-zero)
    _, col_off = _model(grid, sigma)
    out_off = col_off.step(col_off.from_bundle(nat), DT, physics_fn=sink)
    assert abs(masses(out_off.native)[1] - dry0) / dry0 > 1e-12


# ---------------------------------------------------------------------
# the driver lane
# ---------------------------------------------------------------------

def _driver_cfg(tmp_path, **over):
    from legoesm.driver.config import (
        DycoreConfig, ExperimentConfig, GridConfig, OutputConfig)
    grid = GridConfig(grid_type="gaussian", resolution=N_MAX, nlev=NLEV)
    dycore = DycoreConfig(
        model_type="hydrostatic", discretization="spectral",
        dt=over.pop("dt", DT), time_integrator=over.pop("integrator", "ssp_rk54"),
        column_lane=over.pop("column", True))
    base = dict(
        grid=grid, dycore=dycore, days=over.pop("days", 3 * DT / 86400.0),
        radiation="none", convection="none", microphysics="none",
        turbulence="none", gravity_wave_drag="none", precision="fp64",
        output=OutputConfig(diag_days=over.pop("diag_days", 0),
                            checkpoint_days=over.pop("checkpoint_days", 0),
                            output_dir=str(tmp_path)))
    base.update(over)
    return ExperimentConfig(**base)


def _run_driver(tmp_path, **over):
    from legoesm.driver.model_driver import ModelDriver
    cfg = _driver_cfg(tmp_path, **over)
    drv = ModelDriver(cfg, output_dir=tmp_path)
    drv.setup()
    status = drv.run()
    assert status == "COMPLETED", status
    return drv


def test_driver_column_lane_is_the_native_grids_mesh_and_runs(tmp_path):
    drv = _run_driver(tmp_path)
    assert isinstance(drv.model, SpectralColumnModel)
    assert isinstance(drv.state, SpectralColumnState)
    assert drv.grid is drv.model.mesh and drv.sigma is drv.model.sigma_coord
    assert drv.grid.nCells == drv.model.grid.n_lat * drv.model.grid.n_lon
    assert np.asarray(drv._phis_data).shape == (drv.grid.nCells,)
    assert np.asarray(drv._f_land).shape == (drv.grid.nCells,)
    assert drv.model.grid.n_max == N_MAX
    assert np.isfinite(np.asarray(drv.state.native.T_hat.data)).all()
    # a dry deck carries no tracers (the lat-lon / MPAS lanes' rule)
    assert drv.state.tracers is None
    assert np.abs(np.asarray(drv.state.native.T_hat.data)
                  - np.asarray(drv._state_template.native.T_hat.data)).max() > 0.0


def test_driver_column_lane_refusals(tmp_path):
    from legoesm.driver.model_driver import ModelDriver
    for over, frag in ((dict(sponge_enabled=True), "sponge_enabled"),
                       (dict(mpas_qv_smooth_del4_m4s=1e14), "Voronoi mesh"),
                       (dict(distributed=True), "single device"),
                       (dict(integrator="leapfrog_si"), "ssp_rk54 only")):
        drv = ModelDriver(_driver_cfg(tmp_path, **over), output_dir=tmp_path)
        with pytest.raises((ValueError, NotImplementedError), match=frag):
            drv.setup()


def test_checkpoint_carries_the_coefficients_and_refuses_a_file_without_them(tmp_path):
    """The column view is a synthesis: the checkpoint carries the spectral
    coefficients and the mass anchor, a fresh driver restores the native
    state bitwise, and a file stripped of them (an MPAS-lane checkpoint) is
    refused -- the planted violation of this test."""
    from legoesm.driver.model_driver import ModelDriver
    coeffs = ModelDriver._SPECTRALCOL_COEFFS
    a = tmp_path / "a"
    a.mkdir()
    drv = _run_driver(a, days=1.0, checkpoint_days=1)
    ck = a / "checkpoint_day_0001.npz"
    assert ck.is_file(), sorted(p.name for p in a.iterdir())
    with np.load(ck) as d:
        assert str(d["spectralcol_schema"]) == "spectralcol_ckpt_v1"
        for k in coeffs:
            assert d[f"spectralcol_{k}"].dtype == np.complex128
        np.testing.assert_array_equal(d["spectralcol_T_hat"],
                                      np.asarray(drv.state.native.T_hat.data))
        assert float(d["spectralcol_target_mass"]) == float(drv.model.dyn._target_mass)
        files = {k: d[k] for k in d.files}
    b = tmp_path / "b"
    b.mkdir()
    drv_b = ModelDriver(_driver_cfg(b, days=1.0), output_dir=b)
    drv_b.setup()
    step, day = drv_b.load_checkpoint(ck)
    assert step == 144
    _assert_native_equal(drv_b.state.native, drv.state.native)
    np.testing.assert_array_equal(np.asarray(drv_b.state.T.data), np.asarray(drv.state.T.data))
    assert float(drv_b.model.dyn._target_mass) == float(drv.model.dyn._target_mass)
    # planted violations: the same file without the coefficients, and the
    # same file without the mass anchor (an anchored run must restore the
    # ORIGINAL target, never re-snapshot the loaded mass)
    for tag, drop, frag in (
            ("coef", {"spectralcol_schema"} | {f"spectralcol_{k}" for k in coeffs},
             "no spectral coefficients"),
            ("anchor", {"spectralcol_target_mass"}, "spectralcol_target_mass")):
        bad = a / f"bad_{tag}.npz"  # beside the writing run's manifest"
        np.savez(bad, **{k: v for k, v in files.items() if k not in drop})
        c = tmp_path / f"c_{tag}"
        c.mkdir()
        drv_c = ModelDriver(_driver_cfg(c, days=1.0), output_dir=c)
        drv_c.setup()
        assert drv_c.model.config.anchor_mass_to_initial
        with pytest.raises(ValueError, match=frag):
            drv_c.load_checkpoint(bad)
    wide = a / "wide.npz"
    np.savez(wide, **{**files, "spectralcol_tracer_names": np.asarray(["q_i"]),
                      "spectralcol_q_0": np.zeros((4, 4))})
    dd = tmp_path / "d"
    dd.mkdir()
    drv_d = ModelDriver(_driver_cfg(dd, days=1.0), output_dir=dd)
    drv_d.setup()
    with pytest.raises(ValueError, match="tracer"):
        drv_d.load_checkpoint(wide)


def test_checkpoint_round_trip_keeps_a_pending_T_edit_bitwise(tmp_path):
    """A post-step column T edit the driver made (not yet stepped): the
    checkpoint's coefficients carry it written back and the saved T cells
    ARE the edited view, so the resumed state is the continuous run's
    (synthesising the cells instead would drop the unresolved part of the
    edit) and the two continue identically."""
    from legoesm.driver.model_driver import ModelDriver
    a = tmp_path / "a"
    a.mkdir()
    drv = _run_driver(a)
    rng = np.random.default_rng(5)
    st = drv.state
    drv.state = st._replace(T=st.T.replace(
        data=st.T.data + jnp.asarray(rng.normal(0, 0.5, st.T.data.shape))))
    drv.save_checkpoint(3, 1.0)
    ck = a / "checkpoint_day_0001.npz"
    b = tmp_path / "b"
    b.mkdir()
    drv_b = ModelDriver(_driver_cfg(b), output_dir=b)
    drv_b.setup()
    drv_b.load_checkpoint(ck)
    res, edited = drv_b.state, drv.state
    np.testing.assert_array_equal(np.asarray(res.T.data), np.asarray(edited.T.data))
    np.testing.assert_array_equal(np.asarray(res.T_view.data), np.asarray(res.T.data))
    _assert_native_equal(drv_b.model.to_bundle(res), drv.model.to_bundle(edited))
    _assert_native_equal(drv_b.model.step(res, DT).native,
                         drv.model.step(edited, DT).native)


def test_energy_fixer_returns_the_dynamics_loss_as_uniform_heating(grid, sigma):
    """energy_fixer=True: one split step with zero physics conserves the
    global total energy of the synthesised state (enthalpy + KE + phis p_s)
    to roundoff; off, the hyperdiffusion / truncation sink shows (the
    planted violation)."""
    from legoesm.core.conservation import dycore_column_energy
    nat = _native_ic(grid, sigma, tracers=False)
    area = np.asarray(grid.grid_area, dtype=np.float64)

    def energy(st):
        f = _grid_fields(st, grid, sigma)
        ph = np.asarray(sigma.pressure_at_half(f["p_s"]))
        dp = ph[..., 1:] - ph[..., :-1]
        e = dycore_column_energy(np.asarray(f["T"]),
                                 0.5 * (np.asarray(f["u"]) ** 2 + np.asarray(f["v"]) ** 2),
                                 dp, np.asarray(f["phis"]), np.asarray(f["p_s"]))
        return float((np.asarray(e) * area).sum())
    e0 = energy(nat)
    out = {}
    for on in (True, False):
        _, col = _model(grid, sigma, energy_fixer=on)
        out[on] = col.step(col.from_bundle(nat), DT, physics_fn=_zero_physics).native
    assert abs(energy(out[True]) - e0) / abs(e0) < 1e-11
    assert abs(energy(out[False]) - e0) / abs(e0) > 1e-9
    # the fix is a uniform T shift: every coefficient but n = 0 agrees
    dT_hat = np.asarray(out[True].T_hat.data - out[False].T_hat.data)
    np.testing.assert_array_equal(dT_hat[1:], 0.0)
    assert np.all(dT_hat[0] != 0.0)


# ---------------------------------------------------------------------
# the land / radiation deck of the restart-digest gate (G-lane d)
# ---------------------------------------------------------------------

def _spectral_land_stress_cfg(tmp_path, monkeypatch, *, stress, days, **over):
    """The duo digest test's land-stress deck on the spectral column lane:
    RRTMGP, Louis turbulence, the multilayer land with its fluxes and
    stress handed to the boundary layer; land north of the equator only
    (loaders patched: no network, synthetic surface map)."""
    from tests.atmosphere.hydrostatic.unit.test_fv3_duo_column import (
        _fake_surface_map_by_latitude)
    import legoesm.grids.topography as topo
    import legoesm.land.clm_surface_map as clm
    monkeypatch.setattr(clm, "download_clm_surfdata", lambda *a, **k: "synthetic")
    monkeypatch.setattr(clm, "load_clm_surface", _fake_surface_map_by_latitude)
    monkeypatch.setattr(
        topo, "load_land_fraction",
        lambda grid, path, *a, **k: jnp.where(
            jnp.asarray(grid.grid_lat) > 0.0, 0.6, 0.0))
    return _driver_cfg(
        tmp_path, days=days, radiation="rrtmgp", turbulence="louis",
        land_mask_path="synthetic.nc", use_multilayer_land=True,
        multilayer_n_layers=6, multilayer_soil_depth=2.5,
        land_surface_scheme="simple_seb", land_params_refresh=False,
        land_beta_soil=True, land_stress_from_land=stress, **over)


def test_production_deck_dt_is_the_cfl_rule_on_the_true_grid_spacing():
    """The spectral production deck's dt is the decks' rule (0.82 x the
    generic CFL limit 0.8 dx / (390 m/s sqrt 2), rounded down to a divisor
    of 1800 s) on the T31 quadratic grid's TRUE equatorial spacing (the
    lat-lon-48 deck's 48 x 96 columns), it survives the driver's clamp, and
    dt x the cadence counts are CAM6's 1800 s / 3600 s.  FINDING (a separate
    issue, user 2026-10-27): the
    driver's Gaussian dx estimator assumes the LINEAR grid n_lon = 2(n_max+1)
    = 64, 1.5x coarser than the quadratic grid it runs on, so its clamp (906
    s) is loose; a 900 s request would pass it unclamped."""
    import yaml
    from pathlib import Path
    from legoesm.core.cfl import cfl_check_and_adjust, estimate_min_dx_gaussian
    deck = Path(__file__).resolve().parents[4] / "config" / "amip" / "amip_production_spectral_t31.yaml"
    d = yaml.safe_load(deck.read_text())
    dt, n_max = float(d["dt"]), int(d["resolution"])
    assert d["column_lane"] is True and d["discretization"] == "spectral"
    assert d["time_integrator"] == "ssp_rk54" and d["grid_type"] == "gaussian"
    assert dt * d["physics_update_steps"] == 1800.0
    assert dt * d["rad_update_steps"] == 3600.0
    g = create_gaussian_grid(n_max)
    assert (g.n_lat, g.n_lon) == (48, 96)
    dx_true = 2 * np.pi * g.radius / g.n_lon
    limit_true = 0.8 * dx_true / ((50.0 + 340.0) * np.sqrt(2.0))
    assert 600.0 < limit_true < 610.0
    assert dt == max(x for x in (1800.0 / k for k in range(1, 40)) if x <= 0.82 * limit_true)
    assert cfl_check_and_adjust(dt, n_max, model_type="primitive_eq", grid_type="gaussian",
                                verbose=False) == dt
    # the finding, pinned so a fixed estimator turns this line red
    assert estimate_min_dx_gaussian(n_max) == pytest.approx(1.5 * dx_true)
    assert cfl_check_and_adjust(900.0, n_max, model_type="primitive_eq", grid_type="gaussian",
                                verbose=False) == 900.0
