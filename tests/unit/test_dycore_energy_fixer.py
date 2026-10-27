"""CAM-style dycore total-energy fixer (review 2026-10-10 F15).

The KE the dycore damping removes (MPAS del2 sponge / del4 / div4, the duo's
d_con=0 damping) and any numerical dissipation used to leave the total energy
for good.  ``DycoreConfig.energy_fixer`` returns the dynamics window's energy
change as a uniform heating (``core.conservation.energy_fixer_dT``), the
analogue of CAM's ``check_energy_fix``.

Pinned here (x64): the global energy ``sum((c_pd*T + KE) dp/g + phis*p_s/g)``
is unchanged by a step with the fixer on (to round-off) and is NOT unchanged
with it off (non-vacuity), on the Voronoi (MPAS) lane, the cubed-sphere
(FV3-duo column) lane, the lat-lon C-grid lane and the cube C-D grid lane
(F15b); the fixer is differentiable; the driver factory hands the
``DycoreConfig`` switch to every wired core; and every lane that does not run
it refuses ``energy_fixer=True``.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.core.conservation import (  # noqa: E402
    dycore_column_energy,
    energy_fixer_dT,
)
from legoesm.core.operators_voronoi import kinetic_energy_cell_3d  # noqa: E402


@pytest.fixture(autouse=True)
def _drop_compiled_graphs():
    yield
    jax.clear_caches()


def test_dT_restores_the_energy_exactly():
    """E(after + dT) == E(before) for arbitrary columns (dp independent of T)."""
    rng = np.random.default_rng(1)
    ncol, nlev = 40, 12
    area = jnp.asarray(rng.random(ncol) + 0.5)
    dp = jnp.asarray(rng.random((ncol, nlev)) * 4000.0 + 500.0)
    p_s = jnp.sum(dp, axis=-1) + 200.0
    phis = jnp.asarray(rng.random(ncol) * 2e4)

    def E(T, ke):
        return float(jnp.sum(dycore_column_energy(T, ke, dp, phis, p_s) * area))

    T0 = jnp.asarray(250.0 + 20.0 * rng.random((ncol, nlev)))
    ke0 = jnp.asarray(200.0 * rng.random((ncol, nlev)))
    T1, ke1 = T0 - 0.01, ke0 * 0.97            # a lossy "dynamics window"
    mass = float(jnp.sum(jnp.sum(dp, -1) / constants.g * area))
    dT = energy_fixer_dT(E(T0, ke0), E(T1, ke1), mass)
    assert float(dT) > 0.0                     # a loss is returned as warming
    np.testing.assert_allclose(E(T1 + dT, ke1), E(T0, ke0), rtol=1e-14)


# ---------------------------------------------------------------------------
# MPAS (Voronoi) lane
# ---------------------------------------------------------------------------

NLEV, DT = 8, 600.0


@pytest.fixture(scope="module")
def mpas():
    """fp64 state under the fp64 policy: the production deck's precision.  In
    fp32 storage the per-step increment (~1e-5 K) is resolved only to T's ulp
    (1.5e-5 K at 250 K), like every other small increment there."""
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    mesh = create_voronoi_mesh(subdivision_level=2, lloyd_iterations=5)
    sigma = create_sigma_coordinate(NLEV)
    s0 = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64)
        if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jnp.floating) else x,
        baroclinic_wave_init_mpas(mesh, sigma, perturbed=True))
    assert s0.T.data.dtype == jnp.float64
    yield mesh, sigma, s0
    set_policy(prev)


def _mpas_model(mesh, sigma, fixer, fix_mass=False):
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig, MPASPrimitiveEquationModel)
    # damping ON (the sink the fixer closes); no mass fixer, so the step's
    # only energy change is the dynamics window's
    cfg = MPASPrimitiveEquationConfig(
        nu_del2=2.0e5, nu_del4=1.0e19, nu_div4=1.0e19,
        sponge_del2_top_layers=2, sponge_del2_top_factor=8.0,
        fix_mass=fix_mass, energy_fixer=fixer)
    return MPASPrimitiveEquationModel(mesh, sigma, cfg)


def _mpas_energy(state, mesh, sigma):
    ph = sigma.pressure_at_half(state.p_s.data)
    e = dycore_column_energy(state.T.data,
                             kinetic_energy_cell_3d(state.u.data, mesh),
                             ph[..., 1:] - ph[..., :-1], state.phis.data,
                             state.p_s.data)
    return float(jnp.sum(e * mesh.areaCell))


def test_mpas_step_conserves_total_energy_with_the_fixer(mpas):
    mesh, sigma, s0 = mpas
    e0 = _mpas_energy(s0, mesh, sigma)
    off = _mpas_model(mesh, sigma, False).step(s0, DT)
    on = _mpas_model(mesh, sigma, True).step(s0, DT)
    rel_off = (_mpas_energy(off, mesh, sigma) - e0) / e0
    rel_on = (_mpas_energy(on, mesh, sigma) - e0) / e0
    assert on.T.data.dtype == jnp.float64, on.T.data.dtype
    # non-vacuous: the damped dynamics LOSES energy without the fixer...
    assert rel_off < -1e-10, rel_off
    # ...and the fixer returns it to round-off
    assert abs(rel_on) < 1e-13, rel_on
    # the fix is a uniform warming, nothing else moves
    dT = np.asarray(on.T.data - off.T.data)
    assert np.all(dT > 0.0) and np.ptp(dT) < 1e-12   # K: T round-off only
    np.testing.assert_array_equal(np.asarray(on.u.data), np.asarray(off.u.data))


def test_mpas_mass_fixer_is_energy_neutral_with_the_fixer(mpas):
    """The dry-mass fixer runs after physics, OUTSIDE the energy window (F15
    V8): with the fixer on, the energy its p_s shift changes comes back as
    heat, so a step whose fixer moves real mass still closes E."""
    mesh, sigma, s0 = mpas
    model = _mpas_model(mesh, sigma, True, fix_mass=True)
    target = model.compute_mass(s0) * (1.0 + 1.0e-5)   # the fixer adds mass
    on = model.step_pure(s0, DT, target_mass=target)[0]
    e0 = _mpas_energy(s0, mesh, sigma)
    rel_on = (_mpas_energy(on, mesh, sigma) - e0) / e0
    assert abs(rel_on) < 1e-13, rel_on
    # NON-VACUITY: the fixer moved ~1 Pa of air, whose energy at fixed T is
    # ~1e-5 of E (a dynamics-window-only fixer leaves exactly that open)
    dps = float(jnp.mean(on.p_s.data - _mpas_model(mesh, sigma, True)
                         .step(s0, DT).p_s.data))
    assert dps > 0.5, dps


def test_mpas_fixer_is_differentiable(mpas):
    mesh, sigma, s0 = mpas
    model = _mpas_model(mesh, sigma, True)

    def loss(T):
        out = model._step_jit(s0._replace(T=s0.T.replace(data=T)), DT,
                              None, None, None, None)[0]
        return jnp.sum(out.T.data * mesh.areaCell[:, None])

    g = jax.grad(loss)(s0.T.data)
    assert np.all(np.isfinite(np.asarray(g)))


def test_voronoi_fixer_gradient_matches_finite_differences(mpas):
    """The fixer's dT is linear in the energy integrals; its reverse-mode
    gradient w.r.t. the post-dynamics T and u passes check_grads (fp64)."""
    from jax.test_util import check_grads
    from legoesm.core.conservation import voronoi_energy_fixer_dT
    mesh, sigma, s0 = mpas
    s1 = _mpas_model(mesh, sigma, False).step(s0, DT)

    def f(T, u):
        post = s1._replace(T=s1.T.replace(data=T), u=s1.u.replace(data=u))
        return voronoi_energy_fixer_dT(s0, post, mesh, sigma, mesh.areaCell)

    check_grads(f, (s1.T.data, s1.u.data), order=1, modes=["rev"],
                eps=1e-3, atol=1e-10, rtol=1e-6)


def test_voronoi_helper_matches_serial_weights_under_a_split_area(mpas):
    """The MPI form (owned-area weight + a cross-rank sum of the three local
    integrals) is the serial result: two disjoint halves summed == whole."""
    from legoesm.core.conservation import voronoi_energy_fixer_dT
    mesh, sigma, s0 = mpas
    s1 = _mpas_model(mesh, sigma, False).step(s0, DT)
    whole = float(voronoi_energy_fixer_dT(s0, s1, mesh, sigma, mesh.areaCell))
    half = jnp.arange(mesh.nCells) < mesh.nCells // 2
    parts = []

    def capture(x):          # stands in for the allreduce: record the partials
        parts.append(x)
        return x
    for m in (half, ~half):
        voronoi_energy_fixer_dT(s0, s1, mesh, sigma,
                                jnp.where(m, mesh.areaCell, 0.0), sum_fn=capture)
    tot = parts[0] + parts[1]
    split = float(energy_fixer_dT(tot[0], tot[1], tot[2]))
    np.testing.assert_allclose(split, whole, rtol=1e-12)
    assert whole > 0.0


# ---------------------------------------------------------------------------
# FV3-duo column lane (cubed sphere, structured)
# ---------------------------------------------------------------------------

DUO_N, DUO_NG, DUO_KM, DUO_DT = 12, 3, 5, 120.0


@pytest.fixture(scope="module")
def duo():
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoConfig, FV3DuoDynamicsModel)
    from legoesm.grids.factory import create_fv3_duo_grid
    grid = create_fv3_duo_grid(DUO_N, DUO_NG)
    dyn = FV3DuoDynamicsModel(grid, FV3DuoConfig(km=DUO_KM, n_split=2))
    return dyn, dyn.dcmip16_initial_state(n_tracers=3)


def _duo_energy(col, st):
    ph = col.sigma_coord.pressure_at_half(st.p_s.data)
    ke = 0.5 * (st.u.data ** 2 + st.v.data ** 2)
    e = dycore_column_energy(st.T.data, ke, ph[..., 1:] - ph[..., :-1],
                             st.phis.data, st.p_s.data)
    return float(jnp.sum(e * jnp.asarray(col.mesh.areaCell)))


def _zero_physics(state, mesh, coord, phys_state=None, forcing=None):
    from legoesm.core.state import HydrostaticTendencies
    z2 = jnp.zeros_like(state.p_s.data)
    f = lambda like, nm: like.replace(data=jnp.zeros_like(like.data), name=nm)  # noqa: E731
    return HydrostaticTendencies(
        du_dt=f(state.u, "du"), dv_dt=f(state.v, "dv"), dT_dt=f(state.T, "dT"),
        dp_s_dt=state.p_s.replace(data=z2, name="dps"),
        dphis_dt=state.phis.replace(data=z2, name="dphis"))


@pytest.mark.parametrize("physics", [None, _zero_physics])
def test_duo_column_step_conserves_total_energy_with_the_fixer(duo, physics):
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_column import FV3DuoColumnModel
    dyn, ic = duo
    rel = {}
    for fixer in (False, True):
        col = FV3DuoColumnModel(dyn, energy_fixer=fixer)
        s0 = col.from_bundle(ic)
        e0 = _duo_energy(col, s0)
        s1 = col.step(s0, DUO_DT, physics_fn=physics)
        rel[fixer] = (_duo_energy(col, s1) - e0) / e0
    assert abs(rel[False]) > 1e-10, rel        # non-vacuous
    assert abs(rel[True]) < 1e-13, rel


# ---------------------------------------------------------------------------
# lanes that do not run it refuse it
# ---------------------------------------------------------------------------

def test_unwired_lanes_refuse_the_fixer():
    from legoesm.driver.component_factory import (
        energy_fixer_wired, refuse_unwired_energy_fixer)
    from legoesm.driver.config import DycoreConfig, ExperimentConfig
    assert DycoreConfig().energy_fixer is True          # RULE 3: default = fix
    on = ExperimentConfig()
    assert energy_fixer_wired(on, "mpas_primitive_equations")
    duo_col = on._replace(dycore=on.dycore._replace(
        discretization="fv3_duo", column_lane=True))
    assert energy_fixer_wired(duo_col, "fv3_duo_primitive_equations")
    for solver in ("cdgrid_primitive_equations",
                   "latlon_cgrid_primitive_equations"):     # F15b
        assert energy_fixer_wired(on, solver)
        refuse_unwired_energy_fixer(on, solver)
    for solver in ("spectral_primitive_equations",
                   "mpas_compressible_euler", "fv3_duo_primitive_equations"):
        with pytest.raises(ValueError, match="energy_fixer"):
            refuse_unwired_energy_fixer(on, solver)
        off = on._replace(dycore=on.dycore._replace(energy_fixer=False))
        refuse_unwired_energy_fixer(off, solver)        # explicit opt-out


# ---------------------------------------------------------------------------
# Structured lanes (F15b): lat-lon C-grid and cube C-D grid hydrostatic cores
# ---------------------------------------------------------------------------

@pytest.fixture
def fp64_policy():
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(prev)


def _gridded_energy(T, u, v, p_s, phis, sigma, area):
    ph = sigma.pressure_at_half(p_s)
    e = dycore_column_energy(T, 0.5 * (u ** 2 + v ** 2),
                             ph[..., 1:] - ph[..., :-1], phis, p_s)
    return float(jnp.sum(e * area))


def _latlon_setup():
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonHydrostaticState)
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    grid = create_latlon_grid(n_lat=16, radius=constants.R_earth,
                              omega=constants.Omega)
    sigma = create_sigma_coordinate(n_levels=NLEV, dtype=jnp.float64)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    rng = np.random.default_rng(0)
    lat = np.asarray(grid.lat)
    T = (280.0 - 30.0 * np.sin(lat)[:, None, None] ** 2
         + np.linspace(20.0, -20.0, NLEV)[None, None, :])
    state = CGridLatLonHydrostaticState(
        u=jnp.asarray(10.0 * rng.standard_normal((n_lat, n_lon + 1, NLEV))),
        v=jnp.asarray(5.0 * rng.standard_normal((n_lat + 1, n_lon, NLEV))),
        T=jnp.asarray(np.broadcast_to(T, (n_lat, n_lon, NLEV))
                      + 0.1 * rng.standard_normal((n_lat, n_lon, NLEV))),
        p_s=jnp.asarray(1.0e5 + 500.0 * rng.standard_normal((n_lat, n_lon))),
        phis=jnp.asarray(2.0e3 * rng.random((n_lat, n_lon))), tracers={})
    return grid, sigma, state


def _latlon_energy(s, grid, sigma):
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        cgrid_to_hydrostatic)
    h = cgrid_to_hydrostatic(s, grid)
    return _gridded_energy(h.T.data, h.u.data, h.v.data, h.p_s.data,
                           h.phis.data, sigma, grid.area)


def _latlon_step(grid, sigma, state, fixer, physics_fn=None, fix_mass=False,
                 target_mass=None):
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationConfig, CGridLatLonPrimitiveEquationModel)
    # viscosity ON (the sink the fixer closes); no mass fixer, so the step's
    # only energy change is the dynamics window's (plus the physics)
    cfg = CGridLatLonPrimitiveEquationConfig(A_h=1.0e5, fix_mass=fix_mass,
                                             energy_fixer=fixer)
    model = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg, dt=DT)
    return model._step_cgrid(state, DT, target_mass, physics_fn, None)[0]


def test_latlon_step_conserves_total_energy_with_the_fixer(fp64_policy):
    grid, sigma, s0 = _latlon_setup()
    e0 = _latlon_energy(s0, grid, sigma)
    off = _latlon_step(grid, sigma, s0, False)
    on = _latlon_step(grid, sigma, s0, True)
    rel_off = (_latlon_energy(off, grid, sigma) - e0) / e0
    rel_on = (_latlon_energy(on, grid, sigma) - e0) / e0
    assert on.T.dtype == jnp.float64, on.T.dtype
    assert abs(rel_off) > 1e-10, rel_off       # non-vacuous
    assert abs(rel_on) < 1e-13, rel_on
    dT = np.asarray(on.T - off.T)               # a uniform heating, nothing else
    assert np.ptp(dT) < 1e-12 and dT.flat[0] != 0.0
    np.testing.assert_array_equal(np.asarray(on.u), np.asarray(off.u))


def test_latlon_mass_fixer_is_energy_neutral_with_the_fixer(fp64_policy):
    """The lat-lon dry-mass fixer (safety rails, after physics) sits outside
    the energy window (F15 V8): with the fixer on, its energy returns as heat."""
    grid, sigma, s0 = _latlon_setup()
    target = float(jnp.sum(s0.p_s * grid.area)) * (1.0 + 1.0e-5)
    e0 = _latlon_energy(s0, grid, sigma)
    on = _latlon_step(grid, sigma, s0, True, fix_mass=True, target_mass=target)
    rel_on = (_latlon_energy(on, grid, sigma) - e0) / e0
    assert abs(rel_on) < 1e-13, rel_on
    # NON-VACUITY: the fixer moved ~1 Pa of air (energy ~1e-5 of E)
    dps = float(jnp.mean(on.p_s - _latlon_step(grid, sigma, s0, True).p_s))
    assert dps > 0.5, dps


def test_segment_mass_fixer_energy_dT_restores_the_energy():
    """The segment lanes' helper: E(T + dT, p_s_post) == E(T, p_s_pre) on the
    model's own layer masses; ``None`` when the model runs no fixer."""
    from collections import namedtuple

    from legoesm.core.conservation import SigmaLayers
    from legoesm.driver.compiled_segments import mass_fixer_energy_dT
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    grid = create_cubed_sphere(4)
    sig = SigmaLayers(jnp.asarray([0.2, 0.3, 0.5]))
    rng = np.random.default_rng(5)
    c, s2 = (6, 4, 4, 3), (6, 4, 4)
    T = jnp.asarray(260.0 + 10.0 * rng.random(c))
    u, v = (jnp.asarray(10.0 * rng.standard_normal(c)) for _ in range(2))
    phis = jnp.asarray(2.0e3 * rng.random(s2))
    ps0 = jnp.asarray(1.0e5 + 300.0 * rng.standard_normal(s2))
    ps1 = ps0 + 3.0                              # the fixer's uniform shift
    cfg = namedtuple("Cfg", "energy_fixer")
    model = namedtuple("Model", "config sigma_coord")
    dT = mass_fixer_energy_dT(model(cfg(True), sig), T, u, v, phis, ps0, ps1,
                              grid)
    area = grid.area.astype(jnp.float64)
    e0 = _gridded_energy(T, u, v, ps0, phis, sig, area)
    np.testing.assert_allclose(
        _gridded_energy(T + dT, u, v, ps1, phis, sig, area), e0, rtol=1e-14)
    # NON-VACUITY: the shift alone changes E by ~3e-5 of it
    assert abs(_gridded_energy(T, u, v, ps1, phis, sig, area) - e0) / e0 > 1e-5
    assert mass_fixer_energy_dT(model(cfg(False), sig), T, u, v, phis, ps0,
                                ps1, grid) is None


def test_operator_split_lane_mass_fixer_is_energy_neutral(fp64_policy):
    """The lat-lon operator-split segment step (serial layout of the lat-band
    SPMD lane): its dry-mass fixer runs after the dycore's energy window; with
    the fixer on and no physics, the step closes E although the fixer moves
    real mass (anchor offset by 1e-5)."""
    import dataclasses

    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationModel)
    from legoesm.driver.sharded_operator_split_step import (
        make_sharded_operator_split_step)
    from tests.parallel.test_atm_latlon_operator_split_spmd import (
        _build_harness, _mock_step_unified_stateful)
    m0, statics, carry, forcing = _build_harness()
    model = CGridLatLonPrimitiveEquationModel(
        m0.grid, m0.sigma_coord, m0.config._replace(energy_fixer=True))

    def no_physics(*a, **k):
        out, held, tl = _mock_step_unified_stateful(*a, **k)
        z = jnp.zeros_like
        return out._replace(dT_dt=z(out.dT_dt), dq_v_dt=z(out.dq_v_dt),
                            precip=z(out.precip)), held, tl

    statics = dataclasses.replace(statics, step_unified=no_physics,
                                  do_sat_adjust=False, qv_smooth_coeff=0.0)
    carry = carry._replace(target_mass=carry.target_mass * (1.0 + 1.0e-5))
    step = make_sharded_operator_split_step(
        model, None, statics, fix_mass=True, rad_update_steps=1, start_day=0.0)
    out = step(carry, forcing)
    sig, area = model.sigma_coord, model.grid.area

    def E(c):
        return _gridded_energy(c.T, c.u, c.v, c.p_s, c.phis, sig, area)

    e0 = E(carry)
    assert abs(E(out) - e0) / e0 < 1e-13, (E(out) - e0) / e0
    # NON-VACUITY: the fixer added ~1 Pa of air (energy ~1e-5 of E)
    assert float(jnp.mean(out.p_s - carry.p_s)) > 0.5


def test_latlon_cell_centre_step_closes_the_driver_view(fp64_policy):
    """The driver's lanes hand the lat-lon core a CELL-CENTRE state (cell ->
    face -> cell each step): the closed window is that state, so the
    projection's own KE change is fixed too."""
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationConfig, CGridLatLonPrimitiveEquationModel,
        cgrid_to_hydrostatic)
    grid, sigma, c0 = _latlon_setup()
    s0 = cgrid_to_hydrostatic(c0, grid)

    def E(h):
        return _gridded_energy(h.T.data, h.u.data, h.v.data, h.p_s.data,
                               h.phis.data, sigma, grid.area)
    e0, rel = E(s0), {}
    for fixer in (False, True):
        model = CGridLatLonPrimitiveEquationModel(
            grid, sigma, CGridLatLonPrimitiveEquationConfig(
                A_h=1.0e5, fix_mass=False, energy_fixer=fixer), dt=DT)
        rel[fixer] = (E(model.step(s0, DT)) - e0) / e0
    assert abs(rel[False]) > 1e-10, rel       # non-vacuous
    assert abs(rel[True]) < 1e-13, rel


def test_latlon_fixer_runs_before_the_physics(fp64_policy):
    """A known physics source (uniform +1e-4 K/s) stays in the budget: the
    fixer closes only the dynamics window, then the physics adds c_pd*M*dT."""
    from legoesm.core.state import HydrostaticTendencies
    grid, sigma, s0 = _latlon_setup()
    rate = 1.0e-4

    def heat(state, grid_, coord, phys_state=None):
        z = lambda f, nm: f.replace(data=jnp.zeros_like(f.data), name=nm)  # noqa: E731
        return HydrostaticTendencies(
            du_dt=z(state.u, "du"), dv_dt=z(state.v, "dv"),
            dT_dt=state.T.replace(data=jnp.full_like(state.T.data, rate),
                                  name="dT"),
            dp_s_dt=z(state.p_s, "dps"), dphis_dt=z(state.phis, "dphis"))
    e0 = _latlon_energy(s0, grid, sigma)
    on = _latlon_step(grid, sigma, s0, True, physics_fn=heat)
    ph = sigma.pressure_at_half(on.p_s)
    mass = float(jnp.sum(jnp.sum(ph[..., 1:] - ph[..., :-1], -1)
                         / constants.g * grid.area))
    source = constants.c_pd * rate * DT * mass
    np.testing.assert_allclose(_latlon_energy(on, grid, sigma) - e0, source,
                               rtol=1e-9)


def test_gridded_fixer_gradient_matches_finite_differences(fp64_policy):
    from jax.test_util import check_grads
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        cgrid_to_hydrostatic)
    from legoesm.core.conservation import gridded_energy_fixer_dT
    grid, sigma, s0 = _latlon_setup()
    s1 = _latlon_step(grid, sigma, s0, False)
    h0, h1 = cgrid_to_hydrostatic(s0, grid), cgrid_to_hydrostatic(s1, grid)
    pre = (h0.T.data, h0.u.data, h0.v.data, h0.p_s.data, h0.phis.data)

    def f(T, u):
        return gridded_energy_fixer_dT(
            pre, (T, u, h1.v.data, h1.p_s.data, h1.phis.data), grid, sigma)

    check_grads(f, (h1.T.data, h1.u.data), order=1, modes=["rev"],
                eps=1e-3, atol=1e-10, rtol=1e-6)


CUBE_N, CUBE_NLEV = 8, 5


def _cube_setup(fixer):
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationConfig, CDGridPrimitiveEquationModel,
        FV3HydrostaticState)
    from legoesm.core.field import Field
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    grid = create_cubed_sphere(CUBE_N)
    sigma = create_sigma_coordinate(CUBE_NLEV)
    # viscosity + hyperdiffusion + default sponge ON; no mass fixer
    cfg = CDGridPrimitiveEquationConfig(
        A_h=1.0e6, hyperdiff_coeff=1.0e16, use_conservation_fixer=False,
        fix_mass=False, energy_fixer=fixer)
    model = CDGridPrimitiveEquationModel(grid, sigma, cfg)
    rng = np.random.default_rng(3)
    d = (6, CUBE_N + 1, CUBE_N + 1, CUBE_NLEV)
    c = (6, CUBE_N, CUBE_N, CUBE_NLEV)
    s2 = (6, CUBE_N, CUBE_N)
    dd, d2 = ("face", "x", "y", "level"), ("face", "x", "y")
    st = FV3HydrostaticState(
        u_d=Field(data=jnp.asarray(8.0 * rng.standard_normal(d)), name="u_d",
                  dims=dd, units="m/s"),
        v_d=Field(data=jnp.asarray(8.0 * rng.standard_normal(d)), name="v_d",
                  dims=dd, units="m/s"),
        T=Field(data=jnp.asarray(260.0 + rng.standard_normal(c)), name="T",
                dims=dd, units="K"),
        p_s=Field(data=jnp.asarray(1e5 + 300.0 * rng.standard_normal(s2)),
                  name="p_s", dims=d2, units="Pa"),
        phis=Field(data=jnp.zeros(s2), name="phis", dims=d2, units="m^2/s^2"))
    return model, st


def _cube_energy(model, st):
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        fv3_to_hydrostatic)
    h = fv3_to_hydrostatic(st, model.cdgrid)
    return _gridded_energy(h.T.data, h.u.data, h.v.data, h.p_s.data,
                           h.phis.data, model.sigma_coord, model.grid.area)


def test_cube_step_conserves_total_energy_with_the_fixer(fp64_policy):
    m_off, s0 = _cube_setup(False)
    m_on, _ = _cube_setup(True)
    e0 = _cube_energy(m_on, s0)
    off, on = m_off.step(s0, DT), m_on.step(s0, DT)
    rel_off = (_cube_energy(m_off, off) - e0) / e0
    rel_on = (_cube_energy(m_on, on) - e0) / e0
    assert on.T.data.dtype == jnp.float64, on.T.data.dtype
    assert abs(rel_off) > 1e-10, rel_off       # non-vacuous
    assert abs(rel_on) < 1e-13, rel_on
    dT = np.asarray(on.T.data - off.T.data)
    assert np.ptp(dT) < 1e-12 and dT.flat[0] != 0.0
    np.testing.assert_array_equal(np.asarray(on.u_d.data),
                                  np.asarray(off.u_d.data))


def test_cube_cell_centre_step_closes_the_driver_view(fp64_policy):
    """The driver's compiled / per-step lanes hand the cube a CELL-CENTRE
    state (cc -> D -> cc each step): the closed window is that cc state, so
    the round trip's own KE change is fixed too."""
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        fv3_to_hydrostatic)
    m_off, s_d = _cube_setup(False)
    m_on, _ = _cube_setup(True)
    s0 = fv3_to_hydrostatic(s_d, m_on.cdgrid)

    def E(h):
        return _gridded_energy(h.T.data, h.u.data, h.v.data, h.p_s.data,
                               h.phis.data, m_on.sigma_coord, m_on.grid.area)
    e0 = E(s0)
    rel_off = (E(m_off.step(s0, DT)) - e0) / e0
    rel_on = (E(m_on.step(s0, DT)) - e0) / e0
    assert abs(rel_off) > 1e-10, rel_off       # non-vacuous
    assert abs(rel_on) < 1e-13, rel_on


def test_cube_refuses_physics_inside_the_rk_window(fp64_policy):
    model, s0 = _cube_setup(True)
    with pytest.raises(ValueError, match="dynamics-only"):
        model.step(s0, DT, physics_fn=lambda s, g, c: None)


@pytest.mark.parametrize("grid_type,disc,solver_cfg", [
    ("latlon", "latlon_cgrid", "CGridLatLonPrimitiveEquationConfig"),
    ("cubed_sphere", "cdgrid", "CDGridPrimitiveEquationConfig"),
])
def test_factory_hands_the_switch_to_the_core(grid_type, disc, solver_cfg):
    """The DEFAULT driver config builds the core with the fixer ON (RULE 3),
    and the explicit opt-out reaches it as OFF."""
    from legoesm.driver.component_factory import create_atmosphere_dycore
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    from legoesm.grids.vertical import create_sigma_coordinate
    if grid_type == "latlon":
        from legoesm.grids.latlon import create_latlon_grid
        grid = create_latlon_grid(n_lat=16, radius=constants.R_earth,
                                  omega=constants.Omega)
    else:
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        grid = create_cubed_sphere(8)
    sigma = create_sigma_coordinate(5)
    for flag in (True, False):
        cfg = ExperimentConfig(
            grid=GridConfig(grid_type=grid_type, resolution=8, nlev=5),
            dycore=DycoreConfig(discretization=disc, dt=300.0,
                                energy_fixer=flag),
            days=1)
        model = create_atmosphere_dycore(cfg, grid, sigma)
        assert type(model.config).__name__ == solver_cfg
        assert model.config.energy_fixer is flag
    assert DycoreConfig().energy_fixer is True


def test_tiled_cube_lanes_refuse_the_fixer():
    """The sub-face-tiled dynamics do not run the cube step: both driver
    entries refuse energy_fixer=True before building anything."""
    from types import SimpleNamespace
    from legoesm.driver.config import DycoreConfig
    from legoesm.driver.model_driver import ModelDriver
    stub = SimpleNamespace(config=SimpleNamespace(
        enable_tiled_dycore=True, dycore=DycoreConfig()))
    with pytest.raises(ValueError, match="TILED"):
        ModelDriver._maybe_build_tiled_step(stub, 300.0)
    with pytest.raises(ValueError, match="TILED"):
        ModelDriver._run_tiled_cube_spmd(stub)
