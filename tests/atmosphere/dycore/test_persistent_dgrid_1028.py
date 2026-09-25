"""#1028 — the cube's prognostic winds stay in FV3 D staggering between steps.

The cell-centre entry to the cube dycore interpolates the winds to the D-grid
corners on the way in and back to centres on the way out, EVERY step.  A paired
same-commit 200-day C36 Held-Suarez run with that outer projection removed (and
nothing else changed) moved the equilibrated max wind 13.0 -> 38.6 m/s on sigma
levels and 12.6 -> 40.9 on hybrid (PR #1462) — from failing the #1049 dead-jet
floor to clearing the ~30 m/s benchmark.  The dycore already accepted either
staggering; what this covers is the production wiring that CARRIES the D state.

What each test pins, and what it would catch:

* the round trip is NOT a no-op — if it were, the whole issue would be void and
  every other test here would be vacuous, so this one runs first;
* the carry/rebuild path preserves the corner layout, i.e. a compiled segment
  built on the D state does not quietly fall back to cell centres;
* the two helpers the driver uses to touch winds outside the dycore behave as
  their docstrings claim (the Rayleigh decay commutes with the lift; the
  physics increment is lifted with the SAME operator the dycore uses);
* the lanes that are NOT wired refuse loudly instead of silently running the
  damped path.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
    CDGridPrimitiveEquationModel,
    FV3HydrostaticState,
    fv3_to_hydrostatic,
    hydrostatic_to_fv3,
)
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.core.operators_cdgrid import (
    center_to_dgrid_vector,
    dgrid_to_center_vector,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import standard_hybrid_levels


@pytest.fixture(scope="module")
def cube():
    """A cube state with a REAL wind field.

    ``held_suarez_init`` starts from rest, and a projection applied to a zero
    wind is a no-op — a control that perturbs a zero proves nothing.  So the
    fixture paints a smooth zonal jet plus a weaker meridional wave on top of
    the Held-Suarez temperature/pressure state; every wind assertion below is
    then measuring the projection rather than measuring zero.
    """
    n, nlev = 8, 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)

    lat = jnp.asarray(grid.lat)[..., None]          # (6, n, n, 1)
    lon = jnp.asarray(grid.lon)[..., None]
    u = 20.0 * jnp.cos(lat) * jnp.ones((1, 1, 1, nlev))
    v = 2.0 * jnp.cos(lat) * jnp.sin(2.0 * lon) * jnp.ones((1, 1, 1, nlev))
    state_cc = state_cc._replace(
        u=state_cc.u.replace(data=u.astype(state_cc.u.data.dtype)),
        v=state_cc.v.replace(data=v.astype(state_cc.v.data.dtype)),
    )
    assert float(jnp.abs(state_cc.u.data).max()) > 1.0
    return grid, cdgrid, coord, state_cc


def test_the_round_trip_is_not_a_no_op(cube):
    """cc -> D -> cc changes the winds; without this the issue is void.

    This is the premise every other test here rests on.  The projection is a
    4-point average onto the corners followed by an average back, so it is a
    smoother: the difference is what one step of the old path threw away.
    """
    _grid, cdgrid, _coord, state_cc = cube
    back = fv3_to_hydrostatic(hydrostatic_to_fv3(state_cc, cdgrid), cdgrid)
    du = np.asarray(back.u.data) - np.asarray(state_cc.u.data)
    scale = float(np.abs(np.asarray(state_cc.u.data)).max())
    assert scale > 0.0
    assert np.abs(du).max() / scale > 1e-3, (
        "cc->D->cc is nearly the identity on this state, so it cannot be the "
        "eddy damper #1028 measured — check the fixture, not the physics")


def test_one_step_differs_between_the_two_staggerings(cube):
    """Stepping FROM the D state is not the same as stepping from centres.

    Same initial condition, same dt, same config: the only difference is
    whether the prognostic winds were projected to centres and back before the
    step.  A run that carried the D state and still matched the cc path would
    mean the wiring never took effect.
    """
    grid, cdgrid, coord, state_cc = cube
    model = CDGridPrimitiveEquationModel(
        grid, coord, CDGridPrimitiveEquationConfig())
    dt = 300.0

    cc_out = model.step(state_cc, dt)                       # legacy path
    d_out = model.step(hydrostatic_to_fv3(state_cc, cdgrid), dt)
    assert isinstance(d_out, FV3HydrostaticState), (
        "the D-state step must stay D-staggered; returning a cell-centre "
        "state would put the projection back in the prognostic path")
    d_out_cc = fv3_to_hydrostatic(d_out, cdgrid)
    diff = np.abs(np.asarray(d_out_cc.u.data) - np.asarray(cc_out.u.data)).max()
    assert diff > 0.0, "the two staggerings produced identical winds"


def test_carry_round_trip_keeps_the_corner_layout(cube):
    """pack -> rebuild on a D state returns a D state with corner shapes.

    The compiled segment stores raw arrays and rebuilds the state type each
    step.  If the rebuild named the leaves ``u``/``v`` it would either raise or
    construct the wrong state, and the segment would run the damped path.
    """
    from legoesm.driver.compiled_segments import _rebuild_state, pack_carry

    grid, cdgrid, coord, state_cc = cube
    state_d = hydrostatic_to_fv3(state_cc, cdgrid)
    n = grid.n
    assert state_d.u_d.data.shape[1:3] == (n + 1, n + 1)

    zeros = jnp.zeros_like(state_cc.T.data)
    carry = pack_carry(
        state_d, zeros, zeros, zeros,
        held_dT_rad=zeros, held_sw_net_sfc=jnp.zeros_like(state_cc.p_s.data),
        held_lw_net_sfc=jnp.zeros_like(state_cc.p_s.data),
        held_sw_up_toa=jnp.zeros_like(state_cc.p_s.data),
        held_lw_up_toa=jnp.zeros_like(state_cc.p_s.data),
        held_sw_down_toa=jnp.zeros_like(state_cc.p_s.data),
        step_index=0,
    )
    assert carry.u.shape == state_d.u_d.data.shape
    np.testing.assert_array_equal(np.asarray(carry.u),
                                  np.asarray(state_d.u_d.data))

    model = CDGridPrimitiveEquationModel(
        grid, coord, CDGridPrimitiveEquationConfig())
    rebuilt = _rebuild_state(carry, model, persistent_dgrid=True)
    assert isinstance(rebuilt, FV3HydrostaticState)
    np.testing.assert_array_equal(np.asarray(rebuilt.u_d.data),
                                  np.asarray(state_d.u_d.data))


def test_rayleigh_decay_commutes_with_the_lift(cube):
    """A level-dependent decay may be applied on either staggering.

    The driver multiplies the winds by ``exp(-k_f dt)`` outside the dycore.
    ``k_f`` is a function of the vertical coordinate only, so scaling the
    corner winds equals scaling the centre winds and lifting — which is what
    lets the driver apply it directly to the carried D winds.  A HORIZONTALLY
    varying drag would not commute, and this test is the reason that
    restriction is written down at the helper.
    """
    _grid, cdgrid, coord, state_cc = cube
    nlev = state_cc.u.data.shape[-1]
    decay = jnp.exp(-jnp.linspace(0.0, 0.5, nlev))      # level-dependent only

    u_d, v_d = center_to_dgrid_vector(
        state_cc.u.data, state_cc.v.data, cdgrid)
    lift_then_scale_u = u_d * decay
    scale_then_lift_u, _ = center_to_dgrid_vector(
        state_cc.u.data * decay, state_cc.v.data * decay, cdgrid)
    # The two orderings are algebraically identical, so the only difference
    # allowed is floating-point rounding.  Bound it in UNITS OF EPSILON rather
    # than by a hand-picked tolerance: anything larger than a few eps would
    # mean the operations are not the same operation.
    eps = float(np.finfo(np.asarray(u_d).dtype).eps)
    rel = float(np.max(np.abs(np.asarray(lift_then_scale_u)
                              - np.asarray(scale_then_lift_u)))
                / max(float(np.max(np.abs(np.asarray(u_d)))), 1e-30))
    assert rel < 50.0 * eps, (
        f"scale-then-lift and lift-then-scale differ by {rel/eps:.1f} eps "
        "-- that is not rounding, the decay does not commute with the lift")


def test_physics_increment_goes_through_the_driver_helper(cube):
    """The driver's own helper lifts the increment; it is not a second lift.

    Exercised through ``ModelDriver._add_cc_wind_increment`` itself (an earlier
    version of this test compared two identical calls to the shared operator,
    which would have passed with the helper deleted — codex).  Bound to a stand-
    in object carrying only what the helper touches, so it is a unit test of the
    helper rather than of a whole driver.
    """
    from legoesm.driver.model_driver import ModelDriver

    _grid, cdgrid, _coord, state_cc = cube
    dt = 300.0
    du = jnp.asarray(
        np.linspace(-1e-4, 1e-4, state_cc.u.data.size)
    ).reshape(state_cc.u.data.shape).astype(state_cc.u.data.dtype)
    dv = (0.5 * du).astype(state_cc.v.data.dtype)

    class _Stub:
        # the helper asks the driver whether the lane is active, so the stub
        # carries the real predicate bound to itself
        _persistent_dgrid_active = ModelDriver._persistent_dgrid_active

    stub = _Stub()
    stub._persistent_dgrid = True
    stub.model = type("M", (), {"cdgrid": cdgrid})()
    stub.state = hydrostatic_to_fv3(state_cc, cdgrid)
    before_u = np.asarray(stub.state.u_d.data).copy()
    before_v = np.asarray(stub.state.v_d.data).copy()

    ModelDriver._add_cc_wind_increment(stub, du, dv, dt)

    assert isinstance(stub.state, FV3HydrostaticState), (
        "the helper must leave the state D-staggered")
    expect_u, expect_v = center_to_dgrid_vector(du, dv, cdgrid)
    np.testing.assert_allclose(
        np.asarray(stub.state.u_d.data), before_u + dt * np.asarray(expect_u),
        rtol=1e-6, atol=1e-6)
    # BOTH components: asserting only u leaves a helper that drops the
    # meridional update passing (codex round 2 mutated exactly that and the
    # test survived).
    np.testing.assert_allclose(
        np.asarray(stub.state.v_d.data), before_v + dt * np.asarray(expect_v),
        rtol=1e-6, atol=1e-6)
    # ... and it MOVED both winds: an increment that changed nothing would make
    # every assertion above vacuous.
    assert np.abs(np.asarray(stub.state.u_d.data) - before_u).max() > 0.0
    assert np.abs(np.asarray(stub.state.v_d.data) - before_v).max() > 0.0

    # Off the persistent-D lane the same helper takes the cell-centre branch.
    stub_cc = _Stub()
    stub_cc._persistent_dgrid = False
    stub_cc.model = type("M", (), {"cdgrid": cdgrid})()
    stub_cc.state = state_cc
    ModelDriver._add_cc_wind_increment(stub_cc, du, dv, dt)
    np.testing.assert_allclose(
        np.asarray(stub_cc.state.u.data),
        np.asarray(state_cc.u.data) + dt * np.asarray(du),
        rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(stub_cc.state.v.data),
        np.asarray(state_cc.v.data) + dt * np.asarray(dv),
        rtol=1e-6, atol=1e-6)


def test_the_lift_is_linear_so_increment_and_state_agree(cube):
    """Lifting the increment equals lifting the updated state.

    This is the property that makes the driver's increment path equivalent to
    the dycore's own handling of cell-centre physics tendencies; if the lift
    stopped being linear the two would be different operators.
    """
    _grid, cdgrid, _coord, state_cc = cube
    dt = 300.0
    du = jnp.asarray(
        np.linspace(-1e-4, 1e-4, state_cc.u.data.size)
    ).reshape(state_cc.u.data.shape).astype(state_cc.u.data.dtype)
    dv = (0.5 * du).astype(state_cc.v.data.dtype)
    u_d, _v_d = center_to_dgrid_vector(state_cc.u.data, state_cc.v.data, cdgrid)
    a_u, _a_v = center_to_dgrid_vector(du, dv, cdgrid)
    lift_of_sum_u, _ = center_to_dgrid_vector(
        state_cc.u.data + dt * du, state_cc.v.data + dt * dv, cdgrid)
    eps = float(np.finfo(np.asarray(u_d).dtype).eps)
    rel = float(np.max(np.abs(np.asarray(lift_of_sum_u)
                              - np.asarray(u_d + dt * a_u)))
                / max(float(np.max(np.abs(np.asarray(u_d)))), 1e-30))
    assert rel < 50.0 * eps, (
        f"lifting the increment and lifting the updated state differ by "
        f"{rel/eps:.1f} eps -- the lift is not linear here")


def test_unwired_lanes_refuse_rather_than_running_damped():
    """The segment builder refuses persistent-D where it is not implemented.

    Both branches would otherwise keep the per-step projection while the run
    log claimed persistent D winds — the exact silent-fallback class the repo's
    dispatch rule exists to prevent.
    """
    from legoesm.driver.compiled_segments import build_segment_fn

    common = dict(
        model=None, step_unified=None, grid=None, sigma_full=None,
        dsigma=None, dt=600.0, rad_update_steps=1, microphysics="none",
        fix_moisture=False, fix_mass=False, fric_decay=1.0,
        qv_smooth_coeff=0.0, lat=None, lon=None, start_day=0.0,
        persistent_dgrid=True,
    )
    with pytest.raises(ValueError, match="owned_face_ids"):
        build_segment_fn(owned_face_ids=jnp.arange(3), **common)
    with pytest.raises(ValueError, match="tiled"):
        build_segment_fn(tiled_step_fn=(lambda s: s), **common)


def test_driver_refuses_the_lanes_it_did_not_wire(cube):
    """``_enter_persistent_dgrid`` refuses instead of running the damped path.

    A lane that is neither converted nor refused would run the OLD cell-centre
    physics while the run's configuration claims D-staggered winds — the silent
    mismatch this whole change exists to remove.  Checked on the stand-in
    object the helper actually reads, so it does not need a built model.
    """
    from legoesm.driver.model_driver import ModelDriver

    _grid, cdgrid, _coord, state_cc = cube

    class _Cfg:
        pass

    def _stub(grid_type="cubed_sphere", model_type="hydrostatic",
              discretization="cdgrid", coupled=False, ensemble=1):
        st = type("S", (), {})()
        st.config = _Cfg()
        st.config.grid = _Cfg(); st.config.grid.grid_type = grid_type
        st.config.dycore = _Cfg()
        st.config.dycore.persistent_dgrid = True
        st.config.dycore.model_type = model_type
        st.config.dycore.discretization = discretization
        st._ensemble_size = ensemble
        st._owned_face_ids = None
        st._device_config = None
        st._requires_surface_flux_export = coupled
        st._is_spmd_multiprocess = lambda: False
        st.model = type("M", (), {"cdgrid": cdgrid})()
        st.state = state_cc
        return st

    for kwargs, token in (
        (dict(grid_type="latlon"), "grid_type"),
        (dict(model_type="shallow_water"), "model_type"),
        (dict(discretization="fv3_duo"), "discretization"),
        (dict(ensemble=4), "ensemble"),
        (dict(coupled=True), "COUPLED"),
    ):
        st = _stub(**kwargs)
        with pytest.raises(SystemExit, match=token):
            ModelDriver._enter_persistent_dgrid(st)

    # The wired lane converts, once, and records that it did.
    ok = _stub()
    ModelDriver._enter_persistent_dgrid(ok)
    assert ok._persistent_dgrid is True
    assert isinstance(ok.state, FV3HydrostaticState)
    # Idempotent: a restart that already carries D winds is not lifted twice.
    again = np.asarray(ok.state.u_d.data).copy()
    ModelDriver._enter_persistent_dgrid(ok)
    np.testing.assert_array_equal(np.asarray(ok.state.u_d.data), again)

    # Flag off -> untouched, whatever the lane.
    off = _stub(grid_type="latlon")
    off.config.dycore.persistent_dgrid = False
    ModelDriver._enter_persistent_dgrid(off)
    assert off._persistent_dgrid is False
    assert off.state is state_cc


def test_cc_view_is_a_view_not_a_write_back(cube):
    """``fv3_to_hydrostatic`` must not be assigned back over the D winds.

    Nothing enforces this mechanically, so the test states the invariant the
    driver's ``_state_cc`` docstring promises: taking the view leaves the
    D-staggered arrays untouched.  A write-back would reinstate the round trip
    one call site at a time.
    """
    _grid, cdgrid, _coord, state_cc = cube
    state_d = hydrostatic_to_fv3(state_cc, cdgrid)
    before = np.asarray(state_d.u_d.data).copy()
    view = fv3_to_hydrostatic(state_d, cdgrid)
    assert view.u.data.shape != state_d.u_d.data.shape
    np.testing.assert_array_equal(np.asarray(state_d.u_d.data), before)
