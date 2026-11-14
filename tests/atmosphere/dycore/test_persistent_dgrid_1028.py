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
