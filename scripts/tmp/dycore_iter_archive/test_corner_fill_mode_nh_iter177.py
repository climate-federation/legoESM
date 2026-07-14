"""FV3_3D iter 177: cube-vertex corner fill mode reaches NH 3D path.

The legoESM halo machinery has three documented cube-vertex
fill modes (set via ``LEGOESM_CORNER_FILL`` env var or
``set_corner_fill_mode()``):

* ``"avg"`` (default): legacy 2-point average — direction-invariant.
* ``"fv3_agrid_xdir"``: FV3-faithful AGRID-XDir depth-1 mirror
  (``fv_mp_mod.F90:1077``).
* ``"fv3_bgrid_xdir"``: FV3-faithful BGRID-XDir depth-2 mirror
  (``fv_mp_mod.F90:1041``).  PE iter-10 documented this as the
  cleanest win for HS C36: -41 % cube imprint and -15 % max|v|
  vs ``"avg"``.

PE has documented quantitative impact for these modes; NH never
had a test that the modes actually REACH the NH 3D path's
halo calls.  This iter closes that gap with a regression test
that ``set_corner_fill_mode("fv3_bgrid_xdir")`` produces a
measurably different NH trajectory than ``"avg"`` over a few
steps.

A measurable difference is the necessary condition for the mode
to have any impact at all — if avg vs fv3_bgrid_xdir produced
bit-for-bit identical NH output, that would mean the mode
setting is silently ignored by the NH transport and damping
halos.

Tests
-----
1. ``test_fv3_bgrid_xdir_changes_nh_state`` — bit-for-bit
   different NH state after 3 steps with ``fv3_bgrid_xdir`` vs
   ``avg``.
2. ``test_fv3_agrid_xdir_changes_nh_state`` — same for the
   AGRID-XDir mode.
3. ``test_corner_fill_mode_round_trip`` — set/get round-trip
   for the three valid modes and the ``ValueError`` raise on
   an invalid mode.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerConfig,
    CDGridCompressibleEulerModel,
)
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.halo import (
    set_corner_fill_mode, get_corner_fill_mode,
)
from legoesm.grids.vertical import (
    create_height_coordinate, compute_terrain_metric,
)


@pytest.fixture(scope="module")
def small_nh_state():
    """NH state with a sinusoidal divergent perturbation.  Same
    pattern as iter-174 / iter-176 fixtures.  The perturbation
    creates non-trivial halo content at the cube vertices, which
    is where the corner fill mode matters."""
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    i_idx = jnp.arange(n)
    j_idx = jnp.arange(n)
    k_idx = jnp.arange(nlev)
    pattern = (
        jnp.sin(2 * jnp.pi * i_idx[None, :, None, None] / n)
        * jnp.cos(2 * jnp.pi * j_idx[None, None, :, None] / n)
        * jnp.ones_like(k_idx[None, None, None, :], dtype=jnp.float64)
    )
    pattern = jnp.broadcast_to(pattern, (6, n, n, nlev))
    U0 = 5.0
    u_perturb = jnp.asarray(U0 * pattern, dtype=jnp.float64)

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    state = NonHydrostaticState(
        u=Field(data=u_perturb, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n, nlev)), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)), name="w",
                dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                        name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)), name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return grid, height_coord, terrain_metric, state


def _step3(model, state, dt=10.0):
    s = state
    for _ in range(3):
        s = model.step(s, dt)
    return s


def _run_with_mode(grid, height_coord, terrain_metric, state, mode):
    """Run 3 NH steps with the given corner_fill_mode active."""
    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
    )
    # Save & restore the global mode so tests don't leak state.
    saved = get_corner_fill_mode()
    try:
        set_corner_fill_mode(mode)
        # Build the model AFTER setting the mode so any module-level
        # caching (none expected, but defensive) sees the right mode.
        model = CDGridCompressibleEulerModel(
            grid, height_coord, terrain_metric, cfg,
        )
        s = _step3(model, state)
        # Force materialisation before restoring the mode (jax is
        # lazy; we want the halo calls to finalise under the
        # mode-of-interest, not whatever's restored after).
        s.u.data.block_until_ready()
        return s
    finally:
        set_corner_fill_mode(saved)


def test_fv3_bgrid_xdir_changes_nh_state(small_nh_state):
    """``fv3_bgrid_xdir`` mode produces a measurably different NH
    state than ``avg`` mode after 3 steps.  If they were bit-for-bit
    identical, the mode setting is silently ignored by NH halos."""
    grid, height_coord, terrain_metric, state = small_nh_state

    s_avg = _run_with_mode(
        grid, height_coord, terrain_metric, state, "avg",
    )
    s_bgrid = _run_with_mode(
        grid, height_coord, terrain_metric, state, "fv3_bgrid_xdir",
    )

    diff_u = float(jnp.max(jnp.abs(s_avg.u.data - s_bgrid.u.data)))
    base_u = float(jnp.max(jnp.abs(s_avg.u.data)))
    assert diff_u > 1e-6 * base_u, (
        f"fv3_bgrid_xdir mode must change NH state vs avg "
        f"(diff_u={diff_u:.3e}, base_u={base_u:.3e}).  "
        f"If 0, the corner fill mode setting is silently ignored "
        f"by the NH transport / damping halos."
    )


def test_fv3_agrid_xdir_changes_nh_state(small_nh_state):
    """``fv3_agrid_xdir`` mode produces a measurably different NH
    state than ``avg`` mode."""
    grid, height_coord, terrain_metric, state = small_nh_state

    s_avg = _run_with_mode(
        grid, height_coord, terrain_metric, state, "avg",
    )
    s_agrid = _run_with_mode(
        grid, height_coord, terrain_metric, state, "fv3_agrid_xdir",
    )

    diff_u = float(jnp.max(jnp.abs(s_avg.u.data - s_agrid.u.data)))
    base_u = float(jnp.max(jnp.abs(s_avg.u.data)))
    assert diff_u > 1e-6 * base_u, (
        f"fv3_agrid_xdir mode must change NH state vs avg "
        f"(diff_u={diff_u:.3e}, base_u={base_u:.3e})."
    )


def test_corner_fill_mode_round_trip():
    """``set_corner_fill_mode`` round-trips for the three valid
    modes and raises ``ValueError`` for invalid input."""
    saved = get_corner_fill_mode()
    try:
        for mode in ("avg", "fv3_agrid_xdir", "fv3_bgrid_xdir"):
            set_corner_fill_mode(mode)
            assert get_corner_fill_mode() == mode

        with pytest.raises(ValueError, match="Unknown corner fill mode"):
            set_corner_fill_mode("not_a_real_mode")
    finally:
        set_corner_fill_mode(saved)
