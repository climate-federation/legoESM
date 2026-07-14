"""C-grid Coriolis face-placement test for the plane NH dycore.

Codex review 2026-05-24: on a beta-plane the Coriolis parameter
``f(y) = f0 + beta*(y - Ly/2)`` differs between the cell centre
(at ``yc``) and the y-face (at ``yv``). The slow-tendency must
evaluate ``f`` at each target tendency's natural location, not at
the cell centre everywhere. Verified by checking that
``dv_cor = -f_yface · u_at_yface`` uses ``f`` averaged from the
cell centre to the y-face, NOT the raw cell-centre ``f`` value.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    PlaneCompressibleEulerModel,
    interp_cell_to_yface_vlast,
    interp_xface_to_yface_vlast,
    make_flat_plane_terrain_metric,
    make_rest_state,
    plane_compressible_euler_slow_tendencies,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate

jax.config.update("jax_enable_x64", True)


def test_beta_plane_dv_cor_uses_y_face_f():
    """``dv_cor`` at a y-face must equal ``-f_yface · u_at_yface``,
    NOT ``-f_centre · u_at_yface``. On a beta-plane the difference
    is non-trivial because ``f(yc) != f(yv)``.

    Build a state with a uniform non-zero u and check that the
    ``dv_cor`` contribution computed from the slow tendency
    matches the face-interpolated formula to machine epsilon.
    """
    nx = ny = 8
    nlev = 2
    grid = create_plane_grid(
        nx=nx, ny=ny, nlev=nlev, dx=4_000.0, dy=4_000.0,
        coriolis_mode="beta_plane",
        f0=1.0e-4, beta=2.0e-11,
        dtype=jnp.float64,
    )
    hc = create_height_coordinate(nlev, H=2_000.0)
    tm = make_flat_plane_terrain_metric(grid, hc)
    cfg = CompressibleEulerConfig(
        sponge_coeff=0.0, hyperdiff_coeff=0.0,
        hyperdiff_rho_coeff=0.0, hyperdiff_w_coeff=0.0,
        semi_implicit_acoustic=False, use_coriolis=True,
        fix_mass=False, smagorinsky_cs=0.0,
    )
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    # Inject a uniform u; v stays zero so the only non-zero
    # contribution to dv_dt from the slow tendency is -f * u.
    u_val = 5.0
    state = state._replace(
        u=state.u.replace(
            data=jnp.full((ny, nx, nlev), u_val, dtype=jnp.float64),
        ),
    )
    tend = plane_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cfg,
    )
    dv_cor = tend.dv_dt.data

    # Reference: -f_yface * u_at_yface. u is uniform → u_at_yface = u_val.
    # f varies in y; interpolate from cell centre to y-face.
    f_centre = grid.f_y[:, :, None]                # (ny, nx, 1)
    f_yface = interp_cell_to_yface_vlast(
        jnp.broadcast_to(f_centre, (ny, nx, nlev)),
        grid,
    )
    u_yface = interp_xface_to_yface_vlast(state.u.data, grid)
    expected_dv_cor = -f_yface * u_yface

    np.testing.assert_allclose(
        np.asarray(dv_cor), np.asarray(expected_dv_cor),
        rtol=1.0e-12, atol=1.0e-12,
    )

    # And confirm the beta-plane gradient actually matters — using
    # raw cell-centre f would give a different answer at any y
    # where f(yc) != f(yv). At the southernmost ranks of the
    # interior, f(yv) - f(yc) = beta * (-0.5 * dy) != 0.
    naive_dv_cor = -f_centre * u_yface
    diff = float(jnp.max(jnp.abs(dv_cor - naive_dv_cor)))
    assert diff > 0.0, (
        f"f_yface and f_centre give the same answer; beta-plane "
        f"f-variation should be non-zero. max|diff|={diff:.3e}"
    )
