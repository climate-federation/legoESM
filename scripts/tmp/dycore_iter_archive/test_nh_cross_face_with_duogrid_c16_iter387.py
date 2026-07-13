"""FV3_3D iter 387: NH cross_face flag at C16 WITH duogrid grid.
NH counterpart of iter-386.
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
from legoesm.grids.vertical import (
    create_height_coordinate, compute_terrain_metric,
)


def _build_c16():
    n = 16
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=True)
    hc = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    tm = compute_terrain_metric(terrain, hc)
    rng = np.random.default_rng(seed=387)
    u_p = rng.uniform(-2.0, 2.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-2.0, 2.0, size=(6, n, n, nlev))
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)),
                name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                        name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return grid, hc, tm, state


def _imprint_v(state, n_cells, edge_width=3):
    v = state.v.data
    i_idx = jnp.arange(n_cells)
    edge_i = (i_idx < edge_width) | (i_idx >= n_cells - edge_width)
    edge_mask = edge_i[:, None] | edge_i[None, :]
    interior_mask = ~edge_mask
    e = float(jnp.std(v[:, edge_mask, :].reshape(-1)))
    i = float(jnp.std(v[:, interior_mask, :].reshape(-1)))
    return e / max(i, 1e-30)


def _cfg(cross):
    return CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_v=0.030, nord_v=1,
        use_fv3_cross_face_du_proj=cross,
    )


def test_nh_c16_duogrid_cross_face_changes_state():
    grid, hc, tm, state = _build_c16()
    m_off = CDGridCompressibleEulerModel(grid, hc, tm, _cfg(False))
    m_on = CDGridCompressibleEulerModel(grid, hc, tm, _cfg(True))
    s_off = m_off.step(state, 5.0)
    s_on = m_on.step(state, 5.0)
    diff = float(np.max(np.abs(
        np.asarray(s_on.u.data) - np.asarray(s_off.u.data),
    )))
    assert diff > 1e-8


def test_nh_c16_duogrid_cross_face_does_not_amplify():
    grid, hc, tm, state = _build_c16()
    m_off = CDGridCompressibleEulerModel(grid, hc, tm, _cfg(False))
    m_on = CDGridCompressibleEulerModel(grid, hc, tm, _cfg(True))
    s_off = m_off.step(state, 5.0)
    s_on = m_on.step(state, 5.0)
    r_off = _imprint_v(s_off, n_cells=16)
    r_on = _imprint_v(s_on, n_cells=16)
    assert r_on <= r_off * 1.10
