"""FV3_3D iter 360: NH counterpart of iter-359.  NH full
FV3-fidelity stack changes state measurably at C16.

Tests
-----

1. ``test_nh_c16_full_fv3_changes_theta_p`` — full NH FV3 stack
   measurably differs from default at C16 in θ_p field.
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


def _build_c16(use_duogrid):
    n = 16
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=use_duogrid)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)
    rng = np.random.default_rng(seed=360)
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
    return grid, height_coord, terrain_metric, state


def _cfg(all_flags):
    return CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        corner_div_damp_d_con=1.0,
        damp_v=0.030, nord_v=1,
        damp_v_d_con=1.0,
        use_fv3_a2b_zeta_corner=all_flags,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        div_damp_d_con=1.0,
        A_h=1e6, ah_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
        delt_max=1.0,
        use_fv3_d_con_cv=all_flags,
        use_fv3_vector_halo_uv=all_flags,
        use_fv3_dynamic_exner=all_flags,
        use_fv3_metric_aware_d_con=all_flags,
    )


def test_nh_c16_full_fv3_changes_theta_p():
    grid_d, hc, tm, state = _build_c16(use_duogrid=False)
    grid_f, hc2, tm2, state_f = _build_c16(use_duogrid=True)
    m_default = CDGridCompressibleEulerModel(grid_d, hc, tm, _cfg(False))
    m_full = CDGridCompressibleEulerModel(grid_f, hc2, tm2, _cfg(True))
    s_d = m_default.step(state, 5.0)
    s_f = m_full.step(state_f, 5.0)
    diff = float(np.max(np.abs(
        np.asarray(s_d.theta_prime.data)
        - np.asarray(s_f.theta_prime.data),
    )))
    assert diff > 1e-6, (
        f"NH full FV3-fidelity stack at C16 did NOT measurably "
        f"change theta_p vs default: diff={diff:.3e}."
    )
