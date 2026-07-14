"""FV3_3D iter 332: parametrized AD-at-rest regression for the
iter-320 / 325 / 328 NH FV3-fidelity flag combinations.

Background
----------
iter-184 NH umbrella covers AD-at-rest with the full FV3 toolkit +
d_con stack at default flag values (cv=False, duogrid=None,
vector_halo=False).  iter-330 covers the all-flags-ON case.  Neither
covers the INTERMEDIATE combinations:

* cv ON, duogrid OFF, vector_halo OFF
* cv OFF, duogrid ON, vector_halo OFF
* cv OFF, duogrid OFF, vector_halo ON
* cv ON, duogrid ON, vector_halo OFF
* cv ON, duogrid OFF, vector_halo ON
* cv OFF, duogrid ON, vector_halo ON

A future AD hazard could lurk in any of these combinations (e.g.,
the cv flag interacts with the vector halo at the rest state in a
way that the all-on iter-330 path masks via averaging or
cancellation).  iter-332 tests every combination explicitly so any
new AD hazard at the iter-320 / 325 / 328 interaction surface is
caught immediately.

Tests
-----

1. ``test_flag_combo_ad_at_rest`` — parametrized over 6 flag
   combinations × full toolkit, each must produce finite gradient
   under ``jax.grad`` at rest after 2 NH steps.
"""
from __future__ import annotations

from itertools import product

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


def _build_rest(grid):
    n = grid.n
    nlev = 5
    z_top = 30000.0
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    rest = NonHydrostaticState(
        u=Field(data=jnp.zeros((6, n, n, nlev)), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n, nlev)), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)),
                name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                        name="rho_prime",
                        dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return height_coord, terrain_metric, rest


def _toolkit_cfg(use_fv3_d_con_cv: bool, use_fv3_vector_halo_uv: bool):
    """Full FV3 damping toolkit + d_con stack, with the iter-320 +
    iter-328 flags varied."""
    return CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3, corner_div_damp_nord=1,
        damp_v=0.030, nord_v=1,
        use_fv3_a2b_zeta_corner=True,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        A_h=1e6, smagorinsky_cs=0.20,
        damp_w=0.030, nord_w=1,
        damp_w_d_con=1.0, damp_v_d_con=1.0, delt_max=1.0,
        corner_div_damp_d_con=1.0, div_damp_d_con=1.0, ah_d_con=1.0,
        use_fv3_d_con_cv=use_fv3_d_con_cv,
        use_fv3_vector_halo_uv=use_fv3_vector_halo_uv,
    )


# 6 intermediate combinations (excluding default-default per
# iter-184 and all-on per iter-330).
_FLAG_COMBOS = [
    (cv, duo, vec)
    for cv, duo, vec in product([False, True], repeat=3)
    if (cv, duo, vec) not in [(False, False, False), (True, True, True)]
]


@pytest.mark.parametrize("cv,duo,vec", _FLAG_COMBOS)
def test_flag_combo_ad_at_rest(cv, duo, vec):
    """Each iter-320/325/328 flag combination produces finite
    gradient at rest under the full toolkit."""
    grid = create_cubed_sphere(8, use_duogrid=duo)
    hc, tm, rest = _build_rest(grid)
    cfg = _toolkit_cfg(cv, vec)
    m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)

    def loss(amp):
        s = rest._replace(
            u=rest.u.replace(data=amp * jnp.ones_like(rest.u.data)),
        )
        for _ in range(2):
            s = m.step(s, 5.0)
        return jnp.mean(s.theta_prime.data ** 2)

    g = jax.grad(loss)(0.0)
    assert jnp.isfinite(g), (
        f"AD-at-rest FAILED for combination cv={cv}, "
        f"duogrid={duo}, vector_halo={vec}.  This indicates a "
        f"new AD hazard at the iter-320/325/328 interaction "
        f"surface — the all-on (iter-330) and default (iter-184) "
        f"paths masked it."
    )
