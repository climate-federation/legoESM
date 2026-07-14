"""FV3_3D iter 240: sponge-aware ``delt_max`` cap on the
aggregate tendency-based d_con stack (NH).

NH mirror of PE iter-239.  The 3 NH tendency-based d_con
contributions (iter-222 corner-div, iter-224 cell-centre
div_damp, iter-226 A_h) are now AGGREGATED and the cap applies
to the SUM, mirroring FV3 ``dyn_core.F90:1764-1779``.

NH cap policy (FV3 cv_air branch sw_core.F90:1782-1786, in θ_p
tendency form):

* k=0 (top): cap factor 0.1 → ``|dθ_p/dt * Π_ref| ≤ 0.1 * delt_max``
* k=1: cap factor 0.5
* k≥2: cap factor 1.0

Tests
-----

1. ``test_nh_d_con_aggregate_off_when_delt_max_zero`` —
   delt_max=0 (default) preserves baseline (no cap).
2. ``test_nh_d_con_aggregate_caps_interior_and_top`` — strong
   damping + small delt_max + check the cap is applied with
   the correct sponge factor at each layer.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerConfig,
    cdgrid_compressible_euler_slow_tendencies,
)
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import (
    compute_terrain_metric, create_height_coordinate,
)


@pytest.fixture(scope="module")
def small_nh_state_with_winds():
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    rng = np.random.default_rng(seed=240)
    u_p = rng.uniform(-30.0, 30.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-30.0, 30.0, size=(6, n, n, nlev))

    state = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)),
                name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime",
                          dims=dims_3d, units="K"),
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
    return grid, cdgrid, height_coord, terrain_metric, state


def test_nh_d_con_aggregate_off_when_delt_max_zero(small_nh_state_with_winds):
    """delt_max=0 preserves uncapped baseline."""
    grid, cdgrid, hc, tm, state = small_nh_state_with_winds
    common = dict(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        A_h=1e6, smagorinsky_cs=0.20,
    )
    cfg_explicit = CDGridCompressibleEulerConfig(
        **common,
        corner_div_damp_d_con=1.0,
        div_damp_d_con=1.0, ah_d_con=1.0,
        delt_max=0.0,
    )
    cfg_default = CDGridCompressibleEulerConfig(
        **common,
        corner_div_damp_d_con=1.0,
        div_damp_d_con=1.0, ah_d_con=1.0,
    )
    dt = 10.0
    t1 = cdgrid_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cdgrid, cfg_explicit, dt_actual=dt,
    )
    t2 = cdgrid_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cdgrid, cfg_default, dt_actual=dt,
    )
    np.testing.assert_array_equal(
        t1.dtheta_prime_dt.data, t2.dtheta_prime_dt.data,
    )


def test_nh_d_con_aggregate_caps_interior_and_top(small_nh_state_with_winds):
    """Sponge-aware cap on aggregate: k=0 → 0.1×, k=1 → 0.5×,
    k≥2 → 1× of ``delt_max / Π_ref``."""
    grid, cdgrid, hc, tm, state = small_nh_state_with_winds
    common = dict(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        A_h=1e6, smagorinsky_cs=0.20,
    )
    delt_max = 1e-7    # tight cap to force activation
    cfg_no_dcon = CDGridCompressibleEulerConfig(
        **common,
        corner_div_damp_d_con=0.0,
        div_damp_d_con=0.0, ah_d_con=0.0,
        delt_max=0.0,
    )
    cfg_capped = CDGridCompressibleEulerConfig(
        **common,
        corner_div_damp_d_con=1.0,
        div_damp_d_con=1.0, ah_d_con=1.0,
        delt_max=delt_max,
    )
    dt = 10.0
    t_no = cdgrid_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cdgrid, cfg_no_dcon, dt_actual=dt,
    )
    t_cap = cdgrid_compressible_euler_slow_tendencies(
        state, grid, hc, tm, cdgrid, cfg_capped, dt_actual=dt,
    )

    d_con = (
        t_cap.dtheta_prime_dt.data - t_no.dtheta_prime_dt.data
    )
    exner_ref = hc.exner_ref    # (nlev,)

    # Convert dθ_p/dt to T-equivalent: |dT_eq/dt| = |dθ_p/dt * Π|.
    # Cap in T-space = delt_max * sponge_factor.
    sponge = jnp.array([0.1, 0.5] + [1.0] * (exner_ref.shape[0] - 2))

    for k in range(exner_ref.shape[0]):
        max_at_k = float(jnp.max(jnp.abs(d_con[..., k])))
        # Cap in θ_p space at level k = sponge[k] * delt_max / Π_ref[k].
        cap_at_k = float(sponge[k] * delt_max / exner_ref[k])
        assert max_at_k <= cap_at_k + 1e-12, (
            f"NH d_con aggregate at k={k}: max|Δθ_p/dt|="
            f"{max_at_k:.4e} > cap={cap_at_k:.4e} (sponge "
            f"factor {float(sponge[k])}).  Aggregate sponge cap "
            f"is not bounding this layer."
        )
