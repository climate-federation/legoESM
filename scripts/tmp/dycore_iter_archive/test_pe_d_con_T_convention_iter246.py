"""FV3_3D iter 246: regression test for the PE T-convention
claim made in iter-208 / iter-238 / iter-246 audits.

iter-246 audit clarified that FV3 ``dyn_core.F90:1768`` divides
``heat_source`` by ``c_pd * delp * pkz`` because FV3's
prognostic ``pt`` is ``c_p*T/pkz`` (potential-temperature
scaled).  The legoESM PE prognostic T is the actual temperature
directly, so no pkz division is needed in our d_con formula.

This test pins down the convention: a known KE removal of
``ΔKE = X`` per cell should produce a temperature change of
exactly ``-d_con * X / c_pd`` K, NOT ``-d_con * X / (c_pd * pkz)``
or any other pkz-scaled variant.

If a future refactor accidentally introduces an Exner factor in
the PE d_con formula (e.g., copying the iter-207 NH pattern
where ``Π_ref`` IS used because NH stores θ_p), this test would
fail.

Tests
-----

1. ``test_pe_d_con_no_pkz_factor`` — bit-for-bit verify the
   per-cell heat tendency is ``-d_con * dKE / c_pd`` with NO
   pkz factor, by computing the formula independently and
   comparing.  Different from iter-228 in that this test
   builds the EXPECTED formula WITHOUT any Exner factor and
   asserts equality.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
    fv3_hydrostatic_tendencies,
    hydrostatic_to_fv3,
)
from legoesm.atmosphere.held_suarez import held_suarez_init
from legoesm.core.operators_cdgrid import interp_corner_to_center
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import standard_hybrid_levels


def test_pe_d_con_no_pkz_factor():
    """PE d_con formula uses ``-d_con * dKE / c_pd`` with NO
    pkz / Exner factor.  Verify by reconstructing the expected
    formula independently and comparing to the actual model
    output."""
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    rng = np.random.default_rng(seed=246)
    n_corners = n + 1
    u_p = rng.uniform(-30.0, 30.0,
                      size=(6, n_corners, n_corners, nlev))
    v_p = rng.uniform(-30.0, 30.0,
                      size=(6, n_corners, n_corners, nlev))
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )

    common = dict(
        A_h=0.0, hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        div_damp_coeff=0.0, damp_v=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    cfg_no_X = CDGridPrimitiveEquationConfig(
        **common,
        corner_div_damp_d2_bg=0.0,
        corner_div_damp_d_con=0.0,
    )
    cfg_X_off = CDGridPrimitiveEquationConfig(
        **common,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d_con=0.0,
    )
    cfg_X_on = CDGridPrimitiveEquationConfig(
        **common,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d_con=1.0,
    )

    dt = 100.0
    tend_no_X = fv3_hydrostatic_tendencies(
        state, grid, coord, cdgrid, cfg_no_X, dt_actual=dt,
    )
    tend_X_off = fv3_hydrostatic_tendencies(
        state, grid, coord, cdgrid, cfg_X_off, dt_actual=dt,
    )
    tend_X_on = fv3_hydrostatic_tendencies(
        state, grid, coord, cdgrid, cfg_X_on, dt_actual=dt,
    )

    du_d_dt_X = tend_X_off.du_d_dt.data - tend_no_X.du_d_dt.data
    dv_d_dt_X = tend_X_off.dv_d_dt.data - tend_no_X.dv_d_dt.data
    dT_dt_d_con = tend_X_on.dT_dt.data - tend_X_off.dT_dt.data

    # Build the expected formula WITHOUT any pkz/Exner factor.
    u_d = state.u_d.data
    v_d = state.v_d.data
    dKE_dt_corner = u_d * du_d_dt_X + v_d * dv_d_dt_X
    dKE_dt_cc = interp_corner_to_center(dKE_dt_corner)
    expected_no_pkz = -1.0 * dKE_dt_cc / constants.c_pd

    np.testing.assert_allclose(
        dT_dt_d_con, expected_no_pkz,
        rtol=1e-10, atol=1e-12,
        err_msg=(
            "PE d_con formula uses -d_con * dKE / c_pd with NO "
            "pkz factor (per iter-246 audit: FV3's pkz factor "
            "is for FV3's pt = c_p*T/pkz convention; legoESM PE "
            "T is actual temperature directly).  If this fails, "
            "check that no Exner / pkz / Π_ref factor was "
            "accidentally introduced into the PE d_con formula."
        ),
    )

    # Sanity: verify that an Exner-SCALED variant would NOT match
    # (so the test is non-vacuous if pkz_ref ≈ 1).
    # Use a synthetic pkz that varies significantly with level.
    fake_pkz = jnp.array(
        [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )[None, None, None, :]    # (1, 1, 1, nlev)
    expected_with_pkz = expected_no_pkz / fake_pkz

    # Should NOT match expected_with_pkz (since we don't divide
    # by pkz in PE).
    diff = float(jnp.max(jnp.abs(dT_dt_d_con - expected_with_pkz)))
    assert diff > 1e-6, (
        f"PE d_con accidentally matches a pkz-scaled formula "
        f"(diff={diff:.4e}); verify the test isn't vacuous."
    )
