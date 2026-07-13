"""FV3_3D iter 446: shared FV3 sponge boost helper.

The iter-440 NH ``_apply_top_sponge_damp_boost`` was inline
in compressible_euler_cdgrid.py; PE iter-438/439 had a
DIFFERENT inline implementation of the same FV3 logic in
primitive_eq_cdgrid.py.  iter-446 factors both into a single
shared core helper ``legoesm.core.fv3_sponge_boost.
apply_top_sponge_damp_boost`` so NH and PE share one
implementation.

Verifies:
1. ``test_shared_helper_importable`` — module exists and the
   public name is exported.
2. ``test_shared_helper_baseline_noop`` — zero coefficients
   → identity (bit-for-bit).
3. ``test_shared_helper_k1_override`` — d2_bg_k1>0 overrides
   k=0 with ``da_min_c * max(d2_bg, d2_bg_k1)``.
4. ``test_shared_helper_k2_threshold_001`` — d2_bg_k2 below
   0.01 → no override.
5. ``test_shared_helper_k2_threshold_005`` — d2_bg_k2 above
   0.05 → both k=1 and k=2 overridden with 0.2 factor at k=2.
6. ``test_nh_pe_dycores_still_pass`` — sanity check that
   re-running 1 step at C8 with PE + NH still works after the
   refactor.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.fv3_sponge_boost import apply_top_sponge_damp_boost


def test_shared_helper_importable():
    assert callable(apply_top_sponge_damp_boost)


def test_shared_helper_baseline_noop():
    damp = jnp.ones((4, 5, 5, 5))
    out = apply_top_sponge_damp_boost(damp, 1.0, 0.0, 0.0, 0.0)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(damp))


def test_shared_helper_k1_override():
    damp = jnp.ones((4, 5, 5, 5))
    da_min_c = 2.0
    out = apply_top_sponge_damp_boost(damp, da_min_c, 0.0625, 4.0, 0.0)
    # k=0 should equal da_min_c * max(d2_bg, d2_bg_k1)
    expected_k0 = 2.0 * max(0.0625, 4.0)
    np.testing.assert_allclose(
        np.asarray(out[..., 0]), expected_k0,
        rtol=1e-14, atol=1e-14,
    )
    # k=1, 2, 3, 4 unchanged
    np.testing.assert_array_equal(
        np.asarray(out[..., 1:]), np.asarray(damp[..., 1:]),
    )


def test_shared_helper_k2_threshold_001():
    damp = jnp.ones((4, 5, 5, 5))
    # d2_bg_k2 = 0.005 below 0.01 threshold → no change
    out = apply_top_sponge_damp_boost(damp, 1.0, 0.0625, 0.0, 0.005)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(damp))


def test_shared_helper_k2_threshold_005():
    damp = jnp.ones((4, 5, 5, 5))
    da_min_c = 2.0
    d2_bg, d2_bg_k2 = 0.0625, 2.0
    out = apply_top_sponge_damp_boost(damp, da_min_c, d2_bg, 0.0, d2_bg_k2)
    # k=1 override: max(d2_bg, d2_bg_k2) = 2.0
    np.testing.assert_allclose(
        np.asarray(out[..., 1]), da_min_c * d2_bg_k2,
        rtol=1e-14, atol=1e-14,
    )
    # k=2 override: max(d2_bg, 0.2*d2_bg_k2) = 0.4
    np.testing.assert_allclose(
        np.asarray(out[..., 2]), da_min_c * 0.2 * d2_bg_k2,
        rtol=1e-14, atol=1e-14,
    )
    # k=0, k=3, k=4 unchanged
    np.testing.assert_array_equal(
        np.asarray(out[..., 0]), np.asarray(damp[..., 0]),
    )
    np.testing.assert_array_equal(
        np.asarray(out[..., 3:]), np.asarray(damp[..., 3:]),
    )


def test_nh_pe_dycores_still_pass():
    """Sanity: NH + PE 1-step at C8 with the shared helper
    still produces finite state."""
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
        CDGridCompressibleEulerModel,
        make_fv3_faithful_nh_config,
    )
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
        hydrostatic_to_fv3,
        make_fv3_faithful_pe_config,
    )
    from legoesm.atmosphere.held_suarez import held_suarez_init
    from legoesm.core.field import Field
    from legoesm.core.state import NonHydrostaticState
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import (
        create_cubed_sphere_cdgrid,
    )
    from legoesm.grids.vertical import (
        create_height_coordinate,
        compute_terrain_metric,
        standard_hybrid_levels,
    )
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=True)
    hc = create_height_coordinate(nlev, z_top)
    tm = compute_terrain_metric(jnp.zeros((6, n, n)), hc)
    rng = np.random.default_rng(seed=446)
    u_p = rng.uniform(-1.0, 1.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-1.0, 1.0, size=(6, n, n, nlev))
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state_nh = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v", dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)),
                name="w", dims=dims_w, units="m/s"),
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
    cfg_nh = make_fv3_faithful_nh_config(
        n_acoustic_substeps=4,
        damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    m_nh = CDGridCompressibleEulerModel(grid, hc, tm, cfg_nh)
    s_nh = m_nh.step(state_nh, dt=10.0)
    assert jnp.all(jnp.isfinite(s_nh.u.data))

    coord = standard_hybrid_levels(nlev)
    cdg = create_cubed_sphere_cdgrid(grid)
    state_pe = hydrostatic_to_fv3(held_suarez_init(grid, coord), cdg)
    cfg_pe = make_fv3_faithful_pe_config(
        damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
    )
    m_pe = CDGridPrimitiveEquationModel(grid, coord, cfg_pe)
    s_pe = m_pe.step(state_pe, dt=10.0)
    assert jnp.all(jnp.isfinite(s_pe.T.data))
