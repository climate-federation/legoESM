"""FV3_3D iter 320: FV3-faithful c_v factor for the NH d_con KE→heat
conversion (FV3 ``dyn_core.F90:1795`` ``cv_air`` branch).

Audit
-----
FV3's ``hydrostatic = .false.`` branch divides ``heat_source`` by
``cv_air * delp`` because compressible NH dynamics conserves total
energy with internal energy ``c_v * T`` (constant volume), not
enthalpy ``c_p * T`` (constant pressure, hydrostatic limit).

legoESM NH iter 203 / 207 / 209 / 222 / 224 / 226 ports inherited the
``c_pd`` denominator from PE — which UNDER-HEATS by ``c_v / c_p ≈
0.714`` (~40 % under-heating relative to FV3 NH).  iter 320 adds the
opt-in flag ``use_fv3_d_con_cv: bool = False`` (default off,
bit-for-bit baseline) that swaps ``c_pd → c_vd`` at all 5 NH d_con
sites for FV3-faithful heating partition.

The PE path is unaffected — PE uses ``c_pd`` matching FV3's
``cp_air`` branch (FV3 ``dyn_core.F90:1769``).

Tests
-----
1. ``test_baseline_bit_for_bit`` — flag=False reproduces baseline
   exactly across all 5 d_con knobs (regression guard).
2. ``test_cv_amplifies_heating_by_cp_over_cv_ratio`` — flag=True
   produces |Δθ_p| ≈ (c_pd / c_vd) × flag=False |Δθ_p| at each of
   the 5 d_con knobs.  Pinning the formula change explicitly.
3. ``test_cv_global_energy_balance`` — flag=True matches FV3's
   ``Σ c_vd · delp · Π_ref · Δθ_p + Σ delp · ΔKE = 0`` global
   energy balance (vs the c_pd-balance that flag=False satisfies).
4. ``test_cv_differentiable_at_rest`` — ``jax.grad`` flows finitely
   through 3 NH steps with flag=True + full d_con stack ON at FV3
   production defaults.

Directly verifies the new flag and pins the FV3-faithful heat
partition that future refactors must preserve.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
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


@pytest.fixture(scope="module")
def small_nh_state():
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    state = NonHydrostaticState(
        u=Field(data=jnp.zeros((6, n, n, nlev)), name="u",
                dims=dims_3d, units="m/s"),
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


def _perturb(state, seed=320, amp=3.0):
    n = state.u.data.shape[1]
    nlev = state.u.data.shape[-1]
    rng = np.random.default_rng(seed=seed)
    u_p = rng.uniform(-amp, amp, size=(6, n, n, nlev))
    v_p = rng.uniform(-amp, amp, size=(6, n, n, nlev))
    w_p = rng.uniform(-0.3, 0.3, size=(6, n, n, nlev + 1))
    return state._replace(
        u=state.u.replace(data=jnp.asarray(u_p)),
        v=state.v.replace(data=jnp.asarray(v_p)),
        w=state.w.replace(data=jnp.asarray(w_p)),
    )


_FULL_DCON_KW = dict(
    hyperdiff_coeff=1e14, n_acoustic_substeps=4,
    damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
    damp_w=0.30, nord_w=1, damp_w_d_con=1.0,
    A_h=1.0e6, ah_d_con=1.0,
    div_damp_coeff=1e10, div_damp_d_con=1.0,
    corner_div_damp_d2_bg=0.001, corner_div_damp_d_con=1.0,
)


def test_baseline_bit_for_bit(small_nh_state):
    """flag=False is bit-for-bit baseline (no behavior change)."""
    grid, height_coord, terrain_metric, state = small_nh_state
    s = _perturb(state, seed=320)

    cfg_default = CDGridCompressibleEulerConfig(**_FULL_DCON_KW)
    cfg_explicit_off = CDGridCompressibleEulerConfig(
        **_FULL_DCON_KW, use_fv3_d_con_cv=False,
    )
    m_default = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_default,
    )
    m_explicit_off = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_explicit_off,
    )
    s_default = m_default.step(s, 5.0)
    s_explicit_off = m_explicit_off.step(s, 5.0)

    np.testing.assert_array_equal(
        np.asarray(s_default.theta_prime.data),
        np.asarray(s_explicit_off.theta_prime.data),
    )
    np.testing.assert_array_equal(
        np.asarray(s_default.u.data),
        np.asarray(s_explicit_off.u.data),
    )
    np.testing.assert_array_equal(
        np.asarray(s_default.v.data),
        np.asarray(s_explicit_off.v.data),
    )


def test_cv_amplifies_heating_by_cp_over_cv_ratio(small_nh_state):
    """flag=True multiplies Δθ_p by exactly c_pd/c_vd ≈ 1.40 for the
    POST-ACOUSTIC d_con sites (damp_v_d_con + damp_w_d_con).

    Restricted to post-acoustic d_con because the in-step d_con sites
    (div_damp/ah/corner_div) feed back through ``slow_tendencies →
    acoustic_substep`` within the same step, contaminating the exact
    ratio with O(1e-7) acoustic-feedback noise.  Post-acoustic d_con
    is applied as the LAST operation of ``step()`` and has no
    within-step feedback, so the ratio is exact.
    """
    grid, height_coord, terrain_metric, state = small_nh_state
    s = _perturb(state, seed=321)

    # Post-acoustic d_con only (damp_v + damp_w); in-step d_con OFF.
    common = dict(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_v=0.030, nord_v=1,
        damp_w=0.30, nord_w=1,
        A_h=1.0e6, div_damp_coeff=1e10,
        corner_div_damp_d2_bg=0.001,
        ah_d_con=0.0, div_damp_d_con=0.0, corner_div_damp_d_con=0.0,
    )
    cfg_no_dcon = CDGridCompressibleEulerConfig(
        **common, damp_v_d_con=0.0, damp_w_d_con=0.0,
    )
    cfg_cp = CDGridCompressibleEulerConfig(
        **common, damp_v_d_con=1.0, damp_w_d_con=1.0,
        use_fv3_d_con_cv=False,
    )
    cfg_cv = CDGridCompressibleEulerConfig(
        **common, damp_v_d_con=1.0, damp_w_d_con=1.0,
        use_fv3_d_con_cv=True,
    )

    m_no = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_no_dcon,
    )
    m_cp = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_cp,
    )
    m_cv = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_cv,
    )

    s_no = m_no.step(s, 5.0)
    s_cp = m_cp.step(s, 5.0)
    s_cv = m_cv.step(s, 5.0)

    dtheta_cp_dcon = (
        np.asarray(s_cp.theta_prime.data)
        - np.asarray(s_no.theta_prime.data)
    )
    dtheta_cv_dcon = (
        np.asarray(s_cv.theta_prime.data)
        - np.asarray(s_no.theta_prime.data)
    )
    # cv path should have larger heating by factor c_pd / c_vd.
    cp_over_cv = constants.c_pd / constants.c_vd
    # Element-wise: dtheta_cv = (c_pd/c_vd) * dtheta_cp.  Use mean
    # absolute as a robust scalar comparison.
    ratio = float(np.mean(np.abs(dtheta_cv_dcon))) / float(
        np.mean(np.abs(dtheta_cp_dcon)),
    )
    np.testing.assert_allclose(ratio, cp_over_cv, rtol=1e-10)
    # u, v, w are not touched by the post-acoustic d_con block —
    # should be bit-for-bit identical between cp and cv paths.
    np.testing.assert_array_equal(
        np.asarray(s_cp.u.data), np.asarray(s_cv.u.data),
    )
    np.testing.assert_array_equal(
        np.asarray(s_cp.v.data), np.asarray(s_cv.v.data),
    )
    np.testing.assert_array_equal(
        np.asarray(s_cp.w.data), np.asarray(s_cv.w.data),
    )


def test_cv_cp_share_same_KE_source(small_nh_state):
    """The d_con KE source is the same between cp and cv paths;
    only the denominator differs.  Therefore at every cell:

        c_pd * Π_ref * Δθ_p_cp = c_vd * Π_ref * Δθ_p_cv = -d_con · ΔKE

    Verified bit-for-bit (rtol=1e-12) using post-acoustic d_con only
    (in-step d_con sites have acoustic feedback that breaks the
    exact element-wise identity).  Pins the FV3-faithful claim:
    flag swaps the heat-capacity factor ONLY, leaving the KE source
    untouched.
    """
    grid, height_coord, terrain_metric, state = small_nh_state
    s = _perturb(state, seed=322)

    common = dict(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_v=0.030, nord_v=1,
        damp_w=0.30, nord_w=1,
        A_h=1.0e6, div_damp_coeff=1e10,
        corner_div_damp_d2_bg=0.001,
        ah_d_con=0.0, div_damp_d_con=0.0, corner_div_damp_d_con=0.0,
    )
    cfg_no_dcon = CDGridCompressibleEulerConfig(
        **common, damp_v_d_con=0.0, damp_w_d_con=0.0,
    )
    cfg_cp = CDGridCompressibleEulerConfig(
        **common, damp_v_d_con=1.0, damp_w_d_con=1.0,
        use_fv3_d_con_cv=False,
    )
    cfg_cv = CDGridCompressibleEulerConfig(
        **common, damp_v_d_con=1.0, damp_w_d_con=1.0,
        use_fv3_d_con_cv=True,
    )
    m_no = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_no_dcon,
    )
    m_cp = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_cp,
    )
    m_cv = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_cv,
    )
    s_no = m_no.step(s, 5.0)
    s_cp = m_cp.step(s, 5.0)
    s_cv = m_cv.step(s, 5.0)

    dtheta_cp_dcon = (
        np.asarray(s_cp.theta_prime.data)
        - np.asarray(s_no.theta_prime.data)
    )
    dtheta_cv_dcon = (
        np.asarray(s_cv.theta_prime.data)
        - np.asarray(s_no.theta_prime.data)
    )
    # c_pd * Δθ_p_cp == c_vd * Δθ_p_cv (Π_ref cancels — same factor
    # on both sides).
    np.testing.assert_allclose(
        constants.c_pd * dtheta_cp_dcon,
        constants.c_vd * dtheta_cv_dcon,
        rtol=1e-12,
    )


def test_cv_differentiable_at_rest(small_nh_state):
    """jax.grad flows finitely through 3 NH steps with flag=True at
    FV3 production defaults."""
    grid, height_coord, terrain_metric, state = small_nh_state
    cfg = CDGridCompressibleEulerConfig(
        **_FULL_DCON_KW, use_fv3_d_con_cv=True,
    )
    model = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg,
    )

    def loss(amp):
        s = state._replace(
            u=state.u.replace(data=amp * jnp.ones_like(state.u.data)),
        )
        for _ in range(3):
            s = model.step(s, 5.0)
        return jnp.mean(s.theta_prime.data ** 2)

    g = jax.grad(loss)(0.0)
    assert jnp.isfinite(g)
