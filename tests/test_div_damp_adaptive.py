"""Unit tests for FV3-faithful adaptive Smagorinsky-style divergence damping
in the 3D atmospheric C-D grid primitive-equation tendency function.

Faithful port of the formula at FV3 ``sw_core.F90:1720``::

    damp = da_min_c * max(d2_bg, min(0.20, dddmp * abs(div)))

where ``d2_bg = div_damp_coeff / da_min_c`` is the dimensionless
background coefficient.  When ``dddmp > 0``, divergence damping
becomes ADAPTIVE (stronger where local divergence is large), which
in FV3 is intended to suppress halo-amplified spurious divergence at
panel-boundary cells.

Tests
-----

1. ``dddmp = 0`` (default) — bit-for-bit equivalent to the previous
   constant-coefficient path (regression guard for the iter-5 wiring).
2. ``dddmp > 0`` with a large value — adaptive coefficient kicks in,
   tendency MAGNITUDES change in regions of large divergence.
3. ``dddmp > 0`` with a tiny value — adaptive part stays below the
   ``d2_bg`` floor, tendencies match the constant-coefficient path.

The 3D path lives in
``legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid``; the SW analog is
in ``legoesm.core.operators_cdgrid.cdgrid_momentum_tendencies`` (the
existing tested SW path).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
    fv3_hydrostatic_tendencies,
    hydrostatic_to_fv3,
)
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import standard_hybrid_levels


@pytest.fixture(scope="module")
def small_3d_state():
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)
    return grid, cdgrid, coord, state


def test_dddmp_zero_is_bit_for_bit_constant_path(small_3d_state):
    """``dddmp = 0`` (default) must match the existing constant path."""
    grid, cdgrid, coord, state = small_3d_state

    cfg_default = CDGridPrimitiveEquationConfig(
        div_damp_coeff=1e7, div_damp_dddmp=0.0,
    )
    # Use a config with ``div_damp_dddmp`` not set (defaults to 0.0).
    cfg_no_field = CDGridPrimitiveEquationConfig(div_damp_coeff=1e7)

    t_default = fv3_hydrostatic_tendencies(
        state, grid, coord, cdgrid, cfg_default,
    )
    t_no_field = fv3_hydrostatic_tendencies(
        state, grid, coord, cdgrid, cfg_no_field,
    )

    np.testing.assert_array_equal(
        np.asarray(t_default.du_d_dt.data),
        np.asarray(t_no_field.du_d_dt.data),
    )
    np.testing.assert_array_equal(
        np.asarray(t_default.dv_d_dt.data),
        np.asarray(t_no_field.dv_d_dt.data),
    )


def test_tiny_dddmp_below_floor_matches_constant_path(small_3d_state):
    """When ``dddmp * |div|`` is below the d2_bg floor everywhere, the
    adaptive coefficient = d2_bg * da_min_c = div_damp_coeff, so the
    adaptive path produces IDENTICAL tendencies to the constant path."""
    grid, cdgrid, coord, state = small_3d_state

    div_damp_coeff = 1e7

    cfg_const = CDGridPrimitiveEquationConfig(
        div_damp_coeff=div_damp_coeff, div_damp_dddmp=0.0,
    )
    # Use a tiny dddmp so the adaptive part is always below the floor
    # for any plausible divergence magnitude.
    cfg_tiny = CDGridPrimitiveEquationConfig(
        div_damp_coeff=div_damp_coeff, div_damp_dddmp=1e-30,
    )

    t_const = fv3_hydrostatic_tendencies(state, grid, coord, cdgrid, cfg_const)
    t_tiny = fv3_hydrostatic_tendencies(state, grid, coord, cdgrid, cfg_tiny)

    rtol = 1e-12
    np.testing.assert_allclose(
        np.asarray(t_const.du_d_dt.data),
        np.asarray(t_tiny.du_d_dt.data),
        rtol=rtol,
    )
    np.testing.assert_allclose(
        np.asarray(t_const.dv_d_dt.data),
        np.asarray(t_tiny.dv_d_dt.data),
        rtol=rtol,
    )


def test_huge_dddmp_changes_tendencies(small_3d_state):
    """A very large ``dddmp`` triggers the adaptive cap (0.20 * da_min_c)
    on cells with non-zero divergence; tendencies must DIFFER from the
    constant-coefficient path.

    The HS init has zero winds (zero divergence), so we manually
    perturb the state to a non-zero-divergence configuration before
    comparing.
    """
    grid, cdgrid, coord, state = small_3d_state

    # Inject a divergence pattern: small east-west wind variation
    # creates non-zero ``cgrid_divergence`` at cell centres.  Use a
    # spatially-varying perturbation so the adaptive cap kicks in
    # somewhere.
    n = grid.n
    nlev = state.u_d.data.shape[-1]
    rng = np.random.default_rng(seed=11)
    u_perturb = rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1, nlev))
    v_perturb = rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1, nlev))
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_perturb)),
        v_d=state.v_d.replace(data=jnp.asarray(v_perturb)),
    )

    div_damp_coeff = 1e7

    cfg_const = CDGridPrimitiveEquationConfig(
        div_damp_coeff=div_damp_coeff, div_damp_dddmp=0.0,
    )
    cfg_huge = CDGridPrimitiveEquationConfig(
        div_damp_coeff=div_damp_coeff, div_damp_dddmp=1e10,
    )

    t_const = fv3_hydrostatic_tendencies(state, grid, coord, cdgrid, cfg_const)
    t_huge = fv3_hydrostatic_tendencies(state, grid, coord, cdgrid, cfg_huge)

    diff_u = float(jnp.max(jnp.abs(t_const.du_d_dt.data - t_huge.du_d_dt.data)))
    base_u = float(jnp.max(jnp.abs(t_const.du_d_dt.data)))
    # With dddmp = 1e10 the cap (0.20) is hit on every non-zero |div|
    # cell, raising the effective coefficient to 0.20 * da_min_c.
    # For C8 that's ~3e9 → much larger than div_damp_coeff = 1e7.
    # Tendencies must differ by a non-trivial fraction of the base.
    assert diff_u > 0.1 * base_u, (
        f"Adaptive damping with dddmp=1e10 should noticeably change "
        f"tendencies on a non-zero-divergence state; got |Δdu| = "
        f"{diff_u:.3e}, |t_const| = {base_u:.3e}"
    )

    # Stability: tendencies must remain finite.
    assert jnp.all(jnp.isfinite(t_huge.du_d_dt.data))
    assert jnp.all(jnp.isfinite(t_huge.dv_d_dt.data))


def test_adaptive_damping_preserves_stability_short_run(small_3d_state):
    """Default-tuned dddmp (FV3 0.20) preserves stability over 10 steps."""
    grid, cdgrid, coord, _ = small_3d_state

    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
    )

    cfg = CDGridPrimitiveEquationConfig(
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        use_conservation_fixer=False, fix_mass=False,
    )
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)

    # Use HS init for a non-trivial 3D state.
    s = held_suarez_init(grid, coord)
    s = hydrostatic_to_fv3(s, cdgrid)
    for _ in range(10):
        s = model.step(s, 200.0)

    assert jnp.all(jnp.isfinite(s.u_d.data)), "u_d NaN with dddmp=0.20"
    assert jnp.all(jnp.isfinite(s.v_d.data)), "v_d NaN with dddmp=0.20"
    assert jnp.all(jnp.isfinite(s.T.data)), "T NaN with dddmp=0.20"
    assert jnp.all(jnp.isfinite(s.p_s.data)), "p_s NaN with dddmp=0.20"


# --- iter 12: post-step vorticity damping (fv3_del6_vorticity_damping
# reuse from SW backbone) ---


def test_damp_v_zero_is_bit_for_bit_baseline(small_3d_state):
    """damp_v=0 preserves bit-for-bit existing 3D dycore behaviour."""
    grid, cdgrid, coord, _ = small_3d_state

    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
    )

    cfg_zero = CDGridPrimitiveEquationConfig(
        div_damp_coeff=1e7, damp_v=0.0,
    )
    cfg_unset = CDGridPrimitiveEquationConfig(div_damp_coeff=1e7)

    s = held_suarez_init(grid, coord)
    s = hydrostatic_to_fv3(s, cdgrid)

    model_zero = CDGridPrimitiveEquationModel(grid, coord, cfg_zero)
    model_unset = CDGridPrimitiveEquationModel(grid, coord, cfg_unset)

    s_zero = s
    s_unset = s
    for _ in range(5):
        s_zero = model_zero.step(s_zero, 200.0)
        s_unset = model_unset.step(s_unset, 200.0)

    np.testing.assert_array_equal(
        np.asarray(s_zero.u_d.data), np.asarray(s_unset.u_d.data),
    )
    np.testing.assert_array_equal(
        np.asarray(s_zero.v_d.data), np.asarray(s_unset.v_d.data),
    )


def test_damp_v_changes_winds_on_perturbed_state(small_3d_state):
    """damp_v > 0 must produce DIFFERENT winds vs damp_v = 0 after one
    step on a perturbed state."""
    grid, cdgrid, coord, _ = small_3d_state

    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
    )

    s = held_suarez_init(grid, coord)
    s = hydrostatic_to_fv3(s, cdgrid)
    # Perturb winds to non-trivial values so the post-step damping
    # has a non-zero target.
    n = grid.n
    nlev = s.u_d.data.shape[-1]
    rng = np.random.default_rng(seed=21)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1, nlev))
    s = s._replace(
        u_d=s.u_d.replace(data=jnp.asarray(u_p)),
        v_d=s.v_d.replace(data=jnp.asarray(v_p)),
    )

    cfg_zero = CDGridPrimitiveEquationConfig(
        div_damp_coeff=1e7, damp_v=0.0,
    )
    cfg_strong = CDGridPrimitiveEquationConfig(
        div_damp_coeff=1e7, damp_v=0.3,
    )

    model_zero = CDGridPrimitiveEquationModel(grid, coord, cfg_zero)
    model_strong = CDGridPrimitiveEquationModel(grid, coord, cfg_strong)

    s_zero = model_zero.step(s, 100.0)
    s_strong = model_strong.step(s, 100.0)

    diff_u = float(jnp.max(jnp.abs(s_zero.u_d.data - s_strong.u_d.data)))
    base_u = float(jnp.max(jnp.abs(s_zero.u_d.data)))
    assert diff_u > 1e-6 * base_u, (
        f"damp_v=0.3 must change u_d on perturbed state; got "
        f"|Δu|/|u| = {diff_u/(base_u+1e-30):.3e}"
    )
    assert jnp.all(jnp.isfinite(s_strong.u_d.data))
    assert jnp.all(jnp.isfinite(s_strong.v_d.data))


def test_damp_v_stable_short_run(small_3d_state):
    """damp_v at the iter-1009 SW value (0.030) preserves stability for
    20 steps on the HS init."""
    grid, cdgrid, coord, _ = small_3d_state

    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
    )

    cfg = CDGridPrimitiveEquationConfig(
        div_damp_coeff=1e6, damp_v=0.030, nord_v=2,
        use_conservation_fixer=False, fix_mass=False,
    )
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)

    s = held_suarez_init(grid, coord)
    s = hydrostatic_to_fv3(s, cdgrid)
    for _ in range(20):
        s = model.step(s, 200.0)

    assert jnp.all(jnp.isfinite(s.u_d.data)), "damp_v=0.030 NaN"
    assert jnp.all(jnp.isfinite(s.v_d.data))
    assert jnp.all(jnp.isfinite(s.T.data))
    assert jnp.all(jnp.isfinite(s.p_s.data))


# --- iter 16: corner-divergence adaptive damping (FV3 d_sw5 sw_core.F90:1720) ---


def test_corner_div_damp_zero_is_baseline(small_3d_state):
    """corner_div_damp_d2_bg=0 preserves bit-for-bit baseline."""
    grid, cdgrid, coord, _ = small_3d_state

    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
    )

    cfg_zero = CDGridPrimitiveEquationConfig(
        div_damp_coeff=1e7, corner_div_damp_d2_bg=0.0,
    )
    cfg_unset = CDGridPrimitiveEquationConfig(div_damp_coeff=1e7)

    s = held_suarez_init(grid, coord)
    s = hydrostatic_to_fv3(s, cdgrid)

    model_zero = CDGridPrimitiveEquationModel(grid, coord, cfg_zero)
    model_unset = CDGridPrimitiveEquationModel(grid, coord, cfg_unset)

    s_zero = s
    s_unset = s
    for _ in range(5):
        s_zero = model_zero.step(s_zero, 200.0)
        s_unset = model_unset.step(s_unset, 200.0)

    np.testing.assert_array_equal(
        np.asarray(s_zero.u_d.data), np.asarray(s_unset.u_d.data),
    )


def test_corner_div_damp_changes_winds(small_3d_state):
    """corner_div_damp_d2_bg > 0 changes winds on a perturbed state."""
    grid, cdgrid, coord, _ = small_3d_state

    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
    )

    s = held_suarez_init(grid, coord)
    s = hydrostatic_to_fv3(s, cdgrid)
    n = grid.n
    nlev = s.u_d.data.shape[-1]
    rng = np.random.default_rng(seed=33)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1, nlev))
    s = s._replace(
        u_d=s.u_d.replace(data=jnp.asarray(u_p)),
        v_d=s.v_d.replace(data=jnp.asarray(v_p)),
    )

    cfg_zero = CDGridPrimitiveEquationConfig(
        div_damp_coeff=1e7, corner_div_damp_d2_bg=0.0,
    )
    cfg_active = CDGridPrimitiveEquationConfig(
        div_damp_coeff=1e7, corner_div_damp_d2_bg=0.001,
        corner_div_damp_dddmp=0.20,
    )

    model_zero = CDGridPrimitiveEquationModel(grid, coord, cfg_zero)
    model_active = CDGridPrimitiveEquationModel(grid, coord, cfg_active)

    s_zero = model_zero.step(s, 100.0)
    s_active = model_active.step(s, 100.0)

    diff = float(jnp.max(jnp.abs(s_zero.u_d.data - s_active.u_d.data)))
    base = float(jnp.max(jnp.abs(s_zero.u_d.data)))
    assert diff > 1e-6 * base, "corner_div_damp must change winds"
    assert jnp.all(jnp.isfinite(s_active.u_d.data))


def test_corner_div_damp_del4_active_without_d2(small_3d_state):
    """PE mirror of the NH regression: the del-4 pair (d4_bg>0,
    nord>0) activates corner damping WITHOUT d2_bg (FV3 sw_core.F90:1641
    has no d2_bg master switch).  RED under the old ``d2_bg > 0``
    gate (bit-identical to inert)."""
    grid, cdgrid, coord, _ = small_3d_state

    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
    )

    s = held_suarez_init(grid, coord)
    s = hydrostatic_to_fv3(s, cdgrid)
    n = grid.n
    nlev = s.u_d.data.shape[-1]
    rng = np.random.default_rng(seed=34)
    s = s._replace(
        u_d=s.u_d.replace(data=jnp.asarray(
            rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1, nlev)))),
        v_d=s.v_d.replace(data=jnp.asarray(
            rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1, nlev)))),
    )

    cfg_inert = CDGridPrimitiveEquationConfig(div_damp_coeff=1e7)
    cfg_del4 = CDGridPrimitiveEquationConfig(
        div_damp_coeff=1e7,
        corner_div_damp_d2_bg=0.0,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=0.16,
        corner_div_damp_nord=1,
    )

    model_inert = CDGridPrimitiveEquationModel(grid, coord, cfg_inert)
    model_del4 = CDGridPrimitiveEquationModel(grid, coord, cfg_del4)

    s_inert = model_inert.step(s, 100.0)
    s_del4 = model_del4.step(s, 100.0)

    diff = float(jnp.max(jnp.abs(s_inert.u_d.data - s_del4.u_d.data)))
    base = float(jnp.max(jnp.abs(s_inert.u_d.data)))
    assert diff > 1e-6 * base, (
        "del-4 corner damping (d2_bg=0, d4_bg=0.16, nord=1) must be "
        "ACTIVE on the PE path — bit-identical means the d2_bg master "
        f"gate is back (diff={diff:.3e}, base={base:.3e})"
    )
    assert jnp.all(jnp.isfinite(s_del4.u_d.data))


def test_corner_div_damp_stable_short_run(small_3d_state):
    """corner_div_damp_d2_bg=0.001 stable for 20 steps from HS init."""
    grid, cdgrid, coord, _ = small_3d_state

    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
    )

    cfg = CDGridPrimitiveEquationConfig(
        div_damp_coeff=1e6,
        corner_div_damp_d2_bg=0.001, corner_div_damp_dddmp=0.20,
        use_conservation_fixer=False, fix_mass=False,
    )
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)

    s = held_suarez_init(grid, coord)
    s = hydrostatic_to_fv3(s, cdgrid)
    for _ in range(20):
        s = model.step(s, 200.0)

    assert jnp.all(jnp.isfinite(s.u_d.data))
    assert jnp.all(jnp.isfinite(s.v_d.data))


# --- iter 18: higher-order corner-divergence damping (FV3 d_sw5 nord>0) ---


def test_corner_div_damp_d4_disabled_bit_for_bit_with_d2(small_3d_state):
    """When d4_bg=0 OR nord=0, iter-18 is bit-for-bit identical to iter-16.

    The higher-order block in ``primitive_eq_cdgrid.py`` is gated by a
    Python-static ``d4_bg > 0 AND nord > 0`` test, so disabling either
    config knob skips the new code path entirely.  This guards against
    any accidental fall-through that would change the iter-16 baseline.
    """
    grid, cdgrid, coord, _ = small_3d_state

    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
    )

    s = held_suarez_init(grid, coord)
    s = hydrostatic_to_fv3(s, cdgrid)

    # Baseline: iter-16 del-2 only.
    cfg_d2_only = CDGridPrimitiveEquationConfig(
        div_damp_coeff=1e7,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
    )
    # nord=0: gate is False (d4_bg defaults to 0 too).
    cfg_nord0 = CDGridPrimitiveEquationConfig(
        div_damp_coeff=1e7,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=0.16,    # set, but nord=0 disables
        corner_div_damp_nord=0,
    )
    # d4_bg=0: gate is False (nord set, but d4_bg disables).
    cfg_d4_zero = CDGridPrimitiveEquationConfig(
        div_damp_coeff=1e7,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=0.0,
        corner_div_damp_nord=2,
    )

    m_base = CDGridPrimitiveEquationModel(grid, coord, cfg_d2_only)
    m_n0 = CDGridPrimitiveEquationModel(grid, coord, cfg_nord0)
    m_d40 = CDGridPrimitiveEquationModel(grid, coord, cfg_d4_zero)

    s_base, s_n0, s_d40 = s, s, s
    for _ in range(3):
        s_base = m_base.step(s_base, 200.0)
        s_n0 = m_n0.step(s_n0, 200.0)
        s_d40 = m_d40.step(s_d40, 200.0)

    np.testing.assert_array_equal(
        np.asarray(s_base.u_d.data), np.asarray(s_n0.u_d.data),
    )
    np.testing.assert_array_equal(
        np.asarray(s_base.u_d.data), np.asarray(s_d40.u_d.data),
    )
    np.testing.assert_array_equal(
        np.asarray(s_base.v_d.data), np.asarray(s_n0.v_d.data),
    )


def test_corner_div_damp_d4_changes_winds(small_3d_state):
    """Active higher-order damping (nord=1, d4_bg=1e-3) changes winds.

    Note: ``dd8 = (da_min_c * d4_bg)^(nord+1)`` scales super-linearly
    with ``da_min_c``, which is large at the n=8 test resolution (~1.25e12
    m^2).  FV3 production uses ``d4_bg=0.16`` at C96 (n=96 with much
    smaller da_min_c).  At n=8 we need ``d4_bg <= 1e-2`` for stability;
    the ratio of (da_min @ n=8) to (da_min @ C96) is ~ (96/8)^2 ~ 144,
    so ``0.16/144`` ~ 1e-3 is the unit-equivalent test coefficient.
    """
    grid, cdgrid, coord, _ = small_3d_state

    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
    )

    s = held_suarez_init(grid, coord)
    s = hydrostatic_to_fv3(s, cdgrid)
    n = grid.n
    nlev = s.u_d.data.shape[-1]
    rng = np.random.default_rng(seed=181)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1, nlev))
    s = s._replace(
        u_d=s.u_d.replace(data=jnp.asarray(u_p)),
        v_d=s.v_d.replace(data=jnp.asarray(v_p)),
    )

    cfg_d2_only = CDGridPrimitiveEquationConfig(
        div_damp_coeff=1e7,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
    )
    cfg_d4_active = CDGridPrimitiveEquationConfig(
        div_damp_coeff=1e7,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=1,
    )

    m_d2 = CDGridPrimitiveEquationModel(grid, coord, cfg_d2_only)
    m_d4 = CDGridPrimitiveEquationModel(grid, coord, cfg_d4_active)

    s_d2 = m_d2.step(s, 100.0)
    s_d4 = m_d4.step(s, 100.0)

    diff = float(jnp.max(jnp.abs(s_d2.u_d.data - s_d4.u_d.data)))
    base = float(jnp.max(jnp.abs(s_d2.u_d.data)))
    assert diff > 1e-6 * base, "d4_bg>0 + nord=1 must change winds"
    assert jnp.all(jnp.isfinite(s_d4.u_d.data))
    assert jnp.all(jnp.isfinite(s_d4.v_d.data))


def test_corner_div_damp_d4_stable_short_run_nord1(small_3d_state):
    """nord=1, d4_bg=1e-3 is finite for 20 steps from HS init at n=8.

    See ``test_corner_div_damp_d4_changes_winds`` for the rationale on
    the n=8-equivalent ``d4_bg`` value vs FV3 production C96 default.
    """
    grid, cdgrid, coord, _ = small_3d_state

    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
    )

    cfg = CDGridPrimitiveEquationConfig(
        div_damp_coeff=1e6,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=1,
        use_conservation_fixer=False,
        fix_mass=False,
    )
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)

    s = held_suarez_init(grid, coord)
    s = hydrostatic_to_fv3(s, cdgrid)
    for _ in range(20):
        s = model.step(s, 200.0)

    assert jnp.all(jnp.isfinite(s.u_d.data))
    assert jnp.all(jnp.isfinite(s.v_d.data))
    assert jnp.all(jnp.isfinite(s.T.data))
    assert jnp.all(jnp.isfinite(s.p_s.data))


def test_corner_div_damp_fv3_vector_fill_bit_for_bit_nord1(small_3d_state):
    """iter-22: ``corner_div_damp_fv3_vector_fill = True`` is a
    mathematical no-op at nord=1 — end-to-end integration matches
    iter-18 nord=1 (vector fill OFF) within 1-ULP tolerance.

    Unit-level bit-for-bit confirmed in
    ``test_corner_laplacian_vector_fill_is_noop_for_nord1`` (single
    Laplacian iteration: byte-identical output between the two paths).

    Integration-level: 100-sec step through ``CDGridPrimitiveEquationModel
    .step`` produces 1-ULP (∼5×10⁻¹⁷) reorder differences from JAX
    JIT tracing two different code branches.  iter-894: loosened
    from ``assert_array_equal`` to ``assert_allclose(rtol=2e-15)`` to
    accept the trace-reorder ULP drift while still catching any
    non-trivial mathematical divergence.

    Tighter no-op claim remains the unit-level test.
    """
    grid, cdgrid, coord, _ = small_3d_state

    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
    )

    s = held_suarez_init(grid, coord)
    s = hydrostatic_to_fv3(s, cdgrid)
    n = grid.n
    nlev = s.u_d.data.shape[-1]
    rng = np.random.default_rng(seed=222)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1, nlev))
    s = s._replace(
        u_d=s.u_d.replace(data=jnp.asarray(u_p)),
        v_d=s.v_d.replace(data=jnp.asarray(v_p)),
    )

    cfg_no_fill = CDGridPrimitiveEquationConfig(
        div_damp_coeff=1e7,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=1,
        corner_div_damp_fv3_vector_fill=False,
    )
    cfg_with_fill = CDGridPrimitiveEquationConfig(
        div_damp_coeff=1e7,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=1,
        corner_div_damp_fv3_vector_fill=True,    # the iter-22 knob
    )

    m_no = CDGridPrimitiveEquationModel(grid, coord, cfg_no_fill)
    m_yes = CDGridPrimitiveEquationModel(grid, coord, cfg_with_fill)

    s_no = m_no.step(s, 100.0)
    s_yes = m_yes.step(s, 100.0)

    # iter-894: 1-ULP tolerance (was assert_array_equal pre-iter-894).
    # Trace-reorder ULP drift is expected; mathematical equivalence
    # at the Laplacian-iteration level is unit-tested separately.
    np.testing.assert_allclose(
        np.asarray(s_no.u_d.data), np.asarray(s_yes.u_d.data),
        rtol=2e-15, atol=1e-15,
    )
    np.testing.assert_allclose(
        np.asarray(s_no.v_d.data), np.asarray(s_yes.v_d.data),
        rtol=2e-15, atol=1e-15,
    )
    np.testing.assert_allclose(
        np.asarray(s_no.T.data), np.asarray(s_yes.T.data),
        rtol=2e-15, atol=1e-15,
    )
    np.testing.assert_allclose(
        np.asarray(s_no.p_s.data), np.asarray(s_yes.p_s.data),
        rtol=2e-15, atol=1e-15,
    )


def test_corner_div_damp_nord2_pe_runs_and_differs_from_nord1(small_3d_state):
    """FV3_3D iter 886: regression guard for ``corner_div_damp_nord=2``
    PE path with FV3-faithful vector-corner fill.

    Validates the open follow-up "Substantive nord>=2 fidelity
    restructure" (iter-32 / iter-99 stretch goal).  Two checks:

    1. ``corner_div_damp_nord=2`` runs end-to-end one step without
       NaN.  Sanity guard — the higher-order Laplacian iteration
       loop must be JAX-traceable and produce finite output.

    2. ``corner_div_damp_nord=2 + corner_div_damp_fv3_vector_fill=True``
       produces output that differs from ``corner_div_damp_nord=1``.
       At nord=2 the vector-fill path writes cube-vertex halo cells
       that the inner Laplacian iteration reads (FV3 sw_core.F90:1762).
       This contrasts with the bit-for-bit-identical nord=1 behaviour
       (``test_corner_div_damp_fv3_vector_fill_bit_for_bit_nord1``).

    Faithful-implementation goal: nord=2 should be a strict
    extension of nord=1, not a different code path that silently
    fails or returns unchanged output.
    """
    grid, cdgrid, coord, _ = small_3d_state

    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
    )

    s = held_suarez_init(grid, coord)
    s = hydrostatic_to_fv3(s, cdgrid)
    n = grid.n
    nlev = s.u_d.data.shape[-1]
    rng = np.random.default_rng(seed=886)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1, nlev))
    s = s._replace(
        u_d=s.u_d.replace(data=jnp.asarray(u_p)),
        v_d=s.v_d.replace(data=jnp.asarray(v_p)),
    )

    cfg_nord1 = CDGridPrimitiveEquationConfig(
        div_damp_coeff=1e7,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=1,
        corner_div_damp_fv3_vector_fill=True,
    )
    cfg_nord2 = CDGridPrimitiveEquationConfig(
        div_damp_coeff=1e7,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=2,
        corner_div_damp_fv3_vector_fill=True,
    )

    m_nord1 = CDGridPrimitiveEquationModel(grid, coord, cfg_nord1)
    m_nord2 = CDGridPrimitiveEquationModel(grid, coord, cfg_nord2)

    s_nord1 = m_nord1.step(s, 100.0)
    s_nord2 = m_nord2.step(s, 100.0)

    # (1) Both paths finite (no NaN from inner Laplacian iteration).
    assert jnp.all(jnp.isfinite(s_nord2.u_d.data))
    assert jnp.all(jnp.isfinite(s_nord2.v_d.data))
    assert jnp.all(jnp.isfinite(s_nord2.T.data))
    assert jnp.all(jnp.isfinite(s_nord2.p_s.data))

    # (2) nord=2 differs from nord=1 (higher-order Laplacian
    # contributes additional dd8·∇⁴(divg_d) damping vs nord=1's
    # dd4·∇²(divg_d)).
    max_diff_u = float(jnp.max(jnp.abs(s_nord2.u_d.data - s_nord1.u_d.data)))
    assert max_diff_u > 1e-12, (
        "nord=2 path produced bit-for-bit identical output to "
        "nord=1; higher-order Laplacian iteration is silently a no-op."
    )


def test_corner_div_damp_nord3_pe_loop_scales(small_3d_state):
    """FV3_3D iter 888: nord=3 PE stress test.

    Validates that the higher-order Laplacian iteration loop in
    ``primitive_eq_cdgrid.py:508-510``::

        _divg_d_iter = delpc
        for _ in range(config.corner_div_damp_nord):
            _divg_d_iter = _lap_per_level(_divg_d_iter)

    scales beyond nord=2 (which is regression-guarded by iter-886).
    Three checks:

    1. ``corner_div_damp_nord=3`` runs end-to-end without NaN.
       Confirms the ``for _ in range(nord)`` loop is JAX-traceable
       at higher orders, not silently truncating.

    2. nord=3 output differs from nord=2.  Confirms each additional
       iteration contributes non-trivial Laplacian smoothing
       (not silently converging to a fixed point after 2 iters
       due to numerical-precision floor).

    3. dd8 scaling factor ``(da_min_c·d4_bg)^(nord+1)`` increases
       monotonically with nord, so the *damping magnitude* per
       passing time-step grows with nord — verify u_d remains
       finite under the stronger damping.

    FV3 typically uses ``nord ∈ {1, 2, 3}`` (sw_core.F90 namelist
    ranges).  This test pins behaviour at the upper bound.
    """
    grid, cdgrid, coord, _ = small_3d_state

    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
    )

    s = held_suarez_init(grid, coord)
    s = hydrostatic_to_fv3(s, cdgrid)
    n = grid.n
    nlev = s.u_d.data.shape[-1]
    rng = np.random.default_rng(seed=888)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1, nlev))
    s = s._replace(
        u_d=s.u_d.replace(data=jnp.asarray(u_p)),
        v_d=s.v_d.replace(data=jnp.asarray(v_p)),
    )

    cfg_nord2 = CDGridPrimitiveEquationConfig(
        div_damp_coeff=1e7,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=2,
        corner_div_damp_fv3_vector_fill=True,
    )
    cfg_nord3 = CDGridPrimitiveEquationConfig(
        div_damp_coeff=1e7,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=3,
        corner_div_damp_fv3_vector_fill=True,
    )

    m_nord2 = CDGridPrimitiveEquationModel(grid, coord, cfg_nord2)
    m_nord3 = CDGridPrimitiveEquationModel(grid, coord, cfg_nord3)

    s_nord2 = m_nord2.step(s, 100.0)
    s_nord3 = m_nord3.step(s, 100.0)

    # (1) nord=3 runs without NaN.
    assert jnp.all(jnp.isfinite(s_nord3.u_d.data))
    assert jnp.all(jnp.isfinite(s_nord3.v_d.data))
    assert jnp.all(jnp.isfinite(s_nord3.T.data))
    assert jnp.all(jnp.isfinite(s_nord3.p_s.data))

    # (2) nord=3 differs from nord=2 — extra iteration contributes.
    max_diff_u = float(jnp.max(jnp.abs(s_nord3.u_d.data - s_nord2.u_d.data)))
    assert max_diff_u > 1e-12, (
        "nord=3 produced bit-for-bit identical output to nord=2; "
        "Laplacian iteration loop may be silently capped at 2."
    )


def test_smagorinsky_cs_zero_is_bit_for_bit_baseline(small_3d_state):
    """iter 58: ``smagorinsky_cs = 0`` (default) → bit-for-bit
    identical to the iter-19/24 path that does not have Smagorinsky.

    Pin this so the iter-58 wiring is non-disruptive for users who
    don't opt in.
    """
    grid, cdgrid, coord, _ = small_3d_state

    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
    )

    s = held_suarez_init(grid, coord)
    s = hydrostatic_to_fv3(s, cdgrid)
    n = grid.n
    nlev = s.u_d.data.shape[-1]
    rng = np.random.default_rng(seed=58)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1, nlev))
    s = s._replace(
        u_d=s.u_d.replace(data=jnp.asarray(u_p)),
        v_d=s.v_d.replace(data=jnp.asarray(v_p)),
    )

    cfg_no_smag = CDGridPrimitiveEquationConfig(
        div_damp_coeff=1e7, A_h=1e6,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
    )
    cfg_smag_zero = CDGridPrimitiveEquationConfig(
        div_damp_coeff=1e7, A_h=1e6,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        smagorinsky_cs=0.0,    # explicit zero — should be no-op
    )

    m_no = CDGridPrimitiveEquationModel(grid, coord, cfg_no_smag)
    m_zero = CDGridPrimitiveEquationModel(grid, coord, cfg_smag_zero)

    s_no = m_no.step(s, 100.0)
    s_zero = m_zero.step(s, 100.0)

    np.testing.assert_array_equal(
        np.asarray(s_no.u_d.data), np.asarray(s_zero.u_d.data),
    )
    np.testing.assert_array_equal(
        np.asarray(s_no.v_d.data), np.asarray(s_zero.v_d.data),
    )


def test_smagorinsky_cs_active_changes_winds(small_3d_state):
    """iter 58: ``smagorinsky_cs = 0.2`` changes winds vs the
    no-Smagorinsky path on a perturbed initial state (where local
    strain is non-zero).
    """
    grid, cdgrid, coord, _ = small_3d_state

    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
    )

    s = held_suarez_init(grid, coord)
    s = hydrostatic_to_fv3(s, cdgrid)
    n = grid.n
    nlev = s.u_d.data.shape[-1]
    rng = np.random.default_rng(seed=2358)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1, nlev))
    s = s._replace(
        u_d=s.u_d.replace(data=jnp.asarray(u_p)),
        v_d=s.v_d.replace(data=jnp.asarray(v_p)),
    )

    cfg_no = CDGridPrimitiveEquationConfig(
        div_damp_coeff=1e7, A_h=1e6,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        smagorinsky_cs=0.0,
    )
    cfg_yes = CDGridPrimitiveEquationConfig(
        div_damp_coeff=1e7, A_h=1e6,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        smagorinsky_cs=0.2,
    )

    m_no = CDGridPrimitiveEquationModel(grid, coord, cfg_no)
    m_yes = CDGridPrimitiveEquationModel(grid, coord, cfg_yes)

    s_no = m_no.step(s, 100.0)
    s_yes = m_yes.step(s, 100.0)

    diff = float(jnp.max(jnp.abs(s_no.u_d.data - s_yes.u_d.data)))
    base = float(jnp.max(jnp.abs(s_no.u_d.data)))
    assert diff > 1e-8 * base, (
        "smagorinsky_cs=0.2 must change winds vs c_s=0"
    )
    assert jnp.all(jnp.isfinite(s_yes.u_d.data))
    assert jnp.all(jnp.isfinite(s_yes.v_d.data))


def test_corner_div_damp_d4_stable_short_run_nord2(small_3d_state):
    """nord=2 (FV3 production default for del-6) is finite for 20 steps.

    With ``nord=2`` we have ``dd8 = (da_min_c * d4_bg)^3``, which is
    even more sensitive to ``da_min_c``.  We use ``d4_bg=1e-4`` at n=8
    (one decade below the nord=1 setting) to keep the iteration stable.
    """
    grid, cdgrid, coord, _ = small_3d_state

    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
    )

    cfg = CDGridPrimitiveEquationConfig(
        div_damp_coeff=1e6,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-4,
        corner_div_damp_nord=2,
        use_conservation_fixer=False,
        fix_mass=False,
    )
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)

    s = held_suarez_init(grid, coord)
    s = hydrostatic_to_fv3(s, cdgrid)
    for _ in range(20):
        s = model.step(s, 200.0)

    assert jnp.all(jnp.isfinite(s.u_d.data))
    assert jnp.all(jnp.isfinite(s.v_d.data))
    assert jnp.all(jnp.isfinite(s.T.data))
    assert jnp.all(jnp.isfinite(s.p_s.data))


def test_smagorinsky_combined_with_ah_stable_multistep(small_3d_state):
    """iter 64: pin multi-step stability of the combined ``A_h + smag``
    production-recommended configuration.

    iter-58 wired Smagorinsky-style adaptive A_h on top of static
    ``config.A_h``.  ``test_smagorinsky_cs_active_changes_winds`` only
    verifies a SINGLE step.  The iter-62 attempt at a 30-day C72 run
    with the combined ``LEGOESM_AH_SCALE=10 + LEGOESM_SMAG_CS=0.2``
    setting did not complete in budget, leaving multi-step stability
    of the combined path UNVERIFIED in CI.

    This test fills that gap: 20 steps at dt=200s on a perturbed n=8
    HS state with the production-recommended combination
    (``A_h=1e6`` + ``smagorinsky_cs=0.2``) must remain finite and
    not blow up.  Catches regression of the iter-58 corner/center
    wiring under repeated invocation.
    """
    grid, cdgrid, coord, _ = small_3d_state

    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
    )

    s = held_suarez_init(grid, coord)
    s = hydrostatic_to_fv3(s, cdgrid)
    n = grid.n
    nlev = s.u_d.data.shape[-1]
    rng = np.random.default_rng(seed=64)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1, nlev))
    s = s._replace(
        u_d=s.u_d.replace(data=jnp.asarray(u_p)),
        v_d=s.v_d.replace(data=jnp.asarray(v_p)),
    )

    cfg = CDGridPrimitiveEquationConfig(
        div_damp_coeff=1e6,
        A_h=1e6,
        smagorinsky_cs=0.2,
        use_conservation_fixer=False,
        fix_mass=False,
    )
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)

    initial_max_u = float(jnp.max(jnp.abs(s.u_d.data)))

    for _ in range(20):
        s = model.step(s, 200.0)

    assert jnp.all(jnp.isfinite(s.u_d.data))
    assert jnp.all(jnp.isfinite(s.v_d.data))
    assert jnp.all(jnp.isfinite(s.T.data))
    assert jnp.all(jnp.isfinite(s.p_s.data))

    # Sanity bound: combined damping should not let winds explode.
    # iter-62 C72 30-day with ah_x10 + smag_cs=0.2 had max|u| ~ 46
    # m/s; for an n=8 perturbed run we set a generous 5x bound on
    # the initial perturbation magnitude.
    final_max_u = float(jnp.max(jnp.abs(s.u_d.data)))
    assert final_max_u < 5.0 * initial_max_u, (
        f"combined A_h + smag winds grew unphysically: "
        f"final max|u|={final_max_u:.3f} > 5 * initial {initial_max_u:.3f}"
    )
    assert jnp.all(jnp.isfinite(s.p_s.data))
