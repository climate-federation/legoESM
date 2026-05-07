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
``legoesm.atmosphere.dynamics.primitive_eq_cdgrid``; the SW analog is
in ``legoesm.core.operators_cdgrid.cdgrid_momentum_tendencies`` (the
existing tested SW path).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
    fv3_hydrostatic_tendencies,
    hydrostatic_to_fv3,
)
from legoesm.atmosphere.held_suarez import held_suarez_init
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

    from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
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

    from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
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

    from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
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

    from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
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

    from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
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

    from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
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


def test_corner_div_damp_stable_short_run(small_3d_state):
    """corner_div_damp_d2_bg=0.001 stable for 20 steps from HS init."""
    grid, cdgrid, coord, _ = small_3d_state

    from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
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
