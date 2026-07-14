"""Iter-904 sentinels: `use_fv3_dsw1_mass_transport` config flag swaps
the production height-tendency path from the iter-892 CDGrid PPM to
the true-FV3 d_sw1 finite-volume transport.

Per the Ralph-protocol acceptance gate, this flag is default-OFF.
Tests pin:

  1. Default config bit-equality: `fv3_sw_tendencies` outputs are
     bit-identical with vs without the new kwarg passed at default
     False — guards against silent default flip.
  2. ON path changes dh_dt but leaves du_dt/dv_dt unchanged for the
     same input state.
  3. The flag is forwarded by `FV3EdgeShallowWaterModel` (production)
     but NOT by `FV3FBShallowWaterModel` (FB chain emits a __init__
     UserWarning if set there — covered by iter-903b/c sentinels).
  4. iter-900/iter-903 LEFT/RIGHT flags retain their default-False
     after iter-904 (regression guard — iter-904 must not change
     unrelated defaults).
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
import jax.numpy as jnp
import pytest
import warnings

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig,
    FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState,
    FV3FBShallowWaterModel,
)
from legoesm.core.operators_cdgrid import fv3_sw_tendencies


def _make_smooth_state(grid):
    """Smooth balanced state for tendency comparison."""
    n = grid.n
    rng = np.random.default_rng(904)
    h = jnp.asarray(1.0e4 + rng.normal(size=(6, n, n)) * 100.0)
    h_s = jnp.zeros_like(h)
    u_d = jnp.asarray(rng.normal(size=(6, n, n + 1)) * 5.0)
    v_d = jnp.asarray(rng.normal(size=(6, n + 1, n)) * 5.0)
    return h, u_d, v_d, h_s


def test_iter904_default_off_bit_identical_to_pre_iter904():
    """Default `use_fv3_dsw1_mass_transport=False` (or kwarg absent)
    must produce bit-identical fv3_sw_tendencies output to the legacy
    iter-892 path."""
    grid = create_cubed_sphere(n=8, use_duogrid=False)
    cdgrid_data = create_cubed_sphere.__module__  # placeholder
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
    cdgrid = create_cubed_sphere_cdgrid(grid)

    h, u_d, v_d, h_s = _make_smooth_state(grid)

    # Without the kwarg (legacy call signature) — should equal default-False.
    dh_legacy, du_legacy, dv_legacy = fv3_sw_tendencies(
        h, u_d, v_d, h_s, cdgrid,
        boundary_fix=True, dddmp=0.2)

    # With kwarg explicitly False.
    dh_off, du_off, dv_off = fv3_sw_tendencies(
        h, u_d, v_d, h_s, cdgrid,
        boundary_fix=True, dddmp=0.2,
        use_fv3_dsw1_mass_transport=False)

    np.testing.assert_array_equal(np.asarray(dh_legacy),
                                    np.asarray(dh_off))
    np.testing.assert_array_equal(np.asarray(du_legacy),
                                    np.asarray(du_off))
    np.testing.assert_array_equal(np.asarray(dv_legacy),
                                    np.asarray(dv_off))


def test_iter904_on_changes_dh_only_keeps_du_dv_identical():
    """When `use_fv3_dsw1_mass_transport=True`, dh_dt CHANGES but
    du_dt/dv_dt are bit-identical to the default path."""
    grid = create_cubed_sphere(n=8, use_duogrid=False)
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
    cdgrid = create_cubed_sphere_cdgrid(grid)

    h, u_d, v_d, h_s = _make_smooth_state(grid)
    dt = 30.0

    dh_off, du_off, dv_off = fv3_sw_tendencies(
        h, u_d, v_d, h_s, cdgrid,
        boundary_fix=True, dddmp=0.2,
        use_fv3_dsw1_mass_transport=False)

    dh_on, du_on, dv_on = fv3_sw_tendencies(
        h, u_d, v_d, h_s, cdgrid,
        boundary_fix=True, dddmp=0.2,
        use_fv3_dsw1_mass_transport=True, dt=dt)

    # dh changes (the entire purpose of the flag).
    assert not np.allclose(np.asarray(dh_off), np.asarray(dh_on),
                            atol=1e-12), (
        "iter-904 ON path produced bit-identical dh_dt to OFF path — "
        "the flag isn't taking effect.")

    # du/dv unchanged (iter-904 only swaps mass transport).
    np.testing.assert_array_equal(np.asarray(du_off),
                                    np.asarray(du_on),
                                    err_msg=(
                                        "iter-904 ON path changed "
                                        "du_dt — must be bit-identical "
                                        "to OFF (only height transport "
                                        "is swapped)."))
    np.testing.assert_array_equal(np.asarray(dv_off),
                                    np.asarray(dv_on),
                                    err_msg=(
                                        "iter-904 ON path changed "
                                        "dv_dt — must be bit-identical "
                                        "to OFF."))


def test_iter904_on_requires_dt():
    """ON path must raise if `dt` is not provided."""
    grid = create_cubed_sphere(n=8, use_duogrid=False)
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
    cdgrid = create_cubed_sphere_cdgrid(grid)
    h, u_d, v_d, h_s = _make_smooth_state(grid)

    with pytest.raises(ValueError, match="requires `dt`"):
        fv3_sw_tendencies(
            h, u_d, v_d, h_s, cdgrid,
            use_fv3_dsw1_mass_transport=True)


def test_iter904_fb_chain_warns_on_flag_set():
    """FB chain emits the __init__ UserWarning when
    use_fv3_dsw1_mass_transport=True is set on its config."""
    grid = create_cubed_sphere(n=8, use_duogrid=False)
    config = CDGridShallowWaterConfig(use_fv3_dsw1_mass_transport=True)

    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        _ = FV3FBShallowWaterModel(grid, config=config)

    matches = [w for w in w_list
               if issubclass(w.category, UserWarning)
               and "use_fv3_dsw1_mass_transport" in str(w.message)]
    assert matches, (
        "FV3FBShallowWaterModel.__init__ did not warn about the "
        "ignored use_fv3_dsw1_mass_transport flag.  "
        f"Got: {[str(w.message) for w in w_list]}")


def test_iter904_production_model_does_not_warn():
    """Production model.__init__ does NOT emit the FB-chain ignored
    warning when the flag is set, since it correctly consumes the
    flag."""
    grid = create_cubed_sphere(n=8, use_duogrid=False)
    config = CDGridShallowWaterConfig(use_fv3_dsw1_mass_transport=True)

    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        _ = FV3EdgeShallowWaterModel(grid, config=config)

    fb_warnings = [w for w in w_list
                    if issubclass(w.category, UserWarning)
                    and "FV3FBShallowWaterModel" in str(w.message)]
    assert not fb_warnings, (
        "FV3EdgeShallowWaterModel incorrectly emitted the FB-chain "
        f"warning: {fb_warnings}")


def test_iter904_default_config_left_right_flags_remain_off():
    """iter-900/iter-903 LEFT/RIGHT flags must remain default-False
    after iter-904.  Regression guard: iter-904's introduction must
    not silently flip unrelated production defaults."""
    config = CDGridShallowWaterConfig()
    assert config.fortran_faithful_ppm_left is False, (
        "iter-900 fortran_faithful_ppm_left must remain default-False "
        "after iter-904 — silent flip detected.")
    assert config.fortran_faithful_ppm_right is False, (
        "iter-903 fortran_faithful_ppm_right must remain default-False "
        "after iter-904 — silent flip detected.")
    assert config.use_fv3_dsw1_mass_transport is False, (
        "iter-904 use_fv3_dsw1_mass_transport must default to False.")


def test_iter904_production_model_step_runs_with_flag_on():
    """Smoke test: `FV3EdgeShallowWaterModel.step` runs end-to-end
    with `use_fv3_dsw1_mass_transport=True` (no NaN, no exception)."""
    grid = create_cubed_sphere(n=8, use_duogrid=False)
    config = CDGridShallowWaterConfig(
        apply_fortran_xppm_boundary=True,
        use_fv3_dsw1_mass_transport=True,
        damp_v=0.0,
        nord_v=0,
    )
    model = FV3EdgeShallowWaterModel(grid, config=config)
    n = grid.n
    rng = np.random.default_rng(904)
    h = jnp.asarray(1.0e4 + rng.normal(size=(6, n, n)) * 50.0)
    h_s = jnp.zeros_like(h)
    u_d = jnp.zeros((6, n, n + 1))
    v_d = jnp.zeros((6, n + 1, n))
    state = FV3EdgeShallowWaterState(h=h, u_d=u_d, v_d=v_d, h_s=h_s)
    model.set_initial_mass(state)

    state_new = model.step(state, 30.0)
    assert np.all(np.isfinite(np.asarray(state_new.h)))
    assert np.all(np.isfinite(np.asarray(state_new.u_d)))
    assert np.all(np.isfinite(np.asarray(state_new.v_d)))
