"""Per-PFT root depth + plant theta_wp/theta_fc on the LMIP canopy path.

The LMIP path runs ONE global ``root_depth``/``theta_wp``/``theta_fc`` from
``MultiLayerLandConfig`` (1.0 m / 0.15 / 0.30) for every column on Earth.  These
tests cover the opt-in per-PFT tables, and the two ways this could silently fail:
the per-step rebuild dropping the params, and the bare-soil gap-fill fallback
carrying ``None`` where the params carry arrays (a pytree structure mismatch).
"""
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.land.boundary_data import (
    make_step_land_params_updater,
    surface_data_to_land_params,
)
from legoesm.land.boundary_data._internals import tuned_pft_root_arrays
from legoesm.land.canopy import CanopyConfig
from legoesm.land.clm_surface_map import (
    TUNED_PFT_FC_MULTILAYER,
    TUNED_PFT_ROOT_DEPTH_MULTILAYER,
    TUNED_PFT_WP_MULTILAYER,
)
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.multilayer_land import _get
from legoesm.land.surface_params import N_PFT_CLM5

from tests.land.boundary_data.test_glacier_albedo_calibration import _gsd

_BET = 4          # broadleaf evergreen tree — the dominant PFT in the fixture


# --------------------------------------------------------------------------
# The tables themselves
# --------------------------------------------------------------------------
def test_tuned_tables_are_wellformed():
    rt = tuned_pft_root_arrays()
    for k, v in rt.items():
        assert v.shape == (N_PFT_CLM5,), k
        assert np.isfinite(v).all(), k
    # a wilting point at or above field capacity makes beta_root degenerate
    assert (rt["theta_fc"] > rt["theta_wp"]).all()
    assert (rt["root_depth"] > 0.0).all()


def test_tuned_tables_are_public_and_match_the_accessor():
    """Promoted from private (_TUNED_*) so cross-module imports are legal."""
    rt = tuned_pft_root_arrays()
    np.testing.assert_allclose(rt["root_depth"], TUNED_PFT_ROOT_DEPTH_MULTILAYER)
    np.testing.assert_allclose(rt["theta_wp"], TUNED_PFT_WP_MULTILAYER)
    np.testing.assert_allclose(rt["theta_fc"], TUNED_PFT_FC_MULTILAYER)


def test_tuned_tables_span_a_real_range():
    """The whole point: grass and forest must NOT share one rooting depth."""
    rd = np.asarray(TUNED_PFT_ROOT_DEPTH_MULTILAYER)
    assert rd.max() / rd.min() > 5.0


# --------------------------------------------------------------------------
# Default path unchanged
# --------------------------------------------------------------------------
def test_default_leaves_fields_none_and_falls_back_to_config():
    """Absent per-column params must fall back to the SCALAR config values, so
    an existing run is bit-identical."""
    gsd = _gsd([0.0, 0.0])
    lp = surface_data_to_land_params(
        gsd, CanopyConfig(), 15.0, jnp.full(2, 0.2), tuned_root_params=False)
    assert lp.root_depth is None and lp.theta_wp is None and lp.theta_fc is None
    cfg = MultiLayerLandConfig()
    assert _get(lp, "root_depth", cfg.root_depth) == cfg.root_depth
    assert _get(lp, "theta_wp", cfg.theta_wp) == cfg.theta_wp
    assert _get(lp, "theta_fc", cfg.theta_fc) == cfg.theta_fc


def test_get_treats_a_present_none_as_absent():
    """``_get`` uses getattr-with-default; once the field EXISTS with a None
    default, a plain getattr would return None and poison the arithmetic."""
    gsd = _gsd([0.0, 0.0])
    lp = surface_data_to_land_params(
        gsd, CanopyConfig(), 15.0, jnp.full(2, 0.2), tuned_root_params=False)
    assert _get(lp, "root_depth", 1.0) == 1.0          # not None


# --------------------------------------------------------------------------
# Tuned path
# --------------------------------------------------------------------------
def test_tuned_populates_the_dominant_pft_values():
    gsd = _gsd([0.0, 0.0])
    lp = surface_data_to_land_params(
        gsd, CanopyConfig(), 15.0, jnp.full(2, 0.2), tuned_root_params=True)
    for field, table in (("root_depth", TUNED_PFT_ROOT_DEPTH_MULTILAYER),
                         ("theta_wp", TUNED_PFT_WP_MULTILAYER),
                         ("theta_fc", TUNED_PFT_FC_MULTILAYER)):
        v = np.asarray(getattr(lp, field))
        assert v.shape == (2,)
        np.testing.assert_allclose(v, table[_BET], rtol=1e-12)


def test_tuned_differs_from_the_global_default():
    """If the calibrated value happened to equal the config scalar the whole
    change would be a no-op — assert it actually moves."""
    gsd = _gsd([0.0, 0.0])
    lp = surface_data_to_land_params(
        gsd, CanopyConfig(), 15.0, jnp.full(2, 0.2), tuned_root_params=True)
    cfg = MultiLayerLandConfig()
    assert float(np.asarray(lp.root_depth)[0]) != pytest.approx(cfg.root_depth)
    assert float(np.asarray(lp.theta_wp)[0]) != pytest.approx(cfg.theta_wp)
    assert float(np.asarray(lp.theta_fc)[0]) != pytest.approx(cfg.theta_fc)


# --------------------------------------------------------------------------
# The two silent-failure modes
# --------------------------------------------------------------------------
@pytest.mark.parametrize("tuned", [False, True])
def test_step_updater_matches_init_params(tuned):
    """Params are rebuilt EVERY step; if the updater did not also receive
    tuned_root_params the calibration would apply for exactly one timestep."""
    gsd = _gsd([0.0, 0.0])
    lp0 = surface_data_to_land_params(
        gsd, CanopyConfig(), 15.0, jnp.full(2, 0.2), tuned_root_params=tuned)
    upd = make_step_land_params_updater(
        gsd, CanopyConfig(), tuned_root_params=tuned)
    lp1, _ = upd(jnp.full(2, 0.2), jnp.asarray(15.0), jnp.asarray(2000.0))
    for field in ("root_depth", "theta_wp", "theta_fc"):
        a, b = getattr(lp0, field), getattr(lp1, field)
        if tuned:
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12)
        else:
            assert a is None and b is None


def test_bare_gap_fill_structure_matches():
    """The gap-fill combines params with a bare-soil fallback leaf-by-leaf.  A
    fallback carrying None against an array raises
    'None is not a valid value for jnp.array' — this is the regression."""
    gsd = _gsd([0.0, 0.0])
    upd = make_step_land_params_updater(
        gsd, CanopyConfig(), tuned_root_params=True)
    lp, _ = upd(jnp.full(2, 0.2), jnp.asarray(15.0), jnp.asarray(2000.0))
    assert lp.root_depth is not None
    assert np.isfinite(np.asarray(lp.root_depth)).all()


def test_bare_fallback_uses_the_bare_soil_pft_row():
    from legoesm.land.boundary_data.gap_fill import bare_canopy_params
    fb = bare_canopy_params(3, tuned_root_params=True)
    np.testing.assert_allclose(
        np.asarray(fb.root_depth), TUNED_PFT_ROOT_DEPTH_MULTILAYER[0], rtol=1e-12)
    assert bare_canopy_params(3).root_depth is None
