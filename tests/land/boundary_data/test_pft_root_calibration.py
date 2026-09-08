"""Config-supplied per-PFT root-zone water-uptake parameters (LMIP path).

The three ``physics.*_per_pft`` lists (length-17, CLM5 PFT order) replace the
single global ``MultiLayerLandConfig.root_depth``/``theta_wp``/``theta_fc``
with per-column values gathered at the dominant PFT (the calibrated production
values live in the templates, pinned by test_shipped_templates).  Absent lists
must be bit-identical to the pre-existing scalar behaviour.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.land.boundary_data import (
    make_step_land_params_updater,
    surface_data_to_land_params,
)
from legoesm.land.canopy import CanopyConfig
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.multilayer_land import _get
from legoesm.land.surface_params import N_PFT_CLM5

from tests.land.boundary_data.test_glacier_albedo_calibration import _gsd

_BET = 4          # broadleaf evergreen tree — the dominant PFT in the fixture

# Synthetic per-PFT tables with distinct, recognisable values so a wrong gather
# (off-by-one PFT, wrong table) produces a visibly wrong number.
_ROOT = {
    "root_depth": np.linspace(0.1, 1.7, N_PFT_CLM5),
    "theta_wp": np.linspace(0.05, 0.12, N_PFT_CLM5),
    "theta_fc": np.linspace(0.17, 0.30, N_PFT_CLM5),
}


# --------------------------------------------------------------------------
# Default path unchanged
# --------------------------------------------------------------------------
def test_default_leaves_fields_none_and_falls_back_to_config():
    """Absent per-column params must fall back to the SCALAR config values, so
    an existing run is bit-identical."""
    gsd = _gsd([0.0, 0.0])
    lp = surface_data_to_land_params(
        gsd, CanopyConfig(), 15.0, jnp.full(2, 0.2), pft_root_params=None)
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
        gsd, CanopyConfig(), 15.0, jnp.full(2, 0.2), pft_root_params=None)
    assert _get(lp, "root_depth", 1.0) == 1.0          # not None


# --------------------------------------------------------------------------
# Per-PFT path
# --------------------------------------------------------------------------
def test_tables_populate_the_dominant_pft_values():
    gsd = _gsd([0.0, 0.0])
    lp = surface_data_to_land_params(
        gsd, CanopyConfig(), 15.0, jnp.full(2, 0.2), pft_root_params=_ROOT)
    for field in ("root_depth", "theta_wp", "theta_fc"):
        v = np.asarray(getattr(lp, field))
        assert v.shape == (2,)
        np.testing.assert_allclose(v, _ROOT[field][_BET], rtol=1e-12)


def test_tables_differ_from_the_global_default():
    """If the supplied value happened to equal the config scalar the whole
    change would be a no-op — assert the fixture actually moves it."""
    gsd = _gsd([0.0, 0.0])
    lp = surface_data_to_land_params(
        gsd, CanopyConfig(), 15.0, jnp.full(2, 0.2), pft_root_params=_ROOT)
    cfg = MultiLayerLandConfig()
    assert float(np.asarray(lp.root_depth)[0]) != pytest.approx(cfg.root_depth)
    assert float(np.asarray(lp.theta_wp)[0]) != pytest.approx(cfg.theta_wp)
    assert float(np.asarray(lp.theta_fc)[0]) != pytest.approx(cfg.theta_fc)


# --------------------------------------------------------------------------
# The two silent-failure modes
# --------------------------------------------------------------------------
@pytest.mark.parametrize("tables", [None, _ROOT])
def test_step_updater_matches_init_params(tables):
    """Params are rebuilt EVERY step; if the updater did not also receive
    pft_root_params the per-PFT values would apply for exactly one timestep."""
    gsd = _gsd([0.0, 0.0])
    lp0 = surface_data_to_land_params(
        gsd, CanopyConfig(), 15.0, jnp.full(2, 0.2), pft_root_params=tables)
    upd = make_step_land_params_updater(
        gsd, CanopyConfig(), pft_root_params=tables)
    lp1, _ = upd(jnp.full(2, 0.2), jnp.asarray(15.0), jnp.asarray(2000.0))
    for field in ("root_depth", "theta_wp", "theta_fc"):
        a, b = getattr(lp0, field), getattr(lp1, field)
        if tables is not None:
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12)
        else:
            assert a is None and b is None


def test_bare_gap_fill_structure_matches():
    """The gap-fill combines params with a bare-soil fallback leaf-by-leaf.  A
    fallback carrying None against an array raises
    'None is not a valid value for jnp.array' — this is the regression."""
    gsd = _gsd([0.0, 0.0])
    upd = make_step_land_params_updater(
        gsd, CanopyConfig(), pft_root_params=_ROOT)
    lp, _ = upd(jnp.full(2, 0.2), jnp.asarray(15.0), jnp.asarray(2000.0))
    assert lp.root_depth is not None
    assert np.isfinite(np.asarray(lp.root_depth)).all()


def test_bare_fallback_uses_the_bare_soil_pft_row():
    from legoesm.land.boundary_data.gap_fill import bare_canopy_params
    fb = bare_canopy_params(3, pft_root_params=_ROOT)
    np.testing.assert_allclose(
        np.asarray(fb.root_depth), _ROOT["root_depth"][0], rtol=1e-12)
    assert bare_canopy_params(3).root_depth is None


def test_gap_fill_refuses_root_params_it_cannot_match():
    """fill_land_param_gaps has no tables to build a matching bare fallback
    from — feeding it root-carrying params must fail loudly, not crash deep
    inside a pytree map."""
    from legoesm.land.boundary_data.gap_fill import fill_land_param_gaps
    gsd = _gsd([0.0, 0.0])
    lp = surface_data_to_land_params(
        gsd, CanopyConfig(), 15.0, jnp.full(2, 0.2), pft_root_params=_ROOT)
    with pytest.raises(ValueError, match="per-column root fields"):
        fill_land_param_gaps(lp, gsd)
