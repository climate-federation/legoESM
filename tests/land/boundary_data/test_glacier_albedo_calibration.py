"""AMIP-calibrated glacier / snow albedo for the LMIP biophysics path.

The LMIP driver builds its per-column params from the RAW PFT/biome tables, so it
never saw the 2026-07 recalibration that ``clm_multilayer_setup`` injects for the
coupled multilayer land.  ``physics.albedo_calibration='amip_multilayer'`` adopts
it.  The regression these tests exist for: the params are rebuilt EVERY step by
``make_step_land_params_updater``, so an override applied only at init would be
silently reverted after step 0.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.land.boundary_data import (
    GLACIER_ALB_NIR,
    GLACIER_ALB_NIR_TUNED,
    GLACIER_ALB_VIS,
    GLACIER_ALB_VIS_TUNED,
    GLACIER_ALBEDO_DEFAULT,
    GLACIER_ALBEDO_TUNED,
    make_step_land_params_updater,
    surface_data_to_land_params,
)
from legoesm.land.canopy import CanopyConfig
from legoesm.land.clm_surface_map import TUNED_GLACIER_ALBEDO_MULTILAYER
from legoesm.land.global_surface_data import GlobalSurfaceData, GlobalSurfaceDataConfig
from legoesm.land.surface_params import N_PFT_CLM5

_BE = 4          # broadleaf evergreen tree — the vegetated column
_TUNED_PAIR = (GLACIER_ALB_VIS_TUNED, GLACIER_ALB_NIR_TUNED)


def _gsd(glacier_frac):
    """2-column GlobalSurfaceData: column 0 vegetated, column 1 glacier-dominant."""
    ncol, npft = 2, N_PFT_CLM5
    lai = np.zeros((12, ncol, npft))
    lai[:, :, _BE] = 3.0
    htop = np.zeros((12, ncol, npft))
    htop[:, :, _BE] = 20.0
    pft = np.zeros((1, ncol, npft))
    pft[0, :, _BE] = 1.0
    z = np.zeros((ncol, 4))
    gl = np.asarray(glacier_frac, dtype=float)[None, :]      # (1, ncol)
    return GlobalSurfaceData(
        sand_frac=jnp.asarray(np.full((ncol, 4), 0.4)),
        clay_frac=jnp.asarray(np.full((ncol, 4), 0.2)),
        organic=jnp.asarray(z), bulk_density=jnp.asarray(z),
        soil_color=jnp.asarray(np.array([5, 5])),
        cell_area=jnp.ones(ncol),
        years=jnp.asarray([2000.0]),
        f_land=jnp.asarray(1.0 - gl), f_lake=jnp.zeros((1, ncol)),
        f_glacier=jnp.asarray(gl),
        pft_frac=jnp.asarray(pft),
        months=jnp.arange(12.0),
        lai_monthly=jnp.asarray(lai), sai_monthly=jnp.zeros((12, ncol, npft)),
        htop_monthly=jnp.asarray(htop), hbot_monthly=jnp.zeros((12, ncol, npft)),
        config=GlobalSurfaceDataConfig(),
    )


# --------------------------------------------------------------------------
# The derived vis/NIR pair
# --------------------------------------------------------------------------
def test_tuned_pair_broadband_matches_the_calibrated_value():
    """The pair must integrate to the tuned BROADBAND albedo under the 0.5/0.5
    vis-NIR weights the existing GLACIER_ALBEDO_DEFAULT already encodes."""
    assert GLACIER_ALBEDO_DEFAULT == pytest.approx(
        0.5 * (GLACIER_ALB_VIS + GLACIER_ALB_NIR))
    assert 0.5 * (GLACIER_ALB_VIS_TUNED + GLACIER_ALB_NIR_TUNED) == pytest.approx(
        TUNED_GLACIER_ALBEDO_MULTILAYER)
    assert GLACIER_ALBEDO_TUNED == pytest.approx(TUNED_GLACIER_ALBEDO_MULTILAYER)


def test_tuned_pair_is_physical_and_brighter_than_the_default():
    """Ice is brighter in the visible; the calibration RAISES the ice-sheet base
    (ERA5 ~0.85 vs the uncalibrated 0.60)."""
    assert 0.0 < GLACIER_ALB_NIR_TUNED < GLACIER_ALB_VIS_TUNED < 1.0
    assert GLACIER_ALBEDO_TUNED > GLACIER_ALBEDO_DEFAULT
    # contrast preserved from the uncalibrated pair
    assert (GLACIER_ALB_VIS_TUNED - GLACIER_ALB_NIR_TUNED) == pytest.approx(
        GLACIER_ALB_VIS - GLACIER_ALB_NIR)


# --------------------------------------------------------------------------
# Init-time params
# --------------------------------------------------------------------------
def test_init_params_default_is_unchanged():
    """glacier_alb=None must reproduce the uncalibrated constants exactly."""
    gsd = _gsd([0.0, 1.0])
    lp = surface_data_to_land_params(
        gsd, CanopyConfig(), 15.0, jnp.full(2, 0.2), glacier_alb=None)
    assert float(lp.ALB_VIS[1]) == pytest.approx(GLACIER_ALB_VIS)
    assert float(lp.ALB_NIR[1]) == pytest.approx(GLACIER_ALB_NIR)


def test_init_params_apply_the_tuned_pair_on_glacier_columns_only():
    gsd = _gsd([0.0, 1.0])
    lp = surface_data_to_land_params(
        gsd, CanopyConfig(), 15.0, jnp.full(2, 0.2), glacier_alb=_TUNED_PAIR)
    assert float(lp.ALB_VIS[1]) == pytest.approx(GLACIER_ALB_VIS_TUNED)
    assert float(lp.ALB_NIR[1]) == pytest.approx(GLACIER_ALB_NIR_TUNED)
    # the vegetated column keeps its soil-colour background — untouched
    assert float(lp.ALB_VIS[0]) != pytest.approx(GLACIER_ALB_VIS_TUNED)


# --------------------------------------------------------------------------
# THE REGRESSION: the per-step rebuild must not revert the override
# --------------------------------------------------------------------------
@pytest.mark.parametrize("glacier_alb,exp_vis,exp_nir", [
    (None, GLACIER_ALB_VIS, GLACIER_ALB_NIR),
    (_TUNED_PAIR, GLACIER_ALB_VIS_TUNED, GLACIER_ALB_NIR_TUNED),
])
def test_step_updater_preserves_the_override(glacier_alb, exp_vis, exp_nir):
    """``make_step_land_params_updater`` rebuilds CanopyLandParams every step.
    If it did not receive the same glacier_alb, the ice albedo would silently
    revert to the uncalibrated default after step 0 — the whole calibration
    would apply for exactly one timestep."""
    gsd = _gsd([0.0, 1.0])
    upd = make_step_land_params_updater(
        gsd, CanopyConfig(), glacier_alb=glacier_alb)
    lp, _ = upd(jnp.full(2, 0.2), jnp.asarray(15.0), jnp.asarray(2000.0))
    assert float(lp.ALB_VIS[1]) == pytest.approx(exp_vis)
    assert float(lp.ALB_NIR[1]) == pytest.approx(exp_nir)


def test_step_updater_agrees_with_init_params():
    """Init-time and per-step params must carry the SAME ice albedo, else the
    run is discontinuous at step 1."""
    gsd = _gsd([0.0, 1.0])
    lp0 = surface_data_to_land_params(
        gsd, CanopyConfig(), 15.0, jnp.full(2, 0.2), glacier_alb=_TUNED_PAIR)
    upd = make_step_land_params_updater(
        gsd, CanopyConfig(), glacier_alb=_TUNED_PAIR)
    lp1, _ = upd(jnp.full(2, 0.2), jnp.asarray(15.0), jnp.asarray(2000.0))
    np.testing.assert_allclose(np.asarray(lp0.ALB_VIS), np.asarray(lp1.ALB_VIS), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(lp0.ALB_NIR), np.asarray(lp1.ALB_NIR), rtol=1e-12)
