"""LAI source selection for the CLM-ML-JAX canopy scheme.

The CLM-ML canopy accepts leaf area from EITHER of two sources:

  * PRESCRIBED (climatology): the PFT-weighted monthly surfdata LAI/SAI/htop,
    materialised into ``LandSurfaceParams`` by
    ``surface_data_to_land_params`` for a ``CLMMLCanopyConfig`` scheme.
  * PROGNOSTIC (carbon dynamics): ``LAI = C_fol / LCMA`` from the DifferLand
    foliar carbon pool, via ``compute_prognostic_lai`` gated on
    ``CLMMLCanopyConfig.use_prognostic_lai`` — threaded into the CLM-ML
    interface as ``lai_override``.

These tests exercise both paths WITHOUT the optional ``clm-ml-jax`` dependency
(they target the legoESM-side wiring: the surfdata provider, the config flag,
and ``compute_prognostic_lai``).  The end-to-end CLM-ML flux tests live in
``tests/land/unit/test_canopy.py`` (skipped when clm-ml-jax is absent).
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from legoesm.land.global_surface_data import GlobalSurfaceData, GlobalSurfaceDataConfig
from legoesm.land.surface_params import CLM5_PFT_NAMES, N_PFT_CLM5, LandSurfaceParams
from legoesm.land.boundary_data import (
    surface_data_to_land_params,
    prescribed_canopy_structure,
)
from legoesm.land.canopy.config import CLMMLCanopyConfig
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.carbon import CarbonConfig, init_carbon_state
from legoesm.land.surface_scheme import SimpleSEBConfig
from legoesm.land.surface_scheme.two_leaf_canopy import compute_prognostic_lai


# ---------------------------------------------------------------------------
# Synthetic surfdata (mirrors tests/land/boundary_data/test_land_inputs.py)
# ---------------------------------------------------------------------------
def _gsd(ncol=2, nlayer=4):
    npft = N_PFT_CLM5
    pft = np.zeros((1, ncol, npft))
    be = CLM5_PFT_NAMES.index("broadleaf_evergreen_tropical")
    c4 = CLM5_PFT_NAMES.index("c4_grass")
    pft[0, 0, be] = 1.0          # col0 dominant BE-tropical
    pft[0, 1, c4] = 1.0          # col1 dominant C4 grass
    veg = lambda val: np.full((12, ncol, npft), val)
    lai = np.zeros((12, ncol, npft)); lai[:, 0, be] = 5.0; lai[:, 1, c4] = 1.5
    sai = np.zeros((12, ncol, npft)); sai[:, 0, be] = 0.8; sai[:, 1, c4] = 0.3
    htop = np.zeros((12, ncol, npft)); htop[:, 0, be] = 25.0; htop[:, 1, c4] = 1.0
    sand = np.full((ncol, nlayer), 0.40); clay = np.full((ncol, nlayer), 0.20)
    z = lambda: np.zeros((ncol, nlayer))
    return GlobalSurfaceData(
        sand_frac=jnp.asarray(sand), clay_frac=jnp.asarray(clay),
        organic=jnp.asarray(z()), bulk_density=jnp.asarray(z()),
        soil_color=jnp.asarray(np.array([5, 18])),
        cell_area=jnp.ones(ncol),
        years=jnp.asarray([2015.0]),
        f_land=jnp.ones((1, ncol)), f_lake=jnp.zeros((1, ncol)),
        f_glacier=jnp.zeros((1, ncol)), pft_frac=jnp.asarray(pft),
        months=jnp.arange(12.0),
        lai_monthly=jnp.asarray(lai), sai_monthly=jnp.asarray(sai),
        htop_monthly=jnp.asarray(htop), hbot_monthly=jnp.asarray(veg(0.0)),
        config=GlobalSurfaceDataConfig(),
    )


# ===========================================================================
# 1. PRESCRIBED climatology path
# ===========================================================================
def test_prescribed_canopy_structure_returns_lai_sai_htop_only():
    """The helper returns exactly (LAI, SAI, htop) — hbot stays derived."""
    out = prescribed_canopy_structure(_gsd(), day_of_year=180.0)
    assert len(out) == 3
    lai, sai, htop = out
    assert lai.shape == (2,) and sai.shape == (2,) and htop.shape == (2,)


def test_prescribed_canopy_structure_pft_weighted_values():
    """LAI/SAI are the PFT-weighted column aggregate; htop the weighted mean.

    col0 is 100% BE-tropical (LAI 5, SAI 0.8, htop 25); col1 is 100% C4 grass
    (LAI 1.5, SAI 0.3, htop 1)."""
    lai, sai, htop = prescribed_canopy_structure(_gsd(), day_of_year=1.0)
    np.testing.assert_allclose(np.asarray(lai), [5.0, 1.5], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(sai), [0.8, 0.3], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(htop), [25.0, 1.0], rtol=1e-6)


def test_prescribed_uncovered_column_collapses_to_zero():
    """A surfdata-uncovered column (zero pft_frac) gets bare-soil structure
    (~0 leaf/stem area, ~0 height), not a spurious canopy."""
    gsd = _gsd()
    pft = np.asarray(gsd.pft_frac).copy(); pft[0, 1, :] = 0.0   # col1 uncovered
    gsd = gsd._replace(pft_frac=jnp.asarray(pft))
    lai, sai, htop = prescribed_canopy_structure(gsd, day_of_year=1.0)
    # col1 -> bare PFT row (index 0); the synthetic bare row is all-zero.
    assert float(lai[1]) == pytest.approx(0.0)
    assert float(sai[1]) == pytest.approx(0.0)
    assert float(htop[1]) == pytest.approx(0.0)


def test_prescribed_structure_survives_nan_in_zero_weight_pft():
    """A NaN in a ZERO-weight PFT slot must not poison the column via
    ``NaN * 0 == NaN`` (codex F1): the dominant PFT's value must survive.

    col0 is 100% BE-tropical (LAI 5); inject NaN into an unrelated, zero-weight
    PFT slot for col0 — the PFT-weighted LAI must stay 5, not collapse to 0."""
    gsd = _gsd()
    c4 = CLM5_PFT_NAMES.index("c4_grass")
    lai = np.asarray(gsd.lai_monthly).copy()
    lai[:, 0, c4] = np.nan            # zero-weight (col0 is 0% c4) but NaN
    gsd = gsd._replace(lai_monthly=jnp.asarray(lai))
    out_lai, _, _ = prescribed_canopy_structure(gsd, day_of_year=1.0)
    assert np.isfinite(float(out_lai[0]))
    np.testing.assert_allclose(float(out_lai[0]), 5.0, rtol=1e-6)


def test_surface_data_to_land_params_clmml_populates_structure():
    """For a CLMMLCanopyConfig scheme, the dispatcher returns a
    LandSurfaceParams with LAI/SAI/htop populated from the climatology."""
    lp = surface_data_to_land_params(
        _gsd(), CLMMLCanopyConfig(), day_of_year=1.0, theta_top=jnp.full(2, 0.2))
    assert isinstance(lp, LandSurfaceParams)
    assert lp.LAI is not None and lp.SAI is not None and lp.htop is not None
    np.testing.assert_allclose(np.asarray(lp.LAI), [5.0, 1.5], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(lp.htop), [25.0, 1.0], rtol=1e-6)
    # hbot stays derived inside the CLM-ML interface (hbot_frac * htop).
    assert lp.hbot is None


def test_surface_data_to_land_params_simpleseb_unchanged():
    """SimpleSEB does not read canopy structure; the dispatcher must NOT
    populate LAI/SAI/htop (regression guard: byte-identical to before)."""
    lp = surface_data_to_land_params(
        _gsd(), SimpleSEBConfig(), day_of_year=1.0, theta_top=jnp.full(2, 0.2))
    assert isinstance(lp, LandSurfaceParams)
    assert lp.LAI is None and lp.SAI is None and lp.htop is None


# ===========================================================================
# 2. PROGNOSTIC carbon path
# ===========================================================================
def test_clmml_config_has_use_prognostic_lai_default_off():
    cfg = CLMMLCanopyConfig()
    assert hasattr(cfg, "use_prognostic_lai")
    assert cfg.use_prognostic_lai is False


def test_prognostic_lai_equals_cfol_over_lcma():
    """With use_prognostic_lai + differland carbon, LAI = C_fol / LCMA."""
    cfg = MultiLayerLandConfig(
        surface_scheme=CLMMLCanopyConfig(use_prognostic_lai=True),
        carbon=CarbonConfig(scheme="differland"))
    cs = init_carbon_state((4,), cfg.carbon)
    lai = compute_prognostic_lai(cs, cfg, cfg.surface_scheme)
    assert lai is not None
    expected = cfg.carbon.C_fol_init / cfg.carbon.LCMA
    np.testing.assert_allclose(np.asarray(lai), np.full(4, expected), rtol=1e-6)


def test_prognostic_lai_off_returns_none():
    """use_prognostic_lai=False -> None (CLM-ML falls back to prescribed LAI)."""
    cfg = MultiLayerLandConfig(
        surface_scheme=CLMMLCanopyConfig(use_prognostic_lai=False),
        carbon=CarbonConfig(scheme="differland"))
    cs = init_carbon_state((4,), cfg.carbon)
    assert compute_prognostic_lai(cs, cfg, cfg.surface_scheme) is None


def test_prognostic_lai_requires_differland_carbon():
    """Prognostic LAI needs an active DifferLand carbon pool; a 'none' carbon
    scheme returns None even with the flag set (no C_fol to read)."""
    cfg = MultiLayerLandConfig(
        surface_scheme=CLMMLCanopyConfig(use_prognostic_lai=True),
        carbon=CarbonConfig(scheme="none"))
    cs = init_carbon_state((4,), cfg.carbon)
    assert compute_prognostic_lai(cs, cfg, cfg.surface_scheme) is None


def test_prognostic_lai_none_carbon_state_returns_none():
    cfg = MultiLayerLandConfig(
        surface_scheme=CLMMLCanopyConfig(use_prognostic_lai=True),
        carbon=CarbonConfig(scheme="differland"))
    assert compute_prognostic_lai(None, cfg, cfg.surface_scheme) is None


def test_prognostic_lai_differentiable_wrt_cfol():
    """dLAI/dC_fol = 1/LCMA — the prognostic map is differentiable so leaf area
    responds smoothly to the carbon dynamics."""
    cfg = MultiLayerLandConfig(
        surface_scheme=CLMMLCanopyConfig(use_prognostic_lai=True),
        carbon=CarbonConfig(scheme="differland"))
    cs = init_carbon_state((1,), cfg.carbon)

    def lai_of_cfol(cfol):
        # scalar C_fol -> scalar LAI (compute_prognostic_lai only reads C_fol/LCMA)
        return compute_prognostic_lai(
            cs._replace(C_fol=cfol), cfg, cfg.surface_scheme)

    g = jax.grad(lai_of_cfol)(jnp.asarray(200.0))
    assert float(g) == pytest.approx(1.0 / cfg.carbon.LCMA, rel=1e-6)


# ===========================================================================
# 3. CLM-ML interface accepts lai_override (skipped without clm-ml-jax)
# ===========================================================================
def test_compute_clm_ml_canopy_fluxes_accepts_lai_override():
    """The interface entry point exposes the lai_override parameter used by the
    prognostic path (signature contract; does not require clm-ml-jax)."""
    import inspect
    from legoesm.land.canopy.clm_ml_interface import compute_clm_ml_canopy_fluxes
    sig = inspect.signature(compute_clm_ml_canopy_fluxes)
    assert "lai_override" in sig.parameters
