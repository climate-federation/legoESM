"""Model-driver activation of the differentiable multilayer (Richards) land tile.

The slab->coupler differentiable refactor makes the segment forward *capable* of
advancing a per-column MultiLayerLandState (SegmentCarry.land_ml); this test covers
the production-driver wiring that *activates* it: ``use_multilayer_land`` ->
``_setup_multilayer_land`` populates the pipeline ``land_ml_*`` attrs, seeds the
state, and the run loop persists the evolved soil column across segments.

Loaders (CLM surfdata + land mask) are monkeypatched to synthetic data so the test
is offline + portable (no NetCDF, no network, no /tmp dependence).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm.driver.config import (
    ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
)
from legoesm.driver.model_driver import ModelDriver


def _fake_surface_map(path, lat_deg, lon_deg):
    """A uniform-loam, all-bare-soil CLM map sized to the requested columns."""
    n = int(np.asarray(lat_deg).size)
    pft = np.zeros((n, 17)); pft[:, 0] = 1.0          # 100% bare soil
    o = np.ones(n)
    return dict(
        pft_fractions=jnp.asarray(pft),
        theta_wp=jnp.asarray(0.12 * o), theta_fc=jnp.asarray(0.30 * o),
        glacier_frac=jnp.asarray(np.zeros(n)),
        pct_sand=jnp.asarray(40.0 * o), pct_clay=jnp.asarray(20.0 * o),
        theta_r=jnp.asarray(0.05 * o), theta_sat=jnp.asarray(0.45 * o),
        alpha_vg=jnp.asarray(2.0 * o), n_vg=jnp.asarray(1.4 * o),
        K_sat=jnp.asarray(1.0e-5 * o),
    )


def _patch_land_loaders(monkeypatch):
    """Replace the CLM-surfdata + land-mask loaders with synthetic data."""
    import legoesm.land.clm_surface_map as clm
    import legoesm.grids.topography as topo
    monkeypatch.setattr(clm, "download_clm_surfdata", lambda *a, **k: "synthetic")
    monkeypatch.setattr(clm, "load_clm_surface", _fake_surface_map)
    # Half-land everywhere so every column exercises the land tile blend.
    monkeypatch.setattr(
        topo, "load_land_fraction",
        lambda grid, path, *a, **k: jnp.full(grid.lat.shape, 0.5))


def _small_cfg():
    return ExperimentConfig(
        grid=GridConfig(resolution=8, nlev=8),
        dycore=DycoreConfig(dt=600.0),
        output=OutputConfig(diag_days=1),
        days=1, dataset="analytical",
        radiation="gray",
        land_mask_path="synthetic.nc",     # truthy -> land block runs (loader patched)
        use_multilayer_land=True,
        multilayer_n_layers=6, multilayer_soil_depth=2.5,
    )


def test_setup_wires_multilayer_land(monkeypatch, tmp_path):
    """use_multilayer_land -> pipeline land_ml_* attrs + a seeded MultiLayerLandState."""
    from legoesm.land.state import MultiLayerLandState
    _patch_land_loaders(monkeypatch)
    driver = ModelDriver(_small_cfg(), output_dir=tmp_path)
    driver.setup()

    ncol = driver.grid.lat.size
    assert driver.physics.land_ml_cfg is not None
    assert driver.physics.land_ml_params is not None
    assert driver.physics.land_ml_lat is not None
    assert driver.physics.land_ml_lat.shape == (ncol,)
    assert driver.physics.land_ml_cfg.soil_grid.n_layers == 6

    st = driver._land_ml_state
    assert isinstance(st, MultiLayerLandState)
    assert st.T_soil.shape == (ncol, 6)
    # warm-started from near-surface air T (physical soil temperatures)
    assert np.all((np.asarray(st.T_soil) > 180.0) & (np.asarray(st.T_soil) < 340.0))
    assert np.all((np.asarray(st.theta_soil) >= 0.0)
                  & (np.asarray(st.theta_soil) <= 1.0))


def test_multilayer_land_evolves_over_amip_segment(monkeypatch, tmp_path):
    """A 1-day AMIP run completes and the prognostic soil column advances."""
    _patch_land_loaders(monkeypatch)
    driver = ModelDriver(_small_cfg(), output_dir=tmp_path)
    driver.setup()
    T_soil_0 = np.asarray(driver._land_ml_state.T_soil).copy()

    status = driver.run()

    assert status == "COMPLETED"
    st = driver._land_ml_state
    # readback persisted an EVOLVED state (not the seed, not None)
    assert st is not None
    assert not np.allclose(np.asarray(st.T_soil), T_soil_0), "soil column did not advance"
    assert np.all(np.isfinite(np.asarray(st.T_soil)))
    assert np.all((np.asarray(st.theta_soil) >= 0.0)
                  & (np.asarray(st.theta_soil) <= 1.0))


def test_slab_path_leaves_multilayer_inactive(monkeypatch, tmp_path):
    """use_multilayer_land=False -> all land_ml_* stay None (slab path unchanged)."""
    _patch_land_loaders(monkeypatch)
    cfg = _small_cfg()._replace(use_multilayer_land=False)
    driver = ModelDriver(cfg, output_dir=tmp_path)
    driver.setup()
    assert driver._land_ml_state is None
    assert driver.physics.land_ml_cfg is None
    assert driver.physics.land_ml_params is None
    assert driver.physics.land_ml_lat is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
