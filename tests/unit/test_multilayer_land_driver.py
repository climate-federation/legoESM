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


def _latlon_cfg():
    """Lat-lon (res8 -> 8x16=128 cols) twin of _small_cfg. Exercises the
    1-D-lat column construction in _setup_multilayer_land (the cubed-sphere
    path assumes a 2-D per-cell grid.lat; lat-lon grid.lat is 1-D)."""
    return ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=8),
        dycore=DycoreConfig(dt=600.0, discretization="latlon_cgrid"),
        output=OutputConfig(diag_days=1),
        days=1, dataset="analytical",
        radiation="gray",
        land_mask_path="synthetic.nc",
        use_multilayer_land=True,
        multilayer_n_layers=6, multilayer_soil_depth=2.5,
    )


def _patch_land_loaders_latlon(monkeypatch):
    """Like _patch_land_loaders but the land-fraction field is the lat-lon
    per-cell shape (n_lat, n_lon), not grid.lat.shape (which is 1-D here)."""
    import legoesm.land.clm_surface_map as clm
    import legoesm.grids.topography as topo
    monkeypatch.setattr(clm, "download_clm_surfdata", lambda *a, **k: "synthetic")
    monkeypatch.setattr(clm, "load_clm_surface", _fake_surface_map)
    monkeypatch.setattr(
        topo, "load_land_fraction",
        lambda grid, path, *a, **k: jnp.full(grid.grid_shape_2d, 0.5))


def test_setup_multilayer_land_latlon(monkeypatch, tmp_path):
    """Regression: _setup_multilayer_land works on a lat-lon grid (1-D grid.lat).

    Previously ``flatten_2d(grid.lat)`` tried to reshape the (n_lat,) row vector
    into ncol and raised ``cannot reshape array of shape (24,) into shape 1152``.
    The per-column lat/lon must be the broadcast (n_lat, n_lon) meshgrid, flat
    C-order, so land column k aligns with atm column k."""
    _patch_land_loaders_latlon(monkeypatch)
    driver = ModelDriver(_latlon_cfg(), output_dir=tmp_path)
    driver.setup()  # must not raise the reshape TypeError

    n_lat, n_lon = 8, 16
    ncol = n_lat * n_lon
    assert driver.physics.land_ml_lat is not None
    assert driver.physics.land_ml_lat.shape == (ncol,)
    assert driver._land_ml_state.T_soil.shape == (ncol, 6)
    # Column ordering: lat repeats n_lon times (lat-major, lon fastest), matching
    # the atm state's flatten of a (n_lat, n_lon) field — land col k == atm col k.
    lat1d = np.asarray(driver.grid.lat)
    lon1d = np.asarray(driver.grid.lon)
    np.testing.assert_allclose(
        np.asarray(driver.physics.land_ml_lat), np.repeat(lat1d, n_lon), rtol=1e-6)
    # (lon tiled across lat bands is the companion ordering)
    assert np.allclose(
        np.asarray(driver.physics.land_ml_lat)[:n_lon], lat1d[0])  # first band = lat[0]
    del lon1d


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


def test_build_training_segment_land_gradient(monkeypatch, tmp_path):
    """build_training_segment yields a DIFFERENTIABLE coupled segment whose land
    surface temperature carries a finite, non-zero gradient w.r.t. the land params
    fed through the pipeline attribute — the coupled-calibration mechanism
    (scripts/run/train_coupled_land_era5.py).  Synthetic loam soil keeps the
    van-Genuchten backward inside float32 (real stiff-clay soils need fp64)."""
    import jax
    import jax.numpy as jnp
    from legoesm.land.carbon.stomata import StomataConfig
    from legoesm.land.carbon.config import CarbonConfig
    from legoesm.land.carbon.carbon_cycle import init_carbon_state

    _patch_land_loaders(monkeypatch)
    driver = ModelDriver(_small_cfg(), output_dir=tmp_path)
    driver.setup()
    pipe = driver.physics
    # Trainable surface exchange: MOST (z0 active) + Farquhar stomata (Vc_max25/
    # g1/LCMA active via a prescribed carbon state).
    pipe.land_ml_cfg = pipe.land_ml_cfg._replace(
        bulk_scheme="most", stomata=StomataConfig(enabled=True),
        carbon=CarbonConfig(scheme="differland"))
    ncol = int(driver.grid.lat.size)
    pipe.land_ml_carbon = init_carbon_state((ncol,), pipe.land_ml_cfg.carbon)
    base_lp = pipe.land_ml_params

    run_seg, carry0, forcing = driver.build_training_segment(4)
    # forward is finite (real coupled atmosphere + land)
    fin0 = run_seg(carry0, 4, forcing)
    assert jnp.all(jnp.isfinite(fin0.land_ml.T_soil))

    def loss(scale):
        # feed a TRACED scaling of the land roughness through the pipeline attribute
        pipe.land_ml_params = base_lp._replace(z0=base_lp.z0 * scale)
        fin = run_seg(carry0, 4, forcing)
        return jnp.sum(fin.land_ml.T_soil[:, 0] ** 2)

    g = float(jax.grad(loss)(1.0))
    pipe.land_ml_params = base_lp           # drop the escaped tracer
    assert np.isfinite(g), "coupled land gradient is non-finite"
    assert g != 0.0, "coupled land gradient is zero (params not reaching the flux)"


def test_multilayer_no_mask_still_enables_tiled_surface(monkeypatch, tmp_path):
    """use_multilayer_land + surface_tiled WITHOUT a land mask (f_land from
    --topography) must actually ENABLE the tiled turbulent-flux path — i.e. set
    physics.surface_tiled=True.  Regression guard: physics.surface_tiled was
    previously threaded only inside the slab/mask activation branch, so a mask-free
    multilayer config (the SOTA amip_sota.yaml case) validated but silently no-op'd
    the tiled surface.  gaussian topography gives f_land>0 so _has_land is true."""
    _patch_land_loaders(monkeypatch)
    cfg = ExperimentConfig(
        grid=GridConfig(resolution=8, nlev=8),
        dycore=DycoreConfig(dt=600.0),
        output=OutputConfig(diag_days=1),
        days=1, dataset="analytical", radiation="gray",
        topography="gaussian",            # elevation-derived f_land, NO mask
        surface_tiled=True, turbulence="louis",
        use_multilayer_land=True,
        multilayer_n_layers=6, multilayer_soil_depth=2.5,
    )
    cfg.validate_strict()  # the mask-free multilayer+tiled config is valid
    driver = ModelDriver(cfg, output_dir=tmp_path)
    driver.setup()

    assert bool(jnp.any(driver._f_land > 0)), "gaussian topo should give some land"
    assert driver.physics.surface_tiled is True, \
        "mask-free multilayer must still enable the tiled surface (not a silent no-op)"
    assert driver.physics.slab_land_active is False, \
        "multilayer land must not activate the slab tile"
    assert driver.physics.land_ml_cfg is not None, "multilayer land was set up"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
