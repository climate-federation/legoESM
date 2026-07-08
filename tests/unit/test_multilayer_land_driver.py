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


def test_setup_dispatches_land_surface_scheme(monkeypatch, tmp_path):
    """land_surface_scheme dispatches the right surface scheme onto the
    multilayer land config (issue #730): default -> SimpleSEB, 'two_leaf' ->
    the DifferBESS two-leaf canopy (Kelvin h_r bare-soil + stomatal transp.)."""
    from legoesm.land.surface_scheme import SimpleSEBConfig, TwoLeafCanopyConfig
    _patch_land_loaders(monkeypatch)

    driver_seb = ModelDriver(_small_cfg(), output_dir=tmp_path / "seb")
    driver_seb.setup()
    assert isinstance(
        driver_seb.physics.land_ml_cfg.surface_scheme, SimpleSEBConfig)

    driver_two_leaf = ModelDriver(
        _small_cfg()._replace(land_surface_scheme="two_leaf"),
        output_dir=tmp_path / "twoleaf")
    driver_two_leaf.setup()
    assert isinstance(
        driver_two_leaf.physics.land_ml_cfg.surface_scheme, TwoLeafCanopyConfig)


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


def test_setup_seeds_aridity_aware_soil_moisture(monkeypatch, tmp_path):
    """#730/#837: the seeded soil column tracks the IC near-surface RH, not the
    legacy moisture-uniform 0.5*theta_sat.  Verifies the driver wires
    q_v/p_s -> RH -> aridity_theta_init with the per-column CLM theta_wp/theta_fc."""
    from legoesm.land import aridity_theta_init
    from legoesm.thermo import saturation_mixing_ratio
    _patch_land_loaders(monkeypatch)
    driver = ModelDriver(_small_cfg(), output_dir=tmp_path)
    driver.setup()

    ad = driver.physics.adapter
    T_low = ad.flatten_2d(driver.state.T.data[..., -1]).reshape(-1)
    _qv = driver.q_v
    q_v_low = ad.flatten_2d(getattr(_qv, "data", _qv)[..., -1]).reshape(-1)
    p_s = ad.flatten_2d(
        getattr(driver.state.p_s, "data", driver.state.p_s)).reshape(-1)
    rh = np.asarray(q_v_low) / np.maximum(
        np.asarray(saturation_mixing_ratio(jnp.asarray(T_low), jnp.asarray(p_s))), 1e-12)
    # Use the SAME per-column thresholds the driver + tile beta read: the
    # PFT-weighted plant btran theta_wp/theta_fc from clm_multilayer_setup (NOT
    # the raw soil theta_wp/fc in the fake map -- clm_multilayer_setup overrides
    # them with the tuned PFT btran values).
    params = driver.physics.land_ml_params
    wp = np.asarray(params.theta_wp); fc = np.asarray(params.theta_fc)
    expected = np.asarray(aridity_theta_init(jnp.asarray(rh), jnp.asarray(wp), jnp.asarray(fc)))

    theta = np.asarray(driver._land_ml_state.theta_soil)   # (ncol, nlayers)
    # every layer seeded to the per-column aridity value (float32 state -> loose tol)
    np.testing.assert_allclose(theta[:, 0], expected, rtol=2e-3, atol=2e-3)
    for k in range(theta.shape[1]):
        np.testing.assert_allclose(theta[:, k], theta[:, 0], rtol=1e-6)
    # bounded to the per-column plant-available range, and NOT the legacy uniform
    # 0.5*theta_sat seed (which is aridity-blind).
    assert np.all((theta[:, 0] >= wp - 1e-4) & (theta[:, 0] <= fc + 1e-4))
    assert not np.allclose(theta, 0.5 * np.asarray(driver.physics.land_ml_cfg.hydraulics.theta_sat))


# NOTE: multilayer land on the LAT-LON grid is a separate, pre-existing setup
# limitation (``_setup_multilayer_land`` flattens the 2-D-per-cell ``grid.lat``
# via the adapter, but lat-lon ``grid.lat`` is 1-D, so the setup crashes BEFORE
# the aridity seeding runs).  #837's aridity fix is validated on the cubed-sphere
# path (``test_setup_seeds_aridity_aware_soil_moisture``); enabling multilayer
# land on lat-lon is an out-of-scope follow-up.


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_land_ic_path_overrides_cold_start(monkeypatch, tmp_path):
    """#746: a spun-up land restart (land_ic_path) REPLACES the cold-start
    multilayer soil column at setup — the driver loads the equilibrated state
    bit-for-bit and skips init_multilayer_land_state.  Regresses the day-0
    cold-start shock behind the land cloud-albedo cold trap."""
    from legoesm.land.restart import save_land_restart

    _patch_land_loaders(monkeypatch)

    # 1) Build a driver, grab its cold-start state, perturb it to a distinct
    #    "spun-up" column (warmer deep soil, drier top), save as a restart.
    #    (src/dst use SEPARATE output dirs — the run-manifest guard refuses to
    #    mix two configs' provenance in one directory.)
    src = ModelDriver(_small_cfg(), output_dir=tmp_path / "src")
    src.setup()
    ncol = src.grid.lat.size
    n_layers = 6
    seed = src._land_ml_state
    spun = seed._replace(
        T_soil=jnp.asarray(np.asarray(seed.T_soil) + 7.5),      # +7.5 K deep soil
        theta_soil=jnp.asarray(np.asarray(seed.theta_soil) * 0.6),
    )
    ic = tmp_path / "land_ic.npz"
    save_land_restart(ic, spun, land_mode="multilayer",
                      t_end_s=20 * 365 * 86400.0,
                      n_steps_completed=1, metadata={})

    # 2) A fresh driver with land_ic_path set must load THAT column, not the
    #    cold start.
    cfg = _small_cfg()._replace(land_ic_path=str(ic))
    dst = ModelDriver(cfg, output_dir=tmp_path / "dst")
    dst.setup()

    assert dst._land_ml_state.T_soil.shape == (ncol, n_layers)
    # Bit-identical to the saved spun-up column (modulo the storage-dtype cast),
    # and DISTINCT from the cold start (proves the override, not a coincidence).
    np.testing.assert_allclose(
        np.asarray(dst._land_ml_state.T_soil),
        np.asarray(spun.T_soil), rtol=1e-6, atol=1e-4)
    np.testing.assert_allclose(
        np.asarray(dst._land_ml_state.theta_soil),
        np.asarray(spun.theta_soil), rtol=1e-6, atol=1e-6)
    assert float(np.max(np.abs(
        np.asarray(dst._land_ml_state.T_soil) - np.asarray(seed.T_soil)))) > 5.0

    # The skin carry T_land must ALSO seed from the spun-up TOP-SOIL (not the
    # cold-start air temp), so the first step's land turbulent fluxes are
    # consistent with the spun-up column (codex #746: else the day-0 shock
    # leaks into the BL at the closure seam).
    ctx = dst._prepare_run_context(0, cfg.start_day, restore_carry=False)
    T_land = np.asarray(ctx["T_land"])
    expected_skin = np.asarray(spun.T_soil[:, 0]).reshape(T_land.shape)
    np.testing.assert_allclose(T_land, expected_skin, rtol=1e-6, atol=1e-4)


def test_land_ic_path_wrong_grid_raises(monkeypatch, tmp_path):
    """A restart whose n_layers/ncol don't match the run's grid must raise on
    load, not silently reshape (load_land_restart validates shapes)."""
    from legoesm.land.restart import save_land_restart

    _patch_land_loaders(monkeypatch)
    src = ModelDriver(_small_cfg(), output_dir=tmp_path / "src")
    src.setup()
    ic = tmp_path / "land_ic_6lay.npz"
    save_land_restart(ic, src._land_ml_state, land_mode="multilayer",
                      t_end_s=0.0, n_steps_completed=0, metadata={})

    # Run config asks for 10 layers; the restart has 6 -> mismatch -> raise
    # (separate output dir so this is the SHAPE guard, not the manifest guard).
    cfg = _small_cfg()._replace(land_ic_path=str(ic), multilayer_n_layers=10)
    dst = ModelDriver(cfg, output_dir=tmp_path / "dst")
    with pytest.raises((ValueError, AssertionError)):
        dst.setup()


def test_setup_multilayer_land_on_latlon_grid(monkeypatch, tmp_path):
    """#869/#837 follow-up: the LAT-LON grid stores 1-D lat/lon axes; the land
    setup's flatten_2d(grid.lat) raised "cannot reshape (n_lat,) into ncol" and
    killed every latlon use_multilayer_land run at setup (the production
    latlon24 lane).  The setup must broadcast the axes to the (n_lat, n_lon)
    cell grid and seed a full-ncol state."""
    from legoesm.land.state import MultiLayerLandState

    _patch_land_loaders(monkeypatch)
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=8),
        dycore=DycoreConfig(dt=600.0, discretization="latlon_cgrid"),
        output=OutputConfig(diag_days=1),
        days=1, dataset="analytical",
        radiation="gray",
        land_mask_path="synthetic.nc",
        use_multilayer_land=True,
        multilayer_n_layers=6, multilayer_soil_depth=2.5,
    )
    driver = ModelDriver(cfg, output_dir=tmp_path)
    driver.setup()          # raised TypeError (reshape) before the fix

    st = driver._land_ml_state
    assert isinstance(st, MultiLayerLandState)
    ncol = driver.grid.lat.size * driver.grid.lon.size   # n_lat * n_lon
    assert st.T_soil.shape == (ncol, 6)
    assert driver.physics.land_ml_lat.shape == (ncol,)
    # lat must VARY across columns (a broadcast bug that tiled one row would
    # leave it constant).
    import numpy as _np2
    assert _np2.unique(_np2.asarray(driver.physics.land_ml_lat)).size > 1
