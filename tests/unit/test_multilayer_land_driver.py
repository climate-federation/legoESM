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


def test_setup_clm_ml_warm_starts_and_threads_gridinfo(monkeypatch, tmp_path):
    """A coupled clm_ml setup (S2) warm-starts the canopy and threads a per-column
    grid_info into the pipeline — it no longer raises.

    The setup runs ONE eager cold canopy step to build the per-column vertical
    structure, grafts the warm canopy_state onto the cold-start land state, and
    stores the concrete per-column GridInfo tuple on the pipeline so the jitted run
    steps run the ncol>1 traceable canopy.  Requires the clm-ml-jax backend."""
    import pytest
    pytest.importorskip("legoesm.land.canopy.clm_ml_backend.multilayer_canopy")
    import inspect as _inspect
    from legoesm.land.canopy.clm_ml_backend.multilayer_canopy import MLCanopyFluxesMod as _mlmod
    if "cos_zenith_device" not in _inspect.signature(
            _mlmod.MLCanopyFluxes).parameters:
        pytest.skip("clm-ml-jax build lacks MLCanopyFluxes(cos_zenith_device=)")

    _patch_land_loaders(monkeypatch)
    # A tiny grid keeps the eager warm-start + O(ncol) machinery cheap.
    cfg = _small_cfg()._replace(land_surface_scheme="clm_ml",
                                grid=GridConfig(resolution=2, nlev=8))
    driver_clm = ModelDriver(cfg, output_dir=tmp_path / "clmml")
    driver_clm.setup()

    gi = driver_clm.physics.clm_ml_grid_info
    assert isinstance(gi, tuple) and len(gi) >= 1, (
        f"expected a per-column GridInfo tuple, got {type(gi)}")
    ncol = int(driver_clm._land_ml_state.T_soil.shape[0])
    assert len(gi) == ncol, f"grid_info has {len(gi)} entries for {ncol} columns"
    # Each entry is a concrete GridInfo with a warm-started structure (ncan>=1).
    assert all(int(g.ncan) >= 1 for g in gi)
    # The land state carries a WARM canopy (mlcanopy populated), not a cold None.
    cs = driver_clm._land_ml_state.canopy_state
    assert cs is not None and cs.mlcanopy is not None, "canopy not warm-started"


def test_clm_ml_pipeline_step_jits(monkeypatch, tmp_path):
    """The coupled land tile runs the CLM-ML canopy over ncol>1 INSIDE jax.jit (S2
    + coupler threading).  Exercises the exact coupled-segment mechanism — the
    pipeline threads the per-column grid_info + concrete dt into step_multilayer_land
    with a TRACED mlcanopy carry (as in the lax.scan segment) — via one jitted land
    step, ~100x cheaper than a full-day run (which is 288 canopy sub-steps x ncol).

    A crash would surface here (trace or first exec); a finite, advanced land state
    with a still-warm canopy proves the jitted ncol>1 coupled canopy works."""
    import pytest
    pytest.importorskip("legoesm.land.canopy.clm_ml_backend.multilayer_canopy")
    import inspect as _inspect
    import jax
    from legoesm.land.canopy.clm_ml_backend.multilayer_canopy import MLCanopyFluxesMod as _mlmod
    if "cos_zenith_device" not in _inspect.signature(
            _mlmod.MLCanopyFluxes).parameters:
        pytest.skip("clm-ml-jax build lacks MLCanopyFluxes(cos_zenith_device=)")

    _patch_land_loaders(monkeypatch)
    cfg = _small_cfg()._replace(land_surface_scheme="clm_ml",
                                grid=GridConfig(resolution=2, nlev=8))
    driver = ModelDriver(cfg, output_dir=tmp_path)
    driver.setup()
    ncol = int(driver._land_ml_state.T_soil.shape[0])

    # One jitted coupled land step: land_ml (with the warm mlcanopy) is a jit ARG,
    # so canopy_state.mlcanopy is a TRACER exactly like the segment's scan carry;
    # the pipeline supplies grid_info + concrete dt from setup.
    tile = driver.physics._step_multilayer_land_tile

    @jax.jit
    def _step(land_ml, T, p_s, q_v, u, v):
        # Pass a real per-column cos_zenith (as compute_radiation_core threads for
        # clm_ml) to exercise the faithful-zenith param end to end.
        return tile(land_ml, jnp.full(ncol, 400.0), jnp.full(ncol, 350.0),
                    T, p_s, q_v, u, v, None, 600.0,
                    cos_zenith_col=jnp.full(ncol, 0.7))

    land_new, T_sfc_col, _ = _step(
        driver._land_ml_state, driver.state.T.data, driver.state.p_s.data,
        driver.q_v, driver.state.u.data, driver.state.v.data)

    assert jnp.isfinite(T_sfc_col).all(), "coupled CLM-ML skin T not finite"
    assert T_sfc_col.shape[0] == ncol
    # The canopy carry stays warm (structure came from the threaded grid_info, so
    # the traced mlcanopy did not need a host int()).
    assert land_new.canopy_state is not None
    assert land_new.canopy_state.mlcanopy is not None


def test_effective_cos_zenith_physical(monkeypatch, tmp_path):
    """The land-tile solar-zenith helper returns cos in [0,1], finite, on both the
    diurnal (instantaneous) and non-diurnal (daily-mean-effective) branches — the
    value threaded into the CLM-ML canopy radiation.  No backend needed."""
    _patch_land_loaders(monkeypatch)
    driver = ModelDriver(_small_cfg(), output_dir=tmp_path)
    driver.setup()
    pipe = driver.physics
    lat = jnp.linspace(-1.4, 1.4, 8)   # radians
    lon = jnp.linspace(0.0, 6.0, 8)
    for diurnal in (True, False):
        pipe.diurnal_cycle = diurnal
        cz = np.asarray(pipe._effective_cos_zenith(lat, lon, 172.0, 43200.0, 1361.0))
        assert np.all(np.isfinite(cz)), f"non-finite cos_zenith (diurnal={diurnal})"
        assert np.all(cz >= 0.0) and np.all(cz <= 1.0), (
            f"cos_zenith outside [0,1] (diurnal={diurnal}): {cz}")


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


def test_multilayer_land_evolves_under_unfused_radiation(monkeypatch, tmp_path):
    """The UNFUSED radiation path (rad_update_steps>1, unfused_radiation=True)
    must still advance the soil. Before the C4 fix _run_rad discarded
    compute_radiation_core's advanced land state, so the Richards column FROZE
    across the whole run (the radiation cadence is the only place the tile is
    stepped in that mode). Regression: soil must change, not stay at the seed."""
    _patch_land_loaders(monkeypatch)
    cfg = _small_cfg()._replace(rad_update_steps=2, unfused_radiation=True)
    driver = ModelDriver(cfg, output_dir=tmp_path)
    driver.setup()
    T_soil_0 = np.asarray(driver._land_ml_state.T_soil).copy()

    status = driver.run()

    assert status == "COMPLETED"
    st = driver._land_ml_state
    assert st is not None
    assert not np.allclose(np.asarray(st.T_soil), T_soil_0), (
        "unfused-radiation soil column did not advance (frozen — the "
        "_run_rad land_ml discard regression)")
    assert np.all(np.isfinite(np.asarray(st.T_soil)))


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
    from legoesm.land.stomata import StomataConfig
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


# ---------------------------------------------------------------------------
# Transient land-use cover (LULC) — the jitted AMIP dynamic-cover path
# ---------------------------------------------------------------------------


def _write_transient_cover(path, years, pft_by_year):
    """Minimal transient legoesm_surfdata (only the vars the cover loader reads):
    lat/lon/year + pft_frac(year, npft, lat, lon) in PERCENT.  Coarse global grid so
    every driver column nearest-maps into it."""
    import xarray as xr
    from legoesm.land.surface_params import N_PFT_CLM5
    lat = np.array([-60.0, 0.0, 60.0])
    lon = np.array([0.0, 120.0, 240.0])
    pft = np.zeros((len(years), N_PFT_CLM5, lat.size, lon.size))
    for i, idx in enumerate(pft_by_year):
        pft[i, idx] = 90.0                       # 90% of the cell = that PFT
    ds = xr.Dataset(
        {"pft_frac": (("year", "npft", "lat", "lon"), pft)},
        coords={"year": np.asarray(years, float), "npft": np.arange(N_PFT_CLM5),
                "lat": lat, "lon": lon},
    )
    ds.to_netcdf(path)


def _transient_cfg(surfdata_path):
    return _small_cfg()._replace(
        transient_land_cover=True, land_cover_surfdata=str(surfdata_path))


def test_transient_land_ml_params_none_when_off(monkeypatch, tmp_path):
    """Transient cover OFF (default) => _transient_land_ml_params is None, so the
    jitted step falls back to the closure-baked pipeline.land_ml_params (the
    byte-identical static path)."""
    _patch_land_loaders(monkeypatch)
    driver = ModelDriver(_small_cfg(), output_dir=tmp_path)
    driver.setup()
    assert getattr(driver, "_land_cover_transient", "missing") is None
    assert driver._transient_land_ml_params(0.0) is None
    assert driver._transient_land_ml_params(3650.0) is None


def test_transient_cover_setup_reweights_by_year(monkeypatch, tmp_path):
    """Transient cover ON: setup loads the annual series and
    _transient_land_ml_params(day) re-weights the vegetation params per calendar
    year — forest at the start year, crop a decade later => different albedo_veg,
    while the frozen soil map is untouched."""
    from legoesm.land.surface_params import CLM5_PFT_NAMES
    idx = {n: i for i, n in enumerate(CLM5_PFT_NAMES)}
    p = tmp_path / "transient.nc"
    _write_transient_cover(
        p, years=[2000.0, 2010.0],
        pft_by_year=[idx["broadleaf_evergreen_tropical"], idx["crop_c3"]])

    _patch_land_loaders(monkeypatch)
    cfg = _transient_cfg(p)._replace(start_year=2000)
    driver = ModelDriver(cfg, output_dir=tmp_path)
    driver.setup()

    cover, years, rebuild = driver._land_cover_transient
    assert list(np.asarray(years)) == [2000.0, 2010.0]
    ncol = driver.grid.lat.size
    assert cover.shape == (2, ncol, 17)

    lp_2000 = driver._transient_land_ml_params(0.0)          # cover_year 2000
    lp_2010 = driver._transient_land_ml_params(10 * 365.0)   # cover_year 2010
    assert lp_2000 is not None and lp_2010 is not None
    # forest -> crop cover shift re-weights the vegetation albedo
    assert not np.allclose(np.asarray(lp_2000.albedo_veg),
                           np.asarray(lp_2010.albedo_veg))
    # out-of-range years clamp to the series endpoints (interp_annual), so a
    # pre-2000 / post-2010 day reuses the boundary cover rather than extrapolating
    np.testing.assert_allclose(
        np.asarray(driver._transient_land_ml_params(-3650.0).albedo_veg),
        np.asarray(lp_2000.albedo_veg))


def test_transient_cover_amip_run_completes(monkeypatch, tmp_path):
    """End-to-end: a multi-segment AMIP run with transient cover ON completes
    through the real jitted _run_compiled path.  Exercises Stages 1-3 together —
    the per-segment land_ml_params flows as a traced SegmentForcing leaf into the
    compiled step (stable pytree => no retrace/crash)."""
    from legoesm.land.surface_params import CLM5_PFT_NAMES
    idx = {n: i for i, n in enumerate(CLM5_PFT_NAMES)}
    p = tmp_path / "transient.nc"
    _write_transient_cover(
        p, years=[2000.0, 2010.0],
        pft_by_year=[idx["broadleaf_evergreen_tropical"], idx["crop_c3"]])
    _patch_land_loaders(monkeypatch)
    # 2 diagnostic days so the run spans >1 segment (cover advances between them).
    cfg = _transient_cfg(p)._replace(start_year=2000, days=2)
    driver = ModelDriver(cfg, output_dir=tmp_path)
    driver.setup()
    assert driver._land_cover_transient is not None

    status = driver.run()

    assert status == "COMPLETED"
    st = driver._land_ml_state
    assert st is not None and np.all(np.isfinite(np.asarray(st.T_soil)))


def test_jitted_radiation_reads_traced_land_ml_params(monkeypatch, tmp_path):
    """The 5th-issue fix: a JITTED compute_radiation_core reads its per-call
    (traced) land_ml_params, not the closure-baked self.land_ml_params.  A
    brightened cover changes the surface net SW under the jit; passing None
    reproduces the baked-self result byte-for-byte (static path unchanged)."""
    import jax
    from legoesm import constants
    _patch_land_loaders(monkeypatch)
    driver = ModelDriver(_small_cfg(), output_dir=tmp_path)
    driver.setup()
    pipe = driver.physics
    land_ml = driver._land_ml_state

    g2 = tuple(driver.grid.lat.shape)
    nlev = _small_cfg().grid.nlev
    T = jnp.full((*g2, nlev), 285.0)
    p_s = jnp.full(g2, 1.0e5)
    q_v = jnp.full((*g2, nlev), 0.005)
    sst = jnp.full(g2, 290.0)
    sic = jnp.zeros(g2)
    lat = jnp.asarray(driver.grid.lat)
    lon = jnp.asarray(driver.grid.lon)
    u = jnp.full((*g2, nlev), 3.0)
    v = jnp.zeros((*g2, nlev))

    # The multilayer skin T rides SegmentCarry.T_land alongside land_ml (the tile
    # writes T_sfc back into it), so production always carries a concrete T_land.
    T_land = jnp.full(g2, 288.0)

    def _call(land_ml_params):
        return pipe.compute_radiation_core(
            T, p_s, q_v, sst, sic, lat, lon, 1.0, 0.0,
            jnp.zeros(0), constants.S_0, None, None,
            u=u, v=v, dt=600.0, T_land=T_land,
            land_ml=land_ml, land_ml_params=land_ml_params)

    rad = jax.jit(_call)
    base = pipe.land_ml_params
    bright = base._replace(
        albedo_veg=jnp.clip(jnp.zeros_like(base.albedo_veg) + 0.8, 0.0, 1.0))

    out_none = rad(None)      # traced None -> baked self.land_ml_params
    out_base = rad(base)      # explicit self
    out_bright = rad(bright)  # brightened cover

    sw_net_sfc = 1  # compute_radiation_core output index
    # None path == explicitly passing the baked params (byte-identical static path)
    np.testing.assert_array_equal(np.asarray(out_none[sw_net_sfc]),
                                  np.asarray(out_base[sw_net_sfc]))
    # a brighter land cover reflects more SW -> different surface net SW, proving
    # the jitted graph consumed the TRACED param (not the closure-baked attribute)
    assert not np.allclose(np.asarray(out_base[sw_net_sfc]),
                           np.asarray(out_bright[sw_net_sfc]))
