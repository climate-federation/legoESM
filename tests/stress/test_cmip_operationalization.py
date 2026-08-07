"""CMIP operationalization tests.

Covers:
1. perf-mode + CMIP output incompatibility
2. Structured-grid CMIP regridding
3. Grid-aware experiment config factory
4. CMIP diagnostic correctness (no misleading variable names)
5. Incremental CMIP monthly write path
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# =====================================================================
# 1. perf-mode + CMIP output
# =====================================================================

class TestPerfModeCMIP:
    """cmip_output=True must never be silenced by diagnostics_perf_mode."""

    def test_perf_mode_always_with_cmip_produces_files(self, tmp_path):
        """diagnostics_perf_mode='always' + cmip_output=True still writes
        NetCDF files (perf mode is overridden)."""
        from legoesm.forcing.experiments import create_experiment_config
        from legoesm.driver.model_driver import ModelDriver

        cfg = create_experiment_config(
            "piControl",
            resolution=8, nlev=5, dt=600, days=35,
        )
        cfg = cfg._replace(
            output=cfg.output._replace(
                cmip_output=True,
                diagnostics_perf_mode="always",
                diag_days=5,
            ),
        )
        driver = ModelDriver(cfg, output_dir=str(tmp_path))
        driver.setup()
        driver.run()

        cmor_dir = tmp_path / "cmor"
        nc_files = list(cmor_dir.rglob("*.nc"))
        assert len(nc_files) > 0, (
            "cmip_output=True + diagnostics_perf_mode='always' produced "
            "zero NetCDF files — perf mode was not overridden"
        )

    def test_config_warns_perf_mode_cmip(self):
        """Config validation warns when perf_mode='always' + cmip_output."""
        from legoesm.driver.config import ExperimentConfig, OutputConfig
        cfg = ExperimentConfig(
            output=OutputConfig(
                cmip_output=True,
                diagnostics_perf_mode="always",
            ),
        )
        warns = cfg.validate()
        assert any("perf_mode" in w and "cmip" in w.lower() for w in warns), (
            f"Expected warning about perf_mode + CMIP, got: {warns}"
        )


# =====================================================================
# 2. Structured-grid CMIP regridding
# =====================================================================

class TestStructuredGridCMIP:
    """CMIP output works for native lat-lon grids with regridding."""

    def test_structured_regrid_2d(self):
        """Bilinear interpolation from native → CMIP lat-lon works."""
        from legoesm.driver.diagnostics import (
            _build_structured_regrid_weights,
            _apply_structured_regrid_2d,
        )
        # Source: 16x32 grid
        src_lat = np.linspace(-np.pi / 2, np.pi / 2, 16)
        src_lon = np.linspace(0, 2 * np.pi, 32, endpoint=False)

        # Target: different resolution (36x72 = 5-degree)
        w = _build_structured_regrid_weights(src_lat, src_lon, 36, 72)

        # Apply to a smooth test field
        lat2d, lon2d = np.meshgrid(src_lat, src_lon, indexing='ij')
        field = np.cos(lat2d) * np.sin(lon2d)  # (16, 32)
        result = _apply_structured_regrid_2d(field, w)

        assert result.shape == (36, 72)
        assert np.all(np.isfinite(result))
        # Value range should be within source range
        assert result.min() >= field.min() - 0.01
        assert result.max() <= field.max() + 0.01

    def test_structured_regrid_3d(self):
        """3-D bilinear interpolation preserves shape."""
        from legoesm.driver.diagnostics import (
            _build_structured_regrid_weights,
            _apply_structured_regrid_3d,
        )
        src_lat = np.linspace(-np.pi / 2, np.pi / 2, 16)
        src_lon = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        w = _build_structured_regrid_weights(src_lat, src_lon, 36, 72)

        field = np.random.randn(16, 32, 5)
        result = _apply_structured_regrid_3d(field, w)
        assert result.shape == (36, 72, 5)
        assert np.all(np.isfinite(result))

    def test_voronoi_cmip_regrids(self):
        """Voronoi/MPAS grid with CMIP output is regridded to lat-lon (IDW),
        not rejected — the unstructured CMIP path the diagnostics collector
        wires up via ``compute_voronoi_to_latlon_weights``."""
        from legoesm.grids.regridding import (
            compute_voronoi_to_latlon_weights,
            apply_voronoi_to_latlon,
        )
        n_cells = 64
        lat_cell = np.linspace(-np.pi / 2, np.pi / 2, n_cells)  # radians
        lon_cell = np.linspace(0, 2 * np.pi, n_cells, endpoint=False)
        w = compute_voronoi_to_latlon_weights(
            lat_cell, lon_cell, n_lon=72, n_lat=36,
        )
        result = apply_voronoi_to_latlon(np.cos(lat_cell), w)  # (nCells,) -> (36,72)
        assert result.shape == (36, 72)
        assert np.all(np.isfinite(result))


# =====================================================================
# 3. Grid-aware experiment config factory
# =====================================================================

class TestGridAwareExperimentConfig:
    """create_experiment_config produces valid configs for all grid types."""

    def test_cubed_sphere_default_unchanged(self):
        """Default cubed-sphere config keeps cdgrid discretization."""
        from legoesm.forcing.experiments import create_experiment_config
        cfg = create_experiment_config("piControl", resolution=8)
        assert cfg.dycore.discretization == "cdgrid"
        assert cfg.dycore.model_type == "hydrostatic"

    def test_latlon_auto_discretization(self):
        """grid_type='latlon' auto-selects latlon_cgrid discretization."""
        from legoesm.forcing.experiments import create_experiment_config
        cfg = create_experiment_config(
            "piControl", grid_type="latlon", resolution=16,
        )
        assert cfg.dycore.discretization == "latlon_cgrid"
        assert cfg.dycore.model_type == "hydrostatic"
        assert cfg.grid.grid_type == "latlon"

    def test_gaussian_auto_discretization(self):
        """grid_type='gaussian' auto-selects spectral discretization."""
        from legoesm.forcing.experiments import create_experiment_config
        cfg = create_experiment_config(
            "piControl", grid_type="gaussian", resolution=21,
        )
        assert cfg.dycore.discretization == "spectral"
        assert cfg.dycore.model_type == "spectral_pe"

    def test_explicit_override_preserved(self):
        """User-provided discretization overrides auto-detection."""
        from legoesm.forcing.experiments import create_experiment_config
        cfg = create_experiment_config(
            "piControl",
            grid_type="latlon",
            discretization="cdgrid",  # explicit — should be kept
        )
        assert cfg.dycore.discretization == "cdgrid"

    def test_unsupported_grid_type_raises(self):
        """Unknown grid_type raises ValueError."""
        from legoesm.forcing.experiments import create_experiment_config
        with pytest.raises(ValueError, match="Unsupported grid_type"):
            create_experiment_config("piControl", grid_type="hexagonal")


# =====================================================================
# 4. CMIP diagnostic correctness
# =====================================================================

class TestCMIPDiagnosticCorrectness:
    """CMIP output does not publish incorrect standard variables."""

    def test_rsds_rlds_absent_clt_published(self, tmp_path):
        """rsds/rlds stay unpublished (net != downwelling); clt IS published.

        ``clt`` now derives from the model's fractional layer cloud fraction
        (the shared ``compute_cloud_properties``, here sundqvist) reduced by
        maximum-random overlap, replacing the retired near-binary condensate
        mask that saturated it to ~100% (issue #689).
        """
        from legoesm.forcing.experiments import create_experiment_config
        from legoesm.driver.model_driver import ModelDriver

        cfg = create_experiment_config(
            "piControl", resolution=8, nlev=5, dt=600, days=35,
        )
        cfg = cfg._replace(
            cloud_scheme="sundqvist",
            output=cfg.output._replace(cmip_output=True, diag_days=5),
        )
        driver = ModelDriver(cfg, output_dir=str(tmp_path))
        driver.setup()
        driver.run()

        cmor_dir = tmp_path / "cmor"
        nc_files = list(cmor_dir.rglob("*.nc"))
        written_vars = set()
        for nc_path in nc_files:
            # Variable name is the first part of the DRS filename
            parts = nc_path.stem.split("_")
            written_vars.add(parts[0])

        # rsds/rlds remain unpublished (runtime exposes net, not downwelling).
        assert "rsds" not in written_vars, "rsds should not be published (net != downwelling)"
        assert "rlds" not in written_vars, "rlds should not be published (net != downwelling)"
        # clt is published from the real cloud fraction (issue #689).
        assert "clt" in written_vars, (
            "clt should be published from the model cloud fraction "
            "(sundqvist + maximum-random overlap, #689)"
        )

        # Value check: clt must be a bounded CMIP percent and NOT saturated at
        # ~100% everywhere (the original #689 bug from the binary-mask overlap).
        try:
            import xarray as xr
        except ImportError:
            pytest.skip("xarray not installed")
        clt_files = list(cmor_dir.rglob("clt_*.nc"))
        assert clt_files, "clt file should exist"
        ds = xr.open_dataset(clt_files[0])
        clt = ds["clt"].values
        assert np.all(np.isfinite(clt)), "clt must be finite"
        assert clt.min() >= -1e-6 and clt.max() <= 100.0 + 1e-6, (
            f"clt out of [0, 100]%: min={clt.min()}, max={clt.max()}"
        )
        # Fractional cloud + max-random overlap cannot pin every cell to ~100%
        # the way the retired binary condensate mask did.
        assert float(np.mean(clt)) < 99.0, (
            f"clt saturated (mean={np.mean(clt):.2f}%); the fractional "
            "cloud-fraction diagnostic should not report near-total cover"
        )
        ds.close()

    def test_rsdt_is_written(self, tmp_path):
        """rsdt (TOA incoming SW) IS correctly written."""
        from legoesm.forcing.experiments import create_experiment_config
        from legoesm.driver.model_driver import ModelDriver

        cfg = create_experiment_config(
            "piControl", resolution=8, nlev=5, dt=600, days=35,
        )
        cfg = cfg._replace(
            output=cfg.output._replace(cmip_output=True, diag_days=5),
        )
        driver = ModelDriver(cfg, output_dir=str(tmp_path))
        driver.setup()
        driver.run()

        cmor_dir = tmp_path / "cmor"
        nc_files = list(cmor_dir.rglob("*.nc"))
        written_vars = set()
        for nc_path in nc_files:
            parts = nc_path.stem.split("_")
            written_vars.add(parts[0])

        assert "rsdt" in written_vars, (
            f"rsdt should be published (correctly available). Got: {written_vars}"
        )


# =====================================================================
# 5. Incremental CMIP write path
# =====================================================================

class TestIncrementalCMIPWrite:
    """CMIP monthly data is flushed incrementally, not only at end-of-run."""

    def test_pop_completed_months(self):
        """SpatialMonthlyAccumulator.pop_completed_months frees memory."""
        from legoesm.diagnostics.monthly_means import SpatialMonthlyAccumulator

        acc = SpatialMonthlyAccumulator(nlat=4, nlon=8)
        # Add data for Jan, Feb, Mar
        acc.add_2d(15, 0, {"tas": np.ones((4, 8))})   # Jan
        acc.add_2d(45, 0, {"tas": np.ones((4, 8)) * 2})  # Feb
        acc.add_2d(75, 0, {"tas": np.ones((4, 8)) * 3})  # Mar

        # Pop completed months when currently in March (month 3)
        data = acc.pop_completed_months(current_year=0, current_month=3)
        months = data['months']
        assert len(months) == 2, f"Expected Jan+Feb, got {months}"
        assert (0, 1) in months  # Jan
        assert (0, 2) in months  # Feb

        # Jan and Feb data should be gone from accumulator
        assert (0, 1) not in acc._data_2d
        assert (0, 2) not in acc._data_2d
        # March should still be there
        assert (0, 3) in acc._data_2d

    def test_flush_cmip_monthly_writes_incrementally(self, tmp_path):
        """flush_cmip_monthly writes completed months without waiting for save()."""
        from legoesm.driver.diagnostics import DiagnosticCollector
        sigma = np.linspace(0.05, 0.95, 5)
        dsigma = np.ones(5) / 5

        dc = DiagnosticCollector(
            nlev=5,
            sigma_full=sigma,
            dsigma=dsigma,
            cmip_output=True,
            cmip_resolution_deg=90.0,  # very coarse for speed: 2x4 grid
            output_dir=str(tmp_path),
        )
        # Simulate cubed-sphere grid with CMIP regrid weights
        dc._cmip_start_year = 1850
        dc._cmip_grid_type = "latlon"

        # Manually add data for January and February
        dc._spatial_monthly.add_2d(15, 0, {"tas": np.ones((2, 4)) * 280.0})
        dc._spatial_monthly.add_2d(45, 0, {"tas": np.ones((2, 4)) * 285.0})

        # Flush while "currently in March" (day 75 of year 0)
        # day = year * 365 + doy - 1 ≈ 75
        dc.flush_cmip_monthly(current_day=75.0)

        # Should have written files for Jan (tas)
        cmor_dir = tmp_path / "cmor"
        nc_files = list(cmor_dir.rglob("*.nc"))
        assert len(nc_files) > 0, (
            "flush_cmip_monthly did not write any files"
        )

    def test_cfwriter_append_not_rewrite(self, tmp_path):
        """CFWriter.write_field appends without full rewrite when netCDF4 available."""
        from legoesm.io.cmor_output import CFWriter

        writer = CFWriter(
            output_dir=str(tmp_path),
            experiment_id="test",
            model_id="legoESM-test",
        )

        lat = np.linspace(-90, 90, 4)
        lon = np.linspace(0, 350, 8)
        data = np.ones((4, 8))

        # Write first time step
        p1 = writer.write_field(
            "tas", data * 280, time=15.0, time_bounds=(0.0, 30.0),
            lat=lat, lon=lon,
        )

        # Write second time step (appends)
        p2 = writer.write_field(
            "tas", data * 285, time=45.0, time_bounds=(30.0, 60.0),
            lat=lat, lon=lon,
        )

        # ONE file, appended -- not a new file per step.  The path itself is
        # NOT stable any more: a CMIP6 DRS filename ends in the time range it
        # covers, so the file is renamed after every append as its range
        # grows (185001-185001 -> 185001-185003 here).  Assert the invariant
        # that actually matters -- a single growing file -- rather than the
        # pre-time-range assumption that the path never changes.
        tas_files = sorted(tmp_path.rglob("tas_*.nc"))
        assert len(tas_files) == 1, (
            f"expected exactly one appended tas file, got "
            f"{[f.name for f in tas_files]}"
        )
        assert tas_files[0] == p2, (
            f"write_field must return the CURRENT path; got {p2.name}, "
            f"on disk {tas_files[0].name}"
        )
        assert not p1.exists(), (
            f"{p1.name} should have been renamed as the time range grew"
        )
        assert p1 != p2, (
            "filename must encode the time range, so it changes on append"
        )

        # File should have 2 time steps
        import xarray as xr
        ds = xr.open_dataset(p2)
        assert ds.sizes["time"] == 2, f"Expected 2 time steps, got {ds.sizes['time']}"
        assert ds["tas"].shape == (2, 4, 8)
        ds.close()
        writer.close()
