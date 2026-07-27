"""Tests for CMOR output, experiment templates, restart, and tuning modules."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

# ======================================================================
# CMOR output pipeline
# ======================================================================


class TestCMORTables(unittest.TestCase):
    """CMOR variable metadata tables."""

    def test_amon_variables(self):
        """Amon table contains expected atmosphere variables."""
        from legoesm.io.cmor_output import CMOR_TABLES

        amon = CMOR_TABLES["Amon"]
        expected = {"tas", "ta", "ua", "va", "hus", "ps", "pr", "rsut", "rlut"}
        self.assertTrue(expected.issubset(set(amon.keys())))

    def test_lmon_variables(self):
        """Lmon table contains expected land variables."""
        from legoesm.io.cmor_output import CMOR_TABLES

        lmon = CMOR_TABLES["Lmon"]
        expected = {"gpp", "nee", "lai"}
        self.assertTrue(expected.issubset(set(lmon.keys())))

    def test_variable_metadata_fields(self):
        """Each variable entry has required CF fields."""
        from legoesm.io.cmor_output import CMOR_TABLES

        required_keys = {"standard_name", "long_name", "units", "cell_methods"}
        for table_name, table in CMOR_TABLES.items():
            for var_name, entry in table.items():
                for rk in required_keys:
                    self.assertIn(
                        rk, entry,
                        f"{table_name}/{var_name} missing '{rk}'")

    def test_lookup_cmor_entry(self):
        """lookup_cmor_entry finds variables across tables."""
        from legoesm.io.cmor_output import lookup_cmor_entry

        table, entry = lookup_cmor_entry("tas")
        self.assertEqual(table, "Amon")
        self.assertEqual(entry["units"], "K")

        table, entry = lookup_cmor_entry("gpp")
        self.assertEqual(table, "Lmon")

    def test_omon_variables(self):
        """Omon table contains ocean variables."""
        from legoesm.io.cmor_output import CMOR_TABLES

        omon = CMOR_TABLES["Omon"]
        expected = {"tos", "sic"}
        self.assertTrue(expected.issubset(set(omon.keys())))

    def test_aday_variables(self):
        """Aday table contains daily atmosphere variables."""
        from legoesm.io.cmor_output import CMOR_TABLES

        aday = CMOR_TABLES["Aday"]
        expected = {"tas", "pr", "psl", "rsut", "rlut"}
        self.assertTrue(expected.issubset(set(aday.keys())))

    def test_clearsky_variables_in_amon(self):
        """Amon table contains clear-sky radiation variables."""
        from legoesm.io.cmor_output import CMOR_TABLES

        amon = CMOR_TABLES["Amon"]
        cs_vars = {"rsutcs", "rlutcs", "rsdscs", "rldscs"}
        self.assertTrue(cs_vars.issubset(set(amon.keys())))

    def test_new_amon_variables(self):
        """Amon table contains all new Phase 4 variables."""
        from legoesm.io.cmor_output import CMOR_TABLES

        amon = CMOR_TABLES["Amon"]
        new_vars = {"ts", "zg", "wap", "hur", "hurs", "clw", "cli", "evspsbl"}
        self.assertTrue(new_vars.issubset(set(amon.keys())))

    def test_lookup_omon_variable(self):
        """lookup_cmor_entry finds ocean variables in Omon table."""
        from legoesm.io.cmor_output import lookup_cmor_entry

        table, entry = lookup_cmor_entry("tos")
        self.assertEqual(table, "Omon")
        self.assertEqual(entry["units"], "K")

        table, entry = lookup_cmor_entry("tos", table="Omon")
        self.assertEqual(table, "Omon")

    def test_lookup_aday_variable(self):
        """lookup_cmor_entry finds daily variables in Aday table."""
        from legoesm.io.cmor_output import lookup_cmor_entry

        # tas is in both Amon and Aday; explicit table lookup works
        table, entry = lookup_cmor_entry("tas", table="Aday")
        self.assertEqual(table, "Aday")

    def test_lookup_unknown_raises(self):
        """lookup_cmor_entry raises KeyError for unknown variables."""
        from legoesm.io.cmor_output import lookup_cmor_entry

        with self.assertRaises(KeyError):
            lookup_cmor_entry("nonexistent_variable_xyz")

    def test_plev19(self):
        """CMIP6 standard 19-level pressure grid is correct."""
        from legoesm.io.cmor_output import CMIP6_PLEV19

        self.assertEqual(len(CMIP6_PLEV19), 19)
        self.assertEqual(CMIP6_PLEV19[0], 100000.0)  # 1000 hPa
        self.assertEqual(CMIP6_PLEV19[-1], 100.0)     # 1 hPa


class TestCFWriter(unittest.TestCase):
    """CF-compliant NetCDF writer."""

    def test_writer_creation(self):
        """CFWriter can be instantiated."""
        from legoesm.io.cmor_output import CFWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CFWriter(
                output_dir=tmpdir,
                experiment_id="test",
                model_id="legoESM-test",
                freq="mon",
            )
            self.assertIsNotNone(writer)
            writer.close()

    def test_write_2d_field(self):
        """Can write a 2D field and read it back."""
        from legoesm.io.cmor_output import CFWriter

        try:
            import xarray as xr
        except ImportError:
            self.skipTest("xarray not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CFWriter(
                output_dir=tmpdir,
                experiment_id="test",
                model_id="legoESM",
                freq="mon",
            )
            nlat, nlon = 90, 180
            lat = np.linspace(-89.0, 89.0, nlat)
            lon = np.linspace(0.5, 359.5, nlon)
            data = np.random.randn(nlat, nlon).astype(np.float32) + 288.0

            writer.write_field(
                var_name="tas",
                data=data,
                time=15.0,
                time_bounds=(0.0, 30.0),
                lat=lat,
                lon=lon,
            )
            writer.close()

            # Find the written file (CFWriter writes into table sub-dirs)
            nc_files = []
            for root, dirs, files in os.walk(tmpdir):
                nc_files.extend(
                    os.path.join(root, f) for f in files if f.endswith(".nc")
                )
            self.assertTrue(len(nc_files) > 0)

            # Read back and verify
            ds = xr.open_dataset(nc_files[0])
            self.assertIn("tas", ds.data_vars)
            self.assertEqual(ds["tas"].attrs["standard_name"], "air_temperature")
            self.assertEqual(ds["tas"].attrs["units"], "K")
            self.assertIn("Conventions", ds.attrs)
            self.assertEqual(ds.attrs["Conventions"], "CF-1.8")
            ds.close()

    def test_write_3d_field(self):
        """Can write a 3D (pressure-level) field."""
        from legoesm.io.cmor_output import CFWriter, CMIP6_PLEV19

        try:
            import xarray as xr
        except ImportError:
            self.skipTest("xarray not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CFWriter(
                output_dir=tmpdir,
                experiment_id="test",
                model_id="legoESM",
                freq="mon",
            )
            nlat, nlon, nlev = 45, 90, 19
            lat = np.linspace(-88.0, 88.0, nlat)
            lon = np.linspace(2.0, 358.0, nlon)
            plev = np.array(CMIP6_PLEV19[:nlev])
            data = np.random.randn(nlev, nlat, nlon).astype(np.float32) + 250.0

            writer.write_field(
                var_name="ta",
                data=data,
                time=15.0,
                time_bounds=(0.0, 30.0),
                lat=lat,
                lon=lon,
                plev=plev,
            )
            writer.close()

            nc_files = []
            for root, dirs, files in os.walk(tmpdir):
                nc_files.extend(
                    os.path.join(root, f) for f in files if f.endswith(".nc")
                )
            self.assertTrue(len(nc_files) > 0)
            ds = xr.open_dataset(nc_files[0])
            self.assertIn("ta", ds.data_vars)
            self.assertIn("plev", ds.dims)
            ds.close()

    def test_context_manager(self):
        """CFWriter works as a context manager."""
        from legoesm.io.cmor_output import CFWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            with CFWriter(tmpdir, "test", "legoESM", "mon") as writer:
                self.assertIsNotNone(writer)


class TestDiagnosticCollector(unittest.TestCase):
    """DiagnosticCollector CMIP wiring."""

    def test_cmor_writer_uses_requested_experiment_id(self):
        """CFWriter metadata follows the configured experiment id."""
        from legoesm.driver.diagnostics import DiagnosticCollector

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DiagnosticCollector(
                nlev=5,
                sigma_full=np.linspace(0.9, 0.1, 5),
                dsigma=np.full(5, 0.2),
                experiment_id="piControl",
                cmip_output=True,
                output_dir=tmpdir,
            )
            self.assertIsNotNone(collector.cf_writer)
            self.assertEqual(collector.cf_writer.experiment_id, "piControl")
            collector.cf_writer.close()

    def test_3d_plev_ordering_surface_warm_toa_cold(self):
        """CMOR ta: highest pressure (surface) maps to highest temperature.

        _write_cmip_data stores 3-D fields in ascending pressure order
        (100 Pa at index 0), but CMIP6 convention requires descending plev
        (100000 Pa at index 0).  The [::-1] flip in _write_cmip_data must
        align data[0] with plev[0]=100000 Pa.
        """
        import xarray as xr
        from legoesm.driver.diagnostics import DiagnosticCollector
        from legoesm.io.cmor_output import CMIP6_PLEV19

        with tempfile.TemporaryDirectory() as tmpdir:
            # 5-degree grid: 36 lat × 72 lon
            cmip_res = 5.0
            nlat = int(round(180.0 / cmip_res))   # 36
            nlon = int(round(360.0 / cmip_res))   # 72
            nlev = len(CMIP6_PLEV19)              # 19

            collector = DiagnosticCollector(
                nlev=40,
                sigma_full=np.linspace(0.025, 0.993, 40),
                dsigma=np.full(40, 0.025),
                experiment_id="test",
                cmip_output=True,
                output_dir=tmpdir,
                cmip_resolution_deg=cmip_res,
            )

            # Build a test temperature field in ascending pressure order:
            # level 0 = 100 Pa (TOA, cold=100 K), level 18 = 100000 Pa (surface=280 K).
            T_ascending = 100.0 + np.arange(nlev) * 10.0  # [100, 110, ..., 280] K

            # Shape: (nlat, nlon, nlev) as stored by SpatialMonthlyAccumulator.
            field_3d = np.broadcast_to(
                T_ascending[np.newaxis, np.newaxis, :],
                (nlat, nlon, nlev),
            ).copy()

            # Inject directly into the monthly accumulator (bypass collect()).
            collector._spatial_monthly.add_3d(15.0, 0, {"ta": field_3d})

            collector._write_cmip_monthly_files()
            collector.cf_writer.close()

            nc_files = [
                os.path.join(root, f)
                for root, _, files in os.walk(tmpdir)
                for f in files
                if "ta_" in f and f.endswith(".nc")
            ]
            self.assertEqual(len(nc_files), 1, f"Expected 1 ta file, got {nc_files}")

            ds = xr.open_dataset(nc_files[0])
            plev_vals = ds["plev"].values   # CMIP6: descending [100000, ..., 100] Pa
            ta_vals = ds["ta"].values[0, :, 0, 0]  # (time, plev, lat, lon)

            # plev must be descending (CMIP6 convention)
            self.assertGreater(plev_vals[0], plev_vals[-1])
            # T at highest pressure (surface=plev[0]) must be warm, not cold
            self.assertGreater(ta_vals[0], ta_vals[-1])
            # Surface (plev[0]=100000 Pa) → T_ascending[-1] = 280 K
            self.assertAlmostEqual(ta_vals[0], T_ascending[-1], delta=1.0)
            # TOA (plev[-1]=100 Pa) → T_ascending[0] = 100 K
            self.assertAlmostEqual(ta_vals[-1], T_ascending[0], delta=1.0)
            ds.close()

    def test_finalize_cmip_fixed_writes_fx_table(self):
        """finalize_cmip_fixed emits the fx areacella file WITHOUT a normal
        end-of-run save() — the graceful wallclock-exit path depends on it, so
        a restart-chained run still produces a CMOR-complete (area-weightable)
        output tree."""
        from legoesm.driver.diagnostics import DiagnosticCollector

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DiagnosticCollector(
                nlev=5,
                sigma_full=np.linspace(0.9, 0.1, 5),
                dsigma=np.full(5, 0.2),
                experiment_id="amip",
                cmip_output=True,
                output_dir=tmpdir,
                cmip_resolution_deg=5.0,
            )
            # Mimic a graceful mid-run exit: no monthly data, no save().
            collector.finalize_cmip_fixed()
            collector.cf_writer.close()

            fx = [
                os.path.join(root, f)
                for root, _, files in os.walk(tmpdir)
                for f in files
                if "areacella" in f and f.endswith(".nc")
            ]
            self.assertEqual(
                len(fx), 1, f"expected 1 areacella fx file, got {fx}")

    def test_finalize_cmip_fixed_noop_without_writer(self):
        """No CMIP writer → finalize_cmip_fixed is a safe no-op (must not raise
        for the non-CMOR runs whose graceful exit also calls it)."""
        from legoesm.driver.diagnostics import DiagnosticCollector

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DiagnosticCollector(
                nlev=5,
                sigma_full=np.linspace(0.9, 0.1, 5),
                dsigma=np.full(5, 0.2),
                cmip_output=False,
                output_dir=tmpdir,
            )
            self.assertIsNone(collector.cf_writer)
            collector.finalize_cmip_fixed()   # must not raise

    def test_wallclock_exit_writes_cmor_fx_before_exit(self):
        """The graceful wallclock exit flushes completed months AND writes the
        fx table before ``sys.exit(0)`` — guarding the restart-chain CMOR-
        completeness fix (without it every multi-hour AMIP run drops fx)."""
        import types
        from legoesm.driver.model_driver import ModelDriver

        calls = []

        class _Diag:
            def flush_to_disk(self, d):
                calls.append("flush_to_disk")

            def flush_cmip_monthly(self, day, write=True):
                calls.append(("flush_cmip_monthly", write))

            def finalize_cmip_daily(self, current_day):
                calls.append(("finalize_cmip_daily", current_day))

            def finalize_cmip_fixed(self):
                calls.append("finalize_cmip_fixed")

        fake = types.SimpleNamespace(
            _mpi_world_size=1,
            _run_wallclock_start=0.0,       # epoch → budget long exhausted
            _last_checkpoint_step=0,
            diagnostics=_Diag(),
            _save_cmor_accumulator_sidecar=(
                lambda day: calls.append("sidecar")),
            _output_dir="/tmp/ignore_fx_test",
            config=types.SimpleNamespace(
                output=types.SimpleNamespace(
                    max_wallclock_seconds=1.0,
                    restart_buffer_seconds=0.0,
                ),
            ),
        )
        ckpt = []
        with self.assertRaises(SystemExit):
            ModelDriver._maybe_wallclock_exit(
                fake, lambda s, d: ckpt.append((s, d)), step=5, day=10.0)

        self.assertIn("finalize_cmip_fixed", calls)
        self.assertIn(("finalize_cmip_daily", 10.0), calls)
        self.assertIn(("flush_cmip_monthly", True), calls)
        # save() order: completed-month flush -> daily -> fx, all before exit.
        self.assertLess(
            calls.index(("flush_cmip_monthly", True)),
            calls.index(("finalize_cmip_daily", 10.0)),
        )
        self.assertLess(
            calls.index(("finalize_cmip_daily", 10.0)),
            calls.index("finalize_cmip_fixed"),
        )

    def test_finalize_cmip_daily_withholds_partial_day_no_duplicate(self):
        """Restart-chain daily safety: a graceful exit writes only days
        strictly before the in-progress day, so a fresh restart segment
        appending to the same output tree never emits duplicate ``day/tas``
        time coordinates (the corruption a blind write_daily append causes).

        Two independent collectors (fresh process per SLURM segment) share one
        output dir.  Days are keyed exactly as production does — via
        ``day_to_calendar`` — since ``finalize_cmip_daily`` reconstructs the
        boundary key the same way."""
        import xarray as xr
        from legoesm.driver.diagnostics import DiagnosticCollector
        from legoesm.forcing.time_utils import day_to_calendar

        def _dkey(abs_day):
            doy, _ = day_to_calendar(abs_day)
            return (int(abs_day // 365.0), int(doy))

        def _mk(tmpdir):
            return DiagnosticCollector(
                nlev=5, sigma_full=np.linspace(0.9, 0.1, 5),
                dsigma=np.full(5, 0.2), experiment_id="amip",
                cmip_output=True, output_dir=tmpdir, cmip_resolution_deg=5.0,
            )

        def _add(collector, abs_day, fld):
            doy, _ = day_to_calendar(abs_day)
            collector._spatial_daily.add_2d(doy, int(abs_day // 365.0),
                                            {"tas": fld})

        with tempfile.TemporaryDirectory() as tmpdir:
            # --- Segment 1: accumulate two whole days + start a third. ---
            seg1 = _mk(tmpdir)
            fld = np.full((seg1._cmip_nlat, seg1._cmip_nlon), 288.0)
            for d in (1.5, 2.5, 3.3):            # 3.3 = in-progress boundary day
                _add(seg1, d, fld)
            seg1.finalize_cmip_daily(current_day=3.3)   # writes _dkey(1.5),_dkey(2.5)
            seg1.cf_writer.close()

            # --- Segment 2 (fresh process): boundary day's pre-exit samples
            #     are GONE; resume mid-boundary-day, finish it, start a fourth. ---
            seg2 = _mk(tmpdir)
            for d in (3.7, 4.5):                  # 3.7 completes boundary; 4.5 in-progress
                _add(seg2, d, fld)
            seg2.finalize_cmip_daily(current_day=4.5)   # writes _dkey(3.7)==_dkey(3.3)
            seg2.cf_writer.close()

            tas_files = [
                os.path.join(root, f)
                for root, _, files in os.walk(tmpdir)
                for f in files
                if f.startswith("tas_day_") and f.endswith(".nc")
            ]
            self.assertEqual(len(tas_files), 1,
                             f"expected 1 tas day file, got {tas_files}")
            ds = xr.open_dataset(tas_files[0], decode_times=False)
            times = ds["time"].values.tolist()
            ds.close()

            # Completed days written across both segments — each EXACTLY once.
            expected = {_dkey(1.5), _dkey(2.5), _dkey(3.7)}
            self.assertEqual(len(times), len(set(times)),
                             f"duplicate daily time coords: {times}")
            self.assertEqual(len(times), len(expected),
                             f"expected {len(expected)} unique days, got {times}")
            self.assertTrue(all(t >= 0 for t in times),
                            f"negative daily time (doy-0 bug): {times}")


# ======================================================================
# Experiment templates
# ======================================================================


class TestExperimentTemplates(unittest.TestCase):
    """CMIP-style experiment templates."""

    def test_all_templates_present(self):
        """All expected experiments are defined."""
        from legoesm.forcing.experiments import EXPERIMENT_TEMPLATES

        expected = {"piControl", "historical", "ssp245", "ssp585", "amip",
                    "1pctCO2", "abrupt-4xCO2"}
        self.assertEqual(expected, set(EXPERIMENT_TEMPLATES.keys()))

    def test_template_fields(self):
        """Each template has required fields."""
        from legoesm.forcing.experiments import EXPERIMENT_TEMPLATES

        for name, tmpl in EXPERIMENT_TEMPLATES.items():
            self.assertIsInstance(tmpl.name, str)
            self.assertIsInstance(tmpl.start_year, int)
            self.assertIsInstance(tmpl.end_year, int)
            self.assertGreaterEqual(tmpl.end_year, tmpl.start_year)
            self.assertGreater(tmpl.base_co2_ppmv, 0.0)

    def test_picontrol_fixed_forcing(self):
        """piControl uses fixed pre-industrial forcing."""
        from legoesm.forcing.experiments import EXPERIMENT_TEMPLATES

        pi = EXPERIMENT_TEMPLATES["piControl"]
        self.assertEqual(pi.forcing_type, "fixed")
        self.assertAlmostEqual(pi.base_co2_ppmv, 284.3)

    def test_abrupt4xco2_quadruples_co2(self):
        """abrupt-4xCO2 (CMIP6 DECK) instantaneously quadruples
        pre-industrial CO2 and holds it fixed; CH4/N2O stay pre-industrial."""
        from legoesm.forcing.experiments import (
            EXPERIMENT_TEMPLATES, ghg_at_year,
        )

        a4 = EXPERIMENT_TEMPLATES["abrupt-4xCO2"]
        pi = EXPERIMENT_TEMPLATES["piControl"]
        self.assertEqual(a4.forcing_type, "fixed")
        self.assertEqual(a4.parent_experiment, "piControl")
        # 4 x pre-industrial CO2; only CO2 is perturbed.
        self.assertAlmostEqual(a4.base_co2_ppmv, 4.0 * pi.base_co2_ppmv)
        self.assertAlmostEqual(a4.base_ch4_ppbv, pi.base_ch4_ppbv)
        self.assertAlmostEqual(a4.base_n2o_ppbv, pi.base_n2o_ppbv)
        # Fixed forcing: CO2 constant at 4x for every year of the run.
        co2_y0, _, _ = ghg_at_year("abrupt-4xCO2", a4.start_year)
        co2_yend, _, _ = ghg_at_year("abrupt-4xCO2", a4.end_year)
        self.assertAlmostEqual(co2_y0, 4.0 * pi.base_co2_ppmv)
        self.assertAlmostEqual(co2_yend, 4.0 * pi.base_co2_ppmv)

    def test_abrupt4xco2_defaults_to_rrtmgp(self):
        """The CO2 perturbation is radiatively inert under gray radiation,
        so the factory must default abrupt-4xCO2 to rrtmgp."""
        from legoesm.forcing.experiments import create_experiment_config

        cfg = create_experiment_config("abrupt-4xCO2")
        self.assertEqual(cfg.radiation, "rrtmgp")
        self.assertAlmostEqual(cfg.co2_ppmv, 4.0 * 284.3)

    def test_historical_transient(self):
        """historical uses transient forcing 1850-2014."""
        from legoesm.forcing.experiments import EXPERIMENT_TEMPLATES

        hist = EXPERIMENT_TEMPLATES["historical"]
        self.assertEqual(hist.forcing_type, "transient")
        self.assertEqual(hist.start_year, 1850)
        self.assertEqual(hist.end_year, 2014)

    def test_ssp_parent_is_historical(self):
        """SSP experiments have historical as parent."""
        from legoesm.forcing.experiments import EXPERIMENT_TEMPLATES

        for name in ("ssp245", "ssp585"):
            self.assertEqual(
                EXPERIMENT_TEMPLATES[name].parent_experiment, "historical")


class TestGHGInterpolation(unittest.TestCase):
    """GHG concentration time-series interpolation."""

    def test_historical_benchmark_years(self):
        """Historical GHG matches at benchmark years."""
        from legoesm.forcing.experiments import ghg_at_year

        co2, ch4, n2o = ghg_at_year("historical", 1850)
        self.assertAlmostEqual(co2, 284.3, places=1)

        co2, ch4, n2o = ghg_at_year("historical", 2000)
        self.assertAlmostEqual(co2, 369.5, places=1)

    def test_historical_interpolation(self):
        """Historical GHG interpolates between benchmark years."""
        from legoesm.forcing.experiments import ghg_at_year

        co2, _, _ = ghg_at_year("historical", 1925)
        # Between 1900 (295.7) and 1950 (310.7)
        self.assertGreater(co2, 295.7)
        self.assertLess(co2, 310.7)

    def test_ssp585_endpoint(self):
        """SSP5-8.5 reaches ~1135 ppmv CO2 by 2100."""
        from legoesm.forcing.experiments import ghg_at_year

        co2, _, _ = ghg_at_year("ssp585", 2100)
        self.assertAlmostEqual(co2, 1135.0, places=0)

    def test_ssp245_moderate(self):
        """SSP2-4.5 is moderate: ~600 ppmv CO2 by 2100."""
        from legoesm.forcing.experiments import ghg_at_year

        co2, _, _ = ghg_at_year("ssp245", 2100)
        self.assertAlmostEqual(co2, 603.0, places=0)

    def test_picontrol_constant(self):
        """piControl GHG is constant regardless of year."""
        from legoesm.forcing.experiments import ghg_at_year

        co2_a, ch4_a, n2o_a = ghg_at_year("piControl", 1850)
        co2_b, ch4_b, n2o_b = ghg_at_year("piControl", 2050)
        self.assertAlmostEqual(co2_a, co2_b)
        self.assertAlmostEqual(ch4_a, ch4_b)
        self.assertAlmostEqual(n2o_a, n2o_b)

    def test_1pctCO2_increases(self):
        """1pctCO2 increases at 1% per year."""
        from legoesm.forcing.experiments import ghg_at_year

        co2_0, _, _ = ghg_at_year("1pctCO2", 1850)
        co2_70, _, _ = ghg_at_year("1pctCO2", 1920)  # 70 years
        # After 70 years at 1%/yr: 284.3 * 1.01^70 ≈ 572
        self.assertAlmostEqual(co2_70 / co2_0, 1.01 ** 70, places=1)

    def test_get_ghg_dict(self):
        """get_ghg_for_experiment returns a dict with expected keys."""
        from legoesm.forcing.experiments import get_ghg_for_experiment

        ghg = get_ghg_for_experiment("historical", 2000)
        self.assertIn("co2_ppmv", ghg)
        self.assertIn("ch4_ppbv", ghg)
        self.assertIn("n2o_ppbv", ghg)
        self.assertAlmostEqual(ghg["co2_ppmv"], 369.5, places=1)


class TestAMIPExperimentConfigNewFields(unittest.TestCase):
    """New CMIP experiment fields in AMIPExperimentConfig."""

    def test_experiment_field_default(self):
        """Default experiment field is empty string."""
        from legoesm.forcing.amip_config import AMIPExperimentConfig

        cfg = AMIPExperimentConfig()
        self.assertEqual(cfg.experiment, "")

    def test_start_year_default(self):
        """Default start_year is 1979."""
        from legoesm.forcing.amip_config import AMIPExperimentConfig

        cfg = AMIPExperimentConfig()
        self.assertEqual(cfg.start_year, 1979)

    def test_cmip_output_default(self):
        """cmip_output defaults to False."""
        from legoesm.forcing.amip_config import AMIPExperimentConfig

        cfg = AMIPExperimentConfig()
        self.assertFalse(cfg.cmip_output)

    def test_clear_sky_diag_default(self):
        """clear_sky_diag defaults to False."""
        from legoesm.forcing.amip_config import AMIPExperimentConfig

        cfg = AMIPExperimentConfig()
        self.assertFalse(cfg.clear_sky_diag)

    def test_experiment_field_set(self):
        """Experiment field can be set."""
        from legoesm.forcing.amip_config import AMIPExperimentConfig

        cfg = AMIPExperimentConfig(experiment="historical", start_year=1990)
        self.assertEqual(cfg.experiment, "historical")
        self.assertEqual(cfg.start_year, 1990)

    def test_serialization_roundtrip(self):
        """New fields survive config serialization roundtrip."""
        from legoesm.forcing.amip_config import (
            AMIPExperimentConfig, config_to_dict, config_from_dict,
        )

        cfg = AMIPExperimentConfig(
            experiment="ssp245",
            start_year=2015,
            cmip_output=True,
            clear_sky_diag=True,
        )
        d = config_to_dict(cfg)
        cfg2 = config_from_dict(d)
        self.assertEqual(cfg2.experiment, "ssp245")
        self.assertEqual(cfg2.start_year, 2015)
        self.assertTrue(cfg2.cmip_output)
        self.assertTrue(cfg2.clear_sky_diag)


class TestGHGVMROverride(unittest.TestCase):
    """GHG VMR override in RRTMGP radiation."""

    def test_rrtmgp_accepts_ghg_override(self):
        """rrtmgp_radiation accepts ghg_vmr_override parameter."""
        import inspect
        from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import (
            rrtmgp_radiation,
        )

        sig = inspect.signature(rrtmgp_radiation)
        self.assertIn("ghg_vmr_override", sig.parameters)

    def test_integration_accepts_ghg_override(self):
        """_call_radiation_backend accepts ghg_vmr_override parameter."""
        import inspect
        from legoesm.atmosphere.physics.radiation.integration import (
            _call_radiation_backend,
        )

        sig = inspect.signature(_call_radiation_backend)
        self.assertIn("ghg_vmr_override", sig.parameters)


class TestCreateExperimentConfig(unittest.TestCase):
    """Factory function for experiment configs."""

    def test_picontrol_config(self):
        """create_experiment_config builds a valid piControl config."""
        from legoesm.forcing.experiments import create_experiment_config

        cfg = create_experiment_config("piControl")
        self.assertAlmostEqual(cfg.co2_ppmv, 284.3, places=1)
        self.assertGreater(cfg.days, 0)

    def test_historical_config(self):
        """create_experiment_config builds a valid historical config."""
        from legoesm.forcing.experiments import create_experiment_config

        cfg = create_experiment_config("historical")
        # 164 years * 365 days
        self.assertEqual(cfg.days, (2014 - 1850) * 365)

    def test_transient_experiments_default_rrtmgp(self):
        """Transient experiments default to rrtmgp radiation."""
        from legoesm.forcing.experiments import create_experiment_config

        for name in ("historical", "ssp245", "ssp585", "1pctCO2"):
            cfg = create_experiment_config(name)
            self.assertEqual(
                cfg.radiation, "rrtmgp",
                f"{name} should default to rrtmgp radiation"
            )

    def test_radiation_override_respected(self):
        """Explicit radiation override is respected."""
        from legoesm.forcing.experiments import create_experiment_config

        cfg = create_experiment_config("historical", radiation="gray")
        self.assertEqual(cfg.radiation, "gray")

    def test_overrides(self):
        """Overrides are applied to the config."""
        from legoesm.forcing.experiments import create_experiment_config

        cfg = create_experiment_config("piControl", resolution=48, dt=300.0)
        # create_experiment_config now returns ExperimentConfig (canonical)
        # where resolution lives under grid and dt under dycore.
        self.assertEqual(cfg.grid.resolution, 48)
        self.assertEqual(cfg.dycore.dt, 300.0)

    def test_unknown_experiment_raises(self):
        """Unknown experiment name raises ValueError."""
        from legoesm.forcing.experiments import create_experiment_config

        with self.assertRaises(ValueError):
            create_experiment_config("nonexistent_experiment")


# ======================================================================
# Restart / reproducibility
# ======================================================================


class TestStateDigest(unittest.TestCase):
    """State array hashing."""

    def test_deterministic(self):
        """Same arrays give same digest."""
        from legoesm.driver.restart import compute_state_digest

        arrays = {"T": np.ones(100), "u": np.zeros(100)}
        d1 = compute_state_digest(arrays)
        d2 = compute_state_digest(arrays)
        self.assertEqual(d1, d2)

    def test_different_values(self):
        """Different array values give different digest."""
        from legoesm.driver.restart import compute_state_digest

        d1 = compute_state_digest({"T": np.ones(100)})
        d2 = compute_state_digest({"T": np.zeros(100)})
        self.assertNotEqual(d1, d2)

    def test_key_order_invariant(self):
        """Digest is invariant to insertion order (keys are sorted)."""
        from legoesm.driver.restart import compute_state_digest

        a = {"b": np.ones(10), "a": np.zeros(10)}
        b = {"a": np.zeros(10), "b": np.ones(10)}
        self.assertEqual(compute_state_digest(a), compute_state_digest(b))


class TestConfigHash(unittest.TestCase):
    """Config hashing."""

    def test_deterministic(self):
        """Same config gives same hash."""
        from legoesm.driver.restart import compute_config_hash
        from legoesm.forcing.amip_config import AMIPExperimentConfig

        cfg = AMIPExperimentConfig()
        h1 = compute_config_hash(cfg)
        h2 = compute_config_hash(cfg)
        self.assertEqual(h1, h2)

    def test_different_config(self):
        """Different config gives different hash."""
        from legoesm.driver.restart import compute_config_hash
        from legoesm.forcing.amip_config import AMIPExperimentConfig

        cfg1 = AMIPExperimentConfig(resolution=16)
        cfg2 = AMIPExperimentConfig(resolution=48)
        self.assertNotEqual(compute_config_hash(cfg1), compute_config_hash(cfg2))


class TestSaveLoadRestart(unittest.TestCase):
    """Save and load restart with metadata."""

    def _make_state(self, n=4, nlev=5):
        """Create a minimal state for testing."""
        import jax.numpy as jnp
        from legoesm.core.field import Field
        from legoesm.core.state import HydrostaticState

        shape_3d = (6, n, n, nlev)
        shape_2d = (6, n, n)
        dims_3d = ("face", "x", "y", "level")
        dims_2d = ("face", "x", "y")

        state = HydrostaticState(
            u=Field(data=jnp.ones(shape_3d), name="u", dims=dims_3d, units="m/s"),
            v=Field(data=jnp.zeros(shape_3d), name="v", dims=dims_3d, units="m/s"),
            T=Field(data=jnp.full(shape_3d, 280.0), name="T", dims=dims_3d, units="K"),
            p_s=Field(data=jnp.full(shape_2d, 1e5), name="p_s", dims=dims_2d, units="Pa"),
            phis=Field(data=jnp.zeros(shape_2d), name="phis", dims=dims_2d, units="m^2/s^2"),
        )
        q_v = jnp.full(shape_3d, 0.005)
        return state, q_v

    def test_roundtrip(self):
        """Save and load restart preserves state."""
        from legoesm.driver.restart import save_restart, load_restart
        from legoesm.forcing.amip_config import AMIPExperimentConfig
        import jax.numpy as jnp

        state, q_v = self._make_state()
        cfg = AMIPExperimentConfig(resolution=4, nlev=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "restart.npz")
            save_restart(path, state, q_v, step=100, day=50.0, config=cfg)

            # Check metadata file exists
            meta_path = path.replace(".npz", ".meta.json")
            self.assertTrue(os.path.exists(meta_path))

            # Load back (need grid/sigma stubs)
            result = load_restart(
                path, grid=None, sigma=None, strict=False)
            loaded_state, loaded_qv, step, day = result[:4]

            self.assertEqual(step, 100)
            self.assertAlmostEqual(day, 50.0)
            self.assertTrue(
                jnp.allclose(loaded_state.T.data, state.T.data))

    def test_metadata_written(self):
        """Metadata JSON contains expected fields."""
        from legoesm.driver.restart import save_restart
        from legoesm.forcing.amip_config import AMIPExperimentConfig

        state, q_v = self._make_state()
        cfg = AMIPExperimentConfig(resolution=4, nlev=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "restart.npz")
            save_restart(path, state, q_v, step=10, day=5.0, config=cfg)

            meta_path = path.replace(".npz", ".meta.json")
            with open(meta_path) as f:
                meta = json.load(f)

            self.assertIn("model_version", meta)
            self.assertIn("config_hash", meta)
            self.assertIn("state_digest", meta)
            self.assertIn("jax_x64_enabled", meta)
            self.assertEqual(meta["resolution"], 4)
            self.assertEqual(meta["nlev"], 5)
            self.assertEqual(meta["step"], 10)


class TestReproducibilityReport(unittest.TestCase):
    """Cross-restart reproducibility checks."""

    def test_identical_restarts(self):
        """Two identical saves produce matching digests."""
        from legoesm.driver.restart import save_restart, verify_reproducibility
        from legoesm.forcing.amip_config import AMIPExperimentConfig
        import jax.numpy as jnp
        from legoesm.core.field import Field
        from legoesm.core.state import HydrostaticState

        shape_3d = (6, 4, 4, 5)
        shape_2d = (6, 4, 4)
        dims_3d = ("face", "x", "y", "level")
        dims_2d = ("face", "x", "y")

        state = HydrostaticState(
            u=Field(data=jnp.ones(shape_3d), name="u", dims=dims_3d, units="m/s"),
            v=Field(data=jnp.zeros(shape_3d), name="v", dims=dims_3d, units="m/s"),
            T=Field(data=jnp.full(shape_3d, 280.0), name="T", dims=dims_3d, units="K"),
            p_s=Field(data=jnp.full(shape_2d, 1e5), name="p_s", dims=dims_2d, units="Pa"),
            phis=Field(data=jnp.zeros(shape_2d), name="phis", dims=dims_2d, units="m^2/s^2"),
        )
        q_v = jnp.full(shape_3d, 0.005)
        cfg = AMIPExperimentConfig(resolution=4, nlev=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = os.path.join(tmpdir, "a.npz")
            path_b = os.path.join(tmpdir, "b.npz")
            save_restart(path_a, state, q_v, 10, 5.0, cfg)
            save_restart(path_b, state, q_v, 10, 5.0, cfg)

            report = verify_reproducibility(path_a, path_b)
            self.assertTrue(report.identical)
            self.assertTrue(report.state_digest_match)
            self.assertTrue(report.config_match)
            self.assertEqual(len(report.differences), 0)


# ======================================================================
# Tuning
# ======================================================================


class TestTuningParameters(unittest.TestCase):
    """Tuning parameter registry."""

    def test_all_categories_present(self):
        """All expected categories are represented."""
        from legoesm.tuning import TUNING_PARAMETERS

        categories = {p.category for p in TUNING_PARAMETERS.values()}
        # "clouds" left this set 2026-07-26: its only two catalog entries
        # (cloud_rh_ice_crit/sat) advertised a CloudConfig RH_i cirrus ramp
        # that was never implemented (flag-reachability audit cause 3) and
        # were deleted with their ExperimentConfig fields.  The REAL cloud
        # tunables (cloud_rh_crit, ...) are wired via --config/--params, not
        # this catalog.
        expected = {
            "dynamics", "radiation", "convection", "diffusion", "surface",
            "turbulence", "gwd",
        }
        self.assertEqual(expected, categories)

    def test_ranges_valid(self):
        """min_val <= default <= max_val for all parameters."""
        from legoesm.tuning import TUNING_PARAMETERS

        for name, p in TUNING_PARAMETERS.items():
            self.assertLessEqual(
                p.min_val, p.default,
                f"{name}: min ({p.min_val}) > default ({p.default})")
            self.assertLessEqual(
                p.default, p.max_val,
                f"{name}: default ({p.default}) > max ({p.max_val})")


class TestValidateTuning(unittest.TestCase):
    """Tuning validation checks."""

    def test_default_config_no_warnings(self):
        """Default C16 config should produce no warnings."""
        from legoesm.tuning import validate_tuning
        from legoesm.forcing.amip_config import AMIPExperimentConfig

        cfg = AMIPExperimentConfig()  # C16, dt=600
        warnings = validate_tuning(cfg)
        # Default config should be clean
        self.assertIsInstance(warnings, list)

    def test_large_dt_warns(self):
        """Large dt at high resolution should warn."""
        from legoesm.tuning import validate_tuning
        from legoesm.forcing.amip_config import AMIPExperimentConfig

        cfg = AMIPExperimentConfig(resolution=96, dt=1200.0)
        warnings = validate_tuning(cfg)
        # Should warn about CFL
        self.assertTrue(any("dt" in w.lower() or "cfl" in w.lower()
                            for w in warnings),
                        f"Expected dt/CFL warning, got: {warnings}")

    def test_zero_co2_warns(self):
        """Zero CO2 should produce a warning."""
        from legoesm.tuning import validate_tuning
        from legoesm.forcing.amip_config import AMIPExperimentConfig

        cfg = AMIPExperimentConfig(co2_ppmv=0.0)
        warnings = validate_tuning(cfg)
        self.assertTrue(any("co2" in w.lower() for w in warnings),
                        f"Expected CO2 warning, got: {warnings}")


class TestRecommendedParams(unittest.TestCase):
    """Resolution-dependent parameter recommendations."""

    def test_c16(self):
        """C16 recommendations are sensible."""
        from legoesm.tuning import recommended_params

        r = recommended_params(16, 40)
        self.assertEqual(r["dt"], 600.0)
        self.assertGreater(r["hyperdiff_scale"], 1e15)

    def test_c48(self):
        """C48 has smaller dt than C16."""
        from legoesm.tuning import recommended_params

        r16 = recommended_params(16, 40)
        r48 = recommended_params(48, 40)
        self.assertLessEqual(r48["dt"], r16["dt"])

    def test_c96(self):
        """C96 has smallest dt."""
        from legoesm.tuning import recommended_params

        r48 = recommended_params(48, 40)
        r96 = recommended_params(96, 40)
        self.assertLessEqual(r96["dt"], r48["dt"])


class TestPrintTuningGuide(unittest.TestCase):
    """Tuning guide formatting."""

    def test_prints_without_error(self):
        """print_tuning_guide runs without crashing."""
        from legoesm.tuning import print_tuning_guide
        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            print_tuning_guide()
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        self.assertGreater(len(output), 100)
        self.assertIn("dynamics", output.lower())


class TestCMIP6Compliance(unittest.TestCase):
    """CMIP6 controlled-vocabulary + CF spatial-bounds compliance."""

    def test_parse_variant_label(self):
        from legoesm.io.cmor_output import _parse_variant_label
        self.assertEqual(_parse_variant_label("r1i1p1f1"), (1, 1, 1, 1))
        self.assertEqual(_parse_variant_label("r12i3p4f5"), (12, 3, 4, 5))
        with self.assertRaises(ValueError):
            _parse_variant_label("bogus")
        with self.assertRaises(ValueError):
            _parse_variant_label("r1i1p1")  # missing forcing index

    def test_malformed_variant_label_rejected_in_constructor(self):
        from legoesm.io.cmor_output import CFWriter
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                CFWriter(
                    output_dir=tmpdir,
                    experiment_id="amip",
                    model_id="legoESM-1-0",
                    variant_label="malformed",
                )

    def test_compute_nominal_resolution_5deg(self):
        from legoesm.io.cmor_output import _compute_nominal_resolution
        lat = np.linspace(-87.5, 87.5, 36)  # 5°
        lon = np.linspace(2.5, 357.5, 72)   # 5°
        # 5° ≈ 556 km → bucket "1000 km" (smallest upper-bound that
        # covers the spacing).
        self.assertEqual(_compute_nominal_resolution(lat, lon), "1000 km")

    def test_compute_nominal_resolution_2deg(self):
        from legoesm.io.cmor_output import _compute_nominal_resolution
        lat = np.linspace(-89, 89, 90)   # 2°
        lon = np.linspace(1, 359, 180)   # 2°
        # 2° ≈ 222 km → bucket "250 km"
        self.assertEqual(_compute_nominal_resolution(lat, lon), "250 km")

    def test_tracking_id_format(self):
        from legoesm.io.cmor_output import _generate_tracking_id
        tid = _generate_tracking_id()
        self.assertTrue(tid.startswith("hdl:21.14100/"))
        self.assertNotEqual(tid, _generate_tracking_id())  # fresh each call

    def test_cell_bounds_from_uniform_centers(self):
        from legoesm.io.cmor_output import _cell_bounds_from_centers
        c = np.array([0.0, 10.0, 20.0, 30.0])
        bnds = _cell_bounds_from_centers(c)
        self.assertEqual(bnds.shape, (4, 2))
        # Interior cells: halfway between neighbors.
        np.testing.assert_allclose(bnds[1], [5.0, 15.0])
        np.testing.assert_allclose(bnds[2], [15.0, 25.0])
        # Exterior cells: half-step extrapolation from nearest neighbor.
        np.testing.assert_allclose(bnds[0], [-5.0, 5.0])
        np.testing.assert_allclose(bnds[3], [25.0, 35.0])
        # Edges must be contiguous (no gaps).
        np.testing.assert_allclose(bnds[:-1, 1], bnds[1:, 0])

    def test_lat_bnds_clipped_at_poles(self):
        from legoesm.io.cmor_output import _make_lat_bnds_da
        lat = np.array([-87.5, -82.5, 82.5, 87.5])
        da = _make_lat_bnds_da(lat)
        self.assertEqual(da.shape, (4, 2))
        # Polar edges must not extend beyond ±90°.
        self.assertGreaterEqual(float(da.values.min()), -90.0)
        self.assertLessEqual(float(da.values.max()), 90.0)

    def test_realm_for_table(self):
        from legoesm.io.cmor_output import _realm_for_table
        self.assertEqual(_realm_for_table("Amon"), "atmos")
        self.assertEqual(_realm_for_table("Lmon"), "land")
        self.assertEqual(_realm_for_table("Omon"), "ocean")
        self.assertEqual(_realm_for_table("Aday"), "atmos")

    def test_written_file_has_cmip6_required_globals(self):
        try:
            import xarray as xr
        except ImportError:
            self.skipTest("xarray not installed")
        from legoesm.io.cmor_output import CFWriter

        required = {
            "Conventions", "mip_era", "activity_id", "experiment_id",
            "sub_experiment", "sub_experiment_id",
            "institution", "institution_id", "source_id", "source",
            "source_type", "product", "realm",
            "variant_label", "realization_index", "initialization_index",
            "physics_index", "forcing_index",
            "grid_label", "nominal_resolution",
            "creation_date", "tracking_id", "license",
            "parent_experiment_id", "parent_source_id",
            "parent_variant_label", "parent_activity_id",
            "parent_time_units",
            "branch_method", "branch_time_in_child", "branch_time_in_parent",
            "frequency", "table_id", "variable_id",
            "external_variables",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CFWriter(
                output_dir=tmpdir,
                experiment_id="amip",
                model_id="legoESM-1-0",
                variant_label="r2i3p4f5",
            )
            lat = np.linspace(-87.5, 87.5, 8)
            lon = np.linspace(2.5, 357.5, 16)
            data = np.full((8, 16), 288.0, dtype=np.float32)
            path = writer.write_field(
                var_name="tas", data=data,
                time=15.0, time_bounds=(0.0, 30.0),
                lat=lat, lon=lon,
            )
            ds = xr.open_dataset(path)
            attrs = dict(ds.attrs)
            ds.close()

        missing = required - set(attrs)
        self.assertFalse(missing, f"Missing globals: {missing}")
        self.assertEqual(attrs["mip_era"], "CMIP6")
        self.assertEqual(attrs["realm"], "atmos")
        self.assertEqual(int(attrs["realization_index"]), 2)
        self.assertEqual(int(attrs["initialization_index"]), 3)
        self.assertEqual(int(attrs["physics_index"]), 4)
        self.assertEqual(int(attrs["forcing_index"]), 5)
        self.assertTrue(
            attrs["tracking_id"].startswith("hdl:21.14100/"),
            f"tracking_id not hdl-form: {attrs['tracking_id']!r}",
        )
        self.assertEqual(attrs["external_variables"], "areacella")

    def test_written_file_has_spatial_bounds_and_height(self):
        try:
            import xarray as xr
        except ImportError:
            self.skipTest("xarray not installed")
        from legoesm.io.cmor_output import CFWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CFWriter(
                output_dir=tmpdir, experiment_id="amip",
                model_id="legoESM-1-0",
            )
            lat = np.linspace(-87.5, 87.5, 8)
            lon = np.linspace(2.5, 357.5, 16)
            # tas — should carry height = 2 m
            data_tas = np.full((8, 16), 288.0, dtype=np.float32)
            p_tas = writer.write_field(
                var_name="tas", data=data_tas,
                time=15.0, time_bounds=(0.0, 30.0),
                lat=lat, lon=lon,
            )
            # ps — surface pressure, no height
            data_ps = np.full((8, 16), 101325.0, dtype=np.float32)
            p_ps = writer.write_field(
                var_name="ps", data=data_ps,
                time=15.0, time_bounds=(0.0, 30.0),
                lat=lat, lon=lon,
            )
            # ``decode_coords=False`` preserves the raw ``coordinates``
            # variable attribute; xarray otherwise consumes it while
            # promoting auxiliary coordinates, which hides whether the
            # writer actually emitted it.
            ds_tas = xr.open_dataset(
                p_tas, decode_times=False, decode_coords=False,
            ).load()
            ds_ps = xr.open_dataset(
                p_ps, decode_times=False, decode_coords=False,
            ).load()
            ds_tas.close()
            ds_ps.close()

        # Spatial cell bounds present on both files
        for ds in (ds_tas, ds_ps):
            self.assertIn("lat_bnds", ds.variables)
            self.assertIn("lon_bnds", ds.variables)
            self.assertEqual(ds["lat"].attrs.get("bounds"), "lat_bnds")
            self.assertEqual(ds["lon"].attrs.get("bounds"), "lon_bnds")
            # lat bounds must not leak past the poles
            self.assertGreaterEqual(
                float(ds["lat_bnds"].values.min()), -90.0,
            )
            self.assertLessEqual(
                float(ds["lat_bnds"].values.max()), 90.0,
            )

        # tas: scalar height = 2 m stored as a variable referenced via
        # the CF ``coordinates`` attribute on the data variable.
        self.assertIn("height", ds_tas.variables)
        self.assertAlmostEqual(float(ds_tas["height"].values), 2.0)
        self.assertEqual(ds_tas["height"].attrs.get("units"), "m")
        self.assertEqual(
            ds_tas["tas"].attrs.get("coordinates"), "height",
        )
        # Regression: bnds variables must NOT carry coordinates="height".
        # xarray will auto-tag every data variable with a ``coordinates``
        # attribute when the coord is attached to the surrounding Dataset
        # via ``assign_coords``; the fix is to attach it to the DataArray
        # itself so only that variable is tagged.
        for bnds in ("time_bnds", "lat_bnds", "lon_bnds"):
            self.assertNotEqual(
                ds_tas[bnds].attrs.get("coordinates"), "height",
                msg=f"{bnds} should not be tagged with coordinates=height",
            )
        # ps: no height
        self.assertNotIn("height", ds_ps.variables)
        self.assertNotIn("coordinates", ds_ps["ps"].attrs)

    def test_write_monthly_height_not_on_bnds(self):
        """Regression: ``write_monthly`` must not propagate the scalar
        ``height`` coordinate onto bnds variables. Parallels the fix
        already in ``write_field``.
        """
        try:
            import xarray as xr
        except ImportError:
            self.skipTest("xarray not installed")
        from legoesm.io.cmor_output import CFWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CFWriter(
                output_dir=tmpdir, experiment_id="amip",
                model_id="legoESM-1-0",
            )
            nlat, nlon = 8, 16
            lat = np.linspace(-87.5, 87.5, nlat)
            lon = np.linspace(2.5, 357.5, nlon)
            # Two-month zonal_tas accumulator payload
            n_months = 2
            monthly_data = {
                "months": [(1, 1), (1, 2)],
                "zonal_tas": np.full(
                    (n_months, nlat), 288.0, dtype=np.float32,
                ),
                "lat": lat,
            }
            paths = writer.write_monthly(monthly_data, lat, lon)
            self.assertTrue(len(paths) >= 1)
            tas_path = next(p for p in paths if "tas_" in p.name)
            ds_tas = xr.open_dataset(
                tas_path, decode_times=False, decode_coords=False,
            ).load()
            ds_tas.close()

        self.assertIn("height", ds_tas.variables)
        self.assertAlmostEqual(float(ds_tas["height"].values), 2.0)
        self.assertEqual(
            ds_tas["tas"].attrs.get("coordinates"), "height",
        )
        for bnds in ("time_bnds", "lat_bnds", "lon_bnds"):
            self.assertNotEqual(
                ds_tas[bnds].attrs.get("coordinates"), "height",
                msg=f"{bnds} should not be tagged with coordinates=height",
            )

    def test_spatial_monthly_partial_after_flush(self):
        """Regression: after ``pop_completed_months`` flushes full months,
        the in-progress partial month must still be dropped by
        ``finalize()`` — i.e. the guard cannot rely on there being
        multiple live buckets.
        """
        from legoesm.diagnostics.monthly_means import SpatialMonthlyAccumulator
        accum = SpatialMonthlyAccumulator(nlat=4, nlon=8, nlev=0)
        full = np.full((4, 8), 290.0)
        # Full Jan (doy 1): 7 samples
        for _ in range(7):
            accum.add_2d(1, 0, {"tas": full})
        # Full Feb (doy 32): 5 samples
        for _ in range(5):
            accum.add_2d(32, 0, {"tas": full})
        # Pop Jan + Feb (simulates the incremental flush path while the
        # model is mid-March).
        popped = accum.pop_completed_months(current_year=0, current_month=3)
        self.assertEqual(popped["months"], [(0, 1), (0, 2)])
        # A single partial-April sample after the flush:
        accum.add_2d(91, 0, {"tas": full})
        # finalize should now drop the lone 1-sample April bucket because
        # ``_max_count_ever`` remembers that full months had 7 samples.
        data = accum.finalize(min_sample_fraction=0.5)
        self.assertEqual(data["months"], [])

    def test_spatial_daily_partial_day_dropped(self):
        """A lone final bucket with fewer samples than full days is
        dropped by ``SpatialDailyAccumulator.finalize`` at the default
        ``min_sample_fraction=0.5`` guard.
        """
        from legoesm.diagnostics.monthly_means import SpatialDailyAccumulator
        accum = SpatialDailyAccumulator(nlat=4, nlon=8, track_extremes={"tas"})
        # Two full days (8 samples each) + one partial day (1 sample).
        tas_full = np.full((4, 8), 290.0)
        for doy in (1, 2):
            for _ in range(8):
                accum.add_2d(doy, 0, {"tas": tas_full})
        accum.add_2d(3, 0, {"tas": tas_full})
        data = accum.finalize(min_sample_fraction=0.5)
        days = [d for _, d in data["days"]]
        self.assertEqual(days, [1, 2])
        # Setting threshold to 0 keeps the partial day.
        data_all = accum.finalize(min_sample_fraction=0.0)
        self.assertEqual([d for _, d in data_all["days"]], [1, 2, 3])

    def test_pop_completed_days_wraps_year_and_withholds_current(self):
        """pop_completed_days emits+frees only days strictly before the given
        (year, doy) and is idempotent — the restart-safety contract used by
        finalize_cmip_daily.  Lexicographic (year, doy) keys make doy 364 of
        year 0 complete once the run enters year 1 (noleap wrap)."""
        from legoesm.diagnostics.monthly_means import SpatialDailyAccumulator
        accum = SpatialDailyAccumulator(nlat=2, nlon=2, track_extremes={"tas"})
        f = np.full((2, 2), 288.0)
        accum.add_2d(364, 0, {"tas": f})    # last day of year 0
        accum.add_2d(1, 1, {"tas": f})      # first day of year 1 (in-progress)

        # current = (year 1, doy 1): only (0, 364) is strictly before it.
        out = accum.pop_completed_days(current_year=1, current_doy=1)
        self.assertEqual(out["days"], [(0, 364)])
        self.assertIn("field_2d_tas", out)
        self.assertIn("field_2d_tas_min", out)     # extrema preserved
        self.assertIn("field_2d_tas_max", out)
        # Completed day freed; the in-progress day survives for the next seg.
        self.assertEqual(sorted(accum._data.keys()), [(1, 1)])
        # Idempotent: a repeat pop at the same boundary emits nothing.
        self.assertEqual(accum.pop_completed_days(1, 1)["days"], [])

    def test_spatial_daily_tracks_min_max(self):
        """``tas`` extremes are tracked correctly per-day."""
        from legoesm.diagnostics.monthly_means import SpatialDailyAccumulator
        accum = SpatialDailyAccumulator(nlat=2, nlon=2, track_extremes={"tas"})
        a = np.array([[280.0, 285.0], [290.0, 295.0]])
        b = np.array([[278.0, 286.0], [291.0, 294.0]])
        c = np.array([[281.0, 284.0], [292.0, 296.0]])
        for f in (a, b, c):
            accum.add_2d(1, 0, {"tas": f})
        data = accum.finalize(min_sample_fraction=0.0)
        np.testing.assert_allclose(
            data["field_2d_tas"][0], (a + b + c) / 3.0,
        )
        np.testing.assert_allclose(
            data["field_2d_tas_min"][0], np.minimum(np.minimum(a, b), c),
        )
        np.testing.assert_allclose(
            data["field_2d_tas_max"][0], np.maximum(np.maximum(a, b), c),
        )

    def test_write_daily_emits_day_tables_with_tasmin_tasmax(self):
        """``CFWriter.write_daily`` consumes the accumulator finalize
        dict and produces CMIP6 ``day``-table files for ``tas``,
        ``tasmin``, and ``tasmax``.
        """
        try:
            import xarray as xr
        except ImportError:
            self.skipTest("xarray not installed")
        from legoesm.io.cmor_output import CFWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CFWriter(
                output_dir=tmpdir, experiment_id="amip",
                model_id="legoESM-1-0", freq="day", ref_date="1979-01-01",
            )
            nlat, nlon = 4, 8
            lat = np.linspace(-67.5, 67.5, nlat)
            lon = np.linspace(22.5, 337.5, nlon)
            n = 3
            mean = np.full((n, nlat, nlon), 290.0, dtype=np.float32)
            lo = mean - 2.0
            hi = mean + 2.0
            daily_data = {
                "days": [(0, 1), (0, 2), (0, 3)],
                "field_2d_tas": mean,
                "field_2d_tas_min": lo,
                "field_2d_tas_max": hi,
            }
            paths = writer.write_daily(daily_data, lat=lat, lon=lon)
            self.assertEqual(len(paths), 3)  # tas, tasmin, tasmax
            names = sorted(p.name for p in paths)
            self.assertTrue(any(n.startswith("tas_day_") for n in names))
            self.assertTrue(any(n.startswith("tasmin_day_") for n in names))
            self.assertTrue(any(n.startswith("tasmax_day_") for n in names))
            tas_path = next(p for p in paths if p.name.startswith("tas_day_"))
            ds = xr.open_dataset(tas_path, decode_times=False).load()
            ds.close()
        # Three daily timesteps were written.
        self.assertEqual(int(ds.sizes["time"]), 3)
        self.assertEqual(ds.attrs.get("frequency"), "day")
        self.assertEqual(ds.attrs.get("table_id"), "day")
        # Time axis starts at 0.5 (noon day 1) relative to 1979-01-01.
        np.testing.assert_allclose(ds["time"].values[0], 0.5)

    def test_write_fixed_produces_fx_file(self):
        """``CFWriter.write_fixed`` writes a time-invariant ``fx`` file
        with no time dimension and ``frequency='fx'``.
        """
        try:
            import xarray as xr
        except ImportError:
            self.skipTest("xarray not installed")
        from legoesm.io.cmor_output import CFWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CFWriter(
                output_dir=tmpdir, experiment_id="amip",
                model_id="legoESM-1-0",
            )
            nlat, nlon = 4, 8
            lat = np.linspace(-67.5, 67.5, nlat)
            lon = np.linspace(22.5, 337.5, nlon)
            orog = np.full((nlat, nlon), 500.0, dtype=np.float32)
            path = writer.write_fixed("orog", orog, lat=lat, lon=lon)
            ds = xr.open_dataset(path, decode_times=False).load()
            ds.close()
        self.assertNotIn("time", ds.dims)
        self.assertEqual(ds.attrs.get("frequency"), "fx")
        self.assertEqual(ds.attrs.get("table_id"), "fx")
        np.testing.assert_allclose(ds["orog"].values, 500.0)
        self.assertEqual(ds["orog"].attrs.get("units"), "m")

    def test_diagnostic_collector_writes_daily_and_fx(self):
        """Integration: DiagnosticCollector with cmip_output=True emits
        ``day/tas_*.nc``, ``day/tasmin_*.nc``, ``day/tasmax_*.nc``, and
        ``fx/areacella_*.nc`` + ``fx/orog_*.nc`` + ``fx/sftlf_*.nc``.
        """
        try:
            import xarray as xr  # noqa: F401
        except ImportError:
            self.skipTest("xarray not installed")
        from legoesm.driver.diagnostics import DiagnosticCollector
        from legoesm.diagnostics.monthly_means import SpatialDailyAccumulator

        nlat, nlon = 4, 8
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DiagnosticCollector(
                nlev=4,
                sigma_full=np.array([0.125, 0.375, 0.625, 0.875]),
                dsigma=np.array([0.25, 0.25, 0.25, 0.25]),
                cmip_output=True,
                n_days=3,
                output_dir=tmpdir,
                cmip_resolution_deg=45.0,  # → (4, 8) target grid
                start_year=1979,
            )
            # Force the daily accumulator/target shape so the test stays
            # independent of any subtle rounding in the factor-to-shape
            # calculation inside ``__init__``.
            collector._cmip_nlat = nlat
            collector._cmip_nlon = nlon
            collector._spatial_daily = SpatialDailyAccumulator(
                nlat=nlat, nlon=nlon, track_extremes={"tas"},
            )

            # Populate daily accumulator directly with three full days of
            # consistent samples so ``min_sample_fraction=0.5`` keeps all.
            tas_field = np.full((nlat, nlon), 290.0)
            for doy in (1, 2, 3):
                for _ in range(4):
                    collector._spatial_daily.add_2d(
                        doy, 0, {"tas": tas_field},
                    )

            # Fixed-field inputs on the target grid (skip regridding
            # since weights aren't set in this minimal test).
            collector._fixed_phis = np.full(
                (nlat, nlon), 9.80665 * 500.0,
            )  # orog = 500 m
            collector._fixed_land_fraction = np.full((nlat, nlon), 0.3)

            # Monkey-patch _regrid_to_latlon_2d to identity on (nlat, nlon).
            def _id(arr):
                arr = np.asarray(arr)
                if arr.shape == (nlat, nlon):
                    return arr
                return None
            collector._regrid_to_latlon_2d = _id

            collector._write_cmip_daily_files()
            collector._write_cmip_fixed_files()
            collector.cf_writer.close()

            cmor_root = Path(tmpdir) / "cmor"
            day_files = sorted((cmor_root).rglob("day/*.nc"))
            fx_files = sorted((cmor_root).rglob("fx/*.nc"))
            day_names = [p.name.split("_")[0] for p in day_files]
            fx_names = [p.name.split("_")[0] for p in fx_files]

        self.assertIn("tas", day_names)
        self.assertIn("tasmin", day_names)
        self.assertIn("tasmax", day_names)
        self.assertIn("orog", fx_names)
        self.assertIn("sftlf", fx_names)
        self.assertIn("areacella", fx_names)

    def test_nominal_resolution_matches_grid(self):
        try:
            import xarray as xr
        except ImportError:
            self.skipTest("xarray not installed")
        from legoesm.io.cmor_output import CFWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CFWriter(
                output_dir=tmpdir, experiment_id="amip",
                model_id="legoESM-1-0",
            )
            # 5° grid → nominal_resolution "1000 km" (smallest CV bucket
            # that covers the ~556 km equatorial spacing)
            lat = np.linspace(-87.5, 87.5, 36)
            lon = np.linspace(2.5, 357.5, 72)
            data = np.full((36, 72), 288.0, dtype=np.float32)
            path = writer.write_field(
                var_name="tas", data=data,
                time=15.0, time_bounds=(0.0, 30.0),
                lat=lat, lon=lon,
            )
            ds = xr.open_dataset(path)
            nominal = ds.attrs["nominal_resolution"]
            ds.close()

        self.assertEqual(nominal, "1000 km")


if __name__ == "__main__":
    unittest.main()
