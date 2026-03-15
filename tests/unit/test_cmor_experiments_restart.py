"""Tests for CMOR output, experiment templates, restart, and tuning modules."""

import json
import os
import tempfile
import unittest

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


# ======================================================================
# Experiment templates
# ======================================================================


class TestExperimentTemplates(unittest.TestCase):
    """CMIP-style experiment templates."""

    def test_all_templates_present(self):
        """All expected experiments are defined."""
        from legoesm.forcing.experiments import EXPERIMENT_TEMPLATES

        expected = {"piControl", "historical", "ssp245", "ssp585", "amip", "1pctCO2"}
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

    def test_overrides(self):
        """Overrides are applied to the config."""
        from legoesm.forcing.experiments import create_experiment_config

        cfg = create_experiment_config("piControl", resolution=48, dt=300.0)
        self.assertEqual(cfg.resolution, 48)
        self.assertEqual(cfg.dt, 300.0)

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
        from legoesm.io.restart import compute_state_digest

        arrays = {"T": np.ones(100), "u": np.zeros(100)}
        d1 = compute_state_digest(arrays)
        d2 = compute_state_digest(arrays)
        self.assertEqual(d1, d2)

    def test_different_values(self):
        """Different array values give different digest."""
        from legoesm.io.restart import compute_state_digest

        d1 = compute_state_digest({"T": np.ones(100)})
        d2 = compute_state_digest({"T": np.zeros(100)})
        self.assertNotEqual(d1, d2)

    def test_key_order_invariant(self):
        """Digest is invariant to insertion order (keys are sorted)."""
        from legoesm.io.restart import compute_state_digest

        a = {"b": np.ones(10), "a": np.zeros(10)}
        b = {"a": np.zeros(10), "b": np.ones(10)}
        self.assertEqual(compute_state_digest(a), compute_state_digest(b))


class TestConfigHash(unittest.TestCase):
    """Config hashing."""

    def test_deterministic(self):
        """Same config gives same hash."""
        from legoesm.io.restart import compute_config_hash
        from legoesm.forcing.amip_config import AMIPExperimentConfig

        cfg = AMIPExperimentConfig()
        h1 = compute_config_hash(cfg)
        h2 = compute_config_hash(cfg)
        self.assertEqual(h1, h2)

    def test_different_config(self):
        """Different config gives different hash."""
        from legoesm.io.restart import compute_config_hash
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
        from legoesm.io.restart import save_restart, load_restart
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
        from legoesm.io.restart import save_restart
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
        from legoesm.io.restart import save_restart, verify_reproducibility
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
        expected = {"dynamics", "radiation", "convection", "diffusion", "surface"}
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


if __name__ == "__main__":
    unittest.main()
