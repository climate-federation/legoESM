"""Phase 0: Infrastructure validation for CMIP stress test suite."""

import math

import jax
jax.config.update("jax_enable_x64", True)

import pytest

from legoesm.forcing.experiments import (
    create_experiment_config,
    EXPERIMENT_TEMPLATES,
    ghg_at_year,
)
from legoesm.driver.config import (
    ExperimentConfig,
    experiment_config_to_dict,
    experiment_config_from_dict,
)
from legoesm.driver.coupled_config import PRESETS, CoupledConfig
from legoesm.io.cmor_output import CMOR_TABLES


# ======================================================================
# Test 0.1 — Experiment Templates
# ======================================================================

_EXPERIMENT_NAMES = ["piControl", "historical", "ssp245", "ssp585", "amip", "1pctCO2"]


class TestExperimentTemplates:
    """Test 0.1: All experiment templates instantiate correctly."""

    @pytest.mark.parametrize("name", _EXPERIMENT_NAMES)
    def test_all_templates_instantiate(self, name):
        cfg = create_experiment_config(name)
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.days > 0, f"days must be > 0, got {cfg.days}"
        assert cfg.co2_ppmv > 0, f"co2_ppmv must be > 0, got {cfg.co2_ppmv}"


# ======================================================================
# Test 0.2 — GHG Continuity
# ======================================================================

class TestGHGContinuity:
    """Test 0.2: GHG concentration tables are monotonic and well-formed."""

    def test_historical_monotonic(self):
        co2_values = []
        for y in range(1850, 2015):
            co2, ch4, n2o = ghg_at_year("historical", y)
            assert math.isfinite(co2), f"CO2 not finite at year {y}"
            assert math.isfinite(ch4), f"CH4 not finite at year {y}"
            assert math.isfinite(n2o), f"N2O not finite at year {y}"
            co2_values.append(co2)
        for i in range(1, len(co2_values)):
            assert co2_values[i] >= co2_values[i - 1], (
                f"CO2 decreased from year {1850 + i - 1} to {1850 + i}: "
                f"{co2_values[i - 1]} -> {co2_values[i]}"
            )

    def test_ssp585_monotonic(self):
        co2_values = []
        for y in range(2015, 2101):
            co2, ch4, n2o = ghg_at_year("ssp585", y)
            assert math.isfinite(co2), f"CO2 not finite at year {y}"
            assert math.isfinite(ch4), f"CH4 not finite at year {y}"
            assert math.isfinite(n2o), f"N2O not finite at year {y}"
            co2_values.append(co2)
        for i in range(1, len(co2_values)):
            assert co2_values[i] >= co2_values[i - 1], (
                f"CO2 decreased from year {2015 + i - 1} to {2015 + i}: "
                f"{co2_values[i - 1]} -> {co2_values[i]}"
            )

    def test_ssp245_monotonic(self):
        co2_values = []
        for y in range(2015, 2101):
            co2, ch4, n2o = ghg_at_year("ssp245", y)
            assert math.isfinite(co2), f"CO2 not finite at year {y}"
            assert math.isfinite(ch4), f"CH4 not finite at year {y}"
            assert math.isfinite(n2o), f"N2O not finite at year {y}"
            co2_values.append(co2)
        for i in range(1, len(co2_values)):
            assert co2_values[i] >= co2_values[i - 1], (
                f"CO2 decreased from year {2015 + i - 1} to {2015 + i}: "
                f"{co2_values[i - 1]} -> {co2_values[i]}"
            )

    def test_1pctco2_growth_rate(self):
        tmpl = EXPERIMENT_TEMPLATES["1pctCO2"]
        start_year = tmpl.start_year
        for i in range(150):
            y = start_year + i
            co2_now, _, _ = ghg_at_year("1pctCO2", y)
            co2_next, _, _ = ghg_at_year("1pctCO2", y + 1)
            ratio = co2_next / co2_now
            assert abs(ratio - 1.01) < 0.001, (
                f"1pctCO2 growth ratio at year {y}->{y + 1} is {ratio:.6f}, "
                f"expected 1.01 +/- 0.001"
            )


# ======================================================================
# Test 0.3 — Coupled Presets
# ======================================================================

class TestCoupledPresets:
    """Test 0.3: All coupled presets create valid CoupledConfig."""

    @pytest.mark.parametrize("preset_name", list(PRESETS.keys()))
    def test_all_presets_create(self, preset_name):
        cfg = PRESETS[preset_name]()
        assert isinstance(cfg, CoupledConfig)

    def test_aquaplanet_config(self):
        cfg = PRESETS["aquaplanet"]()
        assert cfg.f_land_mode == "zero"
        assert cfg.carbon_active is False

    def test_slab_carbon_config(self):
        cfg = PRESETS["slab_carbon"]()
        assert cfg.carbon_active is True
        assert cfg.co2_tracer is True
        assert cfg.carbon_land == "differland"

    def test_full_coupled_config(self):
        cfg = PRESETS["full_coupled"]()
        assert cfg.land_mode == "multilayer"
        assert cfg.carbon_land == "differland"
        assert not hasattr(cfg, "carbon_ocean")


# ======================================================================
# Test 0.4 — CMOR Completeness
# ======================================================================

_REQUIRED_ENTRY_FIELDS = {"standard_name", "long_name", "units", "cell_methods", "dimensions"}


class TestCMORCompleteness:
    """Test 0.4: CMOR tables contain required variables and metadata."""

    def test_amon_minimum_variables(self):
        amon = CMOR_TABLES["Amon"]
        required = {
            "tas", "ta", "ua", "va", "hus", "ps", "pr", "rsut", "rlut",
            "rsds", "rlds", "rsus", "rlus", "hfss", "hfls", "ts", "psl",
            "prw", "clt", "rsdt", "zg", "wap", "hur", "hurs", "clw", "cli",
            "tauu", "tauv", "evspsbl", "rsutcs", "rlutcs",
        }
        missing = required - set(amon.keys())
        assert len(amon) >= 31, (
            f"Amon has only {len(amon)} variables, expected >= 31"
        )
        assert not missing, f"Amon missing required variables: {sorted(missing)}"

    def test_lmon_variables(self):
        lmon = CMOR_TABLES["Lmon"]
        required = {"gpp", "nee", "lai", "mrso", "mrsos", "tsl"}
        missing = required - set(lmon.keys())
        assert not missing, f"Lmon missing required variables: {sorted(missing)}"

    def test_omon_variables(self):
        omon = CMOR_TABLES["Omon"]
        required = {"tos", "sic"}
        missing = required - set(omon.keys())
        assert not missing, f"Omon missing required variables: {sorted(missing)}"

    def test_aday_variables(self):
        aday = CMOR_TABLES["Aday"]
        # ``rsut`` moved to ``CFday``: the CMIP6 ``day`` table has no
        # ``rsut`` entry, so writing one there was unpublishable.
        required = {"tas", "pr", "psl", "rlut"}
        missing = required - set(aday.keys())
        assert not missing, f"Aday missing required variables: {sorted(missing)}"
        assert "rsut" not in aday
        assert "rsut" in CMOR_TABLES["CFday"]

    def test_all_entries_have_required_fields(self):
        for table_name, table in CMOR_TABLES.items():
            for var_name, entry in table.items():
                entry_keys = set(entry.keys())
                missing = _REQUIRED_ENTRY_FIELDS - entry_keys
                assert not missing, (
                    f"{table_name}.{var_name} missing required fields: "
                    f"{sorted(missing)}"
                )


# ======================================================================
# Test 0.5 — Config Round-trip Serialization
# ======================================================================

class TestConfigRoundtrip:
    """Test 0.5: ExperimentConfig survives dict serialization round-trip."""

    @pytest.mark.parametrize("name", _EXPERIMENT_NAMES)
    def test_experiment_config_roundtrip(self, name):
        original = create_experiment_config(name)
        d = experiment_config_to_dict(original)
        restored = experiment_config_from_dict(d)

        assert restored.days == original.days, (
            f"days mismatch: {restored.days} != {original.days}"
        )
        assert restored.co2_ppmv == original.co2_ppmv, (
            f"co2_ppmv mismatch: {restored.co2_ppmv} != {original.co2_ppmv}"
        )
        assert restored.grid.resolution == original.grid.resolution, (
            f"grid.resolution mismatch: "
            f"{restored.grid.resolution} != {original.grid.resolution}"
        )
