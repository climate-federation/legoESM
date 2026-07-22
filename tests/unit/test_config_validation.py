"""Tests for ExperimentConfig.validate() cross-checking."""

from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig


class TestConfigValidate:
    def test_clean_config_no_warnings(self):
        cfg = ExperimentConfig()
        assert cfg.validate() == []

    def test_aerosol_with_gray_radiation(self):
        cfg = ExperimentConfig(radiation="gray", aerosol_forcing="external")
        warns = cfg.validate()
        assert any("aerosol" in w for w in warns)

    def test_microphysics_with_gray(self):
        cfg = ExperimentConfig(radiation="gray", microphysics="kessler")
        warns = cfg.validate()
        assert any("microphysics" in w.lower() for w in warns)

    def test_large_dt_warns(self):
        cfg = ExperimentConfig(
            dycore=DycoreConfig(dt=1200.0),
            grid=GridConfig(resolution=32),
        )
        warns = cfg.validate()
        assert any("CFL" in w for w in warns)

    def test_small_dt_no_cfl_warning(self):
        cfg = ExperimentConfig(
            dycore=DycoreConfig(dt=300.0),
            grid=GridConfig(resolution=32),
        )
        warns = cfg.validate()
        assert not any("CFL" in w for w in warns)

    def test_joint_parameterization_warns_on_wrong_physical_stack(self):
        cfg = ExperimentConfig(
            convection="sbm",
            turbulence="none",
            physics_parameterization="ml",
            physics_parameterization_checkpoint="model.eqx",
            physics_parameterization_stats="stats.npz",
        )
        warns = cfg.validate()
        assert any("convection='mass_flux'" in w for w in warns)

    def test_joint_parameterization_warns_on_partial_assets(self):
        cfg = ExperimentConfig(
            convection="mass_flux",
            turbulence="louis",
            physics_parameterization="ml",
            physics_parameterization_checkpoint="model.eqx",
            physics_parameterization_stats="",
        )
        warns = cfg.validate()
        assert any("physics_parameterization='ml'" in w for w in warns)


class TestGWDCompositeValidation:
    """`+`-composite gravity_wave_drag membership (issue #834).

    A composite sums multiple GWD sources.  prognostic_spectral is now an
    allowed (single) stateful member; unknown parts and >1 stateful source
    must fail validate_strict.
    """

    import pytest

    def test_combined_mcfarlane_spectral_is_valid(self):
        # The #834 target composite must pass strict validation.
        ExperimentConfig(
            gravity_wave_drag="mcfarlane+prognostic_spectral"
        ).validate_strict()

    def test_stateless_composite_is_valid(self):
        ExperimentConfig(gravity_wave_drag="hines+mcfarlane").validate_strict()

    def test_single_prognostic_spectral_is_valid(self):
        ExperimentConfig(gravity_wave_drag="prognostic_spectral").validate_strict()

    def test_unknown_part_in_composite_raises(self):
        import pytest
        with pytest.raises(ValueError, match="composite gravity_wave_drag"):
            ExperimentConfig(
                gravity_wave_drag="mcfarlane+nonsense"
            ).validate_strict()

    def test_two_stateful_parts_raises(self):
        import pytest
        # e3sm_cam is not composable; two spectral sources share one carry.
        with pytest.raises(ValueError, match="at most one stateful"):
            ExperimentConfig(
                gravity_wave_drag="prognostic_spectral+prognostic_spectral"
            ).validate_strict()

    def test_noncomposable_scheme_in_composite_raises(self):
        import pytest
        with pytest.raises(ValueError, match="composite gravity_wave_drag"):
            ExperimentConfig(
                gravity_wave_drag="mcfarlane+e3sm_cam"
            ).validate_strict()


class TestForcingSelectorValidation:
    """2026-07-21 AMIP/CMIP audit: forcing-source selectors are membership-
    validated in validate_strict (the driver activates each channel with an
    equality gate, so a typo used to silently deactivate the channel)."""

    def test_typo_ozone_forcing_rejected(self):
        import pytest
        with pytest.raises(ValueError, match="ozone_forcing"):
            ExperimentConfig(ozone_forcing="externl").validate_strict()

    def test_typo_ghg_forcing_rejected(self):
        import pytest
        with pytest.raises(ValueError, match="ghg_forcing"):
            ExperimentConfig(ghg_forcing="file").validate_strict()

    def test_typo_solar_source_rejected(self):
        import pytest
        with pytest.raises(ValueError, match="solar_source"):
            ExperimentConfig(solar_source="spectral").validate_strict()

    def test_typo_aerosol_forcing_rejected(self):
        import pytest
        with pytest.raises(ValueError, match="aerosol_forcing"):
            ExperimentConfig(aerosol_forcing="on").validate_strict()

    def test_typo_dataset_rejected(self):
        import pytest
        with pytest.raises(ValueError, match="dataset"):
            ExperimentConfig(dataset="hadsst").validate_strict()

    def test_unknown_experiment_rejected(self):
        import pytest
        with pytest.raises(ValueError, match="experiment"):
            ExperimentConfig(experiment="histrical").validate_strict()

    def test_known_experiments_accepted(self):
        for exp in ("", "amip", "historical", "piControl", "abrupt-4xCO2",
                    "1pctCO2", "ssp245", "ssp585"):
            ExperimentConfig(experiment=exp).validate_strict()

    def test_fixed_experiment_with_external_ghg_rejected(self):
        # abrupt-4xCO2 prescribes FIXED GHG scalars; an external annual file
        # would take precedence in _precompute_external_forcing and silently
        # run the file's (historical) trajectory instead of 4xCO2.
        import pytest
        with pytest.raises(ValueError, match="FIXED GHG"):
            ExperimentConfig(
                experiment="abrupt-4xCO2", ghg_forcing="external",
                ghg_file="ghg.nc",
            ).validate_strict()

    def test_transient_experiment_with_external_ghg_allowed(self):
        ExperimentConfig(
            experiment="historical", ghg_forcing="external", ghg_file="g.nc",
        ).validate_strict()
