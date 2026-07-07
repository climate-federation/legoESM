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
