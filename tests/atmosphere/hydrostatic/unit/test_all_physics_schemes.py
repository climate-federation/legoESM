"""Systematic test of every physics scheme option in legoESM.

Tests each scheme individually for:
1. No crashes / import errors
2. All outputs are finite (no NaN/Inf)
3. Physically reasonable magnitudes
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
from legoesm.atmosphere.physics.radiation.config import RadiationConfig, RRTMGPConfig
from legoesm.atmosphere.physics.convection.config import ConvectionConfig
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
from legoesm.core.field import Field


def _make_hydrostatic_setup():
    """Create a minimal hydrostatic state with realistic profiles."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init

    grid = create_cubed_sphere(8)
    sigma = create_sigma_coordinate(10)
    state = held_suarez_init(grid, sigma)
    n, nlev = grid.n, sigma.n_levels
    # Add realistic wind so turbulence/GWD have something to work on.
    # Pin the wind dtype to the rest of the state's precision (set by
    # the active precision policy) — defaulting to ``jnp.ones`` would
    # produce float64 under JAX_ENABLE_X64=1 even when the policy is
    # float32, which silently promotes the column physics path through
    # surface fluxes / wind_speed / surface_flux into float64.
    _dtype = state.T.data.dtype
    state = state._replace(
        u=Field(data=jnp.ones((6, n, n, nlev), dtype=_dtype) * 10.0,
                name="u", dims=("face", "x", "y", "level"), units="m/s"),
        v=Field(data=jnp.ones((6, n, n, nlev), dtype=_dtype) * 3.0,
                name="v", dims=("face", "x", "y", "level"), units="m/s"),
    )
    return state, grid, sigma


def _none_config(**overrides):
    """Create a PhysicsConfig with all modules set to 'none', then override."""
    base = dict(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    base.update(overrides)
    return PhysicsConfig(**base)


def _check_tendencies(tend, label, max_dT=50.0, max_du=1.0):
    """Check tendencies are finite and physically reasonable.

    Parameters
    ----------
    tend : HydrostaticTendencies
    label : str
        Scheme label for error messages.
    max_dT : float
        Maximum acceptable |dT/dt| in K/s.
    max_du : float
        Maximum acceptable |du/dt| or |dv/dt| in m/s^2.
    """
    # Check finite
    assert jnp.all(jnp.isfinite(tend.dT_dt.data)), f"{label}: dT_dt has NaN/Inf"
    assert jnp.all(jnp.isfinite(tend.du_dt.data)), f"{label}: du_dt has NaN/Inf"
    assert jnp.all(jnp.isfinite(tend.dv_dt.data)), f"{label}: dv_dt has NaN/Inf"

    # Check magnitudes are physically reasonable
    max_dT_actual = float(jnp.max(jnp.abs(tend.dT_dt.data)))
    max_du_actual = float(jnp.max(jnp.abs(tend.du_dt.data)))
    max_dv_actual = float(jnp.max(jnp.abs(tend.dv_dt.data)))

    assert max_dT_actual < max_dT, (
        f"{label}: |dT/dt| = {max_dT_actual:.2e} K/s exceeds {max_dT} K/s"
    )
    assert max_du_actual < max_du, (
        f"{label}: |du/dt| = {max_du_actual:.2e} m/s^2 exceeds {max_du} m/s^2"
    )
    assert max_dv_actual < max_du, (
        f"{label}: |dv/dt| = {max_dv_actual:.2e} m/s^2 exceeds {max_du} m/s^2"
    )


# ============================================================
# RADIATION
# ============================================================

class TestRadiationSchemes:
    """Test each radiation scheme individually."""

    def test_gray(self):
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = _none_config(radiation=RadiationConfig(scheme="gray"))
        tend, _ = make_physics(cfg, "hydrostatic", dt=300.0)(state, grid, sigma)
        _check_tendencies(tend, "radiation/gray")
        assert float(jnp.max(jnp.abs(tend.dT_dt.data))) > 0.0, "gray: zero heating"

    def test_rrtmgp(self):
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = _none_config(radiation=RadiationConfig(scheme="rrtmgp"))
        tend, _ = make_physics(cfg, "hydrostatic", dt=300.0)(state, grid, sigma)
        _check_tendencies(tend, "radiation/rrtmgp")
        assert float(jnp.max(jnp.abs(tend.dT_dt.data))) > 0.0, "rrtmgp: zero heating"


# ============================================================
# CONVECTION
# ============================================================

class TestConvectionSchemes:
    """Test each convection scheme individually."""

    def test_sbm(self):
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = _none_config(convection=ConvectionConfig(scheme="sbm"))
        tend, _ = make_physics(cfg, "hydrostatic", dt=300.0)(state, grid, sigma)
        _check_tendencies(tend, "convection/sbm")

    def test_dca(self):
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = _none_config(convection=ConvectionConfig(scheme="dca"))
        tend, _ = make_physics(cfg, "hydrostatic", dt=300.0)(state, grid, sigma)
        _check_tendencies(tend, "convection/dca")

    def test_kuo(self):
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = _none_config(convection=ConvectionConfig(scheme="kuo"))
        tend, _ = make_physics(cfg, "hydrostatic", dt=300.0)(state, grid, sigma)
        _check_tendencies(tend, "convection/kuo")

    def test_mass_flux(self):
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = _none_config(convection=ConvectionConfig(scheme="mass_flux"))
        tend, _ = make_physics(cfg, "hydrostatic", dt=300.0)(state, grid, sigma)
        _check_tendencies(tend, "convection/mass_flux")

    def test_edmf(self):
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = _none_config(convection=ConvectionConfig(scheme="edmf"))
        tend, _ = make_physics(cfg, "hydrostatic", dt=300.0)(state, grid, sigma)
        _check_tendencies(tend, "convection/edmf")


# ============================================================
# TURBULENCE
# ============================================================

class TestTurbulenceSchemes:
    """Test each turbulence scheme individually."""

    def test_smagorinsky(self):
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = _none_config(turbulence=TurbulenceConfig(scheme="smagorinsky"))
        tend, _ = make_physics(cfg, "hydrostatic", dt=300.0)(state, grid, sigma)
        _check_tendencies(tend, "turbulence/smagorinsky")
        assert float(jnp.max(jnp.abs(tend.du_dt.data))) > 0.0, "smagorinsky: zero wind tendency"

    def test_louis(self):
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = _none_config(turbulence=TurbulenceConfig(scheme="louis"))
        tend, _ = make_physics(cfg, "hydrostatic", dt=300.0)(state, grid, sigma)
        _check_tendencies(tend, "turbulence/louis")
        assert float(jnp.max(jnp.abs(tend.du_dt.data))) > 0.0, "louis: zero wind tendency"

    def test_tke(self):
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = _none_config(turbulence=TurbulenceConfig(scheme="tke"))
        tend, _ = make_physics(cfg, "hydrostatic", dt=300.0)(state, grid, sigma)
        _check_tendencies(tend, "turbulence/tke")

    def test_clubb_lite(self):
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = _none_config(turbulence=TurbulenceConfig(scheme="clubb_lite"))
        tend, _ = make_physics(cfg, "hydrostatic", dt=300.0)(state, grid, sigma)
        _check_tendencies(tend, "turbulence/clubb_lite")

    def test_holtslag_boville(self):
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = _none_config(turbulence=TurbulenceConfig(scheme="holtslag_boville"))
        tend, _ = make_physics(cfg, "hydrostatic", dt=300.0)(state, grid, sigma)
        _check_tendencies(tend, "turbulence/holtslag_boville")

    def test_ysu(self):
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = _none_config(turbulence=TurbulenceConfig(scheme="ysu"))
        tend, _ = make_physics(cfg, "hydrostatic", dt=300.0)(state, grid, sigma)
        _check_tendencies(tend, "turbulence/ysu")

    def test_edmf(self):
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = _none_config(turbulence=TurbulenceConfig(scheme="edmf"))
        tend, _ = make_physics(cfg, "hydrostatic", dt=300.0)(state, grid, sigma)
        _check_tendencies(tend, "turbulence/edmf")

# ============================================================
# MICROPHYSICS
# ============================================================

class TestMicrophysicsSchemes:
    """Test each microphysics scheme individually."""

    def test_kessler(self):
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = _none_config(microphysics=MicrophysicsConfig(scheme="kessler"))
        tend, _ = make_physics(cfg, "hydrostatic", dt=300.0)(state, grid, sigma)
        _check_tendencies(tend, "microphysics/kessler")

    def test_microphysics_exports_surface_precip(self):
        """The lean microphysics bridge carries surface precip on the combined
        tendency (``HydrostaticTendencies.precip``) so the coupled MPAS loop can
        export the ocean P-E / land precip forcing (the seg_precip keystone
        residual). A supersaturated column must rain; a no-microphysics run
        leaves precip=None (byte-identical)."""
        from legoesm.thermo import saturation_mixing_ratio
        state, grid, sigma = _make_hydrostatic_setup()
        p_full = state.p_s.data[..., None] * jnp.asarray(sigma.sigma_full)
        q_sat = saturation_mixing_ratio(state.T.data, p_full)
        _dtype = state.T.data.dtype
        dims4 = ("face", "x", "y", "level")
        z = jnp.zeros_like(state.T.data)
        # Seed rain aloft (q_r>0) so kessler sedimentation delivers surface
        # precip within one step (a q_r=0 IC needs several steps to autoconvert
        # + fall, so it would spuriously read zero).
        state = state._replace(tracers={
            "q_v": Field(data=(1.3 * q_sat).astype(_dtype), name="q_v",
                         dims=dims4, units="kg/kg"),
            "q_c": Field(data=(z + 1.0e-3).astype(_dtype), name="q_c",
                         dims=dims4, units="kg/kg"),
            "q_r": Field(data=(z + 1.0e-3).astype(_dtype), name="q_r",
                         dims=dims4, units="kg/kg"),
        })
        cfg = _none_config(microphysics=MicrophysicsConfig(scheme="kessler"))
        tend, _ = make_physics(cfg, "hydrostatic", dt=300.0)(state, grid, sigma)
        assert tend.precip is not None, "microphysics did not export precip"
        assert jnp.all(jnp.isfinite(tend.precip.data))
        assert float(jnp.sum(tend.precip.data)) > 0.0, (
            "supersaturated column produced zero surface precip — the precip "
            "export tap is not wired")
        # No microphysics -> no precip channel (None default, byte-identical).
        tend_dry, _ = make_physics(
            _none_config(radiation=RadiationConfig(scheme="gray")),
            "hydrostatic", dt=300.0)(state, grid, sigma)
        assert tend_dry.precip is None

    def test_sundqvist(self):
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = _none_config(microphysics=MicrophysicsConfig(scheme="sundqvist"))
        tend, _ = make_physics(cfg, "hydrostatic", dt=300.0)(state, grid, sigma)
        _check_tendencies(tend, "microphysics/sundqvist")

    def test_seifert_beheng(self):
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = _none_config(microphysics=MicrophysicsConfig(scheme="seifert_beheng"))
        tend, _ = make_physics(cfg, "hydrostatic", dt=300.0)(state, grid, sigma)
        _check_tendencies(tend, "microphysics/seifert_beheng")

    def test_morrison(self):
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = _none_config(microphysics=MicrophysicsConfig(scheme="morrison"))
        tend, _ = make_physics(cfg, "hydrostatic", dt=300.0)(state, grid, sigma)
        _check_tendencies(tend, "microphysics/morrison")

    def test_thompson(self):
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = _none_config(microphysics=MicrophysicsConfig(scheme="thompson"))
        tend, _ = make_physics(cfg, "hydrostatic", dt=300.0)(state, grid, sigma)
        _check_tendencies(tend, "microphysics/thompson")

    def test_ml_emulator(self):
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = _none_config(microphysics=MicrophysicsConfig(scheme="ml_emulator"))
        tend, _ = make_physics(cfg, "hydrostatic", dt=300.0)(state, grid, sigma)
        _check_tendencies(tend, "microphysics/ml_emulator")


# ============================================================
# GRAVITY WAVE DRAG
# ============================================================

class TestGWDSchemes:
    """Test each gravity wave drag scheme individually."""

    def test_rayleigh(self):
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = _none_config(gravity_wave_drag=GravityWaveDragConfig(scheme="rayleigh"))
        tend, _ = make_physics(cfg, "hydrostatic", dt=300.0)(state, grid, sigma)
        _check_tendencies(tend, "gwd/rayleigh")
        assert float(jnp.max(jnp.abs(tend.du_dt.data))) > 0.0, "rayleigh: zero wind tendency"

    def test_lindzen(self):
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = _none_config(gravity_wave_drag=GravityWaveDragConfig(scheme="lindzen"))
        tend, _ = make_physics(cfg, "hydrostatic", dt=300.0)(state, grid, sigma)
        _check_tendencies(tend, "gwd/lindzen")

    def test_mcfarlane(self):
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = _none_config(gravity_wave_drag=GravityWaveDragConfig(scheme="mcfarlane"))
        tend, _ = make_physics(cfg, "hydrostatic", dt=300.0)(state, grid, sigma)
        _check_tendencies(tend, "gwd/mcfarlane")

    def test_hines(self):
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = _none_config(gravity_wave_drag=GravityWaveDragConfig(scheme="hines"))
        tend, _ = make_physics(cfg, "hydrostatic", dt=300.0)(state, grid, sigma)
        _check_tendencies(tend, "gwd/hines")

    def test_prognostic_spectral(self):
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = _none_config(gravity_wave_drag=GravityWaveDragConfig(scheme="prognostic_spectral"))
        tend, _ = make_physics(cfg, "hydrostatic", dt=300.0)(state, grid, sigma)
        _check_tendencies(tend, "gwd/prognostic_spectral")

    def test_ml_emulator(self):
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = _none_config(gravity_wave_drag=GravityWaveDragConfig(scheme="ml_emulator"))
        tend, _ = make_physics(cfg, "hydrostatic", dt=300.0)(state, grid, sigma)
        _check_tendencies(tend, "gwd/ml_emulator")


# ============================================================
# COMBINED: all modules active with various combinations
# ============================================================

class TestFullPhysicsCombinations:
    """Test combinations of all five physics modules together."""

    def test_all_defaults(self):
        """Default config (gray + sbm + smagorinsky + none + none)."""
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = PhysicsConfig()
        tend, _ = make_physics(cfg, "hydrostatic", dt=300.0)(state, grid, sigma)
        _check_tendencies(tend, "defaults")

    def test_full_five_module_stack(self):
        """All five modules active simultaneously."""
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = PhysicsConfig(
            radiation=RadiationConfig(scheme="gray"),
            convection=ConvectionConfig(scheme="sbm"),
            turbulence=TurbulenceConfig(scheme="louis"),
            microphysics=MicrophysicsConfig(scheme="kessler"),
            gravity_wave_drag=GravityWaveDragConfig(scheme="rayleigh"),
        )
        tend, _ = make_physics(cfg, "hydrostatic", dt=300.0)(state, grid, sigma)
        _check_tendencies(tend, "full_stack")
        assert float(jnp.max(jnp.abs(tend.dT_dt.data))) > 0.0
        assert float(jnp.max(jnp.abs(tend.du_dt.data))) > 0.0

    def test_advanced_stack(self):
        """All five with more advanced schemes."""
        state, grid, sigma = _make_hydrostatic_setup()
        cfg = PhysicsConfig(
            radiation=RadiationConfig(scheme="gray"),
            convection=ConvectionConfig(scheme="kuo"),
            turbulence=TurbulenceConfig(scheme="tke"),
            microphysics=MicrophysicsConfig(scheme="sundqvist"),
            gravity_wave_drag=GravityWaveDragConfig(scheme="lindzen"),
        )
        tend, _ = make_physics(cfg, "hydrostatic", dt=300.0)(state, grid, sigma)
        _check_tendencies(tend, "advanced_stack")
