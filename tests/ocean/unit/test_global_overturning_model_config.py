"""Unit tests for the shared ``global_overturning_model_config`` factory.

The ~dozen drivers in ``scripts/run/global_overturning/`` previously each
hand-copied the ``LatLonCGridOceanConfig`` construction.  These tests pin the
factory's contract so the drivers can delegate to it without behaviour drift:

* the no-override base reproduces the coefficients carried on
  ``GlobalOverturningConfig`` and leaves every other field at its model default;
* keyword overrides win over the base (the per-run OMIP / implicit-CN stack);
* the GM/Redi path activates exactly when ``use_gm_redi`` is set (or an explicit
  config is passed).
"""

from __future__ import annotations

import dataclasses

import pytest
from legoesm.ocean.experiments import AVAILABLE_EXPERIMENTS
from legoesm.ocean.experiments.global_overturning import (
    GlobalOverturningConfig,
    create_eos_config,
    create_gm_redi_config,
    global_overturning_model_config,
    global_overturning_mpas_model_config,
)
from legoesm.ocean.state import LatLonCGridOceanConfig


class TestBaseAssembly:
    def test_default_latlon_bit_identical_to_pre_catalog_factory(self):
        cfg = GlobalOverturningConfig()
        physics = object()
        eos = create_eos_config(cfg)
        gm_redi = create_gm_redi_config(cfg)
        actual = global_overturning_model_config(
            cfg, physics=physics, eos_config=eos, gm_redi_cfg=gm_redi)
        expected = LatLonCGridOceanConfig.from_flat(
            physics=physics,
            eos="linear",
            eos_linear=eos,
            momentum_advection="vector_invariant",
            tracer_advection="tvd",
            pgf_scheme="adcroft",
            ke_gradient_scheme="centered",
            barotropic_solver="explicit_substep",
            coriolis_scheme="matsuno_split",
            outer_integrator="forward_euler",
            tracer_time_integrator="euler",
            implicit_vertical_mixing=True,
            n_barotropic_substeps=30,
            A_h=cfg.A_h,
            A_v=cfg.A_v,
            K_v=cfg.K_v,
            bottom_drag_r=cfg.bottom_drag_coeff,
            gm_redi=gm_redi,
        )
        assert actual == expected

    def test_base_matches_config_coefficients(self):
        cfg = GlobalOverturningConfig()
        mc = global_overturning_model_config(cfg)
        assert mc.lateral_viscosity.A_h == cfg.A_h
        assert mc.A_v == cfg.A_v
        assert mc.K_v == cfg.K_v
        assert mc.bottom_drag.bottom_drag_r == cfg.bottom_drag_coeff
        assert mc.eos == "linear"
        assert mc.eos_linear is not None
        # T-only buoyancy: linear EOS carries the config's alpha_T, beta_S=0.
        assert mc.eos_linear.alpha_T == cfg.alpha_T
        assert mc.eos_linear.beta_S == 0.0

    def test_base_leaves_dissipation_off(self):
        """Anti-vacuity guard: the factory must NOT silently turn on dissipation
        a driver didn't ask for — these stay at the LatLonCGridOceanConfig
        default (the structural SCHEME fields are pinned separately; see
        test_recipe_snapshots)."""
        mc = global_overturning_model_config(GlobalOverturningConfig())
        default = LatLonCGridOceanConfig.from_flat()
        for field in ("B_h", "C_smag", "C_smag_lap", "slope_foot_alpha",
                      "A_h_lat_scaling", "A_h_floor", "A_h_eq_boost",
                      "A_h_eq_sigma_deg", "A_h_merid", "n_barotropic_substeps",
                      "bottom_drag_bbl_thickness", "bottom_drag_bg_velocity",
                      "maxvel_barotropic"):
            assert mc.flat_get(field) == default.flat_get(field), field

    def test_none_config_uses_defaults(self):
        assert global_overturning_model_config(None).lateral_viscosity.A_h == \
            GlobalOverturningConfig().A_h

    def test_physics_passed_through(self):
        sentinel = object()
        mc = global_overturning_model_config(
            GlobalOverturningConfig(), physics=sentinel)
        assert mc.physics is sentinel

    def test_explicit_eos_config_honoured(self):
        cfg = GlobalOverturningConfig()
        eos = create_eos_config(cfg)
        mc = global_overturning_model_config(cfg, eos_config=eos)
        assert mc.eos_linear is eos


class TestOverrides:
    def test_overrides_win_over_base(self):
        """The genuine per-run OMIP / implicit-CN variations layer on top."""
        cfg = GlobalOverturningConfig()
        mc = global_overturning_model_config(
            cfg,
            barotropic_solver="implicit_cn",
            B_h=5.0e9,
            C_smag=0.2,
            A_h=1.234e5,
            A_h_lat_scaling=True,
            n_barotropic_substeps=30,
            bottom_drag_bbl_thickness=100.0,
        )
        assert mc.barotropic.barotropic_solver == "implicit_cn"
        assert mc.lateral_viscosity.B_h == 5.0e9
        assert mc.lateral_viscosity.C_smag == 0.2
        assert mc.lateral_viscosity.A_h == 1.234e5            # override beats config.A_h
        assert mc.lateral_viscosity.A_h_lat_scaling is True
        assert mc.bottom_drag.bottom_drag_bbl_thickness == 100.0

    def test_unknown_field_raises(self):
        with pytest.raises(TypeError):
            global_overturning_model_config(
                GlobalOverturningConfig(), not_a_real_field=1.0)


class TestGMRedi:
    def test_off_by_default(self):
        mc = global_overturning_model_config(GlobalOverturningConfig())
        assert mc.gm_redi is None

    def test_on_when_use_gm_redi(self):
        cfg = dataclasses.replace(GlobalOverturningConfig(), use_gm_redi=True)
        mc = global_overturning_model_config(cfg)
        assert mc.gm_redi is not None
        assert mc.gm_redi.kappa_GM == cfg.kappa_GM

    def test_explicit_gm_redi_cfg_honoured(self):
        cfg = GlobalOverturningConfig()
        sentinel = object()
        mc = global_overturning_model_config(cfg, gm_redi_cfg=sentinel)
        assert mc.gm_redi is sentinel

    def test_override_beats_gm_redi_cfg_param(self):
        cfg = dataclasses.replace(GlobalOverturningConfig(), use_gm_redi=True)
        mc = global_overturning_model_config(cfg, gm_redi=None)
        assert mc.gm_redi is None


class TestMPASFactory:
    """The MPAS (Voronoi C-grid) counterpart — same shared base on
    MPASOceanConfig, used by the mpas_baseline / mpas_50yr_implicit drivers."""

    def test_default_mpas_bit_identical_to_pre_catalog_factory(self):
        from legoesm.ocean.mpas_config import MPASOceanConfig

        cfg = GlobalOverturningConfig()
        physics = object()
        eos = create_eos_config(cfg)
        gm_redi = create_gm_redi_config(cfg)
        actual = global_overturning_mpas_model_config(
            cfg, physics=physics, eos_config=eos, gm_redi_cfg=gm_redi)
        expected = MPASOceanConfig(
            physics=physics,
            eos="linear",
            eos_linear=eos,
            tracer_advection="upwind",
            pgf_scheme="centered",
            barotropic_solver="explicit_substep",
            pv_scheme="enstrophy",
            implicit_vertical_mixing=True,
            A_h=cfg.A_h,
            A_v=cfg.A_v,
            K_v=cfg.K_v,
            bottom_drag_r=cfg.bottom_drag_coeff,
            gm_redi=gm_redi,
        )
        assert actual == expected

    def test_base_matches_config_coefficients(self):
        cfg = GlobalOverturningConfig()
        mc = global_overturning_mpas_model_config(cfg)
        assert mc.A_h == cfg.A_h
        assert mc.A_v == cfg.A_v
        assert mc.K_v == cfg.K_v
        assert mc.bottom_drag_r == cfg.bottom_drag_coeff  # MPAS: flat (not grouped)
        assert mc.eos == "linear"          # NB: MPASOceanConfig defaults to "wright"
        assert mc.eos_linear is not None

    def test_base_leaves_other_fields_at_mpas_default(self):
        from legoesm.ocean.mpas_config import MPASOceanConfig
        mc = global_overturning_mpas_model_config(GlobalOverturningConfig())
        default = MPASOceanConfig()
        for field in ("n_barotropic_substeps", "barotropic_solver",
                      "barotropic_u_viscosity", "gm_redi"):
            assert getattr(mc, field) == getattr(default, field), field

    def test_overrides_win(self):
        mc = global_overturning_mpas_model_config(
            GlobalOverturningConfig(),
            A_h=9.9e5, barotropic_solver="implicit_cn",
            barotropic_u_viscosity=1.0e3,
        )
        assert mc.A_h == 9.9e5             # override beats config.A_h
        assert mc.barotropic_solver == "implicit_cn"  # MPAS: flat (not grouped)
        assert mc.barotropic_u_viscosity == 1.0e3

    def test_gm_redi_off_by_default(self):
        assert global_overturning_mpas_model_config(
            GlobalOverturningConfig()).gm_redi is None

    def test_registered_in_experiment_config(self):
        assert AVAILABLE_EXPERIMENTS["global_overturning"][
            "create_mpas_model_config"] is global_overturning_mpas_model_config


def test_registered_in_experiment_config():
    assert AVAILABLE_EXPERIMENTS["global_overturning"]["create_model_config"] \
        is global_overturning_model_config
