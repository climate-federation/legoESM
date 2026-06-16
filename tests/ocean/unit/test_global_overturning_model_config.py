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
    global_overturning_model_config,
    global_overturning_mpas_model_config,
)
from legoesm.ocean.state import LatLonCGridOceanConfig


class TestBaseAssembly:
    def test_base_matches_config_coefficients(self):
        cfg = GlobalOverturningConfig()
        mc = global_overturning_model_config(cfg)
        assert mc.A_h == cfg.A_h
        assert mc.A_v == cfg.A_v
        assert mc.K_v == cfg.K_v
        assert mc.bottom_drag_r == cfg.bottom_drag_coeff
        assert mc.eos == "linear"
        assert mc.eos_linear is not None
        # T-only buoyancy: linear EOS carries the config's alpha_T, beta_S=0.
        assert mc.eos_linear.alpha_T == cfg.alpha_T
        assert mc.eos_linear.beta_S == 0.0

    def test_base_leaves_other_fields_at_model_default(self):
        """The factory must NOT silently turn on dissipation a driver didn't ask
        for — every field it doesn't set stays at the LatLonCGridOceanConfig
        default (bit-identical to the old minimal inline construction)."""
        mc = global_overturning_model_config(GlobalOverturningConfig())
        default = LatLonCGridOceanConfig()
        for field in ("B_h", "C_smag", "C_smag_lap", "slope_foot_alpha",
                      "A_h_lat_scaling", "A_h_floor", "A_h_eq_boost",
                      "A_h_eq_sigma_deg", "A_h_merid", "n_barotropic_substeps",
                      "bottom_drag_bbl_thickness", "bottom_drag_bg_velocity",
                      "maxvel_barotropic", "barotropic_solver", "pgf_scheme",
                      "momentum_advection"):
            assert getattr(mc, field) == getattr(default, field), field

    def test_none_config_uses_defaults(self):
        assert global_overturning_model_config(None).A_h == \
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
        assert mc.barotropic_solver == "implicit_cn"
        assert mc.B_h == 5.0e9
        assert mc.C_smag == 0.2
        assert mc.A_h == 1.234e5            # override beats config.A_h
        assert mc.A_h_lat_scaling is True
        assert mc.bottom_drag_bbl_thickness == 100.0

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

    def test_base_matches_config_coefficients(self):
        cfg = GlobalOverturningConfig()
        mc = global_overturning_mpas_model_config(cfg)
        assert mc.A_h == cfg.A_h
        assert mc.A_v == cfg.A_v
        assert mc.K_v == cfg.K_v
        assert mc.bottom_drag_r == cfg.bottom_drag_coeff
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
        assert mc.barotropic_solver == "implicit_cn"
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
