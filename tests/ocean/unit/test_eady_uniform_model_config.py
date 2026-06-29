"""Unit tests for the shared ``eady_uniform_model_config`` factory.

The factory is the single source of the Eady-uniform dycore recipe, consumed by
both ``build_eady_uniform_setup`` (driver path) and the test matrix
(``EXPERIMENT_CONFIG["create_model_config"]``). These tests pin the corrected
eddy-resolving stack so the two consumers cannot drift, and so the matrix can no
longer silently fall back to the legacy adcroft / forward-Euler dycore.
"""

from __future__ import annotations

from legoesm.ocean.experiments import AVAILABLE_EXPERIMENTS
from legoesm.ocean.experiments.eady_uniform import (
    EadyUniformConfig,
    eady_uniform_model_config,
)


class TestDefaults:
    def test_default_bit_identical_to_pre_catalog_factory(self):
        from legoesm.ocean.eos import LinearEOSConfig
        from legoesm.ocean.state import LatLonCGridOceanConfig

        cfg = EadyUniformConfig()
        physics = object()
        eos = LinearEOSConfig(
            alpha_T=cfg.alpha_T,
            rho_ref=cfg.rho_0,
            T_ref=cfg.T_ref_C,
            S_ref=cfg.S_uniform,
        )
        actual = eady_uniform_model_config(cfg, physics=physics, eos_config=eos)
        expected = LatLonCGridOceanConfig.from_flat(
            physics=physics,
            eos="linear",
            eos_linear=eos,
            barotropic_solver="implicit_cn",
            tracer_advection="weno5",
            momentum_advection="weno5",
            ke_gradient_scheme="centered",
            tracer_time_integrator="rk3",
            outer_integrator="ab2",
            pgf_scheme="smc03",
            A_h=0.0,
            B_h=0.0,
            C_smag=cfg.C_smag,
            C_leith=0.0,
            C_smag_lap=0.0,
            smag_cfl_safety=0.0,
            A_v=cfg.A_v,
            K_v=cfg.K_v,
            bottom_drag_r=cfg.bottom_drag_coeff,
            gm_redi=None,
        )
        assert actual == expected

    def test_corrected_dycore_stack(self):
        """The defaults are the Phase-G corrected stack — NOT the legacy
        adcroft / forward-Euler baseline the matrix scrape fell back to."""
        mc = eady_uniform_model_config(EadyUniformConfig())
        assert mc.pgf_scheme == "smc03"
        assert mc.tracer_time_integrator == "rk3"
        assert mc.outer_integrator == "ab2"
        assert mc.barotropic.barotropic_solver == "implicit_cn"
        assert mc.momentum_advection == "weno5"
        assert mc.tracer_advection == "weno5"

    def test_linear_eos_and_no_gm_redi(self):
        cfg = EadyUniformConfig()
        mc = eady_uniform_model_config(cfg)
        assert mc.eos == "linear"
        assert mc.eos_linear is not None
        assert mc.eos_linear.alpha_T == cfg.alpha_T
        assert mc.gm_redi is None          # eddies resolved

    def test_coefficients_from_config(self):
        cfg = EadyUniformConfig()
        mc = eady_uniform_model_config(cfg)
        assert mc.A_v == cfg.A_v
        assert mc.K_v == cfg.K_v
        assert mc.bottom_drag.bottom_drag_r == cfg.bottom_drag_coeff

    def test_none_config(self):
        assert eady_uniform_model_config(None).pgf_scheme == "smc03"


class TestOverrides:
    def test_dissipation_and_track(self):
        mc = eady_uniform_model_config(
            EadyUniformConfig(),
            a_h=1000.0, c_smag=0.1, smag_cfl_safety=0.5,
            momentum_advection="vector_invariant",
            ke_gradient_scheme="hollingsworth")
        assert mc.lateral_viscosity.A_h == 1000.0
        assert mc.lateral_viscosity.C_smag == 0.1
        assert mc.lateral_viscosity.smag_cfl_safety == 0.5
        assert mc.momentum_advection == "vector_invariant"
        assert mc.ke_gradient_scheme == "hollingsworth"

    def test_c_smag_defaults_to_config(self):
        """c_smag=None falls back to config.C_smag (not 0)."""
        cfg = EadyUniformConfig()
        mc = eady_uniform_model_config(cfg)
        assert mc.lateral_viscosity.C_smag == cfg.C_smag

    def test_injected_eos_and_gm_redi(self):
        sentinel_eos, sentinel_gm = object(), object()
        mc = eady_uniform_model_config(
            EadyUniformConfig(), eos_config=sentinel_eos, gm_redi_cfg=sentinel_gm)
        assert mc.eos_linear is sentinel_eos
        assert mc.gm_redi is sentinel_gm


def test_registered_in_experiment_config():
    assert AVAILABLE_EXPERIMENTS["eady_uniform"]["create_model_config"] \
        is eady_uniform_model_config


def test_setup_builder_uses_factory_stack():
    """build_eady_uniform_setup must still emit the corrected stack (it now
    delegates to the factory)."""
    from legoesm.ocean.experiments.eady_uniform import build_eady_uniform_setup
    rec = build_eady_uniform_setup(n_lat=12, n_lon=12, nlev=6,
                                   c_smag=0.1, a_h=1000.0, smag_cfl_safety=0.5)
    mc = rec.model_config
    assert mc.pgf_scheme == "smc03"
    assert mc.outer_integrator == "ab2"
    assert mc.lateral_viscosity.C_smag == 0.1 and mc.lateral_viscosity.A_h == 1000.0 and mc.lateral_viscosity.smag_cfl_safety == 0.5
