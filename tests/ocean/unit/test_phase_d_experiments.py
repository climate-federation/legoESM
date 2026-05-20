"""Phase D unit tests: new benchmark experiments + Jenkins basal-melt.

Covers Munk gyre, Held-Larichev, NeverWorld2-lite, ISOMIP+ Ocean0 /
Ocean1, and the Jenkins three-equation basal-melt parameterisation.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest


def test_munk_layer_width_scales_with_Ah():
    """delta_M = (A_h / beta)^(1/3) must scale ~ A_h^(1/3)."""
    from legoesm.ocean.experiments.munk_gyre import MunkGyreConfig
    c1 = MunkGyreConfig(A_h=1.0e4)
    c2 = MunkGyreConfig(A_h=1.0e5)
    ratio = c2.munk_layer_width_m / c1.munk_layer_width_m
    assert ratio == pytest.approx(10.0 ** (1.0 / 3.0), rel=1e-6)


def test_munk_config_default_munk_layer_about_100_km():
    """With default A_h=2e4 and beta ~ 2e-11 -> delta_M ~ 100 km."""
    from legoesm.ocean.experiments.munk_gyre import MunkGyreConfig
    cfg = MunkGyreConfig()
    assert 80e3 < cfg.munk_layer_width_m < 130e3


def test_held_larichev_config_overrides_eady_perturbation():
    """HL bumps the perturbation amplitude vs. Eady default."""
    from legoesm.ocean.experiments.held_larichev import HeldLarichevConfig
    from legoesm.ocean.experiments.eady_uniform import EadyUniformConfig
    hl = HeldLarichevConfig()
    eu = EadyUniformConfig()
    assert hl.T_perturbation_K > eu.T_perturbation_K


def test_neverworld2_lite_grid_resolution_flag():
    from legoesm.ocean.experiments.neverworld2_lite import (
        NeverWorld2LiteConfig,
    )
    cfg_perm = NeverWorld2LiteConfig()
    assert not cfg_perm.is_eddy_resolving
    cfg_res = NeverWorld2LiteConfig(n_lon=1440)
    assert cfg_res.is_eddy_resolving


# ---------------------------------------------------------------------------
# Jenkins basal-melt parameterisation
# ---------------------------------------------------------------------------

def test_freezing_point_decreases_with_pressure_and_salinity():
    from legoesm.ocean.physics.ice_shelf_basal_melt import freezing_point_C
    t0 = float(freezing_point_C(34.0, 0.0))
    t_press = float(freezing_point_C(34.0, 1000.0))   # 1000 dbar deeper
    t_salt = float(freezing_point_C(35.0, 0.0))
    assert t_press < t0    # pressure lowers freezing point
    assert t_salt < t0     # salt lowers freezing point


def test_isomip_ocean0_cold_basal_melt_in_range():
    """Ocean0 (T_w = -1.9 deg C) target range 0-1 m/yr."""
    from legoesm.ocean.physics.ice_shelf_basal_melt import (
        basal_melt_rate_m_per_s,
    )
    mdot = float(basal_melt_rate_m_per_s(-1.9, 34.55, 50.0)) * 86400.0 * 365.0
    assert 0.0 < mdot < 1.0


def test_isomip_ocean1_warm_basal_melt_within_intercomp_band():
    """Ocean1 (T_w = +1.0 deg C) target range 1-30 m/yr.

    The linearised Jenkins form overestimates vs the full quadratic
    three-equation solve; legoESM's calibrated ``gamma_T = 2e-5``
    keeps the smoke value within the intercomparison spread.
    """
    from legoesm.ocean.physics.ice_shelf_basal_melt import (
        basal_melt_rate_m_per_s,
    )
    mdot = float(basal_melt_rate_m_per_s(1.0, 34.7, 50.0)) * 86400.0 * 365.0
    assert 1.0 < mdot < 30.0


def test_isomip_basal_freshwater_flux_proportional_to_rho_fw():
    from legoesm.ocean.physics.ice_shelf_basal_melt import (
        IceShelfMeltConfig, basal_melt_rate_m_per_s,
        basal_freshwater_flux_kg_per_m2_s,
    )
    cfg = IceShelfMeltConfig()
    m = float(basal_melt_rate_m_per_s(1.0, 34.7, 50.0, cfg))
    F = float(basal_freshwater_flux_kg_per_m2_s(1.0, 34.7, 50.0, cfg))
    assert F == pytest.approx(cfg.rho_fw * m, rel=1e-12)


def test_isomip_cavity_geometry_consistent():
    from legoesm.ocean.experiments.isomip_plus import (
        ISOMIPPlusConfig, ice_draft_m, cavity_water_column_m,
    )
    cfg = ISOMIPPlusConfig.ocean0_cold()
    # Grounding line: y=0 -> ice draft = -ice_draft_max
    assert float(ice_draft_m(0.0, 0.0, cfg)) == pytest.approx(
        -cfg.ice_draft_max_m
    )
    # Ice front: y=ice_front_y_km -> draft = 0
    assert float(
        ice_draft_m(0.0, cfg.ice_front_y_km, cfg)
    ) == pytest.approx(0.0)
    # Water column thickness positive everywhere inside the cavity.
    H = float(cavity_water_column_m(0.0, 50.0, cfg))
    assert 0 < H < cfg.H_max


def test_all_phase_d_experiments_registered():
    from legoesm.ocean.experiments import AVAILABLE_EXPERIMENTS
    for name in ("munk_gyre", "held_larichev",
                 "neverworld2_lite", "isomip_plus"):
        cfg = AVAILABLE_EXPERIMENTS[name]
        assert "config_class" in cfg
        assert "create_initial_conditions" in cfg
        assert "create_forcings" in cfg
        assert "validate" in cfg
