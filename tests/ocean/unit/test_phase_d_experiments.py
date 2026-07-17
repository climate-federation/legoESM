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
    from legoesm.ocean.physics.ice_shelf import freezing_point_C
    t0 = float(freezing_point_C(34.0, 0.0))
    t_press = float(freezing_point_C(34.0, 1000.0))   # 1000 dbar deeper
    t_salt = float(freezing_point_C(35.0, 0.0))
    assert t_press < t0    # pressure lowers freezing point
    assert t_salt < t0     # salt lowers freezing point


def test_isomip_ocean0_cold_basal_melt_plausible():
    """Ocean0 (T_w = -1.9 degC, S = 34.55, p = 50 dbar): the FAITHFUL
    three-equation closure gives a small positive cold-cavity melt
    (~0.26 m/yr ice-equivalent at gamma_T = 1e-4).

    REROUTE NOTE (2026-07-17): the old pin asserted the deleted linearised
    closure sat in the "ISOMIP+ 0-1 m/yr" band — a claim MANUFACTURED by its
    hand-tuned gamma_T (2e-5, ~30x below the ISOMIP+ gamma_T*), which also
    absorbed a rho_sw/rho_fw convention shortcut.  This pin brackets the
    faithful scheme's own value instead."""
    from legoesm.ocean.physics.ice_shelf import three_equation_melt

    r = three_equation_melt(-1.9, 34.55, 50.0)
    mdot = float(r.m_dot_m_s) * 86400.0 * 365.0
    assert 0.05 < mdot < 2.0


def test_isomip_ocean1_warm_exceeds_cold_and_matches_quadratic_oracle():
    """Ocean1 (T_w = +1.0 degC): the faithful closure at the smoke probe
    gives ~60 m/yr — ABOVE the 1-10 m/yr circulating-cavity ensemble band,
    and that is EXPECTED here: this probe feeds the FAR-FIELD T as the
    boundary-layer ambient, so the meltwater-throttling feedback the
    ensemble includes is absent (the old in-band claim came from the
    hand-tuned gamma_T compensating for it).  Pin the value against an
    independent closed-form solve of the SAME quadratic (non-circular
    algebra check) + the physical orderings."""
    from legoesm.ocean.physics.ice_shelf import (
        IceShelfConfig,
        freezing_point_C,
        three_equation_melt,
    )

    cfg = IceShelfConfig()
    r_warm = three_equation_melt(1.0, 34.7, 50.0, config=cfg)
    r_cold = three_equation_melt(-1.9, 34.55, 50.0, config=cfg)
    mdot_warm = float(r_warm.m_dot_m_s) * 86400.0 * 365.0
    # Independent quadratic solve (numpy algebra, coefficients per
    # Holland-Jenkins 1999 as in the module docstring; module-level np).
    alpha = cfg.rho_w * cfg.c_w * cfg.gamma_T
    beta = cfg.rho_w * cfg.gamma_S
    gamma = cfg.rho_ice * cfg.L_f
    delta = cfg.rho_ice
    T_f = float(freezing_point_C(34.7, 50.0, config=cfg))
    theta = 1.0 - T_f
    T_minus_bcp = 1.0 - cfg.freeze_b - cfg.freeze_c * 50.0
    A = gamma * delta
    B = gamma * beta - alpha * delta * T_minus_bcp
    C = -alpha * beta * theta
    m_oracle = (-B + np.sqrt(B * B - 4.0 * A * C)) / (2.0 * A)
    assert float(r_warm.m_dot_m_s) == pytest.approx(m_oracle, rel=1e-12)
    # Physical orderings: warm melts (much) faster than cold; the interface
    # sits between the ambient freezing point and the ambient temperature.
    assert mdot_warm > 10.0 * float(r_cold.m_dot_m_s) * 86400.0 * 365.0
    assert T_f < float(r_warm.T_b_C) < 1.0


def test_isomip_basal_freshwater_flux_is_ice_equivalent():
    """freshwater_to_ocean = rho_ice * m_dot (ICE-equivalent convention of
    the faithful closure) — the deleted module multiplied by rho_fw=1000
    instead, the convention shortcut its tuned gamma_T absorbed."""
    from legoesm.ocean.physics.ice_shelf import (
        IceShelfConfig,
        three_equation_melt,
    )

    cfg = IceShelfConfig()
    r = three_equation_melt(1.0, 34.7, 50.0, config=cfg)
    assert float(r.freshwater_to_ocean) == pytest.approx(
        cfg.rho_ice * float(r.m_dot_m_s), rel=1e-12)
    assert float(r.freshwater_to_ocean) > 0.0


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
