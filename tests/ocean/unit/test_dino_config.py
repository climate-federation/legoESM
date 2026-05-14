"""Unit tests for DINOConfig (Phase 2A).

Verifies the configuration dataclass loads with paper-matching defaults
and that derived helpers (restoring timescales) give the values quoted
in ``docs/ocean_experiments/dino_replication_plan.md``.
"""

from __future__ import annotations

import pytest

from legoesm.ocean.experiments.dino import (
    DINOConfig,
    restoring_timescale_S_days,
    restoring_timescale_T_days,
)


def test_dino_config_defaults_match_paper_table_1():
    cfg = DINOConfig()

    # Reference physical constants (paper Table 1)
    assert cfg.rho_0 == pytest.approx(1026.0)
    assert cfg.c_p == pytest.approx(3991.86)
    assert cfg.A_theta == pytest.approx(40.0)
    assert cfg.A_S == pytest.approx(3.858e-3)


def test_dino_config_domain():
    cfg = DINOConfig()

    # Domain dimensions (Sect 2.2 of paper, Table 2 R1)
    assert cfg.lon_east_deg - cfg.lon_west_deg == pytest.approx(50.0)
    assert cfg.lat_max_deg == pytest.approx(70.0)
    assert cfg.channel_lat_north_deg - cfg.channel_lat_south_deg == pytest.approx(20.0)
    assert cfg.channel_width_deg == pytest.approx(20.0)


def test_dino_config_bathymetry_zenodo_locked():
    cfg = DINOConfig()

    # Locked from vopikamm/DINO@v0.2.0 EXPREF/namelist_cfg
    assert cfg.H_deep == pytest.approx(4000.0)        # rn_H
    assert cfg.H_shallow == pytest.approx(2000.0)     # rn_hborder
    assert cfg.s_lambda_inv_deg == pytest.approx(1.0 / 3.0)  # 1/rn_distLam
    assert cfg.channel_wall_slope == pytest.approx(1.5)      # rn_slp_cha
    assert cfg.sill_gaussian_width_s == pytest.approx(4.0)   # rn_ds_width
    assert cfg.H_sill == pytest.approx(2500.0)               # rn_ds_depth


def test_dino_config_vertical_grid():
    cfg = DINOConfig()

    # Appendix C, eq C3
    assert cfg.n_levels == 36
    assert cfg.dz_min == pytest.approx(10.0)
    assert cfg.k_th == 35
    assert cfg.a_cr == pytest.approx(10.5)


def test_dino_config_wind_knots_match_paper_eq_7():
    cfg = DINOConfig()

    # Sect 2.3, eq 7 — fixed knots interpolated by PCHIP
    assert cfg.wind_tau_lats_deg == (-70.0, -45.0, -15.0, 0.0, 15.0, 45.0, 70.0)
    assert cfg.wind_tau_values == (0.0, 0.2, -0.1, -0.02, -0.1, 0.1, 0.0)
    assert len(cfg.wind_tau_lats_deg) == len(cfg.wind_tau_values)


def test_dino_config_restoring_targets():
    cfg = DINOConfig()

    # Paper Appendix B
    assert cfg.T_star_eq == pytest.approx(27.0)
    assert cfg.T_star_n_mean == pytest.approx(5.0)
    assert cfg.T_star_s_mean == pytest.approx(-0.5)
    assert cfg.S_star_eq == pytest.approx(37.25)
    assert cfg.S_star_n == pytest.approx(35.0)
    assert cfg.S_star_s == pytest.approx(35.1)
    assert cfg.S_star_eq_dip_amp == pytest.approx(1.25)
    assert cfg.S_star_eq_dip_sigma_deg == pytest.approx(7.5)
    assert cfg.L_phi_deg == pytest.approx(140.0)


def test_dino_config_vertical_mixing_namelist_locked():
    cfg = DINOConfig()

    # Zenodo namelist rn_avm0, rn_avt0, rn_evd
    assert cfg.A_v_bg == pytest.approx(1.2e-4)
    assert cfg.K_v_bg == pytest.approx(1.2e-5)
    assert cfg.K_conv == pytest.approx(100.0)


def test_dino_config_gm_redi_visbeck_choice():
    cfg = DINOConfig()

    # Decisions log: Visbeck not Tréguier; α=0.015
    assert cfg.use_gm_redi is True
    assert cfg.visbeck_alpha == pytest.approx(0.015)
    assert cfg.visbeck_kappa_min == pytest.approx(200.0)
    assert cfg.visbeck_kappa_max == pytest.approx(2000.0)
    assert cfg.gm_redi_slope_scheme == "triads"


def test_dino_config_lateral_mixing():
    cfg = DINOConfig()

    # Table 2 R1: U_M=0.27, U_T=0.027 m/s
    assert cfg.U_M == pytest.approx(0.27)
    assert cfg.U_T == pytest.approx(0.027)


def test_dino_config_bottom_drag_and_dt():
    cfg = DINOConfig()

    assert cfg.C_d_bottom == pytest.approx(1.0e-3)
    assert cfg.dt == pytest.approx(2700.0)  # 45 min, Table 2 R1


def test_dino_config_locked_scheme_choices():
    cfg = DINOConfig()

    # Decisions log 2026-05-14
    assert cfg.pgf_scheme == "adcroft"
    assert cfg.barotropic_solver == "implicit_cn"
    assert cfg.tracer_advection == "tvd"
    assert cfg.hi_precision_pressure is True
    assert cfg.barotropic_implicit_theta_eta == pytest.approx(0.55)


def test_dino_config_diagnostic_reference_density():
    cfg = DINOConfig()

    # Paper Sect 2.1 + Figs 5-6 (σ_2 referenced to 2000 m)
    assert cfg.sigma_2_ref_depth == pytest.approx(2000.0)
    assert cfg.rho_ref_z0 == pytest.approx(1026.0)
    assert cfg.rho_ref_z2000 == pytest.approx(1035.0)


def test_restoring_timescale_T_matches_plan():
    """Plan and paper note: τ_T ≈ 11.85 d for default 10 m surface layer."""
    cfg = DINOConfig()
    tau_T = restoring_timescale_T_days(cfg)
    assert tau_T == pytest.approx(11.85, abs=0.05)


def test_restoring_timescale_S_matches_plan():
    """Plan and paper note: τ_S ≈ 30.8 d for default 10 m surface layer."""
    cfg = DINOConfig()
    tau_S = restoring_timescale_S_days(cfg)
    assert tau_S == pytest.approx(30.8, abs=0.1)


def test_dino_config_is_overrideable():
    """Sanity: dataclass fields can be overridden at construction."""
    cfg = DINOConfig(dt=900.0, n_levels=20)
    assert cfg.dt == pytest.approx(900.0)
    assert cfg.n_levels == 20
    # Untouched fields keep defaults
    assert cfg.H_deep == pytest.approx(4000.0)
