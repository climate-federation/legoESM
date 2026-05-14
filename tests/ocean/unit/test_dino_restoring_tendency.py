"""Unit tests for DINO top-layer surface tendency functions
(Phase 2C-extra; paper eqs 7-9).

Paper:
  eq 7: F_u  = τ_u / (ρ₀ · dz_0)              [wind]
  eq 8: F_T  = (A_Θ·(T* − T) − Q_sr) / (ρ₀ c_p dz_0)  [non-solar T]
  eq 9: F_S  = A_S·(S* − S) / (ρ₀ · dz_0)              [salinity]

These functions live in the DINO module (not in legoesm.ocean.physics.
surface_forcing) because the general restoring API uses *timescales* (s)
not heat-flux *coefficients* (W/m²/K), and does not subtract Q_sr.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.ocean.experiments.dino import (
    DINOConfig,
    dino_top_layer_heat_flux_split,
    dino_top_layer_S_tendency,
    dino_top_layer_T_tendency,
    dino_top_layer_u_tendency,
    restoring_timescale_S_days,
    restoring_timescale_T_days,
)


# -------------------- heat-flux split --------------------------------

def test_heat_flux_split_at_radiative_equilibrium():
    """When A_Θ·(T*−T) = Q_sr the non-solar component is zero."""
    cfg = DINOConfig()
    T = jnp.array(20.0)
    T_star = jnp.array(22.0)
    Q_sr = cfg.A_theta * (T_star - T)  # exactly cancels
    Q_ns, _ = dino_top_layer_heat_flux_split(T, T_star, Q_sr, cfg)
    assert float(Q_ns) == pytest.approx(0.0, abs=1e-9)


def test_heat_flux_split_at_T_equals_T_star():
    """When T = T*, the only heat flux is −Q_sr (cooling balances solar)."""
    cfg = DINOConfig()
    T = jnp.array(20.0)
    T_star = jnp.array(20.0)
    Q_sr = jnp.array(150.0)
    Q_ns, _ = dino_top_layer_heat_flux_split(T, T_star, Q_sr, cfg)
    assert float(Q_ns) == pytest.approx(-150.0, abs=1e-9)


def test_heat_flux_split_passes_through_Q_sr_unchanged():
    cfg = DINOConfig()
    Q_sr_in = jnp.array(150.0)
    _, Q_sr_out = dino_top_layer_heat_flux_split(20.0, 22.0, Q_sr_in, cfg)
    assert float(Q_sr_out) == pytest.approx(150.0, abs=1e-9)


# -------------------- T tendency -------------------------------------

def test_T_tendency_negative_when_warmer_than_star_no_solar():
    """T > T* with no solar → cooling."""
    cfg = DINOConfig()
    dT_dt = dino_top_layer_T_tendency(
        T_sfc_C=jnp.array(25.0), T_star=jnp.array(20.0),
        Q_sr=jnp.array(0.0), dz_0=cfg.dz_min, cfg=cfg,
    )
    assert float(dT_dt) < 0.0


def test_T_tendency_positive_when_colder_than_star_no_solar():
    cfg = DINOConfig()
    dT_dt = dino_top_layer_T_tendency(
        T_sfc_C=jnp.array(10.0), T_star=jnp.array(20.0),
        Q_sr=jnp.array(0.0), dz_0=cfg.dz_min, cfg=cfg,
    )
    assert float(dT_dt) > 0.0


def test_T_tendency_matches_restoring_timescale_helper():
    """When Q_sr=0, |dT/dt| / |T-T*| should equal 1/tau_T."""
    cfg = DINOConfig()
    T_star = jnp.array(20.0)
    T = jnp.array(25.0)
    dT_dt = dino_top_layer_T_tendency(T, T_star, jnp.array(0.0),
                                       dz_0=cfg.dz_min, cfg=cfg)
    inv_tau_seconds = float(-dT_dt) / float(T - T_star)
    tau_days = (1.0 / inv_tau_seconds) / 86400.0
    expected_tau_days = restoring_timescale_T_days(cfg)
    assert tau_days == pytest.approx(expected_tau_days, rel=1e-9)


def test_T_tendency_solar_warms_when_at_target():
    """If T = T* and Q_sr > 0, dT/dt < 0 (paper eq 8 subtracts Q_sr from
    the non-solar component; the solar contribution is in the column
    via Jerlov, not here). Sign is intentional."""
    cfg = DINOConfig()
    dT_dt = dino_top_layer_T_tendency(
        T_sfc_C=jnp.array(20.0), T_star=jnp.array(20.0),
        Q_sr=jnp.array(150.0), dz_0=cfg.dz_min, cfg=cfg,
    )
    assert float(dT_dt) < 0.0


def test_T_tendency_array_inputs_broadcast():
    cfg = DINOConfig()
    T = jnp.array([20.0, 25.0, 10.0])
    T_star = jnp.array([22.0, 22.0, 22.0])
    Q_sr = jnp.array([100.0, 100.0, 100.0])
    dT_dt = dino_top_layer_T_tendency(T, T_star, Q_sr,
                                       dz_0=cfg.dz_min, cfg=cfg)
    assert dT_dt.shape == (3,)


# -------------------- S tendency -------------------------------------

def test_S_tendency_negative_when_saltier_than_star():
    cfg = DINOConfig()
    dS_dt = dino_top_layer_S_tendency(
        S_surface=jnp.array(36.0), S_star=jnp.array(35.0),
        dz_0=cfg.dz_min, cfg=cfg,
    )
    assert float(dS_dt) < 0.0


def test_S_tendency_positive_when_fresher_than_star():
    cfg = DINOConfig()
    dS_dt = dino_top_layer_S_tendency(
        S_surface=jnp.array(34.0), S_star=jnp.array(35.0),
        dz_0=cfg.dz_min, cfg=cfg,
    )
    assert float(dS_dt) > 0.0


def test_S_tendency_zero_at_target():
    cfg = DINOConfig()
    dS_dt = dino_top_layer_S_tendency(
        S_surface=jnp.array(35.0), S_star=jnp.array(35.0),
        dz_0=cfg.dz_min, cfg=cfg,
    )
    assert float(dS_dt) == pytest.approx(0.0, abs=1e-12)


def test_S_tendency_matches_restoring_timescale_helper():
    cfg = DINOConfig()
    S_star = jnp.array(35.0)
    S = jnp.array(36.0)
    dS_dt = dino_top_layer_S_tendency(S, S_star, dz_0=cfg.dz_min, cfg=cfg)
    inv_tau_seconds = float(-dS_dt) / float(S - S_star)
    tau_days = (1.0 / inv_tau_seconds) / 86400.0
    expected_tau_days = restoring_timescale_S_days(cfg)
    assert tau_days == pytest.approx(expected_tau_days, rel=1e-9)


# -------------------- u tendency (wind) ------------------------------

def test_u_tendency_positive_for_eastward_wind():
    cfg = DINOConfig()
    du_dt = dino_top_layer_u_tendency(
        tau_u=jnp.array(0.2), dz_0=cfg.dz_min, cfg=cfg,
    )
    assert float(du_dt) > 0.0


def test_u_tendency_negative_for_westward_wind():
    cfg = DINOConfig()
    du_dt = dino_top_layer_u_tendency(
        tau_u=jnp.array(-0.1), dz_0=cfg.dz_min, cfg=cfg,
    )
    assert float(du_dt) < 0.0


def test_u_tendency_zero_for_no_wind():
    cfg = DINOConfig()
    du_dt = dino_top_layer_u_tendency(
        tau_u=jnp.array(0.0), dz_0=cfg.dz_min, cfg=cfg,
    )
    assert float(du_dt) == 0.0


def test_u_tendency_magnitude():
    """Sanity-check magnitude: τ=0.2 N/m², ρ₀=1026, dz=10m → ~2e-5 m/s²."""
    cfg = DINOConfig()
    du_dt = dino_top_layer_u_tendency(
        tau_u=jnp.array(0.2), dz_0=cfg.dz_min, cfg=cfg,
    )
    expected = 0.2 / (cfg.rho_0 * cfg.dz_min)
    assert float(du_dt) == pytest.approx(expected, rel=1e-9)


# -------------------- jit compatibility ------------------------------

def test_T_tendency_jit_compatible():
    import jax
    cfg = DINOConfig()
    fn = jax.jit(lambda T, Tstar, Qsr: dino_top_layer_T_tendency(
        T, Tstar, Qsr, dz_0=cfg.dz_min, cfg=cfg))
    out = fn(jnp.array([20.0, 25.0]), jnp.array([22.0, 22.0]),
             jnp.array([100.0, 100.0]))
    assert out.shape == (2,)


def test_S_tendency_jit_compatible():
    import jax
    cfg = DINOConfig()
    fn = jax.jit(lambda S, Sstar: dino_top_layer_S_tendency(
        S, Sstar, dz_0=cfg.dz_min, cfg=cfg))
    out = fn(jnp.array([35.0, 36.0]), jnp.array([35.0, 35.0]))
    assert out.shape == (2,)
