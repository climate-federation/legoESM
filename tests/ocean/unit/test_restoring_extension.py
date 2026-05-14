"""Unit tests for the extended ``restoring_surface_forcing`` (issue #266).

Covers:
- Back-compat: existing T_profile="cosine" / "constant" paths bit-exact
- Piece 1: T_star_array / S_star_array override built-in formulas
- Piece 1: subtract_qsr applies the eq-8 split correctly
- Piece 1: tau_from_flux_coefficient helper
- Piece 2: implicit=True gives bounded result for any dt
- Piece 2: implicit reduces to forward-Euler at small dt/τ
- Validation: missing dt with implicit=True raises ValueError
- Validation: missing sw_down/rho_0/c_p/dz_0 with subtract_qsr raises
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.physics.surface_forcing.config import (
    RestoringConfig, tau_from_flux_coefficient,
)
from legoesm.ocean.physics.surface_forcing.restoring import (
    restoring_surface_forcing,
)


class _FakeGrid:
    """Minimal grid stub: just provides ``grid_lat`` (radians)."""
    def __init__(self, lat_deg: np.ndarray, n_lon: int):
        lat_rad = jnp.asarray(np.radians(lat_deg))
        self.grid_lat = jnp.broadcast_to(lat_rad[:, None],
                                          (lat_deg.size, n_lon))


@pytest.fixture
def small_state():
    grid = _FakeGrid(np.linspace(-60, 60, 10), n_lon=5)
    nlev = 4
    T = jnp.full((10, 5, nlev), 15.0)
    S = jnp.full((10, 5, nlev), 35.0)
    return grid, T, S, nlev


# -------------------- back-compat (existing API) ----------------------

def test_back_compat_cosine_default(small_state):
    grid, T, S, nlev = small_state
    cfg = RestoringConfig(tau_T=1e6, tau_S=1e6,
                           T_star_eq=25.0, T_star_pole=0.0,
                           T_profile="cosine")
    out = restoring_surface_forcing(T, S, grid, cfg)
    # dT only in surface layer
    assert (np.asarray(out.dT_dt.data if hasattr(out.dT_dt, "data")
                        else out.dT_dt)[..., 1:] == 0.0).all()
    # surface dT is non-zero where lat != 0 (T=15, T*≠15)
    surf = np.asarray(out.dT_dt[..., 0]) if not hasattr(out.dT_dt, "data") else \
        np.asarray(out.dT_dt.data[..., 0])
    assert np.abs(surf).max() > 0.0


def test_back_compat_constant_profile(small_state):
    grid, T, S, _ = small_state
    cfg = RestoringConfig(tau_T=1e6, T_star_eq=20.0, T_profile="constant")
    out = restoring_surface_forcing(T, S, grid, cfg)
    surf = np.asarray(out.dT_dt[..., 0])
    expected = -(15.0 - 20.0) / 1e6  # all cells: T=15, T*=20
    assert np.allclose(surf, expected)


# -------------------- Piece 1: array targets --------------------------

def test_T_star_array_overrides_cosine(small_state):
    grid, T, S, _ = small_state
    custom_T_star = jnp.full(grid.grid_lat.shape, 18.0)
    cfg = RestoringConfig(tau_T=1e6, T_star_array=custom_T_star,
                           T_profile="cosine",  # should be ignored
                           T_star_eq=999.0)     # should be ignored
    out = restoring_surface_forcing(T, S, grid, cfg)
    surf = np.asarray(out.dT_dt[..., 0])
    expected = -(15.0 - 18.0) / 1e6
    assert np.allclose(surf, expected)


def test_S_star_array_overrides_constant(small_state):
    grid, T, S, _ = small_state
    custom_S_star = jnp.full(grid.grid_lat.shape, 36.5)
    cfg = RestoringConfig(tau_S=1e6, S_star_array=custom_S_star,
                           S_star=999.0)  # should be ignored
    out = restoring_surface_forcing(T, S, grid, cfg)
    surf = np.asarray(out.dS_dt[..., 0])
    expected = -(35.0 - 36.5) / 1e6
    assert np.allclose(surf, expected)


# -------------------- Piece 1: subtract_qsr ---------------------------

def test_subtract_qsr_subtracts_solar_from_surface_T_tendency(small_state):
    grid, T, S, _ = small_state
    sw = jnp.full(grid.grid_lat.shape, 100.0)  # 100 W/m² uniform
    rho_0, c_p, dz_0 = 1026.0, 3992.0, 10.0
    cfg = RestoringConfig(tau_T=1e6, T_star_eq=15.0, T_profile="constant",
                           subtract_qsr=True)
    out = restoring_surface_forcing(T, S, grid, cfg, sw_down=sw,
                                     rho_0=rho_0, c_p=c_p, dz_0=dz_0)
    surf = np.asarray(out.dT_dt[..., 0])
    # T = T*, so restoring tendency = 0; only Q_sr contribution remains
    expected = -100.0 / (rho_0 * c_p * dz_0)
    assert np.allclose(surf, expected)


def test_subtract_qsr_requires_sw_down():
    grid = _FakeGrid(np.array([0.0]), n_lon=1)
    T = jnp.full((1, 1, 2), 15.0)
    S = jnp.full((1, 1, 2), 35.0)
    cfg = RestoringConfig(subtract_qsr=True)
    with pytest.raises(ValueError, match="sw_down"):
        restoring_surface_forcing(T, S, grid, cfg)


def test_subtract_qsr_requires_rho_cp_dz():
    grid = _FakeGrid(np.array([0.0]), n_lon=1)
    T = jnp.full((1, 1, 2), 15.0)
    S = jnp.full((1, 1, 2), 35.0)
    cfg = RestoringConfig(subtract_qsr=True)
    sw = jnp.full((1, 1), 100.0)
    with pytest.raises(ValueError, match="rho_0, c_p, dz_0"):
        restoring_surface_forcing(T, S, grid, cfg, sw_down=sw)


# -------------------- Piece 1: helper ---------------------------------

def test_tau_from_flux_coefficient_T():
    # DINO-like: A_θ=40 W/m²/K, ρ₀=1026, c_p=3992, dz_0=10 → ~11.85 days
    tau = tau_from_flux_coefficient(40.0, 1026.0, 3992.0, 10.0)
    assert tau == pytest.approx(11.85 * 86400.0, rel=0.01)


def test_tau_from_flux_coefficient_S():
    # DINO-like: A_S=3.858e-3 kg/m²/s, ρ₀=1026, dz_0=10 → ~30.8 days
    tau = tau_from_flux_coefficient(3.858e-3, 1026.0, 1.0, 10.0)
    assert tau == pytest.approx(30.8 * 86400.0, rel=0.05)


# -------------------- Piece 2: implicit Euler -------------------------

def test_implicit_requires_dt():
    grid = _FakeGrid(np.array([0.0]), n_lon=1)
    T = jnp.full((1, 1, 2), 15.0)
    S = jnp.full((1, 1, 2), 35.0)
    cfg = RestoringConfig(implicit=True)
    with pytest.raises(ValueError, match="implicit=True requires `dt`"):
        restoring_surface_forcing(T, S, grid, cfg)


def test_implicit_reduces_to_explicit_at_small_dt(small_state):
    """For dt << τ, implicit and explicit should give the same tendency
    to leading order."""
    grid, T, S, _ = small_state
    cfg_e = RestoringConfig(tau_T=1e6, T_star_eq=20.0, T_profile="constant",
                             implicit=False)
    cfg_i = RestoringConfig(tau_T=1e6, T_star_eq=20.0, T_profile="constant",
                             implicit=True)
    out_e = restoring_surface_forcing(T, S, grid, cfg_e)
    out_i = restoring_surface_forcing(T, S, grid, cfg_i, dt=1.0)  # dt/τ = 1e-6
    surf_e = np.asarray(out_e.dT_dt[..., 0])
    surf_i = np.asarray(out_i.dT_dt[..., 0])
    assert np.allclose(surf_e, surf_i, rtol=1e-3)


def test_implicit_at_dt_equals_tau_gives_half_relaxation(small_state):
    """At dt = τ, implicit Euler relaxes by exactly 50%
    (analytical: T_new = (T_old + T*)/2)."""
    grid, T, S, _ = small_state
    tau = 1e6
    cfg = RestoringConfig(tau_T=tau, T_star_eq=20.0, T_profile="constant",
                           implicit=True)
    out = restoring_surface_forcing(T, S, grid, cfg, dt=tau)
    surf_dT = np.asarray(out.dT_dt[..., 0])
    # dT_dt_eff = (T*-T)/(τ+dt) = (20-15)/(2τ); applied as T_new = T+dt·dT
    # → T_new = 15 + tau · 5/(2tau) = 15 + 2.5 = 17.5 = (15+20)/2 ✓
    expected_dT_dt = (20.0 - 15.0) / (2 * tau)
    assert np.allclose(surf_dT, expected_dT_dt)


def test_implicit_bounded_for_huge_dt(small_state):
    """Implicit Euler must stay between T_old and T* even at very large
    dt (no overshoot, no oscillation). This is THE motivation."""
    grid, T, S, _ = small_state
    tau = 1e3  # 1000 s — very fast restoring
    dt = 1e6   # 1e6 s — dt/τ = 1000 (forward Euler would oscillate)
    cfg = RestoringConfig(tau_T=tau, T_star_eq=20.0, T_profile="constant",
                           implicit=True)
    out = restoring_surface_forcing(T, S, grid, cfg, dt=dt)
    surf_dT = np.asarray(out.dT_dt[..., 0])
    # T_new = T + dt·dT_dt = 15 + dt·(20-15)/(τ+dt) = 15 + 5·dt/(τ+dt)
    # For dt >> τ: → 15 + 5 = 20 = T* (full relaxation). Bounded.
    T_new = 15.0 + dt * surf_dT
    assert np.all(T_new >= 15.0 - 1e-9)
    assert np.all(T_new <= 20.0 + 1e-9)
    # Should approach T* = 20 since dt >> τ
    assert np.allclose(T_new, 20.0, atol=0.01)


def test_implicit_explicit_diverge_at_large_dt(small_state):
    """At dt = 5τ, forward Euler is unstable (oscillates with growing
    amplitude); implicit stays bounded. Verify they DISAGREE significantly."""
    grid, T, S, _ = small_state
    tau = 1e5
    dt = 5e5  # dt/τ = 5
    cfg_e = RestoringConfig(tau_T=tau, T_star_eq=20.0, T_profile="constant",
                             implicit=False)
    cfg_i = RestoringConfig(tau_T=tau, T_star_eq=20.0, T_profile="constant",
                             implicit=True)
    out_e = restoring_surface_forcing(T, S, grid, cfg_e)
    out_i = restoring_surface_forcing(T, S, grid, cfg_i, dt=dt)
    # Forward Euler: dT/dt = (T*-T)/τ = 5/1e5 = 5e-5
    # T_new_e = 15 + 5e5 · 5e-5 = 15 + 25 = 40 (overshoots T*=20)
    T_new_e = 15.0 + dt * float(np.asarray(out_e.dT_dt[..., 0]).max())
    # Implicit: dT/dt_eff = (T*-T)/(τ+dt) = 5/6e5; T_new = 15 + 5e5·(5/6e5) = 19.17
    T_new_i = 15.0 + dt * float(np.asarray(out_i.dT_dt[..., 0]).max())
    assert T_new_e > 25.0  # overshoot
    assert 15.0 < T_new_i < 20.0  # bounded
