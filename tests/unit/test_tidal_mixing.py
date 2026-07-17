"""Unit tests for the Jayne & St-Laurent abyssal tidal-mixing scheme."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean.physics.vertical_mixing.tidal import (
    TidalMixingConfig,
    brunt_vaisala_cell_from_eos,
    compute_tidal_diffusivity,
    _exp_decay_structure,
    normalize_structure,
    synthetic_baroclinic_tide_energy_from_bathy,
)


# ==============================================================================
# Vertical structure function
# ==============================================================================

class TestExpDecayStructure:

    def test_decay_strongest_at_bottom(self):
        layer_depths = jnp.linspace(0.0, 4000.0, 41)
        H_bathy = jnp.array(4000.0)
        F_raw = _exp_decay_structure(
            layer_depths, H_bathy, h_decay_m=500.0,
        )
        # Bottom (closest to H) has F = 1; top decays to exp(-H/h).
        assert float(F_raw[-1]) == pytest.approx(1.0, abs=1e-6)
        assert float(F_raw[0]) < float(F_raw[-1])
        assert float(F_raw[0]) == pytest.approx(np.exp(-4000.0 / 500.0))

    def test_normalisation_unit_column_integral(self):
        nlev = 40
        layer_depths = jnp.linspace(50.0, 3950.0, nlev)
        h_partial = jnp.full((nlev,), 100.0)
        F_raw = _exp_decay_structure(
            layer_depths, jnp.array(4000.0), h_decay_m=500.0,
        )
        F = normalize_structure(F_raw, h_partial)
        # Column integral should be 1.
        col_int = float(jnp.sum(F * h_partial))
        assert col_int == pytest.approx(1.0, rel=1e-6)


# ==============================================================================
# Main K_tidal calculation
# ==============================================================================

class TestComputeTidalDiffusivity:

    def _make_column(self, nlev=40, H=4000.0, N2=1.0e-5):
        layer_depths = jnp.linspace(
            0.5 * H / nlev, H - 0.5 * H / nlev, nlev,
        )
        h_partial = jnp.full((nlev,), H / nlev)
        H_bathy = jnp.array(H)
        N_squared = jnp.full((nlev,), N2)
        return layer_depths, h_partial, H_bathy, N_squared

    def test_disabled_config_returns_finite(self):
        """Even with ``enabled=False`` the math is well-defined."""
        d, h, H, N2 = self._make_column()
        cfg = TidalMixingConfig(enabled=False)
        K = compute_tidal_diffusivity(
            jnp.array(1.0e-3), d, h, H, N2, config=cfg,
        )
        assert jnp.all(jnp.isfinite(K))

    def test_bottom_intensified(self):
        """K_tidal is largest at the bottom and small at the surface."""
        d, h, H, N2 = self._make_column()
        cfg = TidalMixingConfig(enabled=True, h_decay_m=500.0)
        K = compute_tidal_diffusivity(
            jnp.array(1.0e-3),
            d, h, H, N2,
            config=cfg,
        )
        assert float(K[-1]) > float(K[0])
        # Surface should be tiny (4000 m above bottom, exp(-8) ≈ 3e-4).
        assert float(K[0]) < 0.01 * float(K[-1])

    def test_proportional_to_E_BT(self):
        d, h, H, N2 = self._make_column()
        cfg = TidalMixingConfig(enabled=True)
        K1 = compute_tidal_diffusivity(
            jnp.array(1.0e-3), d, h, H, N2, config=cfg,
        )
        K2 = compute_tidal_diffusivity(
            jnp.array(2.0e-3), d, h, H, N2, config=cfg,
        )
        # K is linear in E_BT (before clipping).
        assert jnp.allclose(K2, 2.0 * K1, rtol=1e-6)

    def test_inverse_proportional_to_N_squared(self):
        d, h, H, _ = self._make_column()
        cfg = TidalMixingConfig(enabled=True, K_max=10.0)
        N2_weak = jnp.full((d.shape[0],), 1.0e-6)
        N2_strong = jnp.full((d.shape[0],), 1.0e-4)
        K_weak = compute_tidal_diffusivity(
            jnp.array(1.0e-3), d, h, jnp.array(4000.0), N2_weak, config=cfg,
        )
        K_strong = compute_tidal_diffusivity(
            jnp.array(1.0e-3), d, h, jnp.array(4000.0), N2_strong, config=cfg,
        )
        # K_weak / K_strong = N2_strong / N2_weak = 100 (before clip).
        ratio = float(K_weak[-1]) / max(float(K_strong[-1]), 1e-30)
        assert ratio > 50.0  # well above the 100× expectation modulo K_max

    def test_K_max_cap(self):
        d, h, H, _ = self._make_column()
        cfg = TidalMixingConfig(enabled=True, K_max=1.0e-4)
        # Tiny N² → would-be K huge; capped at K_max.
        N2 = jnp.full((d.shape[0],), 1.0e-9)
        K = compute_tidal_diffusivity(
            jnp.array(1.0e-2), d, h, H, N2, config=cfg,
        )
        assert jnp.all(K <= 1.0e-4 + 1e-12)
        assert float(jnp.max(K)) == pytest.approx(1.0e-4)

    def test_zero_E_BT_gives_zero_K(self):
        d, h, H, N2 = self._make_column()
        cfg = TidalMixingConfig(enabled=True)
        K = compute_tidal_diffusivity(
            jnp.array(0.0), d, h, H, N2, config=cfg,
        )
        assert jnp.all(K == 0.0)

    def test_land_mask_zeroes_K(self):
        d, h, H, N2 = self._make_column()
        cfg = TidalMixingConfig(enabled=True)
        K = compute_tidal_diffusivity(
            jnp.array(1.0e-3), d, h, H, N2,
            config=cfg, land_mask=jnp.array(0.0),
        )
        assert jnp.all(K == 0.0)

    def test_unstratified_safety_N2_floor(self):
        """K_max cap protects against N² → 0 convective cells."""
        d, h, H, _ = self._make_column()
        cfg = TidalMixingConfig(enabled=True, N_squared_min=1e-7)
        N2 = jnp.zeros((d.shape[0],))  # unstratified
        K = compute_tidal_diffusivity(
            jnp.array(1.0e-3), d, h, H, N2, config=cfg,
        )
        assert jnp.all(jnp.isfinite(K))
        assert jnp.all(K <= cfg.K_max + 1e-12)

    def test_column_integral_recovers_E_BT_times_constants(self):
        """∫ ρ₀ · N² · K dz = Γ · q · E_BT  (energy budget closure).

        The column integral of dissipation rate
        ``ε = K · N²`` times ρ₀ equals ``Γ · q · E_BT`` because
        ``F`` is normalised to integrate to one.
        """
        nlev = 100
        H = 4000.0
        d, h, H_arr, N2 = self._make_column(nlev=nlev, H=H, N2=1.0e-5)
        cfg = TidalMixingConfig(
            enabled=True, K_max=10.0, N_squared_min=1e-10,
        )
        E_BT = 1.0e-3
        K = compute_tidal_diffusivity(
            jnp.array(E_BT), d, h, H_arr, N2, config=cfg,
        )
        # ε = ρ₀ · K · N²; column integral = Γ · q · E_BT.
        col_eps = float(jnp.sum(cfg.rho_0 * K * N2 * h))
        expected = cfg.Gamma * cfg.q_local * E_BT
        assert col_eps == pytest.approx(expected, rel=1e-4)


# ==============================================================================
# Synthetic E_BT
# ==============================================================================

class TestSyntheticEBT:

    def test_zero_at_shallow_shelves(self):
        H = jnp.array([0.0, 100.0, 500.0, 999.0])
        E = synthetic_baroclinic_tide_energy_from_bathy(
            H, deep_threshold_m=1000.0,
        )
        # Shallow → ramp gives < full value but proportional.
        E_np = np.asarray(E)
        assert E_np[0] == 0.0
        # Monotonic increase with depth in the shelf regime.
        assert E_np[1] < E_np[2] < E_np[3]

    def test_deep_ocean_saturates(self):
        H = jnp.array([1000.0, 3000.0, 5000.0, 6000.0])
        E = synthetic_baroclinic_tide_energy_from_bathy(
            H, deep_threshold_m=1000.0,
        )
        E_np = np.asarray(E)
        # All saturate at the same value.
        assert np.allclose(E_np, E_np[0])

    def test_proportional_to_roughness_squared(self):
        H = jnp.array(4000.0)
        E1 = float(synthetic_baroclinic_tide_energy_from_bathy(
            H, rms_roughness_m=100.0,
        ))
        E2 = float(synthetic_baroclinic_tide_energy_from_bathy(
            H, rms_roughness_m=200.0,
        ))
        assert E2 == pytest.approx(4.0 * E1, rel=1e-6)

    def test_proportional_to_u_tide_squared(self):
        H = jnp.array(4000.0)
        E1 = float(synthetic_baroclinic_tide_energy_from_bathy(
            H, u_tide_m_s=0.01,
        ))
        E2 = float(synthetic_baroclinic_tide_energy_from_bathy(
            H, u_tide_m_s=0.02,
        ))
        assert E2 == pytest.approx(4.0 * E1, rel=1e-6)

    def test_magnitude_global_mean_realistic(self):
        """Synthetic E_BT with default params at 4000 m depth should
        be O(1e-3 W/m²) — global-mean tidal conversion in Simmons 2004."""
        H = jnp.array(4000.0)
        E = float(synthetic_baroclinic_tide_energy_from_bathy(H))
        assert 1.0e-5 < E < 1.0e-1


# ==============================================================================
# N² from the model EOS (issue #1111)
# ==============================================================================

_ALPHA_T = 2.0e-4   # test-only linear-EOS coeffs (mirror the old inline block)
_BETA_S = 7.4e-4


def _linear_eos(rho0):
    """A pressure-independent linear EOS fn(T,S,p)->rho for reference checks."""
    return lambda T, S, p: rho0 * (1.0 - _ALPHA_T * T + _BETA_S * S)


class TestBruntVaisalaFromEos:
    def _column(self):
        # Stable: warmer (lighter) on top, cooler below; 5 levels, 100 m each.
        T = jnp.asarray([[20.0, 15.0, 10.0, 6.0, 4.0]])
        S = jnp.asarray([[34.0, 34.5, 34.8, 34.9, 35.0]])
        depth = jnp.asarray([[50.0, 150.0, 250.0, 350.0, 450.0]])
        thick = jnp.full_like(T, 100.0)
        return T, S, depth, thick

    def test_linear_eos_reproduces_the_old_inline_formula(self):
        # With the OLD inline pressure-independent linear EOS the helper reduces
        # (to fp64 round-off) to the old -(g/rho0)(rho_k - rho_{k+1})/dz
        # interface formula, padded — a regression pin on the algebra. The
        # interface-depth choice is irrelevant here (linear EOS ignores p).
        rho0 = float(np.asarray(constants.rho_ocean))
        g = float(constants.g)
        T, S, depth, thick = self._column()
        n2 = np.asarray(brunt_vaisala_cell_from_eos(
            T, S, thick, _linear_eos(rho0),
            rho_0=rho0, n_squared_min=1e-9))
        # Reference: analytic linear N² at interfaces, padded to centres.
        rho = rho0 * (1.0 - _ALPHA_T * np.asarray(T) + _BETA_S * np.asarray(S))
        dz = 0.5 * (np.asarray(thick)[..., :-1] + np.asarray(thick)[..., 1:])
        n2_if = -(g / rho0) * (rho[..., :-1] - rho[..., 1:]) / dz
        avg = 0.5 * (n2_if[..., :-1] + n2_if[..., 1:])
        ref = np.concatenate([n2_if[..., :1], avg, n2_if[..., -1:]], axis=-1)
        ref = np.maximum(ref, 1e-9)
        assert np.allclose(n2, ref, rtol=1e-10, atol=1e-14)

    def test_stable_column_is_positive(self):
        rho0 = float(np.asarray(constants.rho_ocean))
        T, S, depth, thick = self._column()
        n2 = np.asarray(brunt_vaisala_cell_from_eos(
            T, S, thick, _linear_eos(rho0),
            rho_0=rho0, n_squared_min=1e-12))
        assert np.all(n2 > 0.0)
        assert n2.shape == (1, 5)

    def test_model_eos_differs_from_linear(self):
        # The whole point of #1111: swapping in the nonlinear model EOS changes
        # N², proving the helper actually consumes the passed eos_fn.
        from legoesm.ocean.eos import make_eos_fn
        rho0 = float(np.asarray(constants.rho_ocean))
        T, S, depth, thick = self._column()
        n2_lin = np.asarray(brunt_vaisala_cell_from_eos(
            T, S, thick, _linear_eos(rho0),
            rho_0=rho0, n_squared_min=1e-12))
        n2_wright = np.asarray(brunt_vaisala_cell_from_eos(
            T, S, thick, make_eos_fn("wright"),
            rho_0=rho0, n_squared_min=1e-12))
        assert np.all(np.isfinite(n2_wright))
        assert not np.allclose(n2_lin, n2_wright)

    def test_unstable_column_floored(self):
        # Dense-on-top (unstable) → N²<0 clamped to the floor everywhere.
        rho0 = float(np.asarray(constants.rho_ocean))
        floor = 1e-8
        T = jnp.asarray([[2.0, 10.0, 20.0]])       # cold/dense on top
        S = jnp.asarray([[35.0, 34.5, 34.0]])
        depth = jnp.asarray([[50.0, 150.0, 250.0]])
        thick = jnp.full_like(T, 100.0)
        n2 = np.asarray(brunt_vaisala_cell_from_eos(
            T, S, thick, _linear_eos(rho0),
            rho_0=rho0, n_squared_min=floor))
        assert np.allclose(n2, floor)

    def test_single_level_returns_floor(self):
        rho0 = float(np.asarray(constants.rho_ocean))
        floor = 1e-7
        T = jnp.asarray([[10.0]])
        n2 = np.asarray(brunt_vaisala_cell_from_eos(
            T, jnp.asarray([[35.0]]), jnp.asarray([[50.0]]),
            _linear_eos(rho0),
            rho_0=rho0, n_squared_min=floor))
        assert np.allclose(n2, floor)
        assert n2.shape == (1, 1)
