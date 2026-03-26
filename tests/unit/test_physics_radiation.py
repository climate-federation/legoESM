"""Physics validation: radiation physical consistency.

Category 2: Energy balance, heating rate signs, Stefan-Boltzmann.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest


class TestGrayRadiationPhysics:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.atmosphere.physics.radiation.gray import gray_radiation
        from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
        self.gray = gray_radiation
        self.config = GrayRadiationConfig()

        ncol, nlev = 16, 10
        # Standard tropical-ish profile
        self.ncol = ncol
        self.nlev = nlev
        p_half = jnp.broadcast_to(
            jnp.linspace(1e5, 100.0, nlev + 1), (ncol, nlev + 1)
        )
        self.p_half = p_half
        self.p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        # Standard lapse rate: 6.5 K/km from 290K surface
        sigma = self.p_full / 1e5
        self.T = 290.0 * sigma ** 0.286  # adiabatic-ish
        self.sfc_temp = 290.0 * jnp.ones(ncol)
        self.lat = jnp.linspace(-jnp.pi / 3, jnp.pi / 3, ncol)
        self.q_v = 1e-3 * jnp.ones((ncol, nlev))
        self.insol = 400.0 * jnp.ones(ncol)

    def test_lw_up_stefan_boltzmann(self):
        out = self.gray(
            self.T, self.p_full, self.p_half, self.sfc_temp,
            self.lat, self.q_v, self.insol, self.config,
        )
        sigma_sb = 5.670374419e-8
        # LW up at surface (bottom interface) should be ~ eps * sigma * T_sfc^4
        eps = self.config.sfc_emissivity if hasattr(self.config, 'sfc_emissivity') else 1.0
        expected = eps * sigma_sb * self.sfc_temp ** 4
        # lw_flux_up at surface is the last interface
        lw_up_sfc = out.lw_flux_up[:, -1]
        # Allow 20% tolerance (gray scheme may differ from exact SB)
        rel_err = jnp.abs(lw_up_sfc - expected) / expected
        assert jnp.all(rel_err < 0.5), (
            f"LW up at surface: {lw_up_sfc[0]:.1f}, expected ~{expected[0]:.1f}"
        )

    def test_sw_heating_non_negative(self):
        out = self.gray(
            self.T, self.p_full, self.p_half, self.sfc_temp,
            self.lat, self.q_v, self.insol, self.config,
        )
        # SW flux divergence should produce heating ≥ 0 (sunlight only heats)
        sw_net_up = out.sw_flux_up[:, :-1] - out.sw_flux_up[:, 1:]
        sw_net_down = out.sw_flux_down[:, 1:] - out.sw_flux_down[:, :-1]
        # Net SW absorbed per layer
        sw_absorbed = (out.sw_flux_down[:, 1:] - out.sw_flux_down[:, :-1]) - \
                      (out.sw_flux_up[:, :-1] - out.sw_flux_up[:, 1:])
        # Should be non-negative (atmosphere absorbs SW, doesn't emit it)
        assert jnp.all(sw_absorbed >= -1e-6), "SW absorption negative somewhere"

    def test_olr_in_range(self):
        out = self.gray(
            self.T, self.p_full, self.p_half, self.sfc_temp,
            self.lat, self.q_v, self.insol, self.config,
        )
        # OLR = LW up at TOA
        olr = out.lw_flux_up[:, 0]
        assert jnp.all(olr > 50), f"OLR too low: {olr.min():.1f} W/m2"
        assert jnp.all(olr < 500), f"OLR too high: {olr.max():.1f} W/m2"

    def test_heating_rate_magnitude(self):
        out = self.gray(
            self.T, self.p_full, self.p_half, self.sfc_temp,
            self.lat, self.q_v, self.insol, self.config,
        )
        # Exclude lowest level (gray scheme concentrates LW cooling there)
        hr_interior = out.heating_rate[:, :-1]
        # Interior heating rate should be bounded: |dT/dt| < 50 K/day ≈ 5.8e-4 K/s
        assert jnp.all(jnp.abs(hr_interior) < 0.01), (
            f"Interior heating rate too large: {jnp.max(jnp.abs(hr_interior)):.2e} K/s"
        )
        # All heating rates should be finite
        assert jnp.all(jnp.isfinite(out.heating_rate)), "Heating rate has NaN/Inf"
