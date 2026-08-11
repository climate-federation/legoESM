"""Surface forcing utilities: LW-inversion emissivity selection.

The coupled drivers reconstruct gross ``lw_down`` from the held net surface
longwave via ``lw_down = (lw_net + eps*sigma*T^4) / eps``.  For an exact round
trip, ``eps`` MUST equal the emissivity the radiation scheme EMITTED with;
otherwise a persistent O(1 W/m^2) surface-energy bias leaks in.
``surface_emissivity_for_lw_inversion`` returns that matching emissivity.
"""
from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.forcing.surface_utils import (
    snow_for_albedo_deblend,
    surface_emissivity_for_lw_inversion,
    surface_temperature_for_lw_boundary,
)


class TestSnowForAlbedoDeblend:
    """Which snow field the coupled drivers may hand the albedo blend (#1556).

    The driver stashes the RAW segment carry. On an ensemble run the state and
    the held fluxes are unpacked as the ensemble MEAN while that carry stays
    member-shaped, so passing it would make the albedo member-shaped and
    broadcast a 2-D ``sw_net_sfc`` up to 3-D.
    """

    def test_single_member_passes_the_snow_through(self):
        snow = jnp.ones((4, 8)) * 12.0
        assert snow_for_albedo_deblend(snow, 1) is snow

    def test_ensemble_withholds_the_member_shaped_carry(self):
        # (n_members, ny, nx) — the shape that would mis-broadcast.
        snow = jnp.ones((3, 4, 8)) * 12.0
        assert snow_for_albedo_deblend(snow, 3) is None

    def test_absent_snow_stays_absent(self):
        assert snow_for_albedo_deblend(None, 1) is None
        assert snow_for_albedo_deblend(None, 4) is None

    def test_degenerate_sizes_count_as_serial(self):
        """``ModelDriver._create_ensemble`` returns early for ``<= 1``, so a
        config carrying 0 leaves the carry single-member shaped. Gating on
        ``== 1`` would suppress the snow term on a run that is serial."""
        snow = jnp.ones((4, 8)) * 12.0
        assert snow_for_albedo_deblend(snow, 0) is snow

    def test_the_withheld_shape_really_would_mis_broadcast(self):
        """Non-vacuous: show the failure the gate prevents, so the gate is not
        just an unexplained conditional."""
        sw_net = jnp.ones((4, 8)) * 240.0          # mean, 2-D
        member_albedo = jnp.full((3, 4, 8), 0.2)   # what member snow produces
        assert (sw_net / (1.0 - member_albedo)).ndim == 3
        mean_albedo = jnp.full((4, 8), 0.2)
        assert (sw_net / (1.0 - mean_albedo)).shape == sw_net.shape


def _roundtrip_lw_down(eps_emit, eps_inv, lw_down, T):
    """Emit lw_net with eps_emit, reconstruct lw_down with eps_inv (driver math)."""
    sigma = constants.sigma_sb
    # gray.py boundary: lw_up = eps*sigma*T^4 + (1-eps)*F_down  =>
    # lw_net = F_down - lw_up = eps*(F_down - sigma*T^4).
    lw_net = eps_emit * (lw_down - sigma * T ** 4)
    lw_up_inv = eps_inv * sigma * T ** 4
    return (lw_net + lw_up_inv) / jnp.maximum(eps_inv, 0.01)


class TestSurfaceEmissivityForLwInversion:
    def test_gray_and_none_use_black_surface(self):
        # Gray keeps GrayRadiationConfig.sfc_emissivity = 1.0 and never threads
        # the dynamic emis_col -> invert with eps = 1.0 regardless of configs.
        for scheme in ("gray", "none"):
            eps = surface_emissivity_for_lw_inversion(
                scheme, dynamic_emissivity=jnp.full((2,), 0.98),
                static_sfc_emissivity=0.97)
            assert eps == 1.0

    def test_rrtmgp_dynamic_returns_eps_col(self):
        eps_col = jnp.array([0.985, 0.97])
        for scheme in ("rrtmgp", "rrtmg"):
            eps = surface_emissivity_for_lw_inversion(
                scheme, dynamic_emissivity=eps_col, static_sfc_emissivity=0.97)
            assert eps is eps_col

    def test_rrtmgp_static_returns_config(self):
        for scheme in ("rrtmgp", "rrtmg"):
            eps = surface_emissivity_for_lw_inversion(
                scheme, dynamic_emissivity=None, static_sfc_emissivity=0.97)
            assert eps == 0.97

    def test_lw_roundtrip_is_exact_for_each_scheme(self):
        lw_down = jnp.array([340.0, 250.0])
        T = jnp.array([288.0, 265.0])
        cases = [
            # (radiation, emit_eps, dynamic_emissivity, static)
            ("gray", 1.0, None, 0.97),
            ("none", 1.0, None, 0.97),
            ("rrtmgp", jnp.array([0.985, 0.97]),
             jnp.array([0.985, 0.97]), 0.97),         # dynamic feedback
            ("rrtmgp", 0.97, None, 0.97),             # static / pre-first-response
        ]
        for radiation, emit_eps, dyn, static in cases:
            eps_inv = surface_emissivity_for_lw_inversion(
                radiation, dynamic_emissivity=dyn, static_sfc_emissivity=static)
            recon = _roundtrip_lw_down(emit_eps, eps_inv, lw_down, T)
            assert jnp.allclose(recon, lw_down, atol=1e-9), (
                f"{radiation}: round trip not exact (emit {emit_eps}, inv {eps_inv})")

class TestSurfaceTemperatureForLwBoundary:
    def test_rrtmgp_returns_t_rad(self):
        T_rad = jnp.array([248.0, 262.0])
        for scheme in ("rrtmgp", "rrtmg"):
            T = surface_temperature_for_lw_boundary(
                scheme, T_rad=T_rad, lw_up=jnp.array([390.0, 300.0]))
            assert jnp.array_equal(T, T_rad)

    def test_gray_returns_brightness_temp_emitting_full_flux(self):
        # Gray (eps=1) must emit the full upward flux: sigma*T_bb^4 == lw_up.
        lw_up = jnp.array([390.0, 300.0, 450.0])
        for scheme in ("gray", "none"):
            T_bb = surface_temperature_for_lw_boundary(
                scheme, T_rad=jnp.array([248.0, 262.0, 270.0]), lw_up=lw_up)
            assert jnp.allclose(constants.sigma_sb * T_bb ** 4, lw_up, atol=1e-6)

    def test_gray_brightness_differs_from_trad_when_eps_col_below_one(self):
        # T_rad satisfies eps_col*sigma*T_rad^4 = LW_emit; the brightness temp
        # satisfies sigma*T_bb^4 = LW_out.  With eps_col<1 and LW_out>=LW_emit
        # they are genuinely different temperatures (feeding gray T_rad would
        # over-emit by ~1/eps_col).
        eps_col = 0.96
        T_rad = jnp.array([288.0])
        lw_emit = eps_col * constants.sigma_sb * T_rad ** 4
        lw_out = lw_emit  # emission-only worst case (no reflection)
        T_bb = surface_temperature_for_lw_boundary(
            "gray", T_rad=T_rad, lw_up=lw_out)
        # Gray fed T_rad would emit sigma*T_rad^4 = lw_emit/eps_col -> overstated.
        over = constants.sigma_sb * T_rad ** 4
        assert float(over[0]) > float(lw_out[0]) * 1.02   # ~1/eps_col overstatement
        assert float(T_bb[0]) < float(T_rad[0])           # brightness temp is lower

    def test_old_mismatch_would_bias_gray_roundtrip(self):
        # Non-vacuous guard: the PRE-FIX behavior (gray inverted with the
        # ocean emissivity ~0.985 instead of the black-surface 1.0) leaks an
        # O(1 W/m^2) bias.  Pin that the mismatch is real so the fix matters.
        lw_down = jnp.array([340.0])
        T = jnp.array([288.0])
        biased = _roundtrip_lw_down(1.0, constants.emissivity_ocean, lw_down, T)
        assert float(jnp.abs(biased - lw_down)[0]) > 0.3
        # And the fix (eps_inv = 1.0 for gray) removes it.
        fixed = _roundtrip_lw_down(
            1.0,
            surface_emissivity_for_lw_inversion(
                "gray", dynamic_emissivity=None,
                static_sfc_emissivity=constants.emissivity_ocean),
            lw_down, T)
        assert jnp.allclose(fixed, lw_down, atol=1e-9)
