"""Opt-in convective (cumulus) cloud-fraction source tests.

Adjustment convection (sbm Betts-Miller) holds the grid-mean column near
RH_ref~0.7 and detrains no q_c, so the RH/condensate stratiform cloud schemes
give cf~0 in the convecting tropics -> the surface radiates LW straight to
space (measured tropical LW_net_sfc ~-137 W/m2, precip ~1 mm/day, ~4.5 K
coupled cold bias).  ``convective_cloud_fraction`` (Slingo 1987) restores a
*bounded* cumulus cover tied to the convective precip rate, combined with the
stratiform fraction by maximum overlap.  Default-off => byte-identical to the
validated stratiform-only path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from legoesm.atmosphere.physics.clouds.cloud_fraction import (
    compute_cloud_properties,
    convective_cloud_fraction,
)
from legoesm.atmosphere.physics.clouds.config import CloudConfig

jax.config.update("jax_enable_x64", True)


def _column(ncol=4, nlev=20):
    sig = np.linspace(0.025, 0.99, nlev)
    p_full = jnp.asarray(sig[None, :] * 1.0e5 * np.ones((ncol, 1)))
    T = jnp.asarray((280.0 - 40.0 * sig)[None, :] * np.ones((ncol, 1)))
    q_v = jnp.asarray(5.0e-3 * np.ones((ncol, nlev)))
    dp = jnp.asarray(np.full((ncol, nlev), 5000.0))
    return p_full, T, q_v, dp, sig


class TestConvectiveCloudFraction:
    def test_monotone_increasing_bounded_in_precip(self):
        p_full, *_ , sig = _column()
        cfg = CloudConfig(scheme="sundqvist", convective_cloud=True)
        P = jnp.asarray(np.array([0.0, 1.0, 4.0, 10.0, 1000.0]) / 86400.0)
        # broadcast p_full to 5 columns
        p5 = jnp.broadcast_to(p_full[:1], (5, p_full.shape[1]))
        cf = convective_cloud_fraction(P, p5, cfg)
        colmax = np.array([float(cf[i].max()) for i in range(5)])
        assert colmax[0] == 0.0                      # no precip -> no cloud
        assert np.all(np.diff(colmax[:4]) > 0)       # increases with precip
        assert colmax[-1] <= cfg.conv_cloud_max + 1e-9   # capped (no overcast)

    def test_confined_to_free_tropospheric_deck(self):
        p_full, *_, sig = _column()
        cfg = CloudConfig(scheme="sundqvist", convective_cloud=True)
        P = jnp.full((p_full.shape[0],), 5.0 / 86400.0)
        cf = convective_cloud_fraction(P, p_full, cfg)
        # zero in the boundary layer (sigma > base) and above deck top
        below = sig > cfg.conv_cloud_sigma_base
        above = sig < cfg.conv_cloud_sigma_top
        assert float(cf[0][below].max(initial=0.0)) == 0.0
        assert float(cf[0][above].max(initial=0.0)) == 0.0

    def test_default_off_is_byte_identical(self):
        p_full, T, q_v, dp, _ = _column()
        base = compute_cloud_properties(
            T=T, p_full=p_full, q_v=q_v, dp=dp,
            config=CloudConfig(scheme="sundqvist"),
        )
        cfg_on = CloudConfig(scheme="sundqvist", convective_cloud=True)
        # default-off field stays off even if conv_precip is supplied
        off = compute_cloud_properties(
            T=T, p_full=p_full, q_v=q_v, dp=dp,
            config=CloudConfig(scheme="sundqvist"),
            conv_precip=jnp.full((T.shape[0],), 5.0 / 86400.0),
        )
        npt.assert_array_equal(
            np.asarray(base.cloud_fraction), np.asarray(off.cloud_fraction),
        )
        # enabled but zero precip also reproduces stratiform exactly
        on0 = compute_cloud_properties(
            T=T, p_full=p_full, q_v=q_v, dp=dp, config=cfg_on,
            conv_precip=jnp.zeros((T.shape[0],)),
        )
        npt.assert_allclose(
            np.asarray(on0.cloud_fraction), np.asarray(base.cloud_fraction),
        )

    def test_enabled_adds_cloud_and_radiative_condensate(self):
        # In a subsaturated column (sundqvist cf~0) convective precip must
        # create cloud fraction AND, via the existing condensate floor,
        # radiatively-active water (LWP+IWP > 0) where there was none.
        p_full, T, q_v, dp, _ = _column()
        # Genuinely dry in RH terms at ALL levels (q_sat falls fast with the
        # cold upper levels, so a merely-small q_v can still be high-RH up
        # high).  1e-4 kg/kg keeps RH well below rh_crit everywhere => the
        # stratiform sundqvist cf is ~0 and the convective deck is the only
        # cloud source.
        dry = jnp.asarray(1.0e-4 * np.ones_like(np.asarray(q_v)))
        cfg = CloudConfig(scheme="sundqvist", convective_cloud=True)
        base = compute_cloud_properties(
            T=T, p_full=p_full, q_v=dry, dp=dp,
            config=CloudConfig(scheme="sundqvist"),
        )
        on = compute_cloud_properties(
            T=T, p_full=p_full, q_v=dry, dp=dp, config=cfg,
            conv_precip=jnp.full((T.shape[0],), 6.0 / 86400.0),
        )
        # Total cloud amount must increase (max is saturated by the BL
        # stratiform cf=1, so compare the column sum).
        assert float(on.cloud_fraction.sum()) > float(base.cloud_fraction.sum())
        # And the convective anvil deck [sigma_top, sigma_base] must gain cloud
        # where stratiform gave ~0.  Pick a level INSIDE the deck (it is the
        # upper troposphere, not the column mid-point).
        sig = np.asarray(p_full[0]) / float(np.asarray(p_full[0])[-1])
        deck = (sig >= cfg.conv_cloud_sigma_top) & (sig <= cfg.conv_cloud_sigma_base)
        assert deck.any()
        assert float(on.cloud_fraction[:, deck].max()) > float(
            base.cloud_fraction[:, deck].max()
        )
        lwp_iwp = float((on.lwp + on.iwp).sum())
        assert lwp_iwp > 0.0

    def test_enabled_without_conv_precip_raises(self):
        # Dispatch-hardening: requesting the feature but not plumbing the
        # convective precip must fail LOUDLY, never silently no-op.
        p_full, T, q_v, dp, _ = _column()
        cfg = CloudConfig(scheme="sundqvist", convective_cloud=True)
        import pytest
        with pytest.raises(ValueError, match="conv_precip"):
            compute_cloud_properties(
                T=T, p_full=p_full, q_v=q_v, dp=dp, config=cfg, conv_precip=None,
            )

    def test_conv_precip_accepts_ncol_1_shape(self):
        # (ncol, 1) precip must not broadcast to rank-3 output.
        p_full, T, q_v, dp, _ = _column()
        cfg = CloudConfig(scheme="sundqvist", convective_cloud=True)
        cp = compute_cloud_properties(
            T=T, p_full=p_full, q_v=q_v, dp=dp, config=cfg,
            conv_precip=jnp.full((T.shape[0], 1), 5.0 / 86400.0),
        )
        assert cp.cloud_fraction.shape == T.shape

    def test_ad_safe_through_conv_precip(self):
        p_full, T, q_v, dp, _ = _column()
        cfg = CloudConfig(scheme="sundqvist", convective_cloud=True)

        def loss(P):
            cp = compute_cloud_properties(
                T=T, p_full=p_full, q_v=q_v, dp=dp, config=cfg, conv_precip=P,
            )
            return cp.cloud_fraction.sum()

        g = jax.grad(loss)(jnp.full((T.shape[0],), 3.0 / 86400.0))
        assert bool(jnp.all(jnp.isfinite(g)))
