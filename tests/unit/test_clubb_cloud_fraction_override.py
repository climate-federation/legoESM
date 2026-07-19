"""CLUBB cloud-fraction override in ``compute_cloud_properties``.

A moist higher-order closure (CLUBB) diagnoses a sub-grid cloud fraction from
its PDF that is physically less overcast than the RH-diagnosed grid-scale one
over a saturated marine boundary layer.  ``cloud_fraction_override`` routes THAT
fraction into the cloud optics: it REPLACES the RH ``cf`` before the convective
overlap, so the sub-grid condensate floor (``cf * q_c_diagnostic``) — which sets
the marine-Sc liquid water path and hence the planetary albedo — reflects the
moist closure.  This is the radiation READ side of the CLUBB-cf -> RRTMGP wiring
(the turbulence WRITE side is tests/unit/test_turbulence_output_cloud_fraction.py;
the PhysicsState carry is exercised in the combined-physics routing test).

Assertions pin the CONTRACT:
  * ``None`` (default) is byte-identical to the pre-feature path;
  * the override SUPERSEDES the RH fraction (returned cf == the override where no
    convective cloud is added);
  * a LOWER override fraction gives a LOWER liquid water path (the albedo lever
    direction — less overcast => dimmer marine Sc);
  * the override is clipped to a physical [0, 1];
  * the path stays differentiable (finite d LWP / d cf_override).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from legoesm.atmosphere.physics.clouds.cloud_fraction import (
    compute_cloud_properties,
)
from legoesm.atmosphere.physics.clouds.config import CloudConfig

jax.config.update("jax_enable_x64", True)


def _saturated_marine_column(ncol=4, nlev=20):
    """A near-saturated low-level column: RH-diagnosed cf is high, so the
    override (lower) has room to reduce the floor."""
    sig = np.linspace(0.025, 0.99, nlev)
    p_full = jnp.asarray(sig[None, :] * 1.0e5 * np.ones((ncol, 1)))
    # Warm (liquid) BL so f_ice ~ 0 => override maps cleanly to the liquid floor.
    T = jnp.asarray((290.0 - 10.0 * sig)[None, :] * np.ones((ncol, 1)))
    q_v = jnp.asarray(1.2e-2 * np.ones((ncol, nlev)))  # near saturation low down
    dp = jnp.asarray(np.full((ncol, nlev), 5000.0))
    return T, p_full, q_v, dp


class TestCloudFractionOverride:
    def test_none_default_byte_identical(self):
        """Omitting the override (None) reproduces the RH path exactly."""
        T, p_full, q_v, dp = _saturated_marine_column()
        cfg = CloudConfig(scheme="sundqvist", clubb_cf_override_p_min_pa=0.0)
        base = compute_cloud_properties(T=T, p_full=p_full, q_v=q_v, dp=dp, config=cfg)
        same = compute_cloud_properties(
            T=T, p_full=p_full, q_v=q_v, dp=dp, config=cfg,
            cloud_fraction_override=None,
        )
        npt.assert_array_equal(np.asarray(base.cloud_fraction),
                               np.asarray(same.cloud_fraction))
        npt.assert_array_equal(np.asarray(base.lwp), np.asarray(same.lwp))

    def test_override_supersedes_rh_fraction(self):
        """With no convective cloud, the returned cf IS the override (clipped)."""
        T, p_full, q_v, dp = _saturated_marine_column()
        cfg = CloudConfig(scheme="sundqvist", clubb_cf_override_p_min_pa=0.0)
        ovr = jnp.full(T.shape, 0.25)
        props = compute_cloud_properties(
            T=T, p_full=p_full, q_v=q_v, dp=dp, config=cfg,
            cloud_fraction_override=ovr,
        )
        npt.assert_allclose(np.asarray(props.cloud_fraction), 0.25, atol=1e-12)

    def test_lower_override_gives_lower_lwp(self):
        """The albedo lever: a less-overcast CLUBB fraction dims the marine Sc.

        Uses the diagnostic (no explicit condensate) path so lwp ∝ cf directly.
        """
        T, p_full, q_v, dp = _saturated_marine_column()
        cfg = CloudConfig(scheme="sundqvist", clubb_cf_override_p_min_pa=0.0)
        rh = compute_cloud_properties(T=T, p_full=p_full, q_v=q_v, dp=dp, config=cfg)
        cf_rh = float(np.asarray(rh.cloud_fraction).max())
        assert cf_rh > 0.3, "column not overcast enough to exercise the lever"
        low = compute_cloud_properties(
            T=T, p_full=p_full, q_v=q_v, dp=dp, config=cfg,
            cloud_fraction_override=jnp.full(T.shape, 0.2),
        )
        # Lower fraction => strictly lower column liquid water path.
        assert float(np.asarray(low.lwp).sum()) < float(np.asarray(rh.lwp).sum())

    def test_override_is_clipped(self):
        """Unphysical override values are clamped to [0, 1] (no negative cf, no
        >1 overcast)."""
        T, p_full, q_v, dp = _saturated_marine_column()
        cfg = CloudConfig(scheme="sundqvist", clubb_cf_override_p_min_pa=0.0)
        hi = compute_cloud_properties(
            T=T, p_full=p_full, q_v=q_v, dp=dp, config=cfg,
            cloud_fraction_override=jnp.full(T.shape, 2.0),
        )
        lo = compute_cloud_properties(
            T=T, p_full=p_full, q_v=q_v, dp=dp, config=cfg,
            cloud_fraction_override=jnp.full(T.shape, -0.5),
        )
        assert float(np.asarray(hi.cloud_fraction).max()) <= 1.0 + 1e-12
        assert float(np.asarray(lo.cloud_fraction).min()) >= -1e-12

    def test_override_is_differentiable(self):
        """LWP is a smooth function of the override fraction (AD-safe: the clip
        has a finite subgradient on [0, 1], no NaN sentinel)."""
        T, p_full, q_v, dp = _saturated_marine_column()
        cfg = CloudConfig(scheme="sundqvist", clubb_cf_override_p_min_pa=0.0)

        def total_lwp(cf_scalar):
            ovr = jnp.full(T.shape, cf_scalar)
            props = compute_cloud_properties(
                T=T, p_full=p_full, q_v=q_v, dp=dp, config=cfg,
                cloud_fraction_override=ovr,
            )
            return props.lwp.sum()

        g = jax.grad(total_lwp)(0.4)
        assert np.isfinite(float(g))
        assert float(g) > 0.0  # more cloud fraction => more liquid water path

    def test_level_gate_confines_override_to_boundary_layer(self):
        """The real-SST fix: with the default p-min gate the override applies ONLY
        in the BL (p_full >= clubb_cf_override_p_min_pa) — the marine-Sc target —
        and the RH grid-scale fraction is kept ALOFT (avoids CLUBB's high/mid-cloud
        over-diagnosis that collapsed OLR + raised albedo in the real-SST A/B)."""
        T, p_full, q_v, dp = _saturated_marine_column()
        p_min = 70000.0
        # ramp=0 => sharp step for the clean BL-vs-aloft partition assertion.
        cfg = CloudConfig(scheme="sundqvist", clubb_cf_override_p_min_pa=p_min,
                          clubb_cf_override_ramp_pa=0.0)
        rh = compute_cloud_properties(T=T, p_full=p_full, q_v=q_v, dp=dp, config=cfg)
        ovr_val = 0.15
        gated = compute_cloud_properties(
            T=T, p_full=p_full, q_v=q_v, dp=dp, config=cfg,
            cloud_fraction_override=jnp.full(T.shape, ovr_val),
        )
        pf = np.asarray(p_full)
        cf_rh = np.asarray(rh.cloud_fraction)
        cf_gated = np.asarray(gated.cloud_fraction)
        bl = pf >= p_min       # boundary layer: override applies
        aloft = pf < p_min     # above BL: RH cf kept
        assert bl.any() and aloft.any(), "test column must straddle the gate"
        # BL levels take the override; aloft levels are UNCHANGED from the RH cf.
        npt.assert_allclose(cf_gated[bl], ovr_val, atol=1e-12)
        npt.assert_allclose(cf_gated[aloft], cf_rh[aloft], atol=1e-12)

    def test_override_strength_partial_blend(self):
        """strength<1 blends toward CLUBB (strength*CLUBB + (1-strength)*RH) rather
        than fully replacing — a gentler cloud reduction that avoids the
        full-replacement surface-heating runaway.  Checked full-column (p_min=0)
        so the blend is isolated from the level gate."""
        T, p_full, q_v, dp = _saturated_marine_column()
        rh = compute_cloud_properties(
            T=T, p_full=p_full, q_v=q_v, dp=dp,
            config=CloudConfig(scheme="sundqvist", clubb_cf_override_p_min_pa=0.0))
        cf_rh = np.asarray(rh.cloud_fraction)
        ovr_val, a = 0.10, 0.4
        blended = compute_cloud_properties(
            T=T, p_full=p_full, q_v=q_v, dp=dp,
            config=CloudConfig(scheme="sundqvist", clubb_cf_override_p_min_pa=0.0,
                               clubb_cf_override_strength=a),
            cloud_fraction_override=jnp.full(T.shape, ovr_val))
        cf_b = np.asarray(blended.cloud_fraction)
        # exact blend: a*override + (1-a)*RH
        npt.assert_allclose(cf_b, a * ovr_val + (1.0 - a) * cf_rh, atol=1e-10)
        # gentler than full replacement: the cf stays closer to RH than to CLUBB
        assert np.all(np.abs(cf_b - cf_rh) <= np.abs(ovr_val - cf_rh) + 1e-12)

    def test_override_floor_prevents_cloud_collapse(self):
        """clubb_cf_override_floor keeps a minimum BL cloud so the override cannot
        drive cf to ~0 (which triggered the cloud-temperature runaway).  Checked
        full-column with a very low override + strength=1.0."""
        T, p_full, q_v, dp = _saturated_marine_column()
        floor = 0.2
        props = compute_cloud_properties(
            T=T, p_full=p_full, q_v=q_v, dp=dp,
            config=CloudConfig(scheme="sundqvist", clubb_cf_override_p_min_pa=0.0,
                               clubb_cf_override_floor=floor),
            cloud_fraction_override=jnp.full(T.shape, 0.02))  # near-zero cloud
        cf = np.asarray(props.cloud_fraction)
        assert np.all(cf >= floor - 1e-12), "override cf fell below the floor"
        # floor=0 (default) leaves the low override untouched
        props0 = compute_cloud_properties(
            T=T, p_full=p_full, q_v=q_v, dp=dp,
            config=CloudConfig(scheme="sundqvist", clubb_cf_override_p_min_pa=0.0),
            cloud_fraction_override=jnp.full(T.shape, 0.02))
        npt.assert_allclose(np.asarray(props0.cloud_fraction), 0.02, atol=1e-12)

    def test_smooth_ramp_gate_blends_without_discontinuity(self):
        """The default SMOOTH gate (ramp>0) blends the override in over
        [p_min-ramp, p_min] instead of a step — deep BL is the override, high
        aloft is the RH cf, the transition is a monotone blend (no cloud
        discontinuity => removes the step-gate blowup)."""
        T, p_full, q_v, dp = _saturated_marine_column()
        p_min, ramp = 70000.0, 10000.0
        cfg = CloudConfig(scheme="sundqvist", clubb_cf_override_p_min_pa=p_min,
                          clubb_cf_override_ramp_pa=ramp)
        rh = compute_cloud_properties(T=T, p_full=p_full, q_v=q_v, dp=dp, config=cfg)
        ovr_val = 0.15
        sm = compute_cloud_properties(
            T=T, p_full=p_full, q_v=q_v, dp=dp, config=cfg,
            cloud_fraction_override=jnp.full(T.shape, ovr_val),
        )
        pf = np.asarray(p_full); cf_rh = np.asarray(rh.cloud_fraction)
        cf_sm = np.asarray(sm.cloud_fraction)
        deep_bl = pf >= p_min                 # full override
        high_aloft = pf <= (p_min - ramp)     # pure RH cf
        band = (pf > (p_min - ramp)) & (pf < p_min)  # blended
        npt.assert_allclose(cf_sm[deep_bl], ovr_val, atol=1e-12)
        npt.assert_allclose(cf_sm[high_aloft], cf_rh[high_aloft], atol=1e-12)
        if band.any():
            # blended values lie between the override and the local RH cf
            lo = np.minimum(ovr_val, cf_rh[band]); hi = np.maximum(ovr_val, cf_rh[band])
            assert np.all(cf_sm[band] >= lo - 1e-9) and np.all(cf_sm[band] <= hi + 1e-9)
