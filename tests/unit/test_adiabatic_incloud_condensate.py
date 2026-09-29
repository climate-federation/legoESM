"""Adiabatic depth-scaled in-cloud condensate for the stratiform radiative floor.

The diagnostic cloud floor ``q_total_diag = cf * q_c_diagnostic`` uses a CONSTANT
in-cloud water (1 g/kg) at every cloudy level.  Measured against AMIP checkpoints
that floor is the radiative cloud water in the marine boundary layer (~11x the
prognostic q_c, dominating ~77% of BL cells), so a flat value over-brightens THIN
warm marine stratocumulus whose real in-cloud LWC (~0.2-0.5 g/kg) is set by the
shallow cloud DEPTH.  ``diagnostic_condensate_scheme="adiabatic"`` replaces the
constant with ``q_ad = min(adiabatic_lwc_rate * D, q_c_diagnostic)`` where ``D``
is the saturating-indicator cloudy depth (to the layer midpoint) above cloud
base, reset at clear gaps and gated to warm/liquid cells.

These tests prove the physical claims at the scheme level (no full run):

* the helper CAPS at ``q_c_diagnostic`` for a deep cloud and stays BELOW it for a
  shallow one -> deep clouds unchanged, thin marine Sc dimmed;
* the in-cloud water is MONOTONE non-decreasing with cloudy depth above base;
* ``compute_cloud_properties`` with the default ``"constant"`` scheme is
  byte-identical to the legacy floor (no production change);
* the ``"adiabatic"`` scheme DIMS a thin low cloud while leaving a deep cloud at
  the cap (the marine-BL albedo fix, deep-cloud LW_down calibration preserved);
* an unknown scheme is a HARD error (dispatch-hardening), never a silent floor.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.atmosphere.physics.clouds.config import CloudConfig
from legoesm.atmosphere.physics.clouds.cloud_fraction import (
    _adiabatic_incloud_condensate,
    compute_cloud_properties,
)

import pytest

# 40 levels over 2 hPa -> 1000 hPa gives ~200 m near-surface layers (realistic
# L40 boundary layer), so a 2-level marine-Sc deck (~420 m) sits BELOW the cap
# depth q_c_diagnostic/adiabatic_lwc_rate = 1e-3/1.5e-6 ~ 667 m (dimmed) while a
# column-filling deck runs far past it (capped).
NLEV = 40


def _column(cloudy_mask):
    """A single hydrostatic column; ``cloudy_mask`` (nlev,) sets cf per level.

    Surface-last: index 0 = model top, index -1 = surface.  Uniform layer
    thickness in pressure so ``dz`` is well-behaved and the depth accumulation
    is easy to reason about.
    """
    p_half = jnp.linspace(2.0e3, 1.0e5, NLEV + 1)          # top -> surface [Pa]
    p_full = 0.5 * (p_half[:-1] + p_half[1:])[None, :]     # (1, nlev)
    dp = (p_half[1:] - p_half[:-1])[None, :]               # (1, nlev) > 0
    T = jnp.full((1, NLEV), 285.0)                         # warm low column
    cf = jnp.asarray(cloudy_mask, dtype=p_full.dtype)[None, :]
    return cf, dp, T, p_full


def _layer_dz(dp, T, p_full):
    rho = p_full / (constants.R_d * T)
    return dp / jnp.maximum(rho * constants.g, 1.0e-12)


def test_helper_caps_deep_below_shallow():
    cfg = CloudConfig(scheme="sundqvist", diagnostic_condensate_scheme="adiabatic")
    # Deep cloud: every level cloudy.  Shallow: only the 2 near-surface levels.
    deep = _adiabatic_incloud_condensate(*_column([1.0] * NLEV), cfg)
    shallow_mask = [0.0] * (NLEV - 2) + [1.0, 1.0]
    shallow = _adiabatic_incloud_condensate(*_column(shallow_mask), cfg)

    # Non-negative everywhere; never exceeds the calibrated cap.
    assert jnp.all(deep >= 0.0) and jnp.all(shallow >= 0.0)
    assert float(jnp.max(deep)) <= cfg.q_c_diagnostic + 1e-15
    # The DEEP cloud reaches the cap near its top (many layers above base).
    assert float(jnp.max(deep)) == pytest.approx(cfg.q_c_diagnostic, rel=1e-9)
    # The SHALLOW cloud stays strictly BELOW the cap (thin => less water).
    assert float(jnp.max(shallow)) < cfg.q_c_diagnostic
    # And the two cloudy shallow levels carry LESS than the constant floor
    # would give them (the whole point: dim the thin marine Sc).
    assert float(jnp.max(shallow)) < cfg.q_c_diagnostic


def test_helper_monotone_in_cloudy_depth():
    cfg = CloudConfig(scheme="sundqvist", diagnostic_condensate_scheme="adiabatic")
    cf, dp, T, p_full = _column([1.0] * NLEV)
    q_ad = _adiabatic_incloud_condensate(cf, dp, T, p_full, cfg)[0]
    dz = _layer_dz(dp, T, p_full)[0]
    # Depth above base increases from surface (idx -1) toward top (idx 0), so
    # in-cloud water must be non-decreasing going UP until it saturates the cap.
    # Compare each level to the one just BELOW it (higher index).
    below = q_ad[1:]      # levels nearer the surface
    above = q_ad[:-1]     # the level just above each
    assert jnp.all(above + 1e-15 >= below)
    # Uncapped near the base: the bottom cloudy level takes the LAYER-MEAN
    # adiabatic value (midpoint => HALF its own layer thickness), not the layer
    # top — a linear base->top profile averages to the midpoint.
    expect_base = cfg.adiabatic_lwc_rate * 0.5 * float(dz[-1])
    assert float(q_ad[-1]) == pytest.approx(expect_base, rel=1e-6)


def test_adiabatic_phase_aware_ice_keeps_constant_floor():
    """PHASE applied exactly ONCE (codex High): the adiabatic floor dims LIQUID
    cloud but ICE keeps the calibrated constant floor — the liquid gradient never
    leaks into IWP.  Verified at the compute_cloud_properties level (where phase
    is applied), since the helper now returns the liquid value only."""
    mask = [False] * (NLEV - 6) + [True] * 6   # a low-ish deck
    # COLD (T < T_ice_only => f_ice = 1): adiabatic IWP must EQUAL constant IWP
    # (ice unaffected by the liquid gradient) and no spurious liquid appears.
    ad_cold, _ = _sundqvist_column_props("adiabatic", mask, T_val=220.0)
    c_cold, _ = _sundqvist_column_props("constant", mask, T_val=220.0)
    assert float(jnp.max(jnp.abs(ad_cold.iwp - c_cold.iwp))) == 0.0
    assert float(jnp.max(jnp.abs(ad_cold.lwp - c_cold.lwp))) == 0.0
    # WARM (f_ice = 0): adiabatic DIMS the liquid path vs constant; ice stays 0.
    ad_warm, _ = _sundqvist_column_props("adiabatic", mask, T_val=285.0)
    c_warm, _ = _sundqvist_column_props("constant", mask, T_val=285.0)
    assert float(jnp.sum(ad_warm.lwp)) < float(jnp.sum(c_warm.lwp))
    assert float(jnp.max(ad_warm.iwp)) == 0.0


def _sundqvist_column_props(scheme, mask_high_rh, T_val=285.0):
    """Run compute_cloud_properties on a column with a prescribed cloudy layer.

    Builds q_v to make the requested levels near-saturated (cf>0 via Sundqvist)
    and the rest dry, then returns the CloudProperties for the given diagnostic
    condensate scheme.  ``T_val`` sets the (uniform) column temperature so the
    phase split can be exercised (285 K = liquid, 220 K = ice).
    """
    from legoesm.thermo import saturation_specific_humidity

    p_half = jnp.linspace(2.0e3, 1.0e5, NLEV + 1)
    p_full = 0.5 * (p_half[:-1] + p_half[1:])[None, :]
    dp = (p_half[1:] - p_half[:-1])[None, :]
    T = jnp.full((1, NLEV), T_val)
    q_sat = saturation_specific_humidity(T, p_full)
    rh = jnp.where(jnp.asarray(mask_high_rh)[None, :], 0.995, 0.2)
    q_v = rh * q_sat
    cfg = CloudConfig(scheme="sundqvist", diagnostic_condensate_scheme=scheme)
    return compute_cloud_properties(T, p_full, q_v, dp, cfg), cfg


def test_constant_scheme_byte_identical_to_legacy_floor():
    # Default "constant" must reproduce the validated flat-floor path exactly.
    mask = [False] * (NLEV - 3) + [True, True, True]
    const, _ = _sundqvist_column_props("constant", mask)
    # Explicitly build the legacy CloudConfig (no scheme field set) and compare.
    from legoesm.thermo import saturation_specific_humidity
    p_half = jnp.linspace(2.0e3, 1.0e5, NLEV + 1)
    p_full = 0.5 * (p_half[:-1] + p_half[1:])[None, :]
    dp = (p_half[1:] - p_half[:-1])[None, :]
    T = jnp.full((1, NLEV), 285.0)
    q_sat = saturation_specific_humidity(T, p_full)
    q_v = jnp.where(jnp.asarray(mask)[None, :], 0.995, 0.2) * q_sat
    legacy = compute_cloud_properties(T, p_full, q_v, dp,
                                      CloudConfig(scheme="sundqvist"))
    assert float(jnp.max(jnp.abs(const.lwp - legacy.lwp))) == 0.0
    assert float(jnp.max(jnp.abs(const.iwp - legacy.iwp))) == 0.0
    # Non-circular CLOSED-FORM check (independent of the diagnostic-condensate
    # branch): for this warm (f_ice=0) column with no convective cloud the
    # constant floor's LWP is EXACTLY cf * q_c_diagnostic * dp/g, cf the Sundqvist
    # fraction — so a future change to the floor ARITHMETIC would be caught.
    from legoesm.atmosphere.physics.clouds.cloud_fraction import (
        sundqvist_cloud_fraction)
    cf_ref = sundqvist_cloud_fraction(
        q_v / jnp.maximum(q_sat, 1.0e-10), CloudConfig(scheme="sundqvist"))
    ref_lwp = cf_ref * CloudConfig().q_c_diagnostic * dp / constants.g
    assert jnp.allclose(const.lwp, ref_lwp, rtol=1e-6, atol=1e-12)


def test_adiabatic_dims_thin_low_cloud_not_deep():
    # Thin low cloud (2 near-surface levels ~420 m, below the ~667 m cap depth)
    # vs a deep cloud filling the column (runs far past the cap so almost every
    # level saturates the calibrated q_c_diagnostic and is UNCHANGED).
    thin_mask = [False] * (NLEV - 2) + [True, True]
    deep_mask = [True] * NLEV

    thin_c, _ = _sundqvist_column_props("constant", thin_mask)
    thin_a, _ = _sundqvist_column_props("adiabatic", thin_mask)
    deep_c, _ = _sundqvist_column_props("constant", deep_mask)
    deep_a, _ = _sundqvist_column_props("adiabatic", deep_mask)

    # The THIN low cloud is DIMMED by the adiabatic scheme (less liquid path).
    thin_lwp_c = float(jnp.sum(thin_c.lwp))
    thin_lwp_a = float(jnp.sum(thin_a.lwp))
    assert thin_lwp_a < thin_lwp_c
    assert thin_lwp_a < 0.85 * thin_lwp_c   # a MEANINGFUL cut, not a rounding wisp

    # The DEEP cloud upper levels reach the cap, so its total liquid path is
    # nearly UNCHANGED (deep-cloud LW_down calibration preserved).  Only the few
    # near-base layers dim, so the total drops only marginally.
    deep_lwp_c = float(jnp.sum(deep_c.lwp))
    deep_lwp_a = float(jnp.sum(deep_a.lwp))
    assert deep_lwp_a > 0.9 * deep_lwp_c
    # The dimming is RELATIVELY stronger for the thin cloud than the deep one.
    assert (thin_lwp_a / thin_lwp_c) < (deep_lwp_a / deep_lwp_c)


def test_helper_jit_grad_and_fp32_safe():
    """The helper compiles under jit, matches eager, is finite in fp32, and is
    differentiable wrt the tunable rate (forward radiation + future training)."""
    cfg = CloudConfig(scheme="sundqvist", diagnostic_condensate_scheme="adiabatic")
    cf, dp, T, p_full = _column([0.0] * (NLEV - 4) + [1.0, 1.0, 0.0, 1.0])
    eager = _adiabatic_incloud_condensate(cf, dp, T, p_full, cfg)
    jitted = jax.jit(
        lambda a, b, c, d: _adiabatic_incloud_condensate(a, b, c, d, cfg)
    )(cf, dp, T, p_full)
    assert jnp.allclose(eager, jitted)
    assert bool(jnp.all(jnp.isfinite(jitted)))
    # fp32 under JIT (not just eager): finite, no dtype blowup.
    j32 = jax.jit(
        lambda a, b, c, d: _adiabatic_incloud_condensate(a, b, c, d, cfg)
    )(cf.astype(jnp.float32), dp.astype(jnp.float32),
      T.astype(jnp.float32), p_full.astype(jnp.float32))
    assert bool(jnp.all(jnp.isfinite(j32)))
    # Differentiable wrt the tunable rate AND wrt the inputs — the gradient must
    # traverse sigmoid -> reverse -> cumsum -> minimum, not just the linear rate.
    def _loss_rate(rate):
        return jnp.sum(_adiabatic_incloud_condensate(
            cf, dp, T, p_full, cfg._replace(adiabatic_lwc_rate=rate)))
    assert bool(jnp.isfinite(jax.grad(_loss_rate)(1.5e-6)))
    for i, x in enumerate((cf, T, p_full)):
        def _loss_x(v, _i=i):
            args = [cf, dp, T, p_full]
            args[[0, 2, 3][_i]] = v
            return jnp.sum(_adiabatic_incloud_condensate(*args, cfg))
        assert bool(jnp.all(jnp.isfinite(jax.grad(_loss_x)(x))))


def test_reset_isolates_stacked_decks():
    """Hard reset at clear gaps: an upper deck gets its OWN base->top depth, NOT
    the lower deck's inherited depth (codex: stacked decks must not inherit)."""
    cfg = CloudConfig(scheme="sundqvist", diagnostic_condensate_scheme="adiabatic")
    # Surface-last: high deck (idx 8-10), clear gap, low deck (idx NLEV-4..NLEV-2).
    mask = [0.0] * NLEV
    for i in (8, 9, 10):
        mask[i] = 1.0
    for i in (NLEV - 4, NLEV - 3, NLEV - 2):
        mask[i] = 1.0
    cf = jnp.asarray(mask)[None, :]
    _, dp, T, p_full = _column(mask)
    q = _adiabatic_incloud_condensate(cf, dp, T, p_full, cfg)[0]
    # The high deck's BASE (its lowest cloudy level, idx 10) must be ~its own
    # half-layer, clearly BELOW the cap — not the lower deck's accumulated (capped)
    # depth.  If the reset failed it would inherit ~1400 m and hit the cap exactly;
    # isolated it is ~half a (thick upper-tropo) layer, well under the cap.
    assert float(q[10]) < 0.9 * cfg.q_c_diagnostic
    # And the high deck grows base->top (idx 10 -> 8), still isolated.
    assert float(q[8]) >= float(q[9]) >= float(q[10])


def test_constant_deficit_path_byte_identical_incl_fp32():
    """With EXPLICIT condensate (the deficit path — the production sundqvist +
    microphysics route) the constant scheme is byte-identical to the legacy floor
    in BOTH fp64 and fp32 (codex High: the phase restructure must not add division
    rounding to the default)."""
    from legoesm.thermo import saturation_specific_humidity
    mask = [False] * (NLEV - 4) + [True, True, True, True]
    p_half = jnp.linspace(2.0e3, 1.0e5, NLEV + 1)
    p_full = 0.5 * (p_half[:-1] + p_half[1:])[None, :]
    dp = (p_half[1:] - p_half[:-1])[None, :]
    T = jnp.full((1, NLEV), 285.0)
    q_sat = saturation_specific_humidity(T, p_full)
    q_v = jnp.where(jnp.asarray(mask)[None, :], 0.995, 0.2) * q_sat
    # Small explicit condensate below the floor => deficit path is exercised.
    q_cld = jnp.full((1, NLEV), 1.0e-5)
    q_ice = jnp.full((1, NLEV), 1.0e-5)
    for dt in (jnp.float64, jnp.float32):
        a = [T.astype(dt), p_full.astype(dt), q_v.astype(dt), dp.astype(dt)]
        kw = dict(q_cloud=q_cld.astype(dt), q_ice=q_ice.astype(dt))
        const = compute_cloud_properties(
            *a, CloudConfig(scheme="sundqvist",
                            diagnostic_condensate_scheme="constant"), **kw)
        legacy = compute_cloud_properties(
            *a, CloudConfig(scheme="sundqvist"), **kw)
        assert float(jnp.max(jnp.abs(const.lwp - legacy.lwp))) == 0.0
        assert float(jnp.max(jnp.abs(const.iwp - legacy.iwp))) == 0.0


def test_unknown_diagnostic_condensate_scheme_raises():
    from legoesm.thermo import saturation_specific_humidity
    p_half = jnp.linspace(2.0e3, 1.0e5, NLEV + 1)
    p_full = 0.5 * (p_half[:-1] + p_half[1:])[None, :]
    dp = (p_half[1:] - p_half[:-1])[None, :]
    T = jnp.full((1, NLEV), 285.0)
    q_v = 0.5 * saturation_specific_humidity(T, p_full)
    bad = CloudConfig(scheme="sundqvist",
                      diagnostic_condensate_scheme="parabolic")  # typo/unknown
    with pytest.raises(ValueError, match="diagnostic_condensate_scheme"):
        compute_cloud_properties(T, p_full, q_v, dp, bad)
