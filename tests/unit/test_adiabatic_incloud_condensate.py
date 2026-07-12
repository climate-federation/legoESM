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


def test_adiabatic_gated_to_warm_clouds():
    """Cold (ice / mixed-phase) cells keep the CONSTANT floor, not the liquid
    adiabatic gradient — so cirrus is not dimmed with a marine-Sc liquid rate and
    then repartitioned to ice (codex review)."""
    cfg = CloudConfig(scheme="sundqvist", diagnostic_condensate_scheme="adiabatic")
    cf, dp, T_warm, p_full = _column([1.0] * NLEV)
    T_cold = jnp.full_like(T_warm, 250.0)  # below T_freeze => ice regime
    q_cold = _adiabatic_incloud_condensate(cf, dp, T_cold, p_full, cfg)
    # Every cold cloudy cell reverts EXACTLY to the calibrated constant floor.
    assert float(jnp.max(jnp.abs(q_cold - cfg.q_c_diagnostic))) == 0.0
    # Sanity: the SAME column when warm is dimmed somewhere (gate is doing work).
    q_warm = _adiabatic_incloud_condensate(cf, dp, T_warm, p_full, cfg)
    assert float(jnp.min(q_warm)) < cfg.q_c_diagnostic


def _sundqvist_column_props(scheme, mask_high_rh):
    """Run compute_cloud_properties on a column with a prescribed cloudy layer.

    Builds q_v to make the requested levels near-saturated (cf>0 via Sundqvist)
    and the rest dry, then returns the CloudProperties for the given diagnostic
    condensate scheme.
    """
    from legoesm.thermo import saturation_mixing_ratio

    p_half = jnp.linspace(2.0e3, 1.0e5, NLEV + 1)
    p_full = 0.5 * (p_half[:-1] + p_half[1:])[None, :]
    dp = (p_half[1:] - p_half[:-1])[None, :]
    T = jnp.full((1, NLEV), 285.0)
    q_sat = saturation_mixing_ratio(T, p_full)
    rh = jnp.where(jnp.asarray(mask_high_rh)[None, :], 0.995, 0.2)
    q_v = rh * q_sat
    cfg = CloudConfig(scheme="sundqvist", diagnostic_condensate_scheme=scheme)
    return compute_cloud_properties(T, p_full, q_v, dp, cfg), cfg


def test_constant_scheme_byte_identical_to_legacy_floor():
    # Default "constant" must reproduce the validated flat-floor path exactly.
    mask = [False] * (NLEV - 3) + [True, True, True]
    const, _ = _sundqvist_column_props("constant", mask)
    # Explicitly build the legacy CloudConfig (no scheme field set) and compare.
    from legoesm.thermo import saturation_mixing_ratio
    p_half = jnp.linspace(2.0e3, 1.0e5, NLEV + 1)
    p_full = 0.5 * (p_half[:-1] + p_half[1:])[None, :]
    dp = (p_half[1:] - p_half[:-1])[None, :]
    T = jnp.full((1, NLEV), 285.0)
    q_sat = saturation_mixing_ratio(T, p_full)
    q_v = jnp.where(jnp.asarray(mask)[None, :], 0.995, 0.2) * q_sat
    legacy = compute_cloud_properties(T, p_full, q_v, dp,
                                      CloudConfig(scheme="sundqvist"))
    assert float(jnp.max(jnp.abs(const.lwp - legacy.lwp))) == 0.0
    assert float(jnp.max(jnp.abs(const.iwp - legacy.iwp))) == 0.0


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


def test_unknown_diagnostic_condensate_scheme_raises():
    from legoesm.thermo import saturation_mixing_ratio
    p_half = jnp.linspace(2.0e3, 1.0e5, NLEV + 1)
    p_full = 0.5 * (p_half[:-1] + p_half[1:])[None, :]
    dp = (p_half[1:] - p_half[:-1])[None, :]
    T = jnp.full((1, NLEV), 285.0)
    q_v = 0.5 * saturation_mixing_ratio(T, p_full)
    bad = CloudConfig(scheme="sundqvist",
                      diagnostic_condensate_scheme="parabolic")  # typo/unknown
    with pytest.raises(ValueError, match="diagnostic_condensate_scheme"):
        compute_cloud_properties(T, p_full, q_v, dp, bad)
