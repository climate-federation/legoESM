"""Faithfulness pins for the sea-ice melt-pond bookkeeping (CESM / topo-pond).

Target: ``step_ponds`` in ``legoesm.ice.ponds`` — the per-category melt-pond
area/depth update that partitions surface melt + rain into pond storage,
exponential drainage to the ocean, and refreezing back into ice.

Most-trustful source
--------------------
``step_ponds`` is a CESM/CICE-style topo-pond BOOKKEEPING model (no external
solver to port), so an oracle re-deriving the same volume accounting would be
CIRCULAR.  The non-circular certification is a set of FIRST-PRINCIPLES physical
statements + the module's OWN documented closed forms pinned at discriminating
points where the expected value is a KNOWN constant (not the module's arithmetic
re-run):

1. TRUTH-TIER — water-volume conservation.  With ``V = a_pond·h_pond`` per unit
   ice area, the step must satisfy EXACTLY
       V_new + V_drain + V_refreeze == V_old + input
   for any physical pond (a self-contained budget: refreeze and drainage remove
   water, melt+rain add it, the area/depth recovery neither creates nor destroys
   volume — even at the area cap, where depth folds the residual).
2. EXACT FORMS at discriminating points (independent of the module's arithmetic):
   * refreeze fraction = clip((refreeze_threshold − T_air)/refreeze_width_K, 0, 1)
     — pinned at the threshold (0), half-width (0.5), full-width (1), and above
     the threshold (0);
   * drainage fraction = 1 − exp(−dt/τ) — pinned at dt==τ (== 1−e⁻¹ ≈ 0.632, the
     surrogate's exponential relaxation, NOT a linear dt/τ);
   * area/depth split under the ratio constraint — h_pond/a_pond == depth_to_area
     _ratio EXACTLY (uncapped), and a_pond·h_pond == V_new;
   * area cap fold — at the cap a_pond == pond_to_ice_max_area and the residual
     volume goes into depth (a·h == V_new preserved).
3. GATES — thick snow (h_snow > snow_block_threshold) and open water (~ice_mask)
   admit NO new pond water.
4. Process ORDER (refreeze → drain → add input) via a compositional order pin
   (reusing the independently-pinned forms above), config-plumbing checks (every
   parameter passed at a non-default value), differentiability at the empty-pond
   sqrt floor and the cap onset, config defaults.

The existing behavioral tests in tests/unit/test_sea_ice_new_physics.py
(ponds_drain_over_time, ponds_refreeze_below_threshold, water-closure) check the
signs + that drainage/refreeze occur; these pin the exact fractions, the ratio
constraint, the cap fold, and machine-precision conservation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.ice.config import MeltPondConfig
from legoesm.ice.ponds import step_ponds

from legoesm import constants

jax.config.update("jax_enable_x64", True)  # matches the sibling ice faithful tests

_CFG = MeltPondConfig()
_T_FREEZE = float(constants.T_freeze)
_DT = 3600.0
_TAU = 86400.0
_RATIO = 0.8
_MAX_AREA = 0.6
_SNOW_BLOCK = 5.0e-3
_WIDTH = 0.5


def _arr(x):
    return jnp.atleast_1d(jnp.asarray(x, dtype=jnp.float64))


def _call(pond_area, pond_depth, **kw):
    """step_ponds with pond-friendly defaults (ice, snow-free, warm) + overrides."""
    return step_ponds(
        _arr(pond_area), _arr(pond_depth),
        melt_water_m=_arr(kw.get("melt", 0.0)),
        rain_water_m=_arr(kw.get("rain", 0.0)),
        ice_mask=_arr(kw.get("ice", 1.0)) > 0.5,
        h_snow=_arr(kw.get("h_snow", 0.0)),
        T_air=_arr(kw.get("T_air", _T_FREEZE + 5.0)),   # warm => no refreeze
        dt=kw.get("dt", _DT),
        drainage_timescale=kw.get("tau", _TAU),
        refreeze_threshold=kw.get("thr", _T_FREEZE),
        pond_to_ice_max_area=kw.get("max_area", _MAX_AREA),
        depth_to_area_ratio=kw.get("ratio", _RATIO),
        snow_block_threshold=kw.get("snow_block", _SNOW_BLOCK),
        refreeze_width_K=kw.get("width", _WIDTH),
    )


def _f(x):
    return float(x[0])


# ---------------------------------------------------------------------------
# 1. TRUTH-TIER — water-volume conservation: V_new + drain + refreeze == V_old + input.
# ---------------------------------------------------------------------------
def test_water_volume_conserved_all_processes():
    # Refreeze (half), drainage (dt==tau), and melt+rain input all active.
    a0, h0 = 0.4, 0.1                      # V_old = 0.04
    out = _call(a0, h0, melt=0.03, rain=0.02, T_air=_T_FREEZE - _WIDTH / 2,  # rf=0.5
                dt=_TAU, tau=_TAU)
    a_new, h_new, v_drain, v_refreeze = out
    v_old = a0 * h0
    v_input = 0.03 + 0.02
    v_stored = _f(a_new) * _f(h_new)
    np.testing.assert_allclose(
        v_stored + _f(v_drain) + _f(v_refreeze), v_old + v_input,
        rtol=1e-13, atol=1e-15,
    )


def test_water_volume_conserved_at_area_cap():
    # Big input forces the area cap; depth folds the residual so a*h == V_new and
    # conservation still holds to machine precision.
    out = _call(0.0, 0.0, melt=0.5)        # V_new = 0.5 -> a would exceed 0.6 cap
    a_new, h_new, v_drain, v_refreeze = out
    assert _f(a_new) == pytest.approx(_MAX_AREA, abs=1e-12)   # cap active
    v_stored = _f(a_new) * _f(h_new)
    np.testing.assert_allclose(v_stored + _f(v_drain) + _f(v_refreeze),
                               0.0 + 0.5, rtol=1e-13, atol=1e-15)


def test_outputs_nonnegative_and_area_capped():
    out = _call(0.3, 0.2, melt=0.4, rain=0.1, T_air=_T_FREEZE - 0.1)
    a_new, h_new, v_drain, v_refreeze = out
    assert _f(a_new) >= 0.0 and _f(a_new) <= _MAX_AREA + 1e-12
    assert _f(h_new) >= 0.0
    assert _f(v_drain) >= 0.0 and _f(v_refreeze) >= 0.0


# ---------------------------------------------------------------------------
# 2. EXACT refreeze fraction = clip((thr - T_air)/width, 0, 1).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("T_air,frac", [
    (_T_FREEZE, 0.0),                       # at threshold -> no refreeze
    (_T_FREEZE - _WIDTH / 2, 0.5),          # half-width -> half
    (_T_FREEZE - _WIDTH, 1.0),              # full width -> all
    (_T_FREEZE - 2 * _WIDTH, 1.0),          # colder -> clipped at 1
    (_T_FREEZE + 1.0, 0.0),                 # warm -> no refreeze
])
def test_refreeze_fraction_linear_ramp(T_air, frac):
    # V_refreeze is RETURNED before the drainage update, so it equals
    # V_pond * clip((thr - T_air)/width, 0, 1) EXACTLY for any dt (codex R1 L5).
    a0, h0 = 0.5, 0.2                        # V_pond = 0.1
    _, _, _, v_refreeze = _call(a0, h0, T_air=T_air)
    np.testing.assert_allclose(_f(v_refreeze), (a0 * h0) * frac, rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# 3. EXACT drainage fraction = 1 - exp(-dt/tau)  (exponential, NOT linear dt/tau).
# ---------------------------------------------------------------------------
def test_drainage_exponential_relaxation_form():
    # Warm (no refreeze), no input: V_drain == V_pond * (1 - exp(-dt/tau)).
    a0, h0 = 0.5, 0.2                        # V_pond = 0.1
    for dt, tau in [(_TAU, _TAU), (_TAU / 2, _TAU), (2 * _TAU, _TAU)]:
        _, _, v_drain, _ = _call(a0, h0, dt=dt, tau=tau)
        expect = (a0 * h0) * (1.0 - np.exp(-dt / tau))
        np.testing.assert_allclose(_f(v_drain), expect, rtol=1e-12, atol=1e-15)


def test_drainage_departs_from_linear_at_dt_equals_tau():
    # CANARY: at dt==tau the exponential gives 1 - e^-1 = 0.632..., NOT 1.0 (a
    # linear dt/tau relaxation would drain everything).
    a0, h0 = 0.5, 0.2
    _, _, v_drain, _ = _call(a0, h0, dt=_TAU, tau=_TAU)
    np.testing.assert_allclose(_f(v_drain) / (a0 * h0), 1.0 - np.exp(-1.0), rtol=1e-12)
    assert _f(v_drain) / (a0 * h0) < 0.7          # not the linear 1.0


# ---------------------------------------------------------------------------
# 4. Area/depth split under the depth_to_area_ratio constraint.
# ---------------------------------------------------------------------------
def test_depth_to_area_ratio_constraint_uncapped():
    # Small input so the pond stays below the area cap: h_pond/a_pond == ratio
    # EXACTLY, and a_pond*h_pond == V_new.
    out = _call(0.0, 0.0, melt=0.05)        # V_new = 0.05
    a_new, h_new, _, _ = out
    assert _f(a_new) < _MAX_AREA            # uncapped
    np.testing.assert_allclose(_f(h_new) / _f(a_new), _RATIO, rtol=1e-12)
    np.testing.assert_allclose(_f(a_new) * _f(h_new), 0.05, rtol=1e-13)


def test_area_cap_folds_residual_into_depth():
    # Above the cap: a_pond == max_area and depth == V_new / max_area (residual
    # folded to depth, so a*h == V_new).
    out = _call(0.0, 0.0, melt=0.5)         # V_new = 0.5, a would be ~0.79 > 0.6
    a_new, h_new, _, _ = out
    np.testing.assert_allclose(_f(a_new), _MAX_AREA, rtol=1e-13)
    np.testing.assert_allclose(_f(h_new), 0.5 / _MAX_AREA, rtol=1e-13)


# ---------------------------------------------------------------------------
# 5. Gates — thick snow and open water admit no new pond water.
# ---------------------------------------------------------------------------
def test_thick_snow_blocks_pond_input():
    # h_snow > snow_block_threshold => input == 0; with warm air + huge tau the
    # pond is essentially unchanged (no input, negligible drainage).
    v_old = 0.3 * 0.1
    a_new, h_new, v_drain, v_refreeze = _call(
        0.3, 0.1, melt=0.2, rain=0.2, h_snow=2 * _SNOW_BLOCK, dt=1e-6, tau=_TAU)
    np.testing.assert_allclose(_f(a_new) * _f(h_new), v_old, rtol=1e-9, atol=1e-12)
    assert _f(v_refreeze) == 0.0            # warm


def test_thin_snow_does_not_block_pond_input():
    # Snow at exactly the threshold does NOT block (strict >).
    a_new, h_new, _, _ = _call(0.0, 0.0, melt=0.05, h_snow=_SNOW_BLOCK)
    np.testing.assert_allclose(_f(a_new) * _f(h_new), 0.05, rtol=1e-13)  # input admitted


def test_open_water_admits_no_pond_input():
    v_old = 0.2 * 0.1
    a_new, h_new, _, _ = _call(0.2, 0.1, melt=0.3, ice=0.0, dt=1e-6, tau=_TAU)
    np.testing.assert_allclose(_f(a_new) * _f(h_new), v_old, rtol=1e-9, atol=1e-12)


def test_negative_net_input_adds_no_water():
    # input == max(melt + rain, 0): a net-negative water input (sublimation-
    # dominated) adds nothing — and must NOT subtract pond water either
    # (codex R1 L4).
    v_old = 0.3 * 0.1
    a_new, h_new, _, _ = _call(0.3, 0.1, melt=-0.2, rain=0.05, dt=1e-6, tau=_TAU)  # net -0.15
    np.testing.assert_allclose(_f(a_new) * _f(h_new), v_old, rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------
# 6. Process ORDER (refreeze -> drain -> add input) via a COMPOSITIONAL order pin.
#    NOTE (codex R1 M3): _pond_oracle mirrors the module's sequence statement-for-
#    statement, so it is NOT an independent authoritative oracle — the individual
#    FORMS are pinned independently above (refreeze ramp, drainage exp, ratio, cap);
#    this reuses those pinned forms to pin their COMPOSITION ORDER (reordering the
#    processes changes the result), and also drives the all-params-plumbed check.
# ---------------------------------------------------------------------------
def _pond_oracle(a0, h0, melt, rain, T_air, dt, tau, thr, width, ratio, max_area):
    v_pond = a0 * h0
    rf = min(max((thr - T_air) / width, 0.0), 1.0)
    v_refreeze = v_pond * rf
    v_after_rf = max(v_pond - v_refreeze, 0.0)
    df = 1.0 - np.exp(-dt / max(tau, 1.0))
    v_drain = v_after_rf * df
    v_after_drain = max(v_after_rf - v_drain, 0.0)
    v_new = v_after_drain + max(melt + rain, 0.0)
    h_new = np.sqrt(max(ratio * v_new, 1e-14))
    a_new = min(max(v_new / max(h_new, 1e-6), 0.0), max_area)
    if a_new >= max_area:
        a_new = max_area
        h_new = v_new / max(a_new, 1e-6)
    return a_new, h_new, v_drain, v_refreeze


def test_process_order_via_compositional_pin():
    # Reuse the individually-pinned forms in the documented ORDER (refreeze on
    # V_old, drainage on the post-refreeze volume, input added last).  Reordering
    # the processes changes the result, so this pins the ORDER (not independent
    # model validation — the forms are pinned above).
    cases = [
        dict(a0=0.4, h0=0.1, melt=0.03, rain=0.02, T_air=_T_FREEZE - 0.25, dt=_TAU, tau=_TAU),
        dict(a0=0.2, h0=0.3, melt=0.5, rain=0.0, T_air=_T_FREEZE + 2, dt=_DT, tau=_TAU),
        dict(a0=0.1, h0=0.05, melt=0.0, rain=0.0, T_air=_T_FREEZE - 1.0, dt=_TAU / 4, tau=_TAU),
    ]
    for c in cases:
        out = _call(c["a0"], c["h0"], melt=c["melt"], rain=c["rain"],
                    T_air=c["T_air"], dt=c["dt"], tau=c["tau"])
        exp = _pond_oracle(c["a0"], c["h0"], c["melt"], c["rain"], c["T_air"],
                           c["dt"], c["tau"], _T_FREEZE, _WIDTH, _RATIO, _MAX_AREA)
        for got, e in zip(out, exp):
            np.testing.assert_allclose(_f(got), e, rtol=1e-12, atol=1e-15)


def test_all_config_params_plumbed_non_default():
    # Codex R1 H1 / R2 #1: closure params thr/width/tau/ratio passed at NON-DEFAULT
    # values in an UNCAPPED state (a < max_area), so ratio is OBSERVABLE — an active
    # cap would fold depth = V/max_area and mask ratio.  A hard-coded default for
    # any of the four diverges from the oracle.  (max_area is exercised separately.)
    thr, width, tau, ratio, max_area = _T_FREEZE - 1.0, 2.0, _TAU / 3, 0.5, 0.9
    a0, h0, melt = 0.3, 0.2, 0.05
    T_air = thr - width / 2                                  # rf = 0.5 at NON-default thr/width
    out = _call(a0, h0, melt=melt, T_air=T_air, dt=_DT, tau=tau,
                thr=thr, width=width, ratio=ratio, max_area=max_area)
    exp = _pond_oracle(a0, h0, melt, 0.0, T_air, _DT, tau, thr, width, ratio, max_area)
    assert _f(out[0]) < max_area                             # UNCAPPED => ratio observable
    # h/a == the NON-default ratio (masked under a cap):
    np.testing.assert_allclose(_f(out[1]) / _f(out[0]), ratio, rtol=1e-12)
    for got, e in zip(out, exp):
        np.testing.assert_allclose(_f(got), e, rtol=1e-12, atol=1e-15)


def test_max_area_cap_plumbed_non_default():
    # Non-default max_area=0.4 in a capping state: a_new saturates at 0.4 (a
    # hard-coded 0.6 default would give a_new=0.6), depth folds the residual.
    out = _call(0.0, 0.0, melt=0.5, max_area=0.4)            # uncapped a~0.79 > 0.4
    np.testing.assert_allclose(_f(out[0]), 0.4, rtol=1e-13)
    np.testing.assert_allclose(_f(out[1]), 0.5 / 0.4, rtol=1e-13)   # residual -> depth


def test_snow_block_threshold_plumbed_non_default():
    # snow_block at a NON-default 0.02 m: snow of 0.01 m (> the 5 mm DEFAULT but <
    # the passed 0.02) must NOT block, so a hard-coded default threshold fails.
    a_new, h_new, _, _ = _call(0.0, 0.0, melt=0.05, h_snow=0.01, snow_block=0.02)
    np.testing.assert_allclose(_f(a_new) * _f(h_new), 0.05, rtol=1e-13)  # input admitted


def test_signature_defaults_used_when_omitted():
    # Omit refreeze_width_K + snow_block_threshold -> the signature defaults
    # (0.5 K, 5 mm) must apply: refreeze at thr-0.25 gives rf=0.5 (default width),
    # and snow of 0.01 m > the 5 mm default blocks the input.
    out = step_ponds(
        _arr(0.5), _arr(0.2),                                # V_pond = 0.1
        melt_water_m=_arr(0.2), rain_water_m=_arr(0.0),
        ice_mask=_arr(1.0) > 0.5, h_snow=_arr(0.01),         # > 5 mm default => blocked
        T_air=_arr(_T_FREEZE - 0.25), dt=1e-6,
        drainage_timescale=_TAU, refreeze_threshold=_T_FREEZE,
        pond_to_ice_max_area=_MAX_AREA, depth_to_area_ratio=_RATIO,
    )
    a_new, h_new, _, v_refreeze = out
    # default width 0.5 K -> rf = 0.5 at thr-0.25:
    np.testing.assert_allclose(_f(v_refreeze), 0.1 * 0.5, rtol=1e-12)
    # input blocked by default 5 mm snow threshold: stored V == V_pond - refreeze.
    np.testing.assert_allclose(_f(a_new) * _f(h_new), 0.1 - 0.05, rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------
# 7. Differentiability at the empty-pond sqrt floor and the cap kink.
# ---------------------------------------------------------------------------
def test_gradient_finite_at_empty_pond():
    # V_new == 0 (no pond, no input): sqrt'(0) is infinite but the 1e-14 floor
    # must keep d(area,depth)/d(melt) finite.
    def loss(melt):
        a, h, d, r = step_ponds(
            _arr(0.0), _arr(0.0), melt_water_m=melt, rain_water_m=_arr(0.0),
            ice_mask=_arr(1.0) > 0.5, h_snow=_arr(0.0), T_air=_arr(_T_FREEZE + 5.0),
            dt=_DT, drainage_timescale=_TAU, refreeze_threshold=_T_FREEZE,
            pond_to_ice_max_area=_MAX_AREA, depth_to_area_ratio=_RATIO)
        return jnp.sum(a ** 2 + h ** 2 + d ** 2 + r ** 2)
    g = jax.grad(loss)(jnp.asarray([0.0], dtype=jnp.float64))
    assert jnp.all(jnp.isfinite(g))


def test_area_cap_transition_onset_and_differentiable():
    # The cap onset is at V_new == ratio*max_area^2 == 0.8*0.36 == 0.288 (a_new
    # uncapped == sqrt(V/ratio) first reaches max_area there).  Pin the LOCATION:
    # just below the onset a_new is strictly UNCAPPED (< max_area); at/above it
    # saturates at max_area — a wrong cap onset misplaces this transition (codex
    # R2 #2: a*h==V alone only restates volume preservation).  Stored volume stays
    # continuous (== V_new) on both sides, and the grad wrt melt is finite AT onset.
    onset = _RATIO * _MAX_AREA ** 2                        # 0.288
    eps = 1e-6                                             # TIGHT bracket (codex R3 #2)
    a_below, h_below, _, _ = _call(0.0, 0.0, melt=onset - eps)
    a_at, _, _, _ = _call(0.0, 0.0, melt=onset)
    a_above, h_above, _, _ = _call(0.0, 0.0, melt=onset + eps)
    # Just below the onset a_new is strictly uncapped AND equals the uncapped
    # formula sqrt(V/ratio) — pinning both the branch and the onset location to
    # within 1e-6 (a cap displaced further mislocates this).
    assert _f(a_below) < _MAX_AREA
    np.testing.assert_allclose(_f(a_below), np.sqrt((onset - eps) / _RATIO), rtol=1e-12)
    np.testing.assert_allclose(_f(a_at), _MAX_AREA, rtol=1e-9)      # cap starts AT onset
    np.testing.assert_allclose(_f(a_above), _MAX_AREA, rtol=1e-13)  # capped just above
    np.testing.assert_allclose(_f(a_below) * _f(h_below), onset - eps, rtol=1e-12)
    np.testing.assert_allclose(_f(a_above) * _f(h_above), onset + eps, rtol=1e-12)

    def loss(melt):
        a, h, _, _ = step_ponds(
            _arr(0.0), _arr(0.0), melt_water_m=melt, rain_water_m=_arr(0.0),
            ice_mask=_arr(1.0) > 0.5, h_snow=_arr(0.0), T_air=_arr(_T_FREEZE + 5.0),
            dt=_DT, drainage_timescale=_TAU, refreeze_threshold=_T_FREEZE,
            pond_to_ice_max_area=_MAX_AREA, depth_to_area_ratio=_RATIO)
        return jnp.sum(a ** 2 + h ** 2)
    g = jax.grad(loss)(jnp.asarray([onset], dtype=jnp.float64))   # AT the cap onset
    assert jnp.all(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# 8. Config defaults (CESM / CICE provenance).
# ---------------------------------------------------------------------------
def test_meltpond_config_defaults():
    assert _CFG.refreeze_threshold == constants.T_freeze
    assert _CFG.refreeze_width_K == 0.5
    assert _CFG.drainage_timescale_s == 86400.0     # CESM 1-day e-folding
    assert _CFG.pond_to_ice_max_area == 0.6
    assert _CFG.depth_to_area_ratio == 0.8          # CICE volume->area
    assert _CFG.snow_block_threshold == 5.0e-3      # CICE ~5 mm
