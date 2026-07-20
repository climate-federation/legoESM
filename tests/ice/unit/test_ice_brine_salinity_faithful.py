"""Faithfulness pins for the sea-ice bulk-salinity + salt-flux budget.

Target: ``update_salinity_and_salt_flux`` in ``legoesm.ice.brine`` — the
single-bulk-mean ``S_ice`` salt-budget that emits the ocean salt-flux
diagnostic.

Most-trustful source
--------------------
This is a mass-budget bookkeeping model (no external solver to port), so an
oracle that re-derives the same accounting would be CIRCULAR — a shared
coefficient/sign error copied into both would pass.  The non-circular
certification is instead a set of FIRST-PRINCIPLES physical statements, each
independent of the module's internal ``salt_new`` formula:

1. PER-PROCESS SALT EXCHANGE — for a PURE single process (no clamp), the ocean's
   salt change equals the PHYSICAL salt content of the ice that formed or melted:
   ``salt_flux*dt == ± S_process · ΔV_process · ρ_ice · 1e-3``.  New ice
   (lead / basal / flooding / fresh refreeze) carries its own salinity out of the
   ocean (flux < 0); melt returns the melted ice's OLD salinity (flux > 0);
   sublimation removes WATER not salt, so the ice salt mass ``S·V`` is invariant
   and the ocean flux is zero.  Each statement pins one process's salinity
   coefficient AND sign without re-implementing the accounting.  (The existing
   behavioral ``TestBrine`` in tests/unit/test_sea_ice_new_physics.py checks the
   lead/melt/sublimation SIGNS at rel 1e-9; these pin the exact magnitudes and
   add the untested basal, flooding and fresh-refreeze coefficients.)
2. TRUTH-TIER conservation — total salt (ice + ocean) conserved to MACHINE
   precision when the ocean applies the flux, made NON-tautological by a
   clamp-saturating case (the flux is taken from POST-clamp stored salt; a leaky
   pre-clamp-``salt_new`` flux would fail).
3. The sublimation HEADROOM cap — observable only when new low-salinity ice
   dilutes the bulk below the cap (else the final bulk-salinity clip masks it):
   the sublimation salt concentrates the surviving old ice only up to its
   headroom, the rest is rejected.
4. Superposition of non-interacting processes, differentiability, constants.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.ice.brine import (
    PSU_TO_KG_PER_KG,
    aggregate_salt_flux,
    update_salinity_and_salt_flux,
)
from legoesm.ice.config import BrineConfig

from legoesm import constants

jax.config.update("jax_enable_x64", True)  # matches the sibling ice faithful tests

_RHO = constants.rho_ice   # 917 kg/m^3
_DT = 3600.0               # s
_CONV = _RHO * PSU_TO_KG_PER_KG   # PSU·m -> kg(salt)/m^2


def _call(**kw):
    """jnp arrays in, module SaltBudgetResult out (per-category)."""
    def arr(x):
        return jnp.asarray(x, dtype=jnp.float64)
    return update_salinity_and_salt_flux(
        S_ice_old=arr(kw["s_old"]),
        V_ice_old=arr(kw["v_old"]),
        V_ice_new=arr(kw["v_new"]),
        delta_V_lead_freeze=arr(kw.get("dv_lead", 0.0)),
        delta_V_white_ice=arr(kw.get("dv_white", 0.0)),
        rho_ice=_RHO, dt=_DT,
        S_lead_ice=kw.get("s_lead", 4.0),
        S_white_ice=kw.get("s_white", 17.0),
        delta_V_basal_freeze=arr(kw.get("dv_basal", 0.0)),
        S_basal_ice=kw.get("s_basal"),
        delta_V_sublim=arr(kw.get("dv_sublim", 0.0)),
        delta_V_fresh_refreeze=arr(kw.get("dv_fresh", 0.0)),
        S_fresh_ice=kw.get("s_fresh", 0.0),
        S_ice_min=kw.get("s_min", 0.0),
        S_ice_max=kw.get("s_max", 12.0),
        S_drain_target=kw.get("s_drain_target"),
        tau_drain_s=kw.get("tau_drain_s"),
    )


def _flux0(out):
    return float(out.salt_flux_to_ocean[0])


# ---------------------------------------------------------------------------
# 1. FIRST-PRINCIPLES per-process salt exchange (pure process, no clamp).
#    salt_flux*dt == ± S_process · ΔV_process · ρ · 1e-3.
# ---------------------------------------------------------------------------
def test_lead_freeze_exchanges_new_ice_salt():
    out = _call(s_old=[5.0], v_old=[0.5], v_new=[0.6], dv_lead=[0.1], s_lead=4.0)
    expect = -4.0 * 0.1 * _CONV  # new 4-PSU ice took its salt FROM the ocean
    assert _flux0(out) < 0.0
    np.testing.assert_allclose(_flux0(out) * _DT, expect, rtol=1e-13, atol=1e-18)


def test_basal_congelation_exchanges_new_ice_salt():
    out = _call(s_old=[5.0], v_old=[0.5], v_new=[0.6], dv_basal=[0.1], s_basal=3.0)
    expect = -3.0 * 0.1 * _CONV
    assert _flux0(out) < 0.0
    np.testing.assert_allclose(_flux0(out) * _DT, expect, rtol=1e-13, atol=1e-18)


def test_snow_ice_flooding_exchanges_new_ice_salt():
    # s_max=20 so the 17-PSU white ice does NOT clamp -> clean exchange.
    out = _call(s_old=[5.0], v_old=[0.5], v_new=[0.58], dv_white=[0.08],
                s_white=17.0, s_max=20.0)
    expect = -17.0 * 0.08 * _CONV
    assert _flux0(out) < 0.0
    np.testing.assert_allclose(_flux0(out) * _DT, expect, rtol=1e-13, atol=1e-18)


def test_melt_returns_old_ice_salt():
    out = _call(s_old=[6.0], v_old=[0.5], v_new=[0.4])   # ΔV_melt = 0.1
    expect = +6.0 * 0.1 * _CONV                          # melted ice at OLD 6 PSU
    assert _flux0(out) > 0.0
    np.testing.assert_allclose(_flux0(out) * _DT, expect, rtol=1e-13, atol=1e-18)


def test_fresh_refreeze_exchanges_its_own_salinity():
    # Nonzero fresh salinity so the fresh salt COEFFICIENT (not just the residual)
    # is exercised: refrozen ice at 2 PSU takes 2·ΔV salt from the ocean.
    out = _call(s_old=[5.0], v_old=[0.5], v_new=[0.55], dv_fresh=[0.05], s_fresh=2.0)
    expect = -2.0 * 0.05 * _CONV
    assert _flux0(out) < 0.0
    np.testing.assert_allclose(_flux0(out) * _DT, expect, rtol=1e-13, atol=1e-18)


def test_fresh_refreeze_zero_salinity_does_not_bury_old_salt():
    # S_fresh=0: fresh ice adds no salt and must NOT absorb the old ice's salt.
    out = _call(s_old=[5.0], v_old=[0.5], v_new=[0.55], dv_fresh=[0.05], s_fresh=0.0)
    np.testing.assert_allclose(_flux0(out) * _DT, 0.0, atol=1e-18)


def test_sublimation_below_cap_conserves_ice_salt():
    # Sublimation removes WATER not salt: ice salt mass S·V invariant, flux == 0.
    out = _call(s_old=[5.0], v_old=[0.4], v_new=[0.3], dv_sublim=[0.1])  # 5·0.4/0.3=6.7<12
    np.testing.assert_allclose(
        float(out.S_ice_new[0]) * 0.3, 5.0 * 0.4, rtol=1e-13, atol=1e-15
    )
    np.testing.assert_allclose(_flux0(out) * _DT, 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# 2. TRUTH-TIER — total salt conserved to machine precision, NON-tautological
#    via a clamp-saturating white-ice case (flux from POST-clamp stored salt).
# ---------------------------------------------------------------------------
def test_total_salt_conserved_under_clamp_saturation():
    c = dict(s_old=[10.0], v_old=[0.20], v_new=[0.60],
             dv_white=[0.40], s_white=17.0, s_lead=4.0, s_max=12.0)
    out = _call(**c)
    salt_new_unclamped = (10.0 * 0.20 + 17.0 * 0.40) * _CONV
    assert salt_new_unclamped / (0.60 * _CONV) > 12.0        # cap genuinely bites
    assert float(out.S_ice_new[0]) == pytest.approx(12.0, abs=1e-12)
    salt_old = 10.0 * 0.20 * _CONV
    salt_stored = float(out.S_ice_new[0]) * 0.60 * _CONV
    # Conserved to MACHINE precision: a leaky pre-clamp-salt_new flux misses by
    # ~1.6·ρ·1e-3 kg/m^2 here (salt_new_unclamped=8.8 vs stored=7.2 PSU·m).
    assert salt_old == pytest.approx(salt_stored + _flux0(out) * _DT,
                                     rel=1e-13, abs=1e-15)


# ---------------------------------------------------------------------------
# 3. Sublimation HEADROOM cap — observable because new low-salinity lead ice
#    dilutes the bulk BELOW the cap.  The sublimation salt (S_old·ΔV_sublim =
#    1.15 PSU·m) may concentrate the surviving old ice only up to its headroom
#    (S_max-S_old)·V_remain = 0.5·0.3 = 0.15 PSU·m; the remaining 1.0 PSU·m is
#    rejected.  With the cap S_ice_new = 4.4/0.5 = 8.8 PSU; WITHOUT the cap the
#    full 1.15 would be retained -> 5.4/0.5 = 10.8 PSU (both < the 12-PSU clip,
#    so the cap's effect is visible).
# ---------------------------------------------------------------------------
def test_sublimation_headroom_cap_is_observable_and_conserving():
    c = dict(s_old=[11.5], v_old=[0.4], v_new=[0.5],
             dv_lead=[0.2], dv_sublim=[0.1], s_lead=4.0, s_max=12.0)
    out = _call(**c)
    # First-principles capped retention (the physical cap RULE), not re-accounting:
    v_remain = 0.4 - 0.1                                    # V_old - sublim (no melt)
    headroom = (12.0 - 11.5) * v_remain                    # 0.15 PSU·m
    retained = min(11.5 * 0.1, headroom)                   # min(1.15, 0.15) = 0.15
    ice_salt = 11.5 * v_remain + retained + 4.0 * 0.2      # PSU·m
    s_expect = ice_salt / 0.5                              # 8.8 PSU (no-cap -> 10.8)
    assert float(out.S_ice_new[0]) == pytest.approx(s_expect, rel=1e-12)
    assert float(out.S_ice_new[0]) < 10.0                  # discriminates the no-cap 10.8
    # Total salt still conserved to machine precision.
    salt_old = 11.5 * 0.4 * _CONV
    salt_stored = float(out.S_ice_new[0]) * 0.5 * _CONV
    assert salt_old == pytest.approx(salt_stored + _flux0(out) * _DT,
                                     rel=1e-13, abs=1e-15)


# ---------------------------------------------------------------------------
# 4. Superposition — non-interacting freeze processes combine additively.
#    No melt (V_new = V_old + all gains - sublim) and sublimation below the cap
#    (flux contribution 0), so flux == -(sum of freeze salt contents).
# ---------------------------------------------------------------------------
def test_freeze_processes_superpose_additively():
    dv_lead, dv_white, dv_basal, dv_fresh, subl = 0.05, 0.03, 0.04, 0.02, 0.01
    v_old = 0.6
    v_new = v_old + dv_lead + dv_white + dv_basal + dv_fresh - subl   # ΔV_melt = 0
    c = dict(s_old=[3.0], v_old=[v_old], v_new=[v_new],
             dv_lead=[dv_lead], dv_white=[dv_white], dv_basal=[dv_basal],
             dv_fresh=[dv_fresh], dv_sublim=[subl],
             s_lead=4.0, s_white=8.0, s_basal=3.5, s_fresh=1.0, s_max=20.0)
    out = _call(**c)
    expect = -(4.0 * dv_lead + 8.0 * dv_white + 3.5 * dv_basal
               + 1.0 * dv_fresh) * _CONV   # sublimation (below cap) contributes 0
    np.testing.assert_allclose(_flux0(out) * _DT, expect, rtol=1e-12, atol=1e-16)


def test_simultaneous_freeze_and_melt_classifies_residual():
    # Freeze gains AND melt in the SAME step (the pure single-process cases all
    # have ΔV_melt≡0, so they cannot see how a freeze gain enters the melt
    # residual).  The melt volume is the OLD ice that disappeared AFTER crediting
    # every freeze gain,
    #     ΔV_melt = V_old + ΔV_lead + ΔV_white + ΔV_basal + ΔV_fresh − V_new,
    # and the ocean flux is the OLD salt that melt releases MINUS the salt each
    # new-ice gain took from the ocean:
    #     flux·dt = (S_old·ΔV_melt − Σ S_gain·ΔV_gain)·conv.
    # Dropping ANY freeze gain (especially fresh refreeze) from ΔV_melt mis-sizes
    # the melt residual and fails this pin — closing the coverage gap.
    dv_lead, dv_white, dv_basal, dv_fresh = 0.05, 0.03, 0.04, 0.02
    v_old, v_new = 0.6, 0.64
    dvm = v_old + dv_lead + dv_white + dv_basal + dv_fresh - v_new   # = 0.10
    assert dvm > 0.0                                                 # genuine melt
    c = dict(s_old=[10.0], v_old=[v_old], v_new=[v_new],
             dv_lead=[dv_lead], dv_white=[dv_white], dv_basal=[dv_basal],
             dv_fresh=[dv_fresh], s_lead=4.0, s_white=8.0, s_basal=3.5,
             s_fresh=2.0, s_max=20.0)
    out = _call(**c)
    expect = (10.0 * dvm
              - (4.0 * dv_lead + 8.0 * dv_white + 3.5 * dv_basal + 2.0 * dv_fresh)
              ) * _CONV
    assert _flux0(out) > 0.0                       # melt salt release dominates here
    np.testing.assert_allclose(_flux0(out) * _DT, expect, rtol=1e-12, atol=1e-16)


# ---------------------------------------------------------------------------
# 5. Differentiability.
# ---------------------------------------------------------------------------
def test_salt_budget_differentiable_per_category():
    # Every category is in the MELT regime (ΔV_melt>0, no clamp), where the flux
    # is (S_old·ΔV_melt − freeze_salt)/dt — strictly sensitive to BOTH S_ice_old
    # (via S_old·ΔV_melt) and V_ice_new (which sizes ΔV_melt).  So assert BOTH
    # partials finite AND nonzero PER category (not merely max) — no category is
    # structurally disconnected, and none sits on the max(·,0) melt kink.
    s0 = jnp.asarray([5.0, 7.0, 9.0], dtype=jnp.float64)
    # ΔV_melt = V_old + ΔV_lead + ΔV_white − V_new = {0.03, 0.04, 0.05} > 0.
    v_new0 = jnp.asarray([0.53, 0.48, 0.80], dtype=jnp.float64)
    base = dict(v_old=[0.5, 0.5, 0.8], dv_lead=[0.06, 0.0, 0.05],
                dv_white=[0.0, 0.02, 0.0], s_white=8.0, s_max=20.0)

    for i in range(3):
        gs, gv = jax.grad(
            lambda s0_, v_: _call(s_old=s0_, v_new=v_, **base).salt_flux_to_ocean[i],
            argnums=(0, 1),
        )(s0, v_new0)
        assert jnp.all(jnp.isfinite(gs)) and jnp.all(jnp.isfinite(gv))
        assert abs(float(gs[i])) > 0.0 and abs(float(gv[i])) > 0.0


def test_clamp_active_changes_gradient_wrt_s_old():
    # Clamp ACTIVE: unclamped salinity (10·0.2 + 17·0.4)/0.6 = 14.67 > S_max=12,
    # so S_ice_new saturates at 12.  In the clamped plateau the stored salt
    # S_max·V_new is INDEPENDENT of S_ice_old, so the flux's ONLY S_old
    # sensitivity is through salt_old = S_old·V_old:
    #     d(flux)/d(S_old) = V_old·ρ·1e-3/dt   (nonzero).
    # Remove the clamp and salt is conserved regardless of S_old
    # (flux = −freeze_salt/dt, independent of S_old) → the gradient would be 0.
    # Pinning this exact nonzero gradient therefore both asserts the clip is
    # active AND discriminates a removed clamp — which a finite-only check cannot.
    v_old, v_new = 0.20, 0.60

    def flux0(s_old):
        return _call(s_old=s_old, v_old=[v_old], v_new=[v_new],
                     dv_white=[0.40], s_white=17.0, s_max=12.0).salt_flux_to_ocean[0]

    out = _call(s_old=[10.0], v_old=[v_old], v_new=[v_new],
                dv_white=[0.40], s_white=17.0, s_max=12.0)
    assert float(out.S_ice_new[0]) == pytest.approx(12.0, abs=1e-12)  # clip active
    g = jax.grad(flux0)(jnp.asarray([10.0], dtype=jnp.float64))
    assert jnp.all(jnp.isfinite(g))
    np.testing.assert_allclose(float(g[0]), v_old * _CONV / _DT, rtol=1e-10)
    assert abs(float(g[0])) > 0.0   # 0 iff the clamp were removed


# ---------------------------------------------------------------------------
# 6. Constants / config / aggregation.
# ---------------------------------------------------------------------------
def test_published_constants_and_config():
    assert PSU_TO_KG_PER_KG == 1.0e-3
    cfg = BrineConfig()
    assert cfg.enabled is False               # freshwater convention by default
    assert cfg.S_ice_max == 12.0
    assert cfg.S_ice_new == constants.S_ice_bulk_default
    assert cfg.S_ocean_ref == constants.S_ocean_ref


def test_aggregate_salt_flux_sums_categories():
    per_cat = jnp.asarray([[-1.0, 2.0, -0.5]])
    total = aggregate_salt_flux(per_cat, axis=-1)
    np.testing.assert_allclose(np.asarray(total), np.asarray([0.5]), rtol=1e-13)


# ---------------------------------------------------------------------------
# 7. NEMO SI3 nn_icesal=2 mode: new-ice salt ENTRAPMENT (rn_sinew · SSS as an
#    ARRAY S_lead) + GRAVITY DRAINAGE (downward relax toward the mature bulk
#    value, salt-conserving through the (salt_old − salt_stored)/dt residual).
# ---------------------------------------------------------------------------
def test_drainage_relaxes_toward_target_exactly_and_conserves():
    """Pure drainage (no freeze/melt): S follows the exact e-fold toward the
    target, and the drained salt reaches the ocean as a POSITIVE flux equal to
    the stored-salt drop (conservation by construction)."""
    s_old, v, tau = 24.0, 0.5, 10 * 86400.0
    out = _call(s_old=[s_old], v_old=[v], v_new=[v],
                s_max=30.0, s_drain_target=4.0, tau_drain_s=tau)
    expect_S = 4.0 + (s_old - 4.0) * np.exp(-_DT / tau)
    np.testing.assert_allclose(float(out.S_ice_new[0]), expect_S, rtol=1e-13)
    # drained salt -> ocean: flux*dt == (S_old - S_new)·V·conv > 0
    np.testing.assert_allclose(
        _flux0(out) * _DT, (s_old - expect_S) * v * _CONV, rtol=1e-12)
    assert _flux0(out) > 0.0


def test_drainage_is_downward_only():
    """Ice FRESHER than the target must NOT be salted back up (drainage
    removes brine, it never adds salt)."""
    out = _call(s_old=[2.0], v_old=[0.5], v_new=[0.5],
                s_max=30.0, s_drain_target=4.0, tau_drain_s=86400.0)
    np.testing.assert_allclose(float(out.S_ice_new[0]), 2.0, rtol=1e-13)
    np.testing.assert_allclose(_flux0(out), 0.0, atol=1e-18)


def test_drainage_none_is_bit_identical_legacy():
    """S_drain_target/tau None (the defaults) must reproduce the legacy
    budget EXACTLY — the opt-in gate."""
    kw = dict(s_old=[9.0], v_old=[0.4], v_new=[0.55], dv_lead=[0.1],
              dv_basal=[0.05], s_lead=4.0)
    legacy = _call(**kw)
    gated = _call(**kw, s_drain_target=None, tau_drain_s=None)
    np.testing.assert_array_equal(np.asarray(legacy.S_ice_new),
                                  np.asarray(gated.S_ice_new))
    np.testing.assert_array_equal(np.asarray(legacy.salt_flux_to_ocean),
                                  np.asarray(gated.salt_flux_to_ocean))


def test_entrapment_array_s_lead_reduces_formation_frontload():
    """Array S_lead (0.75·SSS ≈ 24 on a 32-PSU shelf) vs legacy const 4:
    the new ice CAPTURES most of the local salt, so the ocean's NEGATIVE
    salt-uptake at formation is ~6x larger in magnitude — i.e. the instant
    (SSS − S_new) brine front-load shrinks from 28 to 8 PSU-equivalent.
    Also proves the array path broadcasts and conserves exactly."""
    dv = 0.1
    legacy = _call(s_old=[0.0], v_old=[0.0], v_new=[dv], dv_lead=[dv],
                   s_lead=4.0, s_max=30.0)
    entrap = _call(s_old=[0.0], v_old=[0.0], v_new=[dv], dv_lead=[dv],
                   s_lead=jnp.asarray([0.75 * 32.0]), s_max=30.0)
    np.testing.assert_allclose(_flux0(legacy) * _DT, -4.0 * dv * _CONV,
                               rtol=1e-13)
    np.testing.assert_allclose(_flux0(entrap) * _DT, -24.0 * dv * _CONV,
                               rtol=1e-13)
    assert _flux0(entrap) < _flux0(legacy) < 0.0
    np.testing.assert_allclose(float(entrap.S_ice_new[0]), 24.0, rtol=1e-13)


def test_entrap_then_drain_releases_gradually_and_closes():
    """Two-step story: (1) freeze at the entrapped 24 PSU (ocean keeps only
    the 8-PSU-equivalent residual), (2) drainage step returns salt to the
    ocean; TOTAL ocean salt over both steps equals the legacy single-step
    front-load once fully drained — same total brine, different timing."""
    dv, tau = 0.1, 5 * 86400.0
    step1 = _call(s_old=[0.0], v_old=[0.0], v_new=[dv], dv_lead=[dv],
                  s_lead=jnp.asarray([24.0]), s_max=30.0,
                  s_drain_target=4.0, tau_drain_s=tau)
    S1 = float(step1.S_ice_new[0])
    assert 4.0 < S1 < 24.0          # already partially drained within step 1
    step2 = _call(s_old=[S1], v_old=[dv], v_new=[dv],
                  s_max=30.0, s_drain_target=4.0, tau_drain_s=tau)
    assert _flux0(step2) > 0.0       # gradual release continues
    # Conservation ledger: ocean uptake at formation + stored + released
    # equals the total salt the freezing seawater carried MINUS what the
    # ocean kept — i.e. every term is accounted, nothing minted.
    ocean_step1 = _flux0(step1) * _DT           # < 0 (uptake into ice)
    ocean_step2 = _flux0(step2) * _DT           # > 0 (drained back)
    stored_end = float(step2.S_ice_new[0]) * dv * _CONV
    np.testing.assert_allclose(-ocean_step1, stored_end + ocean_step2,
                               rtol=1e-12)


def test_brine_config_new_fields_default_off():
    cfg = BrineConfig()
    assert cfg.f_entrap is None
    assert cfg.tau_drain_days is None


# ---------------------------------------------------------------------------
# 8. step_sea_ice static-config fail-early for the entrapment mode.  These
#    raises fire in the pure-config validation block BEFORE any state use, so
#    minimal placeholder args suffice (they must never be touched).
# ---------------------------------------------------------------------------
def _step_with_brine(brine):
    from legoesm.ice.config import SeaIceConfig
    from legoesm.ice.sea_ice import step_sea_ice
    cfg = SeaIceConfig(brine=brine)
    return step_sea_ice(None, None, None, None, None, cfg, 0.0, 3600.0)


def test_step_rejects_entrapment_without_brine_channel():
    with pytest.raises(ValueError, match="brine.enabled"):
        _step_with_brine(BrineConfig(enabled=False, f_entrap=0.75,
                                     tau_drain_days=15.0, S_ice_max=30.0))


def test_step_rejects_entrapment_without_drainage():
    with pytest.raises(ValueError, match="tau_drain_days"):
        _step_with_brine(BrineConfig(enabled=True, f_entrap=0.75,
                                     S_ice_max=30.0))


def test_step_rejects_entrapment_with_legacy_cap():
    """S_ice_max=12 would clamp the entrapped ~24-PSU new ice instantly,
    silently reproducing the front-load the mode removes -> hard reject."""
    with pytest.raises(ValueError, match="S_ice_max"):
        _step_with_brine(BrineConfig(enabled=True, f_entrap=0.75,
                                     tau_drain_days=15.0))


def test_step_rejects_out_of_range_entrapment_and_tau():
    with pytest.raises(ValueError, match="f_entrap"):
        _step_with_brine(BrineConfig(enabled=True, f_entrap=1.5,
                                     tau_drain_days=15.0, S_ice_max=60.0))
    with pytest.raises(ValueError, match="tau_drain_days"):
        _step_with_brine(BrineConfig(enabled=True, tau_drain_days=-1.0))


def test_step_cap_scales_with_entrapment_fraction():
    """The cap rule is f_entrap*40 (most saline plausible surface water), not
    a flat 25/30: f=1.0 on a 30-PSU cap would clip ordinary 32-PSU water at
    formation and silently re-front-load the brine (codex MED)."""
    with pytest.raises(ValueError, match="S_ice_max"):
        _step_with_brine(BrineConfig(enabled=True, f_entrap=1.0,
                                     tau_drain_days=15.0, S_ice_max=30.0))
    # f=0.75 needs exactly 30 -> the driver default passes.
    with pytest.raises(ValueError, match="brine.enabled|dynamics|state"):
        # passes the salinity validation (cap OK) and fails only later on the
        # placeholder state — proving the cap rule accepted f*40 == S_ice_max.
        try:
            _step_with_brine(BrineConfig(enabled=True, f_entrap=0.75,
                                         tau_drain_days=15.0, S_ice_max=30.0))
        except (AttributeError, TypeError) as e:  # placeholder state reached
            raise ValueError(f"state: {e}")       # normalize for the assert


def test_step_rejects_nonfinite_tau():
    """inf never drains (permanently buried salt); nan poisons the state."""
    with pytest.raises(ValueError, match="FINITE"):
        _step_with_brine(BrineConfig(enabled=True, f_entrap=0.75,
                                     tau_drain_days=float("inf"),
                                     S_ice_max=30.0))
    with pytest.raises(ValueError, match="FINITE"):
        _step_with_brine(BrineConfig(enabled=True, tau_drain_days=float("nan")))
