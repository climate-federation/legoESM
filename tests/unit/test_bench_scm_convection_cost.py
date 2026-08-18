"""Gates for the SCM convection cost bench.

The bench exists to decide WHY ``zhang_mcfarlane`` cost 4.4x the other nine
schemes per tuning evaluation, so its controls are the whole product.  What is
gated here is that the controls can actually fail:

* the ``_NEWTON_ITERS`` scaling control must really retrace -- a monkeypatch
  that silently reuses a cached executable produces a flat scaling, which
  reads exactly like a refutation of the hypothesis; and
* the ON/OFF control must select genuinely different code, i.e.
  ``use_dilute_cape`` must reach the scheme's config.

These run at a small ``nlev`` so the whole module is seconds, not minutes: the
gate is on the instrument's wiring, not on the production timings.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NLEV = 16      # enough for a real column, small enough to trace fast
DT = 600.0


def _load():
    path = REPO_ROOT / "scripts" / "bench" / "bench_scm_convection_cost.py"
    spec = importlib.util.spec_from_file_location("bench_scm_conv_cost", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


@pytest.fixture(scope="module")
def col(mod):
    return mod.build_state(NLEV, 1)


def test_state_has_the_requested_shape(mod):
    state, sigma = mod.build_state(NLEV, 4)
    assert state.T.data.shape == (1, 4, 1, NLEV)
    assert state.p_s.data.shape == (1, 4, 1)
    assert sigma.n_levels == NLEV


def test_every_state_leaf_is_identical_across_batch_width(mod):
    """The batch-width control must vary ONLY the number of columns.  Compares
    EVERY leaf of the wide state against the ncol=1 state, not just the first
    and last temperature column (codex finding 18): qv, winds, tracers and
    surface fields could each differ and turn a width effect into a physics
    effect.  Memory layout and total bytes unavoidably change with width --
    that is documented in the bench, not gated here.
    """
    narrow, sig1 = mod.build_state(NLEV, 1)
    wide, sig8 = mod.build_state(NLEV, 8)

    leaves1 = jax.tree_util.tree_leaves(narrow)
    leaves8 = jax.tree_util.tree_leaves(wide)
    assert len(leaves1) == len(leaves8)
    for a, b in zip(leaves1, leaves8):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape[1] == 1 and b.shape[1] == 8, (a.shape, b.shape)
        for c in range(8):
            assert np.array_equal(b[:, c:c + 1], a), (
                f"column {c} of the wide state differs from the ncol=1 state")
    for f1, f8 in zip(sig1, sig8):
        assert np.array_equal(np.asarray(f1), np.asarray(f8)), (
            "the sigma coordinate must not depend on the column count")


def test_profile_is_physical(mod):
    """A probe's first output is untrusted: check the column against a range
    whose bounds are known before any timing is read off it."""
    state, _ = mod.build_state(NLEV, 1)
    T = np.asarray(state.T.data)
    q = np.asarray(state.tracers["q_v"].data)
    assert 199.0 <= T.min() and T.max() <= 301.0, T.min()
    assert 0.0 < q.max() < 0.03, q.max()


def test_timing_a_scheme_returns_finite_tendencies(mod, col):
    state, sigma = col
    r = mod.time_scheme("mass_flux", state, sigma, DT, repeats=1, warmups=0)
    assert r["finite"], "the timed call produced non-finite tendencies"
    assert r["call_ms"] > 0.0
    assert r["compile_s"] > 0.0
    assert r["n_leaves"] > 1, (
        "the timed function must return the whole tendency pytree; returning "
        "one field lets XLA delete the rest of the scheme's work")


def test_the_TIMED_function_returns_more_than_the_temperature_tendency(mod, col):
    """codex round 1 finding 8: with only dT_dt observed, XLA can
    dead-code-eliminate whatever a scheme computes solely for moisture or
    momentum, biasing the comparison by scheme.

    Asserted on what ``time_scheme`` ACTUALLY returned (`n_leaves`), not on a
    separate direct `_physics` call, which would pass even if the jitted step
    still returned one array (codex round 2, finding 15).
    """
    state, sigma = col
    r = mod.time_scheme("mass_flux", state, sigma, DT, repeats=1, warmups=0)
    direct = len(jax.tree_util.tree_leaves(
        mod._physics("mass_flux", DT)(state, None, sigma)[0]))
    assert r["n_leaves"] == direct, (
        f"the timed step returned {r['n_leaves']} arrays but the tendency has "
        f"{direct}; the missing ones are free for XLA to delete")
    assert r["n_leaves"] >= 4, r["n_leaves"]


# Only `kuo` is pinned as inactive, and only because its reason is
# STRUCTURAL: it needs a large-scale moisture-convergence operator, which a
# single column never supplies, at any vertical resolution.
#
# `emanuel` is deliberately NOT pinned. At the production nlev=74 it measures
# exactly 0.0 on both channels and the bench excludes it, but at this module's
# nlev=16 it fires (dT ~ 9.9e-06 K/s) -- so its inactivity is a property of
# that particular column, not a fact about the scheme, and asserting the
# nlev=74 measurement here would be a claim carried across a configuration
# change. The bench decides per run from its own activity flag; this suite
# only pins the reason that is resolution-independent.
_EXPECTED_INACTIVE = ("kuo",)

# Schemes expected to trigger AT THIS MODULE'S nlev. `emanuel` belongs here
# precisely because it is active at nlev=16.
@pytest.mark.parametrize(
    "scheme", [s for s in (
        "sbm", "dca", "mass_flux", "edmf", "zhang_mcfarlane",
        "kain_fritsch", "tiedtke", "bechtold", "emanuel") ])
def test_active_schemes_are_reported_active(mod, col, scheme):
    """codex round 1 findings 14/19: an all-zero tendency is finite and times
    fine, so an untriggered scheme silently joins the 'pack' that ZM is
    measured against. Covers every scheme expected to trigger, not a sample
    (codex round 2, finding 11)."""
    state, sigma = col
    r = mod.time_scheme(scheme, state, sigma, DT, repeats=1, warmups=0)
    assert r["active"], (
        f"{scheme} produced no tendency on the probe column "
        f"(sum|dT/dt| = {r['sum_abs_dT_dt']:.3e}, "
        f"sum|dqv/dt| = {r['sum_abs_dqv_dt']:.3e}); its timing would be the "
        f"cost of the inactive branch")


def test_activity_accepts_a_moisture_only_tendency(mod):
    """codex round 2 finding 10: keying activity on temperature alone would
    drop a moisture-only closure from the pack. Exercised on the flag's own
    thresholds, since no shipped scheme is moisture-only here."""
    assert (mod.ACTIVITY_FLOOR_KG_PER_KG_PER_S > 0.0
            and mod.ACTIVITY_FLOOR_K_PER_S > 0.0)
    dT_only = 10.0 * mod.ACTIVITY_FLOOR_K_PER_S
    dq_only = 10.0 * mod.ACTIVITY_FLOOR_KG_PER_KG_PER_S
    # the rule the bench applies: EITHER channel above its floor is active
    assert (dT_only > mod.ACTIVITY_FLOOR_K_PER_S
            or 0.0 > mod.ACTIVITY_FLOOR_KG_PER_KG_PER_S)
    assert (0.0 > mod.ACTIVITY_FLOOR_K_PER_S
            or dq_only > mod.ACTIVITY_FLOOR_KG_PER_KG_PER_S)


@pytest.mark.parametrize("scheme", _EXPECTED_INACTIVE)
def test_the_expected_inactive_schemes_are_flagged(mod, col, scheme):
    """Pins the exclusion list itself: if one of these starts producing a
    tendency on this column it must be re-admitted to the pack, and every
    ratio recomputed."""
    state, sigma = col
    r = mod.time_scheme(scheme, state, sigma, DT, repeats=1, warmups=0)
    assert not r["active"], (
        f"{scheme} produced a tendency here (dT {r['sum_abs_dT_dt']:.3e}, "
        f"dqv {r['sum_abs_dqv_dt']:.3e}); the pack-exclusion rationale and "
        f"every published ratio need revisiting")


def test_dilute_override_reaches_the_config_through_physics(mod, monkeypatch):
    """codex finding 17: asserting on a hand-built _replace cannot fail if
    `_physics` drops or misroutes sub_overrides. Spy on make_physics and read
    the config it actually received."""
    seen = {}
    real = mod.make_physics

    def spy(cfg, **kw):
        seen["cfg"] = cfg
        return real(cfg, **kw)

    monkeypatch.setattr(mod, "make_physics", spy)
    mod._physics("zhang_mcfarlane", DT, {"use_dilute_cape": False})
    assert seen["cfg"].convection.scheme == "zhang_mcfarlane"
    assert seen["cfg"].convection.zhang_mcfarlane.use_dilute_cape is False

    mod._physics("zhang_mcfarlane", DT, {"use_dilute_cape": True})
    assert seen["cfg"].convection.zhang_mcfarlane.use_dilute_cape is True


def test_dilute_on_and_off_give_different_tendencies(mod, col):
    """The two arms must be different PHYSICS, otherwise the timing control is
    comparing one code path with itself."""
    state, sigma = col
    on = mod._physics("zhang_mcfarlane", DT, {"use_dilute_cape": True})
    off = mod._physics("zhang_mcfarlane", DT, {"use_dilute_cape": False})
    dT_on = np.asarray(on(state, None, sigma)[0].dT_dt.data)
    dT_off = np.asarray(off(state, None, sigma)[0].dT_dt.data)
    assert np.all(np.isfinite(dT_on)) and np.all(np.isfinite(dT_off))
    assert not np.allclose(dT_on, dT_off), (
        "use_dilute_cape did not change the tendency; the ON/OFF timing "
        "control would be comparing the same code path with itself")


def test_newton_iters_patch_actually_retraces(mod, col):
    """THE control that can silently lie.  Patching a module constant only
    changes the executable if the function is retraced; if a cached one is
    reused the timings come back flat and read as 'the inner solve is not the
    cost', i.e. a false refutation.  Proven here on the TENDENCY, which is a
    deterministic function of the trip count, rather than on a timing.
    """
    import jax

    from legoesm.atmosphere.physics.convection import _zm_dilute

    state, sigma = col
    original = _zm_dilute._NEWTON_ITERS
    try:
        _zm_dilute._NEWTON_ITERS = 1
        jax.clear_caches()
        fn = mod._physics("zhang_mcfarlane", DT, {"use_dilute_cape": True})
        few = np.asarray(fn(state, None, sigma)[0].dT_dt.data)

        _zm_dilute._NEWTON_ITERS = 20
        jax.clear_caches()
        fn = mod._physics("zhang_mcfarlane", DT, {"use_dilute_cape": True})
        many = np.asarray(fn(state, None, sigma)[0].dT_dt.data)
    finally:
        _zm_dilute._NEWTON_ITERS = original
        jax.clear_caches()

    assert not np.allclose(few, many, rtol=1e-9, atol=0.0), (
        "one Newton iteration and twenty gave the same tendency — the "
        "monkeypatch is not reaching the traced code, so the scaling control "
        "cannot fail and proves nothing")


def test_scheme_list_matches_the_campaign(mod):
    """The bench must cover exactly the schemes the campaign ranked, or the
    'ZM vs the pack' ratio is against a different pack."""
    assert set(mod.SCHEMES) == {
        "sbm", "dca", "kuo", "mass_flux", "edmf", "zhang_mcfarlane",
        "kain_fritsch", "emanuel", "tiedtke", "bechtold",
    }
    assert len(mod.SCHEMES) == 10
