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


def test_the_timed_function_returns_more_than_the_temperature_tendency(mod, col):
    """codex finding 8: with only dT_dt observed, XLA can dead-code-eliminate
    whatever a scheme computes solely for moisture or momentum, biasing the
    comparison by scheme."""
    import jax

    state, sigma = col
    fn = mod._physics("mass_flux", DT)
    leaves = jax.tree_util.tree_leaves(fn(state, None, sigma)[0])
    assert len(leaves) >= 4, len(leaves)


@pytest.mark.parametrize("scheme", ["mass_flux", "dca", "zhang_mcfarlane",
                                    "emanuel", "tiedtke"])
def test_active_schemes_are_reported_active(mod, col, scheme):
    """codex finding 19/14: an all-zero tendency is finite and times fine, so
    an untriggered scheme silently joins the 'pack' that ZM is measured
    against. The activity flag is what keeps it out."""
    state, sigma = col
    r = mod.time_scheme(scheme, state, sigma, DT, repeats=1, warmups=0)
    assert r["active"], (
        f"{scheme} produced no tendency on the probe column "
        f"(sum|dT/dt| = {r['sum_abs_dT_dt']:.3e}); its timing would be the "
        f"cost of the inactive branch")


def test_kuo_is_reported_inactive_and_would_be_excluded(mod, col):
    """The measured case the flag exists for: kuo cannot convect without a
    large-scale moisture-convergence operator, and at ncol=1 it was ORIGINALLY
    the median of the 'other nine', i.e. the baseline was the cost of doing
    nothing."""
    state, sigma = col
    r = mod.time_scheme("kuo", state, sigma, DT, repeats=1, warmups=0)
    assert not r["active"], (
        "kuo produced a tendency here; if it is genuinely active in this "
        "geometry the pack-exclusion rationale needs revisiting")


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
