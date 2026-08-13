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


def test_columns_are_identical_across_ncol(mod):
    """The vector-width control varies ONLY ncol. If the columns differed,
    a cost difference could be physics rather than width."""
    _, _ = mod.build_state(NLEV, 1)
    state, _ = mod.build_state(NLEV, 8)
    T = np.asarray(state.T.data)
    assert np.allclose(T[:, 0], T[:, 7], rtol=0, atol=0)


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
    r = mod.time_scheme("mass_flux", state, sigma, DT, repeats=1)
    assert r["finite"], "the timed call produced non-finite tendencies"
    assert r["call_ms"] > 0.0
    assert r["compile_s"] > 0.0


def test_dilute_flag_reaches_the_scheme_config(mod):
    """ON/OFF control non-vacuity: the override must change the config the
    scheme actually receives, not be silently dropped."""
    from legoesm.atmosphere.physics import ConvectionConfig

    base = ConvectionConfig(scheme="zhang_mcfarlane")
    assert base.zhang_mcfarlane.use_dilute_cape is True
    off = base._replace(
        zhang_mcfarlane=base.zhang_mcfarlane._replace(use_dilute_cape=False))
    assert off.zhang_mcfarlane.use_dilute_cape is False


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
