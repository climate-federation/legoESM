"""A canopy solve is CONVERGED only at a certified, physical root.

AMIP checkpoints carried leaf temperatures of 13-5655 K and canopy-air humidity
below zero in ~200-320 of 20709 land columns: the relative residual gate
(1e8x below the SEED's residual) is vacuous from a far-off warm-start seed, and
the residual has spurious roots outside the range its formulae are valid in.
The seeds below are copied from those checkpoints (mv3y_vl day 330, the
graupel-off blow-up state day 339); the forcing is a synthetic winter bundle.

Run under ``JAX_ENABLE_X64=1``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.nonlinear import make_implicit_newton_solver
from legoesm.land.canopy.solver import (
    _ROOT_NSQ_MAX,
    canopy_state_admissible,
    solve_canopy_closure_diag,
)

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from test_canopy_warm_start import _CFG, _bundle, _canopy_call, _resid_norm  # noqa: E402

_a = jnp.asarray
_SEED_430 = [1189.11, 4251.002, 373.5, 373.5, 233.287, 0.0]          # day 330, 65.9N 90E
_SEED_9725 = [1151.937, 4308.75, 373.5, 373.5, 247.931, 0.0]         # day 330, 56N 75E
_SEED_BLOWUP = [-1559.9, 79.1, 373.5, 373.5, 79.1, -0.001]           # day 339, same cell


def _winter(lai, night):
    kw = dict(LAI=_a(lai), Ta=_a(248.0), Tv_atm=_a(248.3), Ts_bc=_a(250.0),
              q_atm=_a(4e-4), Ca=_a(373.5), La=_a(200.0), rhoa=_a(1.35),
              Ps=_a(99000.0), ur=_a(3.0))
    if night:
        kw.update(SZA=_a(95.0), fSun=_a(0.0), APAR_Sun=_a(0.0), APAR_Sh=_a(0.0),
                  ASW_Sun=_a(0.0), ASW_Sh=_a(0.0), ASW_Soil=_a(0.0))
    else:
        kw.update(SZA=_a(80.0), fSun=_a(0.3), APAR_Sun=_a(40.0), APAR_Sh=_a(10.0),
                  ASW_Sun=_a(20.0 * lai), ASW_Sh=_a(5.0 * lai), ASW_Soil=_a(30.0))
    return _bundle(**kw)


def _solve(seed, bundle):
    x, _n, conv, nsq, *_ = solve_canopy_closure_diag(_a(seed), bundle, _CFG)
    return x, bool(conv), float(nsq)


def test_admissibility_box():
    ok = _a([[250.0, 251.0, 300.0, 300.0, 249.0, 1e-3],
             [250.0, 251.0, 300.0, 300.0, 249.0, -1e-12]])   # dry-boundary round-off
    bad = _a(_SEED_9725)[None]
    bad = jnp.concatenate([bad, _a([[250.0, 251.0, 300.0, 300.0, 249.0, -1e-3]]),
                           _a([[250.0, jnp.nan, 300.0, 300.0, 249.0, 1e-3]])])
    assert bool(jnp.all(canopy_state_admissible(ok)))
    assert not bool(jnp.any(canopy_state_admissible(bad)))


def test_huge_seed_residual_no_longer_certifies_a_non_root():
    """Relative gate alone: in-box state, |F| ~ 12 K, reported converged (old)."""
    x, conv, nsq = _solve(_SEED_9725, _winter(0.5, night=True))
    if conv:  # converged now means a real root
        assert nsq <= _ROOT_NSQ_MAX and _resid_norm(x, _winter(0.5, True)) < 0.2
    assert not (conv and _resid_norm(x, _winter(0.5, True)) > 1.0)


def test_spurious_root_outside_the_box_is_not_converged():
    """Blow-up seed, low-sun day, LAI 0.1: a genuine residual root at
    Tf_Sun ~ -1301 K, q_c ~ -0.11 (|F| ~ 3e-4) was reported converged."""
    b = _winter(0.1, night=False)
    x, conv, _ = _solve(_SEED_BLOWUP, b)
    assert not conv or bool(canopy_state_admissible(x))


@pytest.mark.parametrize("lai", [0.02, 0.1, 0.5])
@pytest.mark.parametrize("night", [True, False])
@pytest.mark.parametrize("seed", [_SEED_430, _SEED_9725, _SEED_BLOWUP])
def test_checkpoint_seeds_never_certify_garbage(lai, night, seed):
    b = _winter(lai, night)
    x, conv, nsq = _solve(seed, b)
    if conv:
        assert bool(canopy_state_admissible(x))
        assert nsq <= _ROOT_NSQ_MAX


@pytest.mark.parametrize("lai", [0.02, 0.1, 0.5])
@pytest.mark.parametrize("night", [True, False])
def test_cold_start_still_converges_to_a_physical_root(lai, night):
    b = _winter(lai, night)
    x, conv, nsq = _solve([248.0, 248.0, 0.7 * 373.5, 0.7 * 373.5, 248.0, 4e-4], b)
    assert conv and bool(canopy_state_admissible(x)) and nsq < 1e-5


def test_absurd_cache_is_discarded_for_the_cold_start():
    """A checkpoint cache from before the box existed must heal on restart:
    the call seeded with it reproduces the cold call exactly."""
    cold = _canopy_call()
    garbage = jnp.tile(_a(_SEED_9725, dtype=jnp.float32), (8, 1))
    seeded = _canopy_call(seed_arr=garbage)
    for name in ("shflx", "lhflx", "T_surface"):
        a, b = getattr(cold, name), getattr(seeded, name)
        assert float(jnp.max(jnp.abs(jnp.asarray(a) - jnp.asarray(b)))) == 0.0, name
    cx = seeded.canopy_x
    fin = jnp.all(jnp.isfinite(cx), axis=-1)
    assert bool(jnp.all(~fin | canopy_state_admissible(cx)))


def test_canopy_call_jits_on_mixed_columns():
    """Warm (a real converged cache), garbage and never-solved seeds side by side."""
    warm = _canopy_call().canopy_x
    seed = warm.at[3:5].set(_a(_SEED_9725, dtype=warm.dtype)).at[5].set(jnp.nan)
    eager = _canopy_call(seed_arr=seed)
    jitted = jax.jit(lambda s: _canopy_call(seed_arr=s))(seed)
    assert bool(jnp.all(eager.converged == jitted.converged))
    assert float(jnp.max(jnp.abs(eager.shflx - jitted.shflx))) < 1e-3


# ---- the Picard control flow, driven by a stand-in solver -------------------
# The stand-in "converges" exactly when its seed's leaf temperature is below
# _COLD_MAX (the _canopy_call cold state is Ta = 290-298 K), and returns its seed
# as the solution.  It records the ground flux the soil callback receives.

_COLD_MAX = 300.0
_WARM_BAD = [330.0, 330.0, 280.0, 280.0, 330.0, 0.01]   # admissible, but "leads astray"


def _stub_run(monkeypatch, seed_arr, converge_on_passes=None):
    import legoesm.land.surface_scheme.two_leaf_canopy as tl
    calls = {"n": 0}

    def fake(x0, bun):
        k = calls["n"]
        ok = x0[0] < _COLD_MAX
        if converge_on_passes is not None:
            ok = ok & (k in converge_on_passes)
        z = jnp.zeros((), x0.dtype)
        return (x0, jnp.array(1), ok, z, z, z, z)

    def counted(*a, **k):
        out = jax.vmap(fake)(*a)
        calls["n"] += 1
        return out

    monkeypatch.setattr(tl.jax, "vmap", _VmapProxy(tl.jax.vmap, fake, counted))
    seen_G = []
    out = _canopy_call(seed_arr=seed_arr, soil_record=seen_G)
    return out, seen_G, calls["n"]


class _VmapProxy:
    """jax.vmap, except vmap(_solve_one_col) is replaced by the stand-in."""
    def __init__(self, real, fake, counted):
        self.real, self.fake, self.counted = real, fake, counted

    def __call__(self, f, *a, **k):
        if getattr(f, "__name__", "") == "_solve_one_col":
            return self.counted
        return self.real(f, *a, **k)


def test_a_warm_seed_that_fails_gets_a_cold_retry(monkeypatch):
    seed = jnp.tile(_a(_WARM_BAD, dtype=jnp.float32), (8, 1))
    out, seen_G, n = _stub_run(monkeypatch, seed)
    assert n == 6
    # pass 0 (warm seed) fails everywhere, and its ground flux does not reach the soil
    assert float(jnp.max(jnp.abs(seen_G[0]))) == 0.0
    # pass 1 (cold) converges; the soil sees a real flux from then on
    assert float(jnp.max(jnp.abs(seen_G[1]))) > 0.0
    assert bool(jnp.all(out.converged))
    assert bool(jnp.all(out.canopy_x[:, 0] < _COLD_MAX))


def test_acceptance_is_the_last_pass(monkeypatch):
    """Converged on pass 0 only: cached, but the call reports NOT converged."""
    out, seen_G, _ = _stub_run(monkeypatch, None, converge_on_passes={0})
    assert not bool(jnp.any(out.converged))
    assert bool(jnp.all(jnp.isfinite(out.canopy_x)))
    assert all(float(jnp.max(jnp.abs(g))) == 0.0 for g in seen_G[1:])


# ---- shared solver: the new gates are opt-in and drive the adjoint ----------

def _quad_solver(max_iters=60, **kw):
    # root of x^2 - p at x = +/- sqrt(p); admissible() can veto the negative one
    return make_implicit_newton_solver(
        lambda x, p: x * x - p, x_scale=_a([1.0]), f_scale=_a([1.0]),
        max_iters=max_iters, **kw)


def test_default_contract_unchanged():
    """Defaults: an exact seed stops at iteration 0 (the initial n_sq_0 <= atol
    gate), and the negative root is accepted as before."""
    solve = _quad_solver()
    x, n, conv, *_ = solve(_a([2.0]), _a([4.0]))
    assert bool(conv) and int(n) == 0 and float(x[0]) == 2.0
    x, _, conv, *_ = solve(_a([-3.0]), _a([4.0]))
    assert bool(conv) and abs(float(x[0]) + 2.0) < 1e-4


def test_an_exact_but_inadmissible_seed_is_not_converged():
    solve = _quad_solver(admissible=lambda x: x[0] > 0.0)
    _, _, conv, *_ = solve(_a([-2.0]), _a([4.0]))
    assert not bool(conv)


def test_inadmissible_exact_root_is_rejected_and_has_zero_gradient():
    solve = _quad_solver(admissible=lambda x: x[0] > 0.0)
    x_neg, _, conv_neg, *_ = solve(_a([-3.0]), _a([4.0]))
    assert abs(float(x_neg[0]) + 2.0) < 1e-6 and not bool(conv_neg)
    g_neg = jax.grad(lambda p: solve(_a([-3.0]), p)[0][0])(_a([4.0]))
    assert float(g_neg[0]) == 0.0
    x_pos, _, conv_pos, *_ = solve(_a([3.0]), _a([4.0]))
    assert bool(conv_pos)
    g_pos = jax.grad(lambda p: solve(_a([3.0]), p)[0][0])(_a([4.0]))
    assert abs(float(g_pos[0]) - 0.25) < 1e-4          # d sqrt(p)/dp at p=4


def test_absolute_ceiling_rejects_a_relative_only_pass():
    """Seed x=1e3 on x^2=4: the seed residual is ~1e12, so the relative gate
    alone stops near x~8 (|F|~60) and calls it converged; the ceiling makes the
    solve continue to the real root."""
    x_rel, _, conv_rel, *_ = _quad_solver()(_a([1e3]), _a([4.0]))
    assert bool(conv_rel) and abs(float(x_rel[0])) > 4.0      # the old vacuity
    x, _, conv, nsq, *_ = _quad_solver(n_sq_max=1e-10)(_a([1e3]), _a([4.0]))
    assert bool(conv) and float(nsq) <= 1e-10 and abs(float(x[0]) - 2.0) < 1e-6


def test_a_rejected_nan_iterate_leaves_finite_gradients(monkeypatch):
    """Round-1 fix: a failed column's iterate is NaN; the fluxes and the soil
    boundary it would feed must still give finite reverse-mode gradients."""
    import legoesm.land.surface_scheme.two_leaf_canopy as tl
    real_vmap = tl.jax.vmap

    def fake(x0, bun):
        z = jnp.zeros((), x0.dtype)
        return (jnp.full_like(x0, jnp.nan), jnp.array(1), jnp.asarray(False), z, z, z, z)

    monkeypatch.setattr(tl.jax, "vmap", _VmapProxy(real_vmap, fake, real_vmap(fake)))

    def loss(scale):
        out = _canopy_call(soil_record=None, lw_scale=scale)
        return jnp.sum(out.Ts_solve) + jnp.sum(jnp.nan_to_num(out.T_surface))
    g = jax.grad(loss)(jnp.asarray(1.0, jnp.float32))
    assert bool(jnp.isfinite(g))
