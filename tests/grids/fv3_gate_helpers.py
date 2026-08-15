"""Comparison and gradient helpers shared by the FV3 JAX-lane gate files.

Not a conftest: these are plain functions, and a conftest would be
imported twice (once by pytest under its own module name, once by a test
module) the moment anything here grew into a fixture.  Nothing here is a
fixture, and nothing here builds state.

WHY THIS FILE EXISTS.  Every JAX-lane test module carries its own
private ``_cmp``/``_check_adjoint``/``_counted``; the newest copy
(``test_fv3_dsw_phase_3d``) is annotated "FIFTH COPY, and it is
deliberate", with a FOLLOW-UP asking for exactly this shared home.  The
D-grid tail's gate file would have been the sixth.

SCOPE, deliberately narrow.  The existing five copies are NOT converged
onto this file in the same change.  They differ from each other -- the
newest is the corrected metric (exact-value fill classification,
per-element relative difference, refusal on an all-fill pair and on an
identically-zero reference), the oldest is not -- so converging them
would change what several dozen calibrated bounds are measuring, and
every one of those bounds would have to be re-measured before it could
be trusted again.  That is a separate, measurable change.  What this
file stops is the GROWTH.

The metric implemented here is the corrected one, copied from
``test_fv3_dsw_phase_3d`` rather than from any older copy.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

# The workspace FILL CONSTANTS the duo lane round-trips, BY VALUE.
# 1e30 is `d_sw2_duo`'s `workspace_sentinel` default and 1e25 is
# `divergence_corner_duo`'s `divg_d` init.
#
# 0.0 is NOT a fill even though `d_sw1` is called with
# `workspace_sentinel=0.0`: zero is a legitimate value of every field in
# these phases, and classifying it as a fill would silently drop real
# cells from every comparison.  A magnitude-threshold classifier was
# tried in this campaign and RETRACTED -- exact values only.
FILL_VALUES = (1.0e30, 1.0e25)


def cmp_fields(got, ref, name, tol, fills=FILL_VALUES, floor_pct=10.0,
               n_over_at=1.0e-13):
    """Mask-aware, PER-ELEMENT relative comparison.

    It must be able to fail, and must not be able to fail spuriously:

    * NON-FINITE CLASS equality, per kind -- NaN, +Inf and -Inf are
      compared as SEPARATE masks.  A cell the oracle never writes must
      stay a tripwire in both lanes, and a lane that swapped a NaN for
      an infinity (a dead branch leaking, say) would be invisible to a
      single ``~isfinite`` mask;
    * EXACT-FILL mask equality AND fill VALUE equality -- ``1e30``
      passes every ``isfinite`` guard, so a drifted fill is a real
      defect wearing a finite disguise, and the two fill constants must
      not be interchangeable either;
    * every other cell: ``|a-b| / (|b| + floor)``.  A floor is needed so
      a near-zero reference cell does not divide a rounding-scale
      difference into a spurious catastrophe.

    ``floor_pct`` sets that floor as a LOW PERCENTILE of ``|ref|`` over
    the compared cells, not the median.  The median is wrong whenever
    large values are a MAJORITY: with half the field at 1e30, a
    unit-scale cell would inherit a 1e30 denominator and a 100 % error on
    it would report as 1e-30.  A low percentile tracks the small end of
    the population, which is the end that needs protecting.

    Returns ``(rel, n_over)``.  ``n_over`` is DESCRIPTIVE only -- the
    number of compared cells whose per-element ratio exceeds
    ``n_over_at`` -- and carries no causal interpretation: a sparse index
    error and a limiter flip both give "a handful", and a limiter flip
    downstream of a wide stencil can give "all of them".
    """
    a = np.asarray(got, dtype=np.float64)
    b = np.asarray(ref, dtype=np.float64)
    assert a.shape == b.shape, (name, a.shape, b.shape)

    for kind, fn in (("NaN", np.isnan), ("+Inf", np.isposinf),
                     ("-Inf", np.isneginf)):
        ma, mb = fn(a), fn(b)
        assert np.array_equal(ma, mb), (
            f"{name}: {kind} masks differ (jax {int(ma.sum())} vs numpy "
            f"{int(mb.sum())} cells of {a.size}); the three non-finite "
            f"kinds are compared separately so a NaN cannot be swapped "
            f"for an infinity unnoticed")

    fa = np.zeros(a.shape, bool)
    fb = np.zeros(b.shape, bool)
    for v in fills:
        fa |= (a == v)
        fb |= (b == v)
    assert np.array_equal(fa, fb), (
        f"{name}: workspace-FILL masks differ (jax {int(fa.sum())} vs "
        f"numpy {int(fb.sum())} cells of {a.size}); the fill constants "
        f"are {list(fills)}")
    # ... and the fill VALUES themselves, or one sentinel could be
    # replaced by the other everywhere and the cells would then be
    # EXCLUDED from the comparison below rather than reported.
    assert np.array_equal(a[fa], b[fa]), (
        f"{name}: the fill masks agree but the fill VALUES differ at "
        f"{int((a[fa] != b[fa]).sum())} of {int(fa.sum())} fill cells "
        f"(the constants {list(fills)} are not interchangeable)")

    ok = np.isfinite(a) & ~fa
    assert ok.any(), (
        f"{name}: every cell is a fill or a tripwire -- there is nothing "
        f"to compare and this gate would pass vacuously")
    assert np.any(b[ok] != 0.0), (
        f"{name}: the REFERENCE is identically zero over all "
        f"{int(ok.sum())} compared cells, so any `got` that is also zero "
        f"passes and this comparison CANNOT FAIL. Either the fixture "
        f"drives the field to a structural zero (fix the fixture), or "
        f"the zero is the contract (assert it explicitly, do not route "
        f"it through a relative comparison).")
    diff = np.abs(a[ok] - b[ok])
    mag = np.abs(b[ok])
    floor = float(np.percentile(mag[mag > 0.0], floor_pct)) \
        if np.any(mag > 0.0) else 0.0
    if not (floor > 0.0):
        floor = max(float(mag.max()), 1e-300)
    per = diff / (mag + floor)
    rel = float(per.max())
    n_over = int((per > n_over_at).sum())
    assert rel <= tol, (
        f"{name}: MEASURED per-element rel {rel:.3e} > {tol:.3e}; "
        f"{n_over} of {int(ok.sum())} compared cells exceed {n_over_at:g} "
        f"(descriptive only, no cause implied); p{floor_pct:g}|ref| "
        f"{floor:.3e}, median|ref| {float(np.median(mag)):.3e}, "
        f"max|diff| {float(diff.max()):.3e}, "
        f"bitwise={np.array_equal(a[ok], b[ok])}")
    return rel, n_over


def bitwise_equal(a, b) -> bool:
    """Exact equality with ``NaN == NaN``.

    ONLY for pure index copies.  Every call site must independently
    assert that the compared region contains real numbers, because this
    predicate is TRUE for two all-NaN arrays -- which is precisely the
    vacuous comparison the campaign has been bitten by.
    """
    return np.array_equal(np.asarray(a, dtype=np.float64),
                          np.asarray(b, dtype=np.float64), equal_nan=True)


def assert_real(arr, name):
    """Refuse to build a claim on a slab with nothing real in it."""
    a = np.asarray(arr, dtype=np.float64)
    fin = np.isfinite(a)
    assert fin.any(), (
        f"{name}: entirely non-finite -- any equality assertion over it "
        f"would be vacuous")
    assert np.any(a[fin] != 0.0), (
        f"{name}: every finite cell is exactly 0.0 -- an equality or "
        f"'did it move' assertion over it cannot fail")
    return a


def tree_dot(a, b) -> float:
    """Plain inner product -- NO ``nan_to_num``.

    Sanitising here would silently repair a NaN the adjoint identity
    exists to EXPOSE, so a non-finite leaf must propagate to the
    assertion in :func:`check_adjoint` and name itself there.
    """
    la = jax.tree_util.tree_leaves(a)
    lb = jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb), (len(la), len(lb))
    return float(sum(
        np.dot(np.asarray(x, dtype=np.float64).ravel(),
               np.asarray(y, dtype=np.float64).ravel())
        for x, y in zip(la, lb)))


def adjoint_residual(f, primals, seed=0):
    """Relative residual of ``<J v, w> == <v, J^T w>``.

    NO finite differences: ``J v`` comes from ``jax.jvp`` and ``J^T w``
    from ``jax.vjp``, so the identity is exact in exact arithmetic and
    the residual is pure floating-point roundoff.  Its power does not
    depend on an FD step, on operand scaling, or on the output's dynamic
    range -- which is why it, and not ``check_grads``, is the primary
    gradient gate for this port.
    """
    rng = np.random.default_rng(seed)
    primals = tuple(jnp.asarray(p) for p in primals)
    v = tuple(jnp.asarray(rng.standard_normal(p.shape)) for p in primals)
    _, jv = jax.jvp(f, primals, v)
    for i, leaf in enumerate(jax.tree_util.tree_leaves(jv)):
        assert np.isfinite(np.asarray(leaf)).all(), (
            f"J v leaf {i} is not finite -- the objective window includes "
            f"cells the phase never writes, or a dead branch is leaking a "
            f"NaN into the gradient (R1b)")
    _, vjp_fn = jax.vjp(f, *primals)
    w = jax.tree_util.tree_map(
        lambda x: jnp.asarray(rng.standard_normal(x.shape)), jv)
    jtw = vjp_fn(w)
    lhs = tree_dot(jv, w)
    rhs = tree_dot(v, jtw)
    return abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300), lhs, rhs


def check_adjoint(name, f, primals, tol, seed=0):
    """:func:`adjoint_residual` plus its two non-vacuity assertions."""
    r, lhs, rhs = adjoint_residual(f, primals, seed=seed)
    assert abs(lhs) > 0.0, (
        f"{name}: <J v, w> == 0 -- the identity is satisfied trivially, "
        f"so this gate proves nothing (check the window/scale)")
    assert np.isfinite(lhs) and np.isfinite(rhs), (
        f"{name}: the inner products are not finite ({lhs}, {rhs})")
    assert r <= tol, (
        f"{name}: adjoint residual {r:.3e} > {tol:.3e} "
        f"(<J v, w>={lhs:.12e}, <v, J^T w>={rhs:.12e}) -- MEASURED "
        f"value is {r:.3e}")
    return r


def counted(fn):
    """``(traced-call counter, wrapper)`` for the retrace assertions.

    ``functools.wraps`` is load-bearing, not cosmetic: ``jax.jit``
    resolves ``static_argnums``/``static_argnames`` against
    ``inspect.signature(fun)``, and a bare ``(*a, **k)`` wrapper would
    hand it a signature that does not contain ``km`` or ``cfg``.
    ``wraps`` sets ``__wrapped__``, which ``inspect.signature`` follows.
    """
    box = {"n": 0}

    @functools.wraps(fn)
    def wrapper(*a, **k):
        box["n"] += 1
        return fn(*a, **k)

    return box, wrapper


def deepcopy_faces(per_face: list) -> list:
    """Six per-face dicts with every array COPIED.

    Required, not tidiness: the NumPy phases MUTATE their inputs (the
    post-p_grad_c exchanges rewrite ``uc``/``vc``/``divg_d``, the
    pressure phases scratch ``pk``/``gz``, the NH tail rewrites the
    carry), so calling one on a module-scoped fixture would silently
    corrupt every later test in the file.
    """
    return [{k: np.array(v, copy=True) for k, v in d.items()}
            for d in per_face]


def stack_np(per_face: list) -> dict:
    """Six NumPy per-face dicts -> the face-stacked JAX container."""
    keys = tuple(per_face[0])
    return {k: jnp.asarray(np.stack([np.asarray(d[k], dtype=np.float64)
                                     for d in per_face]))
            for k in keys}
