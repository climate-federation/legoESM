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
import os

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

_MEASURE_ENV = "LEGOESM_FV3_TOL_MEASURE"


def measure_mode() -> bool:
    """True when the campaign's tolerance-measurement sweep is running.

    In that mode every gate prints one ``TOLMEASURE {name!r}: ...`` line
    (one grep harvests the whole run) and skips ONLY its numeric bound;
    every structural assert stays live, so a measurement run cannot
    silently bless a broken comparison.
    """
    return os.environ.get(_MEASURE_ENV) == "1"


def gate_scalar(name, measured, tol, quantity="rel"):
    """Tolerance gate on an ALREADY-COMPUTED scalar measure.

    For call sites whose comparison is a bespoke expression (a
    telescoping residual, a conservation defect, a one-sided FD
    agreement, a growth ratio) rather than a field pair ``cmp_fields``
    could take.  The caller keeps every structural assert live -- only
    the numeric bound routes through here.  ``quantity`` names WHAT the
    scalar measures, because an adjoint residual, an FD gap and a field
    parity are different quantities and their printed lines must say so.
    """
    measured = float(measured)
    if measure_mode():
        print(f"TOLMEASURE {name!r}: rel {measured:.3e} "
              f"(bound {tol:.3e}, quantity {quantity})", flush=True)
        return measured
    assert measured <= tol, (
        f"{name}: MEASURED {quantity} {measured:.3e} > {tol:.3e}")
    return measured


# A smallest-passing check_grads tolerance above this is a gradient
# DEFECT finding, never a bound candidate; see gated_check_grads. 1e-3
# sits an order above the worst honest FD-vs-AD gap measured on a
# conditioning-limited group (2.1e-4, nh_core job 9425294) and well
# below a plainly wrong Jacobian (O(1)).
_GRAD_DEFECT_CAP = 1.0e-3


def gated_check_grads(name, f, args, order, modes=("fwd", "rev"),
                      eps=None, atol=None, rtol=None):
    """``jax.test_util.check_grads`` with the campaign's measure mode.

    Normal mode: plain ``check_grads`` at the given bounds.  Measure
    mode: the smallest power of ten at which ``check_grads`` ITSELF
    passes (``atol=rtol`` swept downward; its tangents come from a
    seeded ``np.random.RandomState(0)``, so the scan is deterministic --
    verified in the installed ``jax._src.public_test_util.check_jvp``),
    printed as the measured value.  The quantity is check_grads' own
    FD-vs-AD agreement, NOT field parity, and the printed line says so.
    """
    from jax.test_util import check_grads
    if measure_mode():
        passing = None
        for expo in range(-1, -16, -1):
            try:
                check_grads(f, args, order=order, modes=modes, eps=eps,
                            atol=10.0 ** expo, rtol=10.0 ** expo)
            except AssertionError:
                break
            passing = 10.0 ** expo
        shown = float("inf") if passing is None else passing
        print(f"TOLMEASURE {name!r}: rel {shown:.3e} "
              f"(bound {float(atol):.3e}, quantity smallest-passing "
              f"check_grads atol=rtol, order={order}, eps={eps})",
              flush=True)
        # LAUNDERING CAP (codex MINOR, taken): a smallest-passing
        # tolerance above _GRAD_DEFECT_CAP is a gradient FINDING, not
        # a bound candidate -- the mechanical x10 protocol must not be
        # able to bless it. The value is printed first (the census
        # needs it), then the sweep is stopped loudly.
        if shown > _GRAD_DEFECT_CAP:
            raise AssertionError(
                f"{name}: smallest-passing check_grads tolerance "
                f"{shown:.3e} exceeds _GRAD_DEFECT_CAP="
                f"{_GRAD_DEFECT_CAP:g} -- a derivative this far from "
                f"its FD is a defect to investigate, not a bound to "
                f"write.")
        return
    check_grads(f, args, order=order, modes=modes, eps=eps, atol=atol,
                rtol=rtol)


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
    # MEASUREMENT MODE (LEGOESM_FV3_TOL_MEASURE=1): print every
    # comparison's measured value and DO NOT raise on the tolerance.
    # Exists so one run can replace a whole file's provisional bounds
    # with `measured x 10` instead of one first-failure per test per
    # run. Structural checks above (shapes, non-finite classes, fills,
    # anti-vacuity) still raise -- only the numeric bound is suspended,
    # so a measurement run cannot silently bless a broken comparison.
    if os.environ.get("LEGOESM_FV3_TOL_MEASURE") == "1":
        print(f"TOLMEASURE {name!r}: rel {rel:.3e} (bound {tol:.3e}, "
              f"n_over {n_over}, max|diff| {float(diff.max()):.3e})",
              flush=True)
        return rel, n_over
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
    if measure_mode():
        print(f"TOLMEASURE {name!r}: rel {r:.3e} (bound {tol:.3e}, "
              f"quantity adjoint identity residual)", flush=True)
        return r
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


# ---------------------------------------------------------------------
# the FD-vs-AD discriminator
#
# The adjoint identity above compares two transformations of the SAME
# program, so it is blind to a wrong Jacobian (codex BLOCKER, job
# 9417397).  A finite difference IS independent -- it compares the
# derivative against the function -- but on a limiter-heavy map a plain
# `check_grads` at a fixed step straddles a switching surface and
# certifies a jump: measured 6.6 % on the D-grid tail's wind group (job
# 9417462), which says nothing about which side is wrong.
#
# What CAN be asserted without a tolerance is the SCALING.  A central
# difference has truncation error ~eps^2 and roundoff ~1/eps, so halving
# eps quarters the gap if truncation dominates, doubles it if roundoff
# does, and leaves it UNCHANGED if the reverse mode is wrong.  The
# verdict is the ratio, and the ratio needs no bound on the gap itself
# -- which is the number in dispute.
#
# Promoted here from `test_fv3_pgrad`'s private copies for the same
# reason `cmp_fields` was: the D-grid tail would have been the second
# copy.  The pgrad copies keep their calibrated ladders and are not
# touched in the same change.
# ---------------------------------------------------------------------

_U_MACH_F64 = float(np.finfo(np.float64).eps)      # 2.220446e-16


def vjp_fd_projection(f, primals, eps, seed=0):
    """``(|<v, J^T w> - <J_FD v, w>|, <v, J^T w>)`` at one step size."""
    rng = np.random.default_rng(seed)
    primals = tuple(jnp.asarray(p) for p in primals)
    v = tuple(jnp.asarray(rng.standard_normal(p.shape)) for p in primals)
    out, vjp_fn = jax.vjp(f, *primals)
    w = jax.tree_util.tree_map(
        lambda x: jnp.asarray(rng.standard_normal(x.shape)), out)
    ad = tree_dot(v, vjp_fn(w))
    plus = f(*[p + eps * t for p, t in zip(primals, v)])
    minus = f(*[p - eps * t for p, t in zip(primals, v)])
    fd = jax.tree_util.tree_map(lambda a, b: (a - b) / (2.0 * eps),
                                plus, minus)
    return abs(ad - tree_dot(fd, w)), ad


def fd_roundoff_floor(f, primals, eps) -> float:
    """``u_mach * ||f||_inf * sqrt(n_out) / eps`` -- the FD's own floor.

    It GROWS as eps shrinks, which is why a ladder that only shrinks eps
    walks AWAY from the truncation regime rather than toward it.
    """
    out = f(*[jnp.asarray(p) for p in primals])
    leaves = jax.tree_util.tree_leaves(out)
    scale = max(float(np.abs(np.asarray(x)).max()) for x in leaves)
    n = sum(int(np.asarray(x).size) for x in leaves)
    return _U_MACH_F64 * scale * float(np.sqrt(n)) / eps


def assert_fd_truncation_scaling(name, f, primals, seed=0,
                                 steps=(4.0e-4, 2.0e-4, 1.0e-4)):
    """Is a VJP-vs-FD gap the FD's fault or the VJP's?

    eps^2 truncation -> ratio ~4 per halving; roundoff -> ~0.5; a WRONG
    reverse-mode derivative -> ~1, because its error does not move with
    eps at all.  The precondition that the largest-eps gap clears the
    roundoff floor by 10x is ASSERTED, not assumed -- without it the
    ladder is in the roundoff regime and its ratio measures nothing.
    """
    r = [vjp_fd_projection(f, primals, e, seed=seed)[0] for e in steps]
    ratios = [r[i] / max(r[i + 1], 1e-300) for i in range(len(r) - 1)]
    floor0 = fd_roundoff_floor(f, primals, steps[0])
    detail = (f"{name}: |<v,J^T w> - <J_FD v,w>| at eps={steps} is "
              f"{[f'{x:.6e}' for x in r]}, halving ratios "
              f"{[f'{x:.3f}' for x in ratios]}, roundoff floor at "
              f"eps={steps[0]:g} is {floor0:.6e} (eps^2 truncation -> ~4, "
              f"roundoff -> ~0.5, wrong reverse-mode derivative -> ~1)")
    assert all(x > 0.0 for x in r), detail
    assert r[0] > 10.0 * floor0, (
        detail + " -- PRECONDITION FAILED: the largest-eps gap does not "
        "clear the roundoff floor, so this ladder is in the roundoff "
        "regime and its ratio is meaningless.  Use LARGER eps, or -- if "
        "f is affine in these operands -- an affine/roundoff gate.")
    for x in ratios:
        assert 2.5 < x < 6.0, detail
    return ratios


def assert_fd_gap_at_roundoff_floor(name, f, primals, margin, eps=1.0e-4,
                                    seed=0):
    """The independent FD check when there is NO truncation term to see.

    :func:`assert_fd_truncation_scaling` needs the eps^2 term to dominate
    the FD's own roundoff.  When it does not -- because the map is affine
    in these operands, or because its curvature is simply too small at
    the output's scale -- the ladder's ratio measures nothing, and the
    precondition there says so rather than reporting a number.

    What is still assertable, and still INDEPENDENT of the AD (it
    compares the derivative against the FUNCTION), is that the gap sits
    AT the central difference's own floor rather than anywhere above it:

        floor ~ u_mach * ||f||_inf * sqrt(n_out) / eps

    A wrong Jacobian puts the gap ABOVE that floor by whatever the error
    is, which at these scales is orders of magnitude, so the gate can
    still fail on the defect it exists for.
    """
    gap, ad = vjp_fd_projection(f, primals, eps, seed=seed)
    floor = fd_roundoff_floor(f, primals, eps)
    assert gap <= margin * floor, (
        f"{name}: FD projection gap {gap:.6e} exceeds {margin:g}x the "
        f"central-difference roundoff floor {floor:.6e} "
        f"(u_mach*||f||_inf*sqrt(n)/eps at eps={eps:g}); AD projection "
        f"<v,J^T w> = {ad:.6e}.  MEASURED gap/floor = "
        f"{gap / max(floor, 1e-300):.3f}")
    return gap, floor
