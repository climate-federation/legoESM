"""Certification of the JAX 3-D C-grid phase against the NumPy lane.

Authority: ``legoesm.core.fv3_native_cgrid_phase_3d`` is the
SPECIFICATION (hop B of
``docs/atmosphere/fv3_duo_jax_lane_strategy.md``).  The pinned Fortran is
quoted only where a line number says WHY a step exists; it is never what
a bound here is calibrated against.

The module under test contributes CADENCE, not arithmetic -- every
operation inside it happens in a kernel gated by
``test_fv3_duo_sw_core.py``, ``test_fv3_pgrad.py`` or
``test_fv3_nh_core.py``.  So the gates below interrogate the JOINS,
which is exactly what per-kernel gates cannot see:

1. **parity** -- JAX vs the NumPy twin from BYTE-IDENTICAL inputs, per
   output, with a bound MEASURED in job 9425294 (measured x 10).
   Both pressure phases are fed the SAME NumPy ``csw_phase_3d`` output
   rather than each lane's own, so the comparison is one-variable: a
   c_sw parity difference cannot leak into a pressure-phase verdict;
2. **cadence** -- perturb one LEVEL and one FACE and require only that
   level / that face to move.  This is the gate that a per-kernel suite
   structurally cannot provide, and it is the whole point of a 3-D
   assembler;
3. **jit vs eager** -- an ASSERTION, plus a trace counter proving no
   retrace on a new ``dt2`` AND (non-vacuity) that the counter DOES move
   on a new ``nord`` -- ``nord`` and not ``km``, because a new ``km``
   also changes every array SHAPE and would prove nothing about the
   static split.  **Nothing here is asserted bitwise across the
   jit boundary.** Every output of all three routines flows through a
   kernel containing ``Sum w*v``, which XLA contracts into an FMA when
   jitted and not when eager; the only pure index copies in this module
   are the state adapters and the level/face assembly, and those are
   asserted bitwise in EAGER-only gates where no such contraction site
   exists;
4. **guards** -- every entry gate, each shown NON-VACUOUS by
   monkeypatching the validator to a no-op and asserting the call then
   proceeds.  A guard whose removal changes nothing is decoration;
5. **the NH refusal** -- asserted, and shown non-vacuous the same way:
   with ``_refuse_nh_pressure`` neutered, the call RUNS and returns
   finite numbers, i.e. the refusal is the only thing standing between a
   caller and a hydrostatic-formula NH column;
6. **gradients** -- the PRIMARY gate is the tolerance-free adjoint
   identity ``<J v, w> == <v, J^T w>`` (``J v`` from ``jax.jvp``,
   ``J^T w`` from ``jax.vjp``), per operand group, with an explicit
   non-vacuity assert that ``<J v, w> != 0``.  ``check_grads`` is a
   SCOPED supplement with ``order=1`` and ``order=2`` as separate
   parametrised IDs, so an FD-resolution failure can never be confused
   with a wrong Jacobian (STATE lesson 12).

FIRST COVERAGE OF THE SPEC ITSELF.  ``cgrid_pressure_phase_3d`` and
``cgrid_nh_pressure_phase_3d`` have no caller and no test anywhere in
the repository -- ``csw_phase_3d`` is the only one of the three that was
ever exercised.  The parity gates below are therefore the first
execution of those two NumPy routines as well as of their JAX twins, and
a red gate must be triaged accordingly: the expectation, the JAX lane
and the NumPy lane are all candidates.

TOLERANCE POLICY.  ``c_sw`` runs the upwind selects and the divergence
terms; ``geopk`` is an accumulating recurrence; ``sim1_solver`` is a
sequential Thomas sweep with a ``p_fac`` floor.  Per strategy section 4
that spans the pointwise, accumulating and branch-switching classes, so
every numeric bound below is MEASURED (job 9425294, the
LEGOESM_FV3_TOL_MEASURE sweep) and set to measured x 10, keeping its
class label.  Worst figures: nh vc parity 8.594e-14, and the one gate
above 1e-9 in this file -- jit-vs-eager nh.ws3 at 1.625e-09
(cancellation-amplified FMA in ws = (zs - gz_bot)/dt).

COST.  These gates run six faces x three levels of ``c_sw`` per call, so
they are minutes, not seconds; the jvp/vjp programs add more.  That is
inherent to gating a composition and is stated here so the measurement
job's wall clock is not read as a hang.
"""
from __future__ import annotations

import functools
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402


from tests.grids.fv3_gate_helpers import gated_check_grads  # noqa: E402
from legoesm.core import fv3_cgrid_phase_3d as jphase  # noqa: E402
from legoesm.core import fv3_native_cgrid_phase_3d as npphase  # noqa: E402
from legoesm.core import fv3_native_duo_stepper as npstep  # noqa: E402
from legoesm.core.fv3_duo_stepper import (  # noqa: E402
    DuoStepperContext,
    build_jax_duo_stepper_context,
)
from legoesm.core.fv3_native_state_3d import (  # noqa: E402
    STATE_FIELDS,
    build_state_3d,
    field_shape,
)
from legoesm.core.fv3_native_sw_core import BIG_NUMBER  # noqa: E402
from legoesm.core.fv3_pgrad import geopk  # noqa: E402
from legoesm.grids.fv3_native_gridstruct import (  # noqa: E402
    FV3_CP_AIR,
    FV3_GRAV,
    FV3_KAPPA,
    FV3_RDGAS,
)

# C12 is the smallest resolution the duo corner-region Lagrange fill
# admits and is what every other JAX-lane test module uses, so the
# fixtures are comparable across the port.
N, NG = 12, 3
MA = N + 2 * NG
NPX = N + 1

# km = 3 is the only depth that clears BOTH ends of the window this
# phase runs in: `require_no_remap_needed` refuses km > 4
# (fv_dynamics.F90:568 gates Lagrangian_to_Eulerian on `npz > 4`), and
# `update_dz_c` / `sim1_solver` refuse km < 2 (the top/bottom
# extrapolation ratios read dp0[1] and dp0[km-2]).
KM = 3

# The C-grid half step.  Small enough that `delpc = delp - dt2*div`
# stays comfortably positive on this IC -- which is a PRECONDITION of
# the geopk chain, not a nicety, and is what `check_delpc` exists to
# catch when it is violated.
DT2 = 10.0

# Deck constants for the 3-D hydrostatic column.  ptop from the analytic
# coordinate `fv_eta.F90:241` uses at npz=5 (`ptop = 500e2`); this test
# runs a shallower artificial column, so 100 Pa is used and stated
# rather than implied.
PTOP = 100.0
P_FAC = 0.05
# a_imp = 1. selects SIM1, the only arm ported (nh_utils.F90:392-401);
# riem_solver_c RAISES on anything <= 0.999.
A_IMP = 1.0

# The workspace FILL CONSTANTS this lane round-trips, by VALUE: 1e30
# (c_sw's ut/vt, ptc, delpc outside the B ring; geopk's `unwritten_fill`
# default BIG_NUMBER) and 1e25 (divergence_corner_duo's divg_d init).  A
# cell is a fill only if it holds one of these EXACTLY -- a magnitude
# threshold was tried in this campaign and RETRACTED, because it also
# swept up the sentinel-PROPAGATED cascade (1e22 ... 1e111), which is
# ordinary arithmetic and therefore subject to FMA contraction.
_FILL_VALUES = (1.0e30, 1.0e25)

# geopk writes `pkz` ONLY on the D-grid call.  `dyn_core.F90:2781` gates
# the whole pkz loop on `.not. CG` (the write is at :2784), and the
# C-grid call site `:533` passes `.true.` -- so on THIS call `pkz` is
# 100 % `unwritten_fill` in BOTH lanes by construction.  That is a
# CONTRACT, not a windowing mistake and not something to compare: there
# are no non-fill cells anywhere in the array, at any origin.
# `test_pkz_is_unwritten_on_the_cgrid_call` asserts the contract instead
# (with a cg=False control proving the field IS writable), which is
# strictly stronger than dropping the field from the loop.
_CGRID_GEOPK_COMPARED = ("pk", "gz", "pe", "peln")
_CGRID_GEOPK_UNWRITTEN = ("pkz",)


# =====================================================================
# comparison helpers
#
# FOURTH COPY, and it is deliberate: pytest modules are not an import
# surface (importing one test module from another would execute its
# module-level jax config), and the campaign's three existing copies are
# private.  This one is the CORRECTED metric -- exact-value fill
# classification and a PER-ELEMENT relative difference -- adopted from
# `test_fv3_duo_sw_core._cmp` after job 9404093 retracted the
# magnitude-based classifier and the global max/max metric.
# FOLLOW-UP, already open on the stepper's copy: converge all four onto
# one shared `tests/grids/conftest.py` helper and re-measure every bound
# under it.
# =====================================================================

def _cmp(got, ref, name, tol):
    """Mask-aware, PER-ELEMENT relative comparison -- and it MUST be
    able to fail without also being able to fail spuriously.

    * NON-FINITE mask equality -- a cell the oracle never writes must
      stay a tripwire in BOTH lanes, and ``assert_array_equal`` treats
      ``NaN == NaN`` as equal, so an unchecked comparison over a
      mostly-NaN halo passes while proving nothing;
    * EXACT-FILL mask and value equality -- ``1e30`` passes every
      ``isfinite`` guard, so a drifted fill is a real defect wearing a
      finite disguise;
    * every other cell: ``|a-b| / (|b| + median|b|)``.  A robust floor
      stops one huge cell from setting the scale and dividing every
      physical discrepancy to nothing.

    Returns ``(rel, n_over)``; ``n_over`` counts cells above a
    rounding-scale reference, which is the BRANCH-FLIP signature (a
    handful far out with the rest at 1e-16 is a limiter flip, all of
    them out is something systematic).
    """
    a = np.asarray(got, dtype=np.float64)
    b = np.asarray(ref, dtype=np.float64)
    assert a.shape == b.shape, (name, a.shape, b.shape)

    na, nb = ~np.isfinite(a), ~np.isfinite(b)
    assert np.array_equal(na, nb), (
        f"{name}: non-finite masks differ (jax {int(na.sum())} vs numpy "
        f"{int(nb.sum())} cells of {a.size})")

    fa = np.zeros(a.shape, bool)
    fb = np.zeros(b.shape, bool)
    for v in _FILL_VALUES:
        fa |= (a == v)
        fb |= (b == v)
    assert np.array_equal(fa, fb), (
        f"{name}: workspace-FILL masks differ (jax {int(fa.sum())} vs "
        f"numpy {int(fb.sum())} cells of {a.size}); the fill constants "
        f"are {list(_FILL_VALUES)}")

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
        f"it through a relative comparison). Added after job 9411351, "
        f"where the NH stage's ws3 was compared 0-against-0 and the "
        f"per-element metric returned rel = 0.0 -- the same shape as "
        f"this campaign's earlier `_cmp` that returned 0.0 on an "
        f"all-non-finite pair.")
    diff = np.abs(a[ok] - b[ok])
    scale = float(np.median(np.abs(b[ok])))
    if not (scale > 0.0):
        scale = max(float(np.abs(b[ok]).max()), 1e-300)
    per = diff / (np.abs(b[ok]) + scale)
    rel = float(per.max())
    n_over = int((per > 1e-13).sum())
    # MEASUREMENT MODE (LEGOESM_FV3_TOL_MEASURE=1): print and skip ONLY
    # the tolerance assert; the structural checks above still raise.
    if os.environ.get("LEGOESM_FV3_TOL_MEASURE") == "1":
        print(f"TOLMEASURE {name!r}: rel {rel:.3e} (bound {tol:.3e}, "
              f"n_over {n_over}, max|diff| {float(diff.max()):.3e})",
              flush=True)
        return rel, n_over
    assert rel <= tol, (
        f"{name}: MEASURED per-element rel {rel:.3e} > {tol:.3e}; "
        f"{n_over} of {int(ok.sum())} compared cells exceed 1e-13 "
        f"(a handful => BRANCH FLIP, all of them => systematic); "
        f"median|ref| {scale:.3e}, max|diff| {float(diff.max()):.3e}, "
        f"bitwise={np.array_equal(a[ok], b[ok])}")
    return rel, n_over


def _tree_dot(a, b) -> float:
    """Plain inner product -- NO ``nan_to_num``.

    Sanitising here would silently repair a NaN the adjoint identity
    exists to EXPOSE (an R1b dead-branch leak shows up as exactly
    that), so a non-finite leaf must propagate to the assertion in
    :func:`_check_adjoint` and name itself there.
    """
    la = jax.tree_util.tree_leaves(a)
    lb = jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb), (len(la), len(lb))
    return float(sum(
        np.dot(np.asarray(x, dtype=np.float64).ravel(),
               np.asarray(y, dtype=np.float64).ravel())
        for x, y in zip(la, lb)))


def _adjoint_residual(f, primals, seed=0):
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
            f"J v leaf {i} is not finite -- the objective window "
            f"includes cells the phase never writes, or a dead branch is "
            f"leaking a NaN into the gradient (R1b)")
    _, vjp_fn = jax.vjp(f, *primals)
    w = jax.tree_util.tree_map(
        lambda x: jnp.asarray(rng.standard_normal(x.shape)), jv)
    jtw = vjp_fn(w)
    lhs = _tree_dot(jv, w)
    rhs = _tree_dot(v, jtw)
    return abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300), lhs, rhs


def _check_adjoint(name, f, primals, tol, seed=0):
    r, lhs, rhs = _adjoint_residual(f, primals, seed=seed)
    assert abs(lhs) > 0.0, (
        f"{name}: <J v, w> == 0 -- the identity is satisfied trivially, "
        f"so this gate proves nothing (check the window/scale)")
    assert np.isfinite(lhs) and np.isfinite(rhs), (
        f"{name}: the inner products are not finite ({lhs}, {rhs})")
    if os.environ.get("LEGOESM_FV3_TOL_MEASURE") == "1":
        print(f"TOLMEASURE {name!r}: rel {r:.3e} (bound {tol:.3e}, "
              f"quantity adjoint identity residual)", flush=True)
        return r
    assert r <= tol, (
        f"{name}: adjoint residual {r:.3e} > {tol:.3e} "
        f"(<J v, w>={lhs:.12e}, <v, J^T w>={rhs:.12e}) -- MEASURED "
        f"value is {r:.3e}")
    return r


def _counted(fn):
    """(traced-call counter, wrapper) for the retrace assertions.

    ``functools.wraps`` is load-bearing, not cosmetic: ``jax.jit``
    resolves ``static_argnums``/``static_argnames`` against
    ``inspect.signature(fun)`` (``jax/_src/api_util.py:443``
    ``resolve_argnums``), and a bare ``(*a, **k)`` wrapper would hand it
    a signature that does not contain ``km`` or ``nord``.  ``wraps``
    sets ``__wrapped__``, which ``inspect.signature`` follows.
    """
    box = {"n": 0}

    @functools.wraps(fn)
    def wrapper(*a, **k):
        box["n"] += 1
        return fn(*a, **k)

    return box, wrapper


# The strict COMPUTE square, is..ie x js..je, in storage indices: the
# NumPy lane puts Fortran (isd, jsd) = (1-ng, 1-ng) at storage [0, 0],
# so Fortran i = is = 1 sits at index ng.  Used for every gradient
# OBJECTIVE, because the halo legitimately carries fills and NaN
# tripwires and an objective that touches them is not differentiable.
# Exclude by WINDOW, never by a nan-aware reduction.
_CS = slice(NG, NG + N)


def _win(a):
    """Compute square of a face-stacked ``(6, i, j, k)`` array."""
    return a[:, _CS, _CS, :]


def _deepcopy_faces(per_face: list) -> list:
    """Six per-face dicts with every array COPIED.

    Required, not tidiness: the NumPy pressure phases MUTATE
    ``csw_outs[t]["uc"]``/``["vc"]`` (p_grad_c's oracle contract) and
    the NH one also mutates ``gz6``/``ws3_6``, so calling them on a
    module-scoped fixture would silently corrupt every later test.
    """
    return [{k: np.array(v, copy=True) for k, v in d.items()}
            for d in per_face]


def _stack_np(per_face: list) -> dict:
    """Six NumPy per-face dicts -> the face-stacked JAX container."""
    keys = tuple(per_face[0])
    return {k: jnp.asarray(np.stack([np.asarray(d[k], dtype=np.float64)
                                     for d in per_face]))
            for k in keys}


# =====================================================================
# fixtures
# =====================================================================

@pytest.fixture(scope="module")
def ctx():
    """The NumPy context, EXT-BUNDLE lane, ORACLE conventions.

    ``oracle_conventions=True`` is the BOUNDED-conventions gridstruct
    the Zenodo duo runs execute; ``duogrid`` forces it
    (``fv_arrays.F90:1512``) and ``c_sw`` refuses the combination
    without it.  ``use_ext_bundle=True`` because
    ``build_jax_duo_stepper_context`` implements the faithful exchange
    lane only.
    """
    return npstep.build_six_face_duo_context(
        N, NG, use_ext_bundle=True, oracle_conventions=True)


@pytest.fixture(scope="module")
def jctx(ctx):
    return build_jax_duo_stepper_context(ctx)


# A smooth GLOBAL surface for the NH fixtures, 200-600 m, strictly
# positive and non-uniform on EVERY face.  Deliberately not
# Williamson-5's cone: W5 is exactly zero outside its cone
# (`test_cases.F90:1181-1189` takes `min(r0**2, r**2)`), so five of the
# six faces would carry a UNIFORM surface and the ws3 gate would be
# right back where job 9411351 found it.
_TOPO_H0_M = 400.0


def _topo(lon, lat):
    """``topo_fn(lon, lat) -> phis`` [m^2/s^2], the ctx builder's
    contract (`fv3_native_duo_stepper.py:265-274`); it is evaluated on
    the full data domain and then ext-exchanged, so the halo ring
    `update_dz_c` reads carries k2e-consistent values."""
    return FV3_GRAV * _TOPO_H0_M * (1.0 + 0.5 * np.sin(lat)
                                    * np.cos(lon))


@pytest.fixture(scope="module")
def ctx_topo():
    """A SECOND context, identical except for real topography.

    Separate rather than shared ON PURPOSE: the hydrostatic gates are
    green on the flat-`hs` context and re-using one context for both
    would change their fixture at the same time as this one, which is
    the confound the controlled-comparison rule exists to stop.  The
    cost is one extra halo-table build at module scope.
    """
    return npstep.build_six_face_duo_context(
        N, NG, use_ext_bundle=True, oracle_conventions=True,
        topo_fn=_topo)


@pytest.fixture(scope="module")
def jctx_topo(ctx_topo):
    return build_jax_duo_stepper_context(ctx_topo)


def _seeded_state(km, seed=0, hydrostatic=True):
    """A PHYSICAL 3-D column, distinct per face AND per level.

    Magnitudes matter and are not decoration: ``delp ~ 1e4 Pa`` and
    ``pt ~ 280 K`` keep ``delpc = delp - dt2*div`` positive (the geopk
    log's precondition) and keep the hydrostatic ``dz`` build below
    physical.  The NumPy spec's own test uses ``delp ~ 1``, which is
    fine for a c_sw-only cadence check and would put this file's
    pressure chain straight into the ``check_delpc`` guard.

    Distinct values per face and per level so that a face mix-up or a
    level mix-up cannot hide behind symmetry.
    """
    rng = np.random.default_rng(seed)
    st = build_state_3d(N, NG, km, hydrostatic=hydrostatic)
    for t, face in enumerate(st):
        for k in range(km):
            face["delp"][:, :, k] = (
                1.0e4 * (1.0 + 0.02 * t + 0.05 * k)
                + 300.0 * rng.standard_normal((MA, MA)))
            face["pt"][:, :, k] = (
                280.0 + 2.0 * t + 5.0 * k
                + 3.0 * rng.standard_normal((MA, MA)))
            face["w"][:, :, k] = 0.1 * rng.standard_normal((MA, MA))
            face["u"][:, :, k] = (
                1.0 + 0.1 * t + rng.standard_normal(face["u"].shape[:2]))
            face["v"][:, :, k] = (
                -1.0 + 0.1 * t + rng.standard_normal(face["v"].shape[:2]))
        # positive layer mass everywhere, including the halo the
        # stencils read
        face["delp"][:] = np.abs(face["delp"]) + 1.0e3
    return st


@pytest.fixture(scope="module")
def state_np():
    return _seeded_state(KM, seed=11)


@pytest.fixture(scope="module")
def jstate(state_np):
    return jphase.state_3d_to_jax(state_np)


@pytest.fixture(scope="module")
def csw_np(ctx, state_np):
    """The NumPy lane's c_sw phase output -- the SHARED input to both
    lanes' pressure phases, so those gates are one-variable."""
    return npphase.csw_phase_3d(ctx, _deepcopy_faces(state_np),
                                dt2=DT2, km=KM, nord=2, duogrid=True)


@pytest.fixture(scope="module")
def csw_np_nh(ctx_topo, state_np):
    """Built on the TOPOGRAPHY context so the whole NH chain -- c_sw,
    update_dz_c, Riem_Solver_C, p_grad_c -- runs on one gridstruct."""
    return npphase.csw_phase_3d(ctx_topo, _deepcopy_faces(state_np),
                                dt2=DT2, km=KM, nord=2, duogrid=True,
                                hydrostatic=False)


def _nh_column(ctx_topo, km=KM, seed=23):
    """(dp0, zs6, gz6, ws3_6) -- a hydrostatically consistent height
    column standing on the ctx's REAL surface.

    ⛔ EARNED, job 9411351.  The first version put a FLAT surface at
    ``z = 0`` on every face, which made ``ws3`` come out identically
    zero -- and not by a small margin, but EXACTLY, for any wind.  The
    reason is structural, not a tuning miss: ``update_dz_c``'s bottom
    interface is updated in FLUX FORM,
    ``gz_new = (gz*area + div(F)) / (area + div(u))`` with
    ``F = u * gz_upwind`` (`nh_utils.F90:166-171`).  With
    ``gz[:, :, km] == 0`` everywhere, every upwind donor is 0, so the
    numerator is 0 and ``gz_new[km] = 0`` exactly; then
    ``ws = (zs - gz_new[km])/dt = 0``.  A control that perturbs a zero
    is not a control.

    The fix mirrors what the certified ``update_dz_c`` fixtures do
    (`update_dz_c_zero_wind_certificate`, which builds a spatially
    VARYING gz and gets ws ~ 0 only because its WIND is zero): give the
    surface real structure.  ``zs = hs / g`` keeps the two arguments the
    NH stage takes from different places -- ``zs`` (height) to
    ``update_dz_c`` and ``hs`` (geopotential) to ``Riem_Solver_C`` --
    consistent by construction.

    ``dz`` is the exact inverse of the solver's EOS at zero
    perturbation, which is what the certified ``riem_solver_c`` fixtures
    use; an arbitrary monotone column would put sim1 far from the
    balanced state it is gated on.
    """
    rng = np.random.default_rng(seed)
    gama = 1.0 / (1.0 - FV3_KAPPA)
    hs6 = ctx_topo["hs6"]
    assert hs6 is not None, (
        "ctx_topo carries no hs6 -- build it with topo_fn, or this "
        "fixture is back to the flat zero surface that made ws3 "
        "identically zero")
    gz6, ws6, zs6 = [], [], []
    for _t in range(6):
        zs = np.asarray(hs6[_t], dtype=np.float64) / FV3_GRAV
        delp = np.abs(1.0e4 + 300.0 * rng.standard_normal((MA, MA, km)))
        pt = 280.0 + 5.0 * rng.standard_normal((MA, MA, km))
        pem = np.zeros((MA, MA, km + 1))
        pem[:, :, 0] = PTOP
        for k in range(km):
            pem[:, :, k + 1] = pem[:, :, k] + delp[:, :, k]
        pm = np.empty((MA, MA, km))
        for k in range(km):
            pm[:, :, k] = delp[:, :, k] / np.log(pem[:, :, k + 1]
                                                 / pem[:, :, k])
        dzh = -(delp / FV3_GRAV) * FV3_RDGAS * pt / np.exp(
            np.log(pm) / gama)
        gz = np.zeros((MA, MA, km + 1))
        gz[:, :, km] = zs          # the column STANDS on the surface
        for k in range(km - 1, -1, -1):
            gz[:, :, k] = gz[:, :, k + 1] - dzh[:, :, k]
        gz6.append(gz)
        # ws3 enters as the workspace update_dz_c FILLS; zero in, and
        # the gate below requires it to be non-zero out.
        ws6.append(np.zeros((MA, MA)))
        zs6.append(zs)
    dp0 = np.full(km, 1.0e4)
    return dp0, zs6, gz6, ws6


@pytest.fixture(scope="module")
def nh_inputs(ctx_topo):
    return _nh_column(ctx_topo)


def test_nh_fixture_surface_is_not_uniform(nh_inputs):
    """The precondition the ws3 gate rests on, asserted rather than
    assumed: a face whose ``zs`` is CONSTANT cannot produce a non-zero
    ``ws3``, because the flux-form bottom update of a constant field
    reproduces that constant exactly.  This is the check that would have
    caught the flat-zero fixture before it certified anything."""
    _dp0, zs6, gz6, _ws6 = nh_inputs
    for t in range(6):
        spread = float(np.ptp(zs6[t]))
        assert spread > 1.0, (
            f"face {t + 1}: zs varies by only {spread:.3e} m -- the ws3 "
            f"gate would be measuring a constant field")
        assert np.array_equal(gz6[t][:, :, KM], zs6[t]), (
            f"face {t + 1}: the column does not stand on its own "
            f"surface, so ws3 would start from a spurious offset")


# =====================================================================
# A. layout, adapters, tier 0
# =====================================================================

def test_state_round_trip_is_bitwise(state_np):
    """C1's PyTree contract, pinned as strategy section 7a requires.

    BITWISE is legitimate here and stated as such: both adapters are
    pure index copies (``np.stack`` / a per-face slice), so there is no
    ``x*y + z`` anywhere for XLA to contract into an FMA.  This is the
    ONLY class of gate in this file allowed to assert bitwise.
    """
    back = jphase.state_3d_to_numpy(jphase.state_3d_to_jax(state_np))
    assert len(back) == 6
    for t in range(6):
        assert set(back[t]) == set(state_np[t])
        for k in state_np[t]:
            np.testing.assert_array_equal(
                back[t][k], state_np[t][k],
                err_msg=f"face {t + 1} {k}: round trip is not an identity")


def test_state_3d_to_jax_declares_every_shape(jstate):
    for name in STATE_FIELDS:
        assert jstate[name].shape == (6,) + field_shape(name, N, NG, KM)
        assert jstate[name].dtype == jnp.float64


def _f32_pt(state_np):
    """An f32 ``pt`` on EVERY face.

    Demoting one face only would not test what it looks like it tests:
    ``stack6`` runs ``np.stack``, which PROMOTES a mixed f32/f64 list
    back to f64, and the gate would never see the f32 at all.
    """
    bad = _deepcopy_faces(state_np)
    for face in bad:
        face["pt"] = face["pt"].astype(np.float32)
    return bad


def test_state_3d_to_jax_refuses_f32(state_np):
    with pytest.raises(TypeError, match="float64"):
        jphase.state_3d_to_jax(_f32_pt(state_np))


def test_f64_gate_is_non_vacuous(state_np, monkeypatch):
    """Neutering the gate must let the f32 state through -- otherwise
    the test above is passing for some other reason."""
    monkeypatch.setattr(jphase, "require_f64_jax", lambda *a, **k: None)
    out = jphase.state_3d_to_jax(_f32_pt(state_np))
    assert out["pt"].dtype == jnp.float32


def test_state_3d_to_jax_refuses_a_per_face_key_split(state_np):
    bad = _deepcopy_faces(state_np)
    del bad[3]["w"]
    with pytest.raises(KeyError, match="face 4"):
        jphase.state_3d_to_jax(bad)


def test_state_3d_to_numpy_refuses_the_wrong_container(state_np):
    with pytest.raises(TypeError, match="face-stacked dict"):
        jphase.state_3d_to_numpy(state_np)


def test_csw_out_like_matches_the_spec():
    """Drift gate for the ONE table that had to be re-stated (C7).

    The spec's ``_out_like`` is private, so it cannot be imported by the
    module under test (the empty-allowlist ratchet
    ``tests/test_no_private_cross_imports.py``); a test may read it, and
    that is what keeps the two copies from diverging silently.
    """
    for name in jphase.CSW_OUT_LIKE:
        assert jphase.CSW_OUT_LIKE[name] == npphase._out_like(name), name
    for name in npphase.CSW_OUT_2D + ("wc",):
        assert name in jphase.CSW_OUT_LIKE, (
            f"{name} is a c_sw output the JAX shape table does not map")


def test_eval_shape_declares_the_csw_layout(jctx, jstate):
    """Tier 0 -- shapes and dtypes without running anything expensive."""
    out = jax.eval_shape(
        lambda st: jphase.csw_phase_3d(jctx, st, DT2, KM), jstate)
    assert set(out) == set(npphase.CSW_OUT_2D)
    for name, spec in out.items():
        want = (6,) + field_shape(jphase.CSW_OUT_LIKE[name], N, NG, KM)
        assert spec.shape == want, (name, spec.shape, want)
        assert spec.dtype == jnp.float64, name


def test_eval_shape_nh_adds_wc(jctx, jstate):
    out = jax.eval_shape(
        lambda st: jphase.csw_phase_3d(jctx, st, DT2, KM,
                                       hydrostatic=False), jstate)
    assert "wc" in out
    assert out["wc"].shape == (6,) + field_shape("w", N, NG, KM)


# =====================================================================
# B. parity vs the NumPy lane (tier 1)
# =====================================================================

@pytest.mark.parametrize("hydrostatic", [True, False],
                         ids=["hydro", "nh"])
def test_csw_phase_3d_matches_numpy_lane(ctx, jctx, state_np, jstate,
                                         hydrostatic):
    """gate 1 -- byte-identical inputs, per output, whole padded array.

    The comparison covers the HALO as well as the compute window: the
    fills and NaN tripwires a lane carries there are part of its
    contract, and ``_cmp``'s mask checks are what make that assertable
    instead of vacuous.
    """
    ref = npphase.csw_phase_3d(ctx, _deepcopy_faces(state_np), dt2=DT2,
                               km=KM, nord=2, duogrid=True,
                               hydrostatic=hydrostatic)
    got = jphase.csw_phase_3d(jctx, jstate, DT2, KM, nord=2,
                              duogrid=True, hydrostatic=hydrostatic)
    names = npphase.CSW_OUT_2D + (() if hydrostatic else ("wc",))
    assert set(got) == set(names)
    for name in names:
        for t in range(6):
            # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), every field/face/arm; bound =
            # 1e-15 eps guard (measured exactly 0.0).
            _cmp(got[name][t], ref[t][name],
                 f"csw_phase_3d.{name}[face {t + 1}]", 1e-15)


def test_csw_phase_3d_km1_equals_the_certified_2d_kernel(jctx):
    """The assembler must add CADENCE, not change math.

    BITWISE, and eager: the level assembly is ``jnp.stack`` of the very
    array the direct call returns, i.e. a pure index copy of identical
    values, with no contraction site between them.  Run outside ``jit``
    so no fusion can reassociate either side.
    """
    from legoesm.core.fv3_duo_sw_core import c_sw

    st = _seeded_state(1, seed=3)
    jst = jphase.state_3d_to_jax(st)
    outs = jphase.csw_phase_3d(jctx, jst, 7.5, 1, nord=2, duogrid=True)
    for t in range(6):
        direct = c_sw(jst["delp"][t][:, :, 0], jst["pt"][t][:, :, 0],
                      jst["w"][t][:, :, 0], jst["u"][t][:, :, 0],
                      jst["v"][t][:, :, 0], jctx.gs6[t], jctx.bd,
                      NPX, NPX, 7.5, nord=2, duogrid=True,
                      bounded_domain=jctx.flags6[t].bounded_domain)
        for name in npphase.CSW_OUT_2D:
            got = np.asarray(outs[name][t][:, :, 0])
            want = np.asarray(direct[name])
            # Non-vacuity: assert_array_equal treats NaN == NaN as
            # equal, so require the compute square to be FINITE before
            # trusting an equality over the padded array.
            assert np.isfinite(want[_CS, _CS]).all(), (
                f"face {t + 1} {name}: reference is non-finite in the "
                f"compute square; the equality below would be vacuous")
            np.testing.assert_array_equal(
                got, want,
                err_msg=f"face {t + 1} {name}: the 3-D path diverged "
                        f"from the certified 2-D kernel at km=1")


def test_each_level_is_computed_from_its_own_level_only(jctx, jstate):
    """THE cadence gate. Perturb ONE level of the input; only that level
    of the output may move.  If the k loop leaked, or a 2-D kernel
    broadcast a column, other levels would change too."""
    base = jphase.csw_phase_3d(jctx, jstate, DT2, KM)
    pert = dict(jstate)
    pert["delp"] = jstate["delp"].at[0, :, :, 1].add(50.0)
    got = jphase.csw_phase_3d(jctx, pert, DT2, KM)

    d = np.abs(np.asarray(_win(got["delpc"])[0])
               - np.asarray(_win(base["delpc"])[0]))
    assert d[:, :, 1].max() > 1e-9, "level 1 must respond"
    assert d[:, :, 0].max() == 0.0, "level 0 must be untouched"
    assert d[:, :, 2].max() == 0.0, "level 2 must be untouched"


def test_perturbing_one_face_does_not_change_another(jctx, jstate):
    """c_sw is face-local; cross-face coupling only enters at the halo
    exchanges (``dyn_core.F90:652``/``:655``), which are the next
    unit."""
    base = jphase.csw_phase_3d(jctx, jstate, DT2, KM)
    pert = dict(jstate)
    pert["pt"] = jstate["pt"].at[2, :, :, 0].add(1.0)
    got = jphase.csw_phase_3d(jctx, pert, DT2, KM)
    assert np.abs(np.asarray(_win(got["ptc"])[2])
                  - np.asarray(_win(base["ptc"])[2])).max() > 1e-9
    for t in (0, 1, 3, 4, 5):
        np.testing.assert_array_equal(
            np.asarray(_win(got["ptc"])[t]),
            np.asarray(_win(base["ptc"])[t]),
            err_msg=f"face {t + 1} moved when face 3 was perturbed")


def test_cgrid_pressure_phase_3d_matches_numpy_lane(ctx, jctx, csw_np):
    """gate 1 for the hydrostatic pressure chain, FED THE SAME INPUT.

    Both lanes consume the NumPy ``csw_phase_3d`` output, so this is a
    one-variable test of the pressure phase: a c_sw parity difference
    cannot leak in and be read as a geopk/p_grad_c defect.

    NOTE this is also the FIRST execution of the NumPy
    ``cgrid_pressure_phase_3d`` anywhere -- it has no other caller and
    no other test.
    """
    ref_in = _deepcopy_faces(csw_np)
    ref = npphase.cgrid_pressure_phase_3d(
        ctx, ref_in, KM, dt2=DT2, ptop=PTOP, akap=FV3_KAPPA,
        cp_air=FV3_CP_AIR, a2b_ord=4)
    got = jphase.cgrid_pressure_phase_3d(
        jctx, _stack_np(csw_np), KM, dt2=DT2, ptop=PTOP,
        akap=FV3_KAPPA, cp_air=FV3_CP_AIR, a2b_ord=4)

    assert set(got) == {"pk", "gz", "pe", "peln", "pkz", "uc", "vc"}
    for t in range(6):
        for name in _CGRID_GEOPK_COMPARED:
            # MEASURED (job 9425294 sweep): worst gz 1.188e-15 (pk ~1.1e-16, pe/peln 0.0); bound = measured x 10 =
            # 1.2e-14.
            _cmp(got[name][t], ref[t][name],
                 f"cgrid_pressure.{name}[face {t + 1}]", 1.2e-14)
        for name in ("uc", "vc"):
            # p_grad_c MUTATES uc/vc in place in the NumPy lane
            # (dyn_core.F90:2073-2132); the JAX twin RETURNS them (R4 /
            # C4), so the reference is the MUTATED input array.
            # MEASURED (job 9425294 sweep): worst vc (face 5) 4.267e-14; bound = measured x 10 =
            # 4.3e-13.
            _cmp(got[name][t], ref_in[t][name],
                 f"cgrid_pressure.{name}[face {t + 1}]", 4.3e-13)


def test_pkz_is_unwritten_on_the_cgrid_call(ctx, jctx, csw_np):
    """``pkz`` is NOT a comparable output of the C-grid geopk call.

    ``dyn_core.F90:2781`` opens the pkz block with
    ``if ( .not. CG .and. j .ge. js .and. j .le. je )`` and the write is
    at ``:2784``; the C-grid call site ``:533`` passes ``.true.``.  So
    the oracle never writes pkz here, and both lanes leave the whole
    array at ``unwritten_fill``.  This asserts that CONTRACT -- which
    goes red the moment either lane starts writing it, whereas simply
    dropping the field from the parity loop would go quiet forever.

    The ``cg=False`` control at the end is what makes this non-vacuous:
    it proves the field IS writable by this same kernel on this same
    state, so the all-fill result above is about ``CG`` and not about
    pkz being structurally impossible to fill.
    """
    kw = dict(dt2=DT2, ptop=PTOP, akap=FV3_KAPPA, cp_air=FV3_CP_AIR)
    ref = npphase.cgrid_pressure_phase_3d(ctx, _deepcopy_faces(csw_np),
                                          KM, a2b_ord=4, **kw)
    got = jphase.cgrid_pressure_phase_3d(jctx, _stack_np(csw_np), KM,
                                         a2b_ord=4, **kw)
    for t in range(6):
        assert np.all(np.asarray(got["pkz"][t]) == BIG_NUMBER), (
            f"face {t + 1}: the JAX lane WROTE pkz on a CG call; "
            f"dyn_core.F90:2781 gates that block on `.not. CG`")
        assert np.all(np.asarray(ref[t]["pkz"]) == BIG_NUMBER), (
            f"face {t + 1}: the NumPy lane WROTE pkz on a CG call")

    # CONTROL: the same kernel, the same state, cg=False -- pkz must
    # now be written, or the assertions above are about the wrong thing.
    d_out = geopk(jnp.asarray(csw_np[0]["delpc"]),
                  jnp.asarray(csw_np[0]["ptc"]), jctx.hs6[0], ctx["bd"],
                  km=KM, ptop=PTOP, akap=FV3_KAPPA, cp_air=FV3_CP_AIR,
                  cg=False, duogrid=True, computehalo=False,
                  npx=NPX, npy=NPX, a2b_ord=4, bounded_domain=False,
                  sw_dynamics=False)
    written = np.asarray(d_out["pkz"]) != BIG_NUMBER
    assert written.all(), (
        "the cg=False control did not fill pkz either, so the CG "
        "attribution above is unproven")


def test_cgrid_pressure_phase_3d_actually_moves_uc(jctx, csw_np):
    """Non-vacuity for the gate above: if ``p_grad_c`` never wrote,
    comparing the returned ``uc`` to the input would pass trivially."""
    got = jphase.cgrid_pressure_phase_3d(
        jctx, _stack_np(csw_np), KM, dt2=DT2, ptop=PTOP,
        akap=FV3_KAPPA, cp_air=FV3_CP_AIR)
    before = np.asarray(_stack_np(csw_np)["uc"])
    after = np.asarray(got["uc"])
    moved = np.abs(after[:, _CS, _CS, :] - before[:, _CS, _CS, :]).max()
    assert moved > 0.0, (
        "p_grad_c returned uc unchanged in the compute square -- the "
        "parity gate on uc would then be comparing an input to itself")


def test_cgrid_nh_pressure_phase_3d_matches_numpy_lane(ctx_topo,
                                                       jctx_topo,
                                                       csw_np_nh,
                                                       nh_inputs):
    """gate 1 for the NH pressure chain, FED THE SAME INPUT.

    Also the FIRST execution of the NumPy
    ``cgrid_nh_pressure_phase_3d`` anywhere.  The NumPy lane mutates
    ``gz6``/``ws3_6``/``uc``/``vc`` in place, so each lane gets its own
    copies and the reference for a mutated field is the mutated array
    (R4 / C4).
    """
    dp0, zs6, gz6, ws6 = nh_inputs
    ref_in = _deepcopy_faces(csw_np_nh)
    gz_ref = [np.array(g, copy=True) for g in gz6]
    ws_ref = [np.array(w, copy=True) for w in ws6]
    ref = npphase.cgrid_nh_pressure_phase_3d(
        ctx_topo, ref_in, gz_ref, ws_ref, KM, dt2=DT2, ptop=PTOP,
        akap=FV3_KAPPA, cp_air=FV3_CP_AIR, p_fac=P_FAC, a_imp=A_IMP,
        dp0=dp0, hs6=ctx_topo["hs6"], zs6=zs6)

    got = jphase.cgrid_nh_pressure_phase_3d(
        jctx_topo, _stack_np(csw_np_nh), jnp.asarray(np.stack(gz6)),
        jnp.asarray(np.stack(ws6)), KM, dt2=DT2, ptop=PTOP,
        akap=FV3_KAPPA, cp_air=FV3_CP_AIR, p_fac=P_FAC, a_imp=A_IMP,
        dp0=jnp.asarray(dp0), zs6=jnp.asarray(np.stack(zs6)))

    assert set(got) == {"pkc", "gz", "ws3", "uc", "vc"}
    for t in range(6):
        # MEASURED (job 9425294 sweep): worst 4.203e-16 (face 6); bound = measured x 10 =
        # 4.3e-15.
        _cmp(got["pkc"][t], ref[t]["pkc"],
             f"cgrid_nh.pkc[face {t + 1}]", 4.3e-15)
        # MEASURED (job 9425294 sweep): worst 1.661e-15 (face 6); bound = measured x 10 =
        # 1.7e-14.
        _cmp(got["gz"][t], gz_ref[t], f"cgrid_nh.gz[face {t + 1}]", 1.7e-14)
        # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), every face; bound =
        # 1e-15 eps guard (measured exactly 0.0).
        _cmp(got["ws3"][t], ws_ref[t],
             f"cgrid_nh.ws3[face {t + 1}]", 1e-15)
        for name in ("uc", "vc"):
            # MEASURED (job 9425294 sweep): worst vc (face 6) 8.594e-14; bound = measured x 10 =
            # 8.6e-13.
            _cmp(got[name][t], ref_in[t][name],
                 f"cgrid_nh.{name}[face {t + 1}]", 8.6e-13)


def test_nh_stage_rebuilds_gz_and_fills_ws(jctx_topo, csw_np_nh,
                                           nh_inputs):
    """Non-vacuity for the NH parity gate: ``gz`` must LEAVE different
    from how it arrived (height in, geopotential out) and ``ws3`` must
    stop being all zeros, or those two comparisons are input-to-input.

    PER FACE, not globally.  A global ``max > 0`` passes as soon as ONE
    face responds, which is exactly how a surface that is flat on five
    faces would slip through -- the failure this gate has already had
    once (job 9411351), in its stronger form.
    """
    dp0, zs6, gz6, ws6 = nh_inputs
    got = jphase.cgrid_nh_pressure_phase_3d(
        jctx_topo, _stack_np(csw_np_nh), jnp.asarray(np.stack(gz6)),
        jnp.asarray(np.stack(ws6)), KM, dt2=DT2, ptop=PTOP,
        akap=FV3_KAPPA, cp_air=FV3_CP_AIR, p_fac=P_FAC, a_imp=A_IMP,
        dp0=jnp.asarray(dp0), zs6=jnp.asarray(np.stack(zs6)))
    gz_in = np.stack(gz6)
    gz_out = np.asarray(got["gz"])
    ws_out = np.asarray(got["ws3"])
    for t in range(6):
        moved = np.abs(gz_out[t, _CS, _CS, :]
                       - gz_in[t, _CS, _CS, :]).max()
        assert moved > 0.0, (
            f"face {t + 1}: gz came back unchanged -- the NH stage is "
            f"dead on this face and the gz parity comparison is "
            f"input-against-input")
        wmax = np.abs(ws_out[t, _CS, _CS]).max()
        assert wmax > 0.0, (
            f"face {t + 1}: ws3 is identically zero. update_dz_c's "
            f"bottom update is FLUX FORM (nh_utils.F90:166-171), so a "
            f"UNIFORM surface reproduces itself exactly and ws = "
            f"(zs - gz[km])/dt is structurally 0 for any wind -- check "
            f"that this face's zs actually varies")


# =====================================================================
# C. jit vs eager, and the retrace policy (tier 2)
# =====================================================================

def test_csw_phase_3d_jit_matches_eager(jctx, jstate):
    """gate 3 -- an ASSERTION, not a comment.

    NOT bitwise, and the reason is structural rather than empirical:
    ``c_sw`` computes flux divergences and KE as ``Sum w*v``, and XLA
    contracts mul+add into an FMA in the jitted lowering and not in the
    eager one, so a few-ULP gap is correct behaviour.
    """
    fn = jphase.make_csw_phase_3d_jit()
    eager = jphase.csw_phase_3d(jctx, jstate, DT2, KM)
    fast = fn(jctx, jstate, DT2, KM)
    for name in npphase.CSW_OUT_2D:
        # MEASURED (job 9425294 sweep): worst divg_d 1.323e-15; bound = measured x 10 =
        # 1.4e-14.
        _cmp(fast[name], eager[name], f"jit-vs-eager csw.{name}", 1.4e-14)


def test_cgrid_pressure_phase_3d_jit_matches_eager(jctx, csw_np):
    """``check_delpc=False`` is REQUIRED on the compiled path and is
    passed explicitly (C5) -- the default would raise, by design."""
    fn = jphase.make_cgrid_pressure_phase_3d_jit()
    kw = dict(dt2=DT2, ptop=PTOP, akap=FV3_KAPPA, cp_air=FV3_CP_AIR)
    stacked = _stack_np(csw_np)
    eager = jphase.cgrid_pressure_phase_3d(jctx, stacked, KM,
                                           check_delpc=False, **kw)
    fast = fn(jctx, stacked, KM, check_delpc=False, **kw)
    for name in eager:
        if name in _CGRID_GEOPK_UNWRITTEN:
            # 100 % unwritten_fill on a CG call (dyn_core.F90:2781);
            # `_cmp` correctly refuses an all-fill pair, and the
            # contract is asserted by
            # test_pkz_is_unwritten_on_the_cgrid_call.  Compilation
            # cannot change a constant fill, so there is nothing for a
            # jit-vs-eager gate to say about it.
            assert np.array_equal(np.asarray(fast[name]),
                                  np.asarray(eager[name])), name
            continue
        # MEASURED (job 9425294 sweep): worst vc 2.283e-15; bound = measured x 10 =
        # 2.3e-14.
        _cmp(fast[name], eager[name], f"jit-vs-eager pressure.{name}",
             2.3e-14)


def test_cgrid_nh_pressure_phase_3d_jit_matches_eager(jctx_topo,
                                                      csw_np_nh,
                                                      nh_inputs):
    dp0, zs6, gz6, ws6 = nh_inputs
    fn = jphase.make_cgrid_nh_pressure_phase_3d_jit()
    args = (jctx_topo, _stack_np(csw_np_nh), jnp.asarray(np.stack(gz6)),
            jnp.asarray(np.stack(ws6)), KM)
    kw = dict(dt2=DT2, ptop=PTOP, akap=FV3_KAPPA, cp_air=FV3_CP_AIR,
              p_fac=P_FAC, a_imp=A_IMP, dp0=jnp.asarray(dp0),
              zs6=jnp.asarray(np.stack(zs6)))
    eager = jphase.cgrid_nh_pressure_phase_3d(*args, **kw)
    fast = fn(*args, **kw)
    for name in eager:
        # MEASURED (job 9425294 sweep): worst ws3 1.625e-09, 265
        # cells, max|diff| 1.137e-14 (uc/vc <= 6.0e-14, others <=
        # 2.2e-15); bound = measured x 10 = 1.7e-08. The only cgrid
        # gate above 1e-9. UNEXPLAINED (codex MAJOR: the first version
        # of this comment presented a mechanism as established).
        # PLAUSIBLE mechanism only: ws = (zs - gz_bot)/dt is a
        # difference of nearly equal terms, so FMA contraction under
        # jit would be amplified by the cancellation -- but no stage
        # split or contraction-control experiment has been run, and a
        # reader picking this up starts from that measurement, not from
        # a conclusion.
        _cmp(fast[name], eager[name], f"jit-vs-eager nh.{name}", 1.7e-8)


def test_a_new_dt2_does_not_retrace_but_a_new_nord_does(jctx, jstate):
    """C3, asserted BOTH ways.

    ``dt2`` dynamic is the whole reason a production time loop does not
    recompile; the non-vacuity half is that the counter DOES move when a
    genuinely static argument changes, otherwise the first assertion
    could be satisfied by a counter that never increments.

    ``nord`` is the lever rather than ``km`` ON PURPOSE: a different
    ``km`` also changes every array SHAPE, so a retrace there would be
    attributable to the shapes and would prove nothing about the static
    split.  ``nord`` leaves every shape identical.
    """
    box, wrapped = _counted(jphase.csw_phase_3d)
    fn = jphase.make_csw_phase_3d_jit(wrapped)
    fn(jctx, jstate, 5.0, KM, nord=2)
    assert box["n"] == 1
    fn(jctx, jstate, 9.0, KM, nord=2)
    assert box["n"] == 1, "a new dt2 must NOT retrace (C3)"
    fn(jctx, jstate, 9.0, KM, nord=1)
    assert box["n"] == 2, "a new nord MUST retrace (it is static)"


# =====================================================================
# D. guards -- each asserted AND shown non-vacuous
# =====================================================================

def test_km_above_the_remap_window_is_refused(jctx):
    st = jphase.state_3d_to_jax(_seeded_state(4, seed=5))
    with pytest.raises(ValueError, match="fv_mapz"):
        jphase.csw_phase_3d(jctx, st, DT2, 8)


def test_km_must_be_a_python_int(jctx, jstate):
    with pytest.raises(TypeError, match="Python int"):
        jphase.csw_phase_3d(jctx, jstate, DT2, jnp.asarray(3))


def test_remap_follows_must_be_a_bool(jctx, jstate):
    with pytest.raises(TypeError, match="remap_follows"):
        jphase.csw_phase_3d(jctx, jstate, DT2, KM, remap_follows=1)


def test_nord_must_be_integral(jctx, jstate):
    with pytest.raises(ValueError, match="integral damping order"):
        jphase.csw_phase_3d(jctx, jstate, DT2, KM, nord=2.7)


def test_nord_guard_is_non_vacuous(jctx, jstate, monkeypatch):
    """Neutered, the float reaches ``c_sw`` and something else objects
    (or it silently runs a different set of divergence terms) -- either
    way the guard is what produces the clean refusal."""
    monkeypatch.setattr(jphase, "require_nord", lambda f, n, v: v)
    with pytest.raises(Exception) as exc:
        jphase.csw_phase_3d(jctx, jstate, DT2, KM, nord=2.7)
    assert "integral damping order" not in str(exc.value)


def test_a2b_ord_dispatch_raises_on_unknown(jctx, csw_np):
    """Dispatch hardening AT ENTRY, on the static value: the phase names
    itself, so the refusal is attributable to this layer rather than to
    whichever kernel happens to guard the same field."""
    with pytest.raises(ValueError,
                       match=r"cgrid_pressure_phase_3d: unknown a2b_ord"):
        jphase.cgrid_pressure_phase_3d(
            jctx, _stack_np(csw_np), KM, dt2=DT2, ptop=PTOP,
            akap=FV3_KAPPA, cp_air=FV3_CP_AIR, a2b_ord=3,
            check_delpc=False)


def test_a2b_ord_guard_is_non_vacuous(jctx, csw_np, monkeypatch):
    monkeypatch.setattr(jphase, "_require_a2b_ord", lambda f, v: v)
    with pytest.raises(Exception) as exc:
        jphase.cgrid_pressure_phase_3d(
            jctx, _stack_np(csw_np), KM, dt2=DT2, ptop=PTOP,
            akap=FV3_KAPPA, cp_air=FV3_CP_AIR, a2b_ord=3,
            check_delpc=False)
    # geopk's own guard is what fires once ours is gone -- so ours is
    # the one that names the phase, and neither is silent.
    assert "unknown a2b_ord=3" in str(exc.value)
    assert "cgrid_pressure_phase_3d" not in str(exc.value)


def _ctx_with_grid_type(jctx, value):
    """A ctx clone whose six gridstructs claim a non-cubed-sphere
    ``grid_type``.  ``DuoStepperContext`` uses ``__slots__``, so the
    clone copies each slot explicitly rather than a ``__dict__``."""
    out = DuoStepperContext()
    for slot in DuoStepperContext.__slots__:
        setattr(out, slot, getattr(jctx, slot))
    out.flags6 = tuple(f._replace(grid_type=value) for f in jctx.flags6)
    return out


def test_grid_type_guard_refuses_a_non_cubed_sphere_gridstruct(jctx,
                                                               jstate):
    bad = _ctx_with_grid_type(jctx, 4)
    with pytest.raises(ValueError, match="grid_type"):
        jphase.csw_phase_3d(bad, jstate, DT2, KM)


def test_grid_type_guard_is_non_vacuous(jctx, jstate, monkeypatch):
    """Neutered, the phase RUNS on the mislabelled gridstruct and
    returns numbers -- silently using cubed-sphere formulas, which is
    precisely what the guard exists to stop."""
    monkeypatch.setattr(jphase, "_require_grid_type_zero",
                        lambda *a, **k: None)
    bad = _ctx_with_grid_type(jctx, 4)
    out = jphase.csw_phase_3d(bad, jstate, DT2, KM)
    assert np.isfinite(np.asarray(_win(out["delpc"]))).all()


def test_states_must_be_the_face_stacked_dict(jctx, state_np):
    with pytest.raises(TypeError, match="face-stacked dict"):
        jphase.csw_phase_3d(jctx, state_np, DT2, KM)


def test_states_missing_a_field_names_it(jctx, jstate):
    partial = {k: v for k, v in jstate.items() if k != "w"}
    with pytest.raises(KeyError, match="'w'"):
        jphase.csw_phase_3d(jctx, partial, DT2, KM)


def test_a_stagger_slip_raises_instead_of_broadcasting(jctx, jstate):
    """``u`` and ``v`` differ only by which axis carries the extra
    node, so a swap is the easiest slip to make and the hardest to see:
    without a shape gate it BROADCASTS inside a later op."""
    bad = dict(jstate)
    bad["u"] = jnp.swapaxes(jstate["u"], 1, 2)
    with pytest.raises(ValueError, match=r"states\['u'\]"):
        jphase.csw_phase_3d(jctx, bad, DT2, KM)


def test_shape_gate_is_non_vacuous(jctx, jstate, monkeypatch):
    monkeypatch.setattr(jphase, "validate_stacked", lambda *a, **k: None)
    bad = dict(jstate)
    bad["u"] = jnp.swapaxes(jstate["u"], 1, 2)
    with pytest.raises(Exception) as exc:
        jphase.csw_phase_3d(jctx, bad, DT2, KM)
    assert "states['u']" not in str(exc.value)


def test_a_dropped_c_sw_output_is_named(jctx, jstate, monkeypatch):
    """The assembler must not silently drop a stage output.

    Stated precisely: without this guard the bare ``got[name]`` lookup
    would ALSO raise ``KeyError`` -- what the guard adds is the contract
    in the message (which routine, which keys were returned), not the
    raise itself.  That is why the assertion below is on the message.
    """
    real = jphase.c_sw

    def dropping(*a, **k):
        out = dict(real(*a, **k))
        out.pop("ua")
        return out

    monkeypatch.setattr(jphase, "c_sw", dropping)
    with pytest.raises(KeyError, match="must not"):
        jphase.csw_phase_3d(jctx, jstate, DT2, KM)


def test_check_delpc_catches_a_non_positive_layer(jctx, csw_np):
    stacked = _stack_np(csw_np)
    stacked["delpc"] = stacked["delpc"].at[0, NG + 2, NG + 2, 1].set(-1.0)
    with pytest.raises(ValueError, match="face 1"):
        jphase.cgrid_pressure_phase_3d(
            jctx, stacked, KM, dt2=DT2, ptop=PTOP, akap=FV3_KAPPA,
            cp_air=FV3_CP_AIR)


def test_check_delpc_is_non_vacuous(jctx, csw_np):
    """Turned off, the same state runs -- so the guard is what stops it,
    and the opt-out really is an opt-out."""
    stacked = _stack_np(csw_np)
    stacked["delpc"] = stacked["delpc"].at[0, NG + 2, NG + 2, 1].set(-1.0)
    out = jphase.cgrid_pressure_phase_3d(
        jctx, stacked, KM, dt2=DT2, ptop=PTOP, akap=FV3_KAPPA,
        cp_air=FV3_CP_AIR, check_delpc=False)
    assert set(out) == {"pk", "gz", "pe", "peln", "pkz", "uc", "vc"}


def test_check_delpc_refuses_a_tracer_instead_of_skipping(jctx, csw_np):
    """C5 -- under ``jit`` the value is not available, and the two
    honest options are raise or skip.  Skipping silently is how a guard
    becomes decoration, so this lane raises and names the opt-out."""
    fn = jphase.make_cgrid_pressure_phase_3d_jit()
    with pytest.raises(ValueError, match="check_delpc"):
        fn(jctx, _stack_np(csw_np), KM, dt2=DT2, ptop=PTOP,
           akap=FV3_KAPPA, cp_air=FV3_CP_AIR)


# =====================================================================
# E. the NH refusal (mirrored from the spec)
# =====================================================================

def test_hydrostatic_only_pressure_phase_refuses_nh(jctx, csw_np):
    with pytest.raises(NotImplementedError,
                       match="cgrid_nh_pressure_phase_3d"):
        jphase.cgrid_pressure_phase_3d(
            jctx, _stack_np(csw_np), KM, dt2=DT2, ptop=PTOP,
            akap=FV3_KAPPA, cp_air=FV3_CP_AIR, hydrostatic=False,
            check_delpc=False)


def test_nh_refusal_is_non_vacuous(jctx, csw_np, monkeypatch):
    """Remove the raise and the call RUNS, returning finite numbers from
    the hydrostatic formulas -- i.e. the refusal is the only thing
    between a caller and a silently wrong NH column.  This is why the
    refusal lives in ``_refuse_nh_pressure`` and not in an inline
    ``if``: an inline raise cannot be shown non-vacuous.
    """
    monkeypatch.setattr(jphase, "_refuse_nh_pressure",
                        lambda *a, **k: None)
    out = jphase.cgrid_pressure_phase_3d(
        jctx, _stack_np(csw_np), KM, dt2=DT2, ptop=PTOP,
        akap=FV3_KAPPA, cp_air=FV3_CP_AIR, hydrostatic=False,
        check_delpc=False)
    assert np.isfinite(np.asarray(out["pk"])[:, _CS, _CS, :]).all(), (
        "the neutered call must PRODUCE numbers -- if it failed for some "
        "other reason this test proves nothing about the refusal")


# =====================================================================
# F. gradients
# =====================================================================

def test_csw_phase_3d_adjoint_identity_scalars(jctx, jstate):
    """gate 6, PRIMARY -- operand group (delp, pt, w).

    Tolerance-free: ``J v`` from ``jax.jvp``, ``J^T w`` from
    ``jax.vjp``, so the identity is exact in exact arithmetic and the
    residual is pure roundoff.  Neither an FD step nor the array's
    dynamic range can make it fail spuriously, which is what sank
    ``check_grads(order=2)`` on this port's first run.
    """
    u0, v0 = jstate["u"], jstate["v"]

    def f(delp, pt, w):
        out = jphase.csw_phase_3d(
            jctx, {"delp": delp, "pt": pt, "w": w, "u": u0, "v": v0},
            DT2, KM)
        return {"delpc": _win(out["delpc"]), "ptc": _win(out["ptc"])}

    # MEASURED (job 9425294 sweep): adjoint residual 6.347e-15; bound = measured x 10 =
    # 6.4e-14.
    _check_adjoint("csw_phase_3d d(delp,pt,w)", f,
                   (jstate["delp"], jstate["pt"], jstate["w"]), 6.4e-14)


def test_csw_phase_3d_adjoint_identity_winds(jctx, jstate):
    """gate 6, PRIMARY -- operand group (u, v)."""
    d0, p0, w0 = jstate["delp"], jstate["pt"], jstate["w"]

    def f(u, v):
        out = jphase.csw_phase_3d(
            jctx, {"delp": d0, "pt": p0, "w": w0, "u": u, "v": v},
            DT2, KM)
        return {"uc": _win(out["uc"]), "vc": _win(out["vc"]),
                "divg_d": _win(out["divg_d"])}

    # MEASURED (job 9425294 sweep): adjoint residual 2.802e-16; bound = measured x 10 =
    # 2.9e-15.
    _check_adjoint("csw_phase_3d d(u,v)", f,
                   (jstate["u"], jstate["v"]), 2.9e-15)


def test_cgrid_pressure_phase_3d_adjoint_identity(jctx, csw_np):
    """gate 6, PRIMARY -- the hydrostatic pressure chain.

    ``check_delpc=False`` because the operands are tracers here; the
    positivity of this fixture's ``delpc`` is established by the parity
    gate, which runs the check.
    """
    stacked = _stack_np(csw_np)
    uc0, vc0 = stacked["uc"], stacked["vc"]

    def f(delpc, ptc):
        out = jphase.cgrid_pressure_phase_3d(
            jctx, {"delpc": delpc, "ptc": ptc, "uc": uc0, "vc": vc0},
            KM, dt2=DT2, ptop=PTOP, akap=FV3_KAPPA, cp_air=FV3_CP_AIR,
            check_delpc=False)
        return {"gz": _win(out["gz"]), "uc": _win(out["uc"]),
                "vc": _win(out["vc"])}

    # MEASURED (job 9425294 sweep): adjoint residual 2.030e-15; bound = measured x 10 =
    # 2.1e-14.
    _check_adjoint("cgrid_pressure d(delpc,ptc)", f,
                   (stacked["delpc"], stacked["ptc"]), 2.1e-14)


def test_cgrid_nh_pressure_phase_3d_adjoint_identity(jctx_topo,
                                                     csw_np_nh,
                                                     nh_inputs):
    """gate 6, PRIMARY -- the NH chain, including the sim1 Thomas sweep
    and the ``p_fac`` floor.  Run ON the state, not away from it: the
    adjoint identity is meaningful at a switching surface, which is
    exactly where ``check_grads`` is not."""
    dp0, zs6, gz6, ws6 = nh_inputs
    stacked = _stack_np(csw_np_nh)
    gz0 = jnp.asarray(np.stack(gz6))
    ws0 = jnp.asarray(np.stack(ws6))
    zs0 = jnp.asarray(np.stack(zs6))
    dp0j = jnp.asarray(dp0)

    def f(delpc, ptc, wc):
        out = jphase.cgrid_nh_pressure_phase_3d(
            jctx_topo, {**stacked, "delpc": delpc, "ptc": ptc,
                        "wc": wc},
            gz0, ws0, KM, dt2=DT2, ptop=PTOP, akap=FV3_KAPPA,
            cp_air=FV3_CP_AIR, p_fac=P_FAC, a_imp=A_IMP, dp0=dp0j,
            zs6=zs0)
        return {"pkc": _win(out["pkc"]), "gz": _win(out["gz"]),
                "uc": _win(out["uc"])}

    # MEASURED (job 9425294 sweep): adjoint residual 3.338e-15; bound = measured x 10 =
    # 3.4e-14.
    _check_adjoint("cgrid_nh d(delpc,ptc,wc)", f,
                   (stacked["delpc"], stacked["ptc"], stacked["wc"]),
                   3.4e-14)


@pytest.mark.parametrize("order", [1, 2])
def test_csw_phase_3d_check_grads(jctx, jstate, order):
    """gate 6, SUPPLEMENT -- smooth-region finite differences.

    ``order=1`` and ``order=2`` are SEPARATE parametrised IDs so an
    FD-resolution failure at order 2 can never be confused with a wrong
    Jacobian (STATE lesson 12).

    Differentiated with respect to two SCALAR multipliers rather than
    the fields themselves: ``delp`` spans 1e3 to 1e4 across the column,
    and finite-differencing a 3-D field with that spread leaves only a
    few significant digits in the difference -- an FD failure that says
    nothing about the Jacobian.  Scaling keeps the FD well conditioned
    while exercising the same directional derivative.
    """
    one = jnp.asarray(1.0)
    d0, p0 = jstate["delp"], jstate["pt"]
    w0, u0, v0 = jstate["w"], jstate["u"], jstate["v"]

    def f(a, b):
        out = jphase.csw_phase_3d(
            jctx, {"delp": a * d0, "pt": b * p0, "w": w0, "u": u0,
                   "v": v0}, DT2, KM)
        return (jnp.sum(_win(out["delpc"]) ** 2)
                + jnp.sum(_win(out["ptc"]) ** 2))

    # MEASURED (job 9425294 sweep): smallest-passing check_grads atol=rtol 1e-7
    # (order=2; order=1 passed at 1e-11); bound = one decade up = 1e-6.
    gated_check_grads(f"csw_phase_3d check_grads order={order}", f,
                      (one, one), order=order,
                      modes=("fwd", "rev"),
                      atol=1e-6, rtol=1e-6, eps=1e-4)


# =====================================================================
# G. the face-batched arm (C2a -- face-batching ladder steps 1-2)
#
# The vmapped arm is an OPT-IN twin of the certified loop path: same
# kernels, same cadence, the `for t in range(6)` replaced by one
# `jax.vmap` over `build_batched_gs`'s stacked view.  The gates are
# (i) batched == loop per output at rtol 1e-13 / atol 1e-12 -- the
# few-ulp slack is XLA reassociating across the added batch axis; NaN
# masks (the halo tripwires) must agree in POSITION, which
# `assert_allclose(equal_nan=True)` enforces -- (ii) the stacked view
# round-trips the REAL context exactly, and (iii) the common-mode
# assert fires from the phase on a synthetic flags6 with ONE differing
# static field.  `build_batched_gs`'s own unit gates (key split, shape
# skip, cache identity) live in test_fv3_phase3d_common.py.
# =====================================================================

from legoesm.core.fv3_phase3d_common import (  # noqa: E402
    build_batched_gs,
)

# rtol/atol for batched-vs-loop: reassociation-class, NOT measured-x10
# (no cross-lane comparison here -- both arms run the same kernels on
# the same device); equal_nan keeps the structural halo NaNs comparable
# by position.
_BATCH_RTOL, _BATCH_ATOL = 1e-13, 1e-12


def _assert_batched_matches_loop(got_b, got_l, label):
    assert set(got_b) == set(got_l), (
        f"{label}: batched arm returned keys {sorted(got_b)}, loop arm "
        f"{sorted(got_l)}")
    for name in got_l:
        b, ref = np.asarray(got_b[name]), np.asarray(got_l[name])
        assert b.shape == ref.shape, (label, name, b.shape, ref.shape)
        # BIG_NUMBER sentinel cells (1e30/1e25 fills in halo scratch) are
        # GARBAGE by contract: jit-vs-eager FMA differences on them are
        # absolute-huge and meaningless. Compare them only for "both are
        # fills"; real cells at the reassociation bound. (Same masking
        # discipline as fv3_gate_helpers.cmp_fields.)
        fill = np.abs(ref) >= 1.0e20
        assert (np.abs(b)[fill] >= 1.0e20).all() if fill.any() else True, \
            f"{label}.{name}: a sentinel cell became a real value"
        keep = ~fill
        np.testing.assert_allclose(
            b[keep], ref[keep], rtol=_BATCH_RTOL, atol=_BATCH_ATOL,
            equal_nan=True,
            err_msg=f"{label}.{name}: batched arm != loop arm")


def test_batched_gs_view_roundtrips_the_real_context(jctx):
    """Every gridstruct key stacks (the six faces share shapes at one
    resolution), and unstacking equals the per-face dicts BITWISE --
    ``jnp.stack`` is a pure index copy."""
    view = build_batched_gs(jctx)
    assert view["unstacked_keys"] == ()
    assert set(view["gs"]) == set(jctx.gs6[0])
    for key in view["gs"]:
        for t in range(6):
            assert np.array_equal(np.asarray(view["gs"][key][t]),
                                  np.asarray(jctx.gs6[t][key])), (key, t)
    for t in range(6):
        assert float(view["da_min6"][t]) == jctx.flags6[t].da_min
        assert float(view["da_min_c6"][t]) == jctx.flags6[t].da_min_c
    # built once: the second call must return the CACHED view
    assert build_batched_gs(jctx) is view


@pytest.mark.parametrize("hydrostatic", [True, False],
                         ids=["hydro", "nh"])
def test_csw_phase_3d_batched_matches_loop(jctx, jstate, hydrostatic):
    loop = jphase.csw_phase_3d(jctx, jstate, DT2, KM, nord=2,
                               duogrid=True, hydrostatic=hydrostatic)
    bat = jphase.csw_phase_3d(jctx, jstate, DT2, KM, nord=2,
                              duogrid=True, hydrostatic=hydrostatic,
                              batched=True)
    _assert_batched_matches_loop(bat, loop, "csw_phase_3d")


def test_cgrid_pressure_phase_3d_batched_matches_loop(jctx, csw_np):
    kw = dict(dt2=DT2, ptop=PTOP, akap=FV3_KAPPA, cp_air=FV3_CP_AIR,
              a2b_ord=4)
    loop = jphase.cgrid_pressure_phase_3d(jctx, _stack_np(csw_np), KM,
                                          **kw)
    bat = jphase.cgrid_pressure_phase_3d(jctx, _stack_np(csw_np), KM,
                                         batched=True, **kw)
    _assert_batched_matches_loop(bat, loop, "cgrid_pressure_phase_3d")


def test_cgrid_nh_pressure_phase_3d_batched_matches_loop(jctx_topo,
                                                         csw_np_nh,
                                                         nh_inputs):
    dp0, zs6, gz6, ws6 = nh_inputs
    args = (jctx_topo, _stack_np(csw_np_nh), jnp.asarray(np.stack(gz6)),
            jnp.asarray(np.stack(ws6)), KM)
    kw = dict(dt2=DT2, ptop=PTOP, akap=FV3_KAPPA, cp_air=FV3_CP_AIR,
              p_fac=P_FAC, a_imp=A_IMP, dp0=jnp.asarray(dp0),
              zs6=jnp.asarray(np.stack(zs6)))
    loop = jphase.cgrid_nh_pressure_phase_3d(*args, **kw)
    bat = jphase.cgrid_nh_pressure_phase_3d(*args, batched=True, **kw)
    _assert_batched_matches_loop(bat, loop, "cgrid_nh_pressure_phase_3d")


def test_batched_jit_static_split_accepts_the_flag(jctx):
    """The jit factories declare ``batched`` static; a km=1 column keeps
    the compile to ONE vmapped c_sw trace.  jit-batched vs eager-batched
    at the same reassociation-class bound (FMA contraction under jit is
    the known, structural difference in this file)."""
    st1 = jphase.state_3d_to_jax(_seeded_state(1, seed=7))
    eager = jphase.csw_phase_3d(jctx, st1, DT2, 1, batched=True)
    jitted = jphase.make_csw_phase_3d_jit()(jctx, st1, DT2, 1,
                                            batched=True)
    # jit-vs-eager FMA contraction is a LARGER class than batched-vs-loop
    # reassociation: measured worst 2.7e-13 rel on one ut cell (job
    # 9502342); bound = measured x4. The batched-vs-loop comparisons
    # above stay at 1e-13.
    for name in eager:
        b, ref = np.asarray(jitted[name]), np.asarray(eager[name])
        fill = np.abs(ref) >= 1.0e20
        keep = ~fill
        np.testing.assert_allclose(
            b[keep], ref[keep], rtol=1.0e-12, atol=1e-12, equal_nan=True,
            err_msg=f"csw_phase_3d[jit, batched].{name}")


def test_batched_defaults_off_on_all_three_phases():
    """RULE 3 guard: the certified loop path is the DEFAULT; the vmap
    arm is opt-in.  A flipped default would silently move every caller
    (the acoustic assembler passes no flag) onto the uncertified arm."""
    import inspect

    for fn in (jphase.csw_phase_3d, jphase.cgrid_pressure_phase_3d,
               jphase.cgrid_nh_pressure_phase_3d):
        assert inspect.signature(fn).parameters["batched"].default \
            is False, fn.__name__


def test_batched_must_be_a_bool(jctx, jstate):
    with pytest.raises(TypeError, match="batched"):
        jphase.csw_phase_3d(jctx, jstate, DT2, KM, batched=1)


def test_batched_common_mode_assert_fires_from_the_phase(jctx, jstate):
    """The GLM guard, exercised THROUGH a phase call: a ctx whose faces
    disagree on ONE static GridFlags field must raise naming the field,
    not broadcast face 1's value into the vmapped kernel.  The loop
    path (control) still RUNS on the same ctx -- it reads the per-face
    flags -- which is what makes this a batched-arm gate and not a
    context-builder gate."""
    bad = DuoStepperContext()
    for slot in DuoStepperContext.__slots__:
        setattr(bad, slot, getattr(jctx, slot))
    # Flip face 3's flag relative to whatever the context carries
    # (oracle-conventions gridstructs hold the corner flags False, so a
    # hardcoded False would be a no-op and the gate vacuous).
    bad.flags6 = tuple(
        f._replace(ne_corner=not f.ne_corner) if t == 2 else f
        for t, f in enumerate(jctx.flags6))
    with pytest.raises(ValueError, match="ne_corner"):
        jphase.csw_phase_3d(bad, jstate, DT2, KM, batched=True)
    out = jphase.csw_phase_3d(bad, jstate, DT2, KM)   # control
    assert np.isfinite(np.asarray(_win(out["delpc"]))).all()
