"""Certification of the JAX 3-D D-grid transport phase, and of BARRIER 1.

Authority: ``legoesm.core.fv3_native_dsw_phase_3d`` is the
SPECIFICATION (hop B of
``docs/atmosphere/fv3_duo_jax_lane_strategy.md``).  The pinned Fortran
is quoted only where a line number says WHY a step exists; it is never
what a bound here is calibrated against.

The module under test contributes CADENCE, not arithmetic -- every
operation inside it happens in a kernel gated by
``test_fv3_duo_sw_core.py`` or ``test_fv3_duo_halos.py``.  So the gates
below interrogate the JOINS, and one join in particular:

* **BARRIER 1 and its SLOT EXCLUSION.**  ``dyn_core.F90:855`` opens
  ``do iq=1,4+nq`` and ``:856`` reads ``if (iq==1 .or. iq==4 .or.
  iq>4 ) then``, so slots 2 (``w``) and 3 (``q_con``) are NOT averaged.
  The gate here proves the property survives the 3-D lift, and a
  MUTATION control re-includes slot 2 in ``tab.allflux_slots`` and
  shows the gate then goes red.  Modelled on
  ``test_fv3_duo_halos.test_barrier1_skips_slot2_w_and_slot3_qcon_exactly``,
  which caught the same mutation at km=1.

* the barrier sits BETWEEN ``d_sw1`` and ``d_sw2`` (``:872``, between
  the ``:744`` and ``:914`` k-nests), which is the whole reason the
  oracle splits ``d_sw`` into six routines: the seam fluxes are
  averaged BEFORE they are applied.  A gate asserts the ordering
  directly -- the ``delp`` update must be a function of the AVERAGED
  fluxes, not the raw ones.

Gate classes, following the sibling 3-D module's file:

1. **parity** -- JAX vs the NumPy twin from BYTE-IDENTICAL inputs, per
   output, with a measured bound carrying a ``TOL-PENDING`` marker.
   Both lanes are fed the SAME NumPy ``csw_phase_3d`` output, so a
   ``c_sw`` parity difference cannot leak into a transport verdict;
2. **cadence** -- perturb one LEVEL and one FACE and require the
   response the cadence predicts.  Note the face gate is NOT "only
   that face moves": this phase contains two cross-face couplings (the
   exchanges and the barrier), which is exactly what distinguishes it
   from the C-grid phase;
3. **jit vs eager** -- an ASSERTION.  **Nothing that flows through a
   kernel is asserted bitwise across the jit boundary**: ``d_sw1``'s
   flux divergences, ``d_sw2``'s update and every ``ext_*`` exchange
   (whose ``k2e``/Lagrange/``c2l`` stages are ``Sum w*v``) are
   contraction sites XLA fuses when jitted and not when eager.  The
   only BITWISE gate is on the pure index-copy assembly -- ``jnp.stack``
   of a level -- and it is run EAGER on both sides;
4. **guards** -- every entry gate, each shown NON-VACUOUS by
   monkeypatching the validator to a no-op and asserting the call then
   proceeds (or fails elsewhere);
5. **gradients** -- the PRIMARY gate is the tolerance-free adjoint
   identity ``<J v, w> == <v, J^T w>``, per operand group, with an
   explicit non-vacuity assert that ``<J v, w> != 0``.  ``check_grads``
   is a SCOPED supplement with ``order=1`` and ``order=2`` as separate
   parametrised IDs (STATE lesson 12);
6. **anti-vacuity, everywhere** -- ``_cmp`` refuses an all-fill pair
   and an identically-zero reference, and every "these are equal"
   assertion over an array that MAY be all-NaN is preceded by a
   finiteness assert.  Two vacuous comparisons were caught on the
   sibling module's first run; the allflux stack makes that hazard
   worse here, because THREE of its five slots are NaN by construction
   on this lane (see ``_SLOTS_*`` below).

TOLERANCE POLICY.  ``d_sw1`` runs the ``xppm``/``yppm`` limiter
pipelines and ``del6_vt_flux``; ``d_sw2`` is a flux-divergence update;
the exchanges are weighted stencil sums.  Per strategy section 4 that
spans the accumulating and branch-switching classes, so every numeric
bound below carries a ``TOL-PENDING`` marker plus a one-word class
label.

COST.  These gates run six faces x three levels of ``d_sw1`` AND
``d_sw2`` per call, plus three barrier applications and three
six-face exchange chains; the jvp/vjp programs add more.  Minutes, not
seconds -- stated so the measurement job's wall clock is not read as a
hang.
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
from jax.test_util import check_grads  # noqa: E402
from legoesm.core import fv3_dsw_phase_3d as jdsw  # noqa: E402
from legoesm.core import fv3_native_cgrid_phase_3d as npcg  # noqa: E402
from legoesm.core import fv3_native_dsw_phase_3d as npdsw  # noqa: E402
from legoesm.core import fv3_native_duo_stepper as npstep  # noqa: E402
from legoesm.core.fv3_cgrid_phase_3d import state_3d_to_jax  # noqa: E402
from legoesm.core.fv3_duo_stepper import (  # noqa: E402
    SWConfig,
    build_jax_duo_stepper_context,
)
from legoesm.core.fv3_duo_sw_core import d_sw1_duo  # noqa: E402
from legoesm.core.fv3_native_state_3d import build_state_3d  # noqa: E402
from legoesm.core.fv3_native_tracer2d import (  # noqa: E402
    alloc_flux_capacitors,
)

# C12 is the smallest resolution the duo corner-region Lagrange fill
# admits and is what every other JAX-lane test module uses, so the
# fixtures are comparable across the port.
N, NG = 12, 3
MA = N + 2 * NG
NPX = N + 1

# km = 3: deep enough that "perturb level 1, levels 0 and 2 must not
# move" is a real cadence statement, and inside `require_no_remap_needed`'s
# km <= 4 window (fv_dynamics.F90:568).  Same depth as the sibling 3-D
# module's fixtures, so the two files are comparable.
KM = 3

# The FULL acoustic step -- dyn_core.F90:831 passes `dt` to d_sw1, not
# `dt2`.  Small enough that the transport stays well inside CFL on this
# IC.
DT = 20.0
DT2 = 0.5 * DT

# The workspace FILL CONSTANTS this lane round-trips, BY VALUE.  1e30 is
# `d_sw2_duo`'s `workspace_sentinel` default (its `dw` when the del-6
# block does not run) and 1e25 is `divergence_corner_duo`'s `divg_d`
# init, which arrives here through `csw_outs`.
#
# 0.0 is NOT a fill here even though `d_sw1` is called with
# `workspace_sentinel=0.0` (the spec's choice, mirroring the km=1
# stepper): zero is a legitimate value of every field in this phase, and
# classifying it as a fill would silently drop real cells from every
# comparison.  A magnitude-threshold classifier was tried in this
# campaign and RETRACTED -- exact values only.
_FILL_VALUES = (1.0e30, 1.0e25)

# ---------------------------------------------------------------------
# BARRIER-1 SLOT MAP -- 0-based, i.e. Fortran `iq` minus one.
#
# dyn_core.F90:855  `do iq=1,4+nq ! 1delp 2w 3qcon 4temp 5>q`
# dyn_core.F90:856  `if (iq==1 .or. iq==4 .or. iq>4 ) then`
#
# so the AVERAGED set is iq in {1, 4, 5} -> indices {0, 3, 4} and the
# EXCLUDED set is iq in {2, 3} -> indices {1, 2}.
#
# What each slot actually CONTAINS on this lane matters as much as
# whether it is averaged, because three of the five are never written
# and a bitwise "unchanged" assertion over an all-NaN slab is vacuous:
#
#   0 (delp)   written always      -> compared, and must MOVE
#   1 (w)      written on NH ONLY  -> the load-bearing exclusion gate;
#                                     run on the NH arm so it is finite
#   2 (q_con)  never written       -> NaN; contract asserted, not compared
#   3 (temp)   written always      -> compared, and must MOVE
#   4 (tracer) never written       -> NaN; `inline_q=False` on this lane,
#                                     so d_sw1 transports no tracer.  It
#                                     IS in the averaged set and the
#                                     blend does run on it -- producing
#                                     0.5*(NaN + NaN) = NaN.  It
#                                     therefore CANNOT be asserted to
#                                     move, and saying otherwise would be
#                                     a gate that can only fail.
_SLOT_DELP, _SLOT_W, _SLOT_QCON, _SLOT_TEMP, _SLOT_TRACER = 0, 1, 2, 3, 4
_SLOTS_AVERAGED = (_SLOT_DELP, _SLOT_TEMP, _SLOT_TRACER)
_SLOTS_EXCLUDED = (_SLOT_W, _SLOT_QCON)
_SLOTS_WRITTEN_HYDRO = (_SLOT_DELP, _SLOT_TEMP)
_SLOTS_WRITTEN_NH = (_SLOT_DELP, _SLOT_W, _SLOT_TEMP)
_SLOTS_NEVER_WRITTEN_NH = (_SLOT_QCON, _SLOT_TRACER)
_NSLOT = 5

# The d_sw1 stage outputs compared against the spec's per-level dicts.
# `allflux_*` are handled separately (pre- vs post-barrier) and
# `delp`/`pt`/`w` are compared as d_sw2's UPDATE, not d_sw1's
# corner-rotated input.
_DSW1_COMPARED = ("crx_adv", "cry_adv", "xfx_adv", "yfx_adv",
                  "ra_x", "ra_y", "ut", "vt")


# =====================================================================
# comparison helpers
#
# FIFTH COPY, and it is deliberate: pytest modules are not an import
# surface (importing one test module from another would execute its
# module-level jax config), and the campaign's existing copies are
# private.  This is the CORRECTED metric -- exact-value fill
# classification, a PER-ELEMENT relative difference, and refusal on both
# an all-fill pair and an identically-zero reference -- copied from
# `test_fv3_cgrid_phase_3d._cmp`, NOT from `fv3_duo_stepper`'s older one.
# FOLLOW-UP, already open on the other copies: converge onto one shared
# `tests/grids/conftest.py` helper and re-measure every bound under it.
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
        f"it through a relative comparison).")
    diff = np.abs(a[ok] - b[ok])
    scale = float(np.median(np.abs(b[ok])))
    if not (scale > 0.0):
        scale = max(float(np.abs(b[ok]).max()), 1e-300)
    per = diff / (np.abs(b[ok]) + scale)
    rel = float(per.max())
    n_over = int((per > 1e-13).sum())
    assert rel <= tol, (
        f"{name}: MEASURED per-element rel {rel:.3e} > {tol:.3e}; "
        f"{n_over} of {int(ok.sum())} compared cells exceed 1e-13 "
        f"(a handful => BRANCH FLIP, all of them => systematic); "
        f"median|ref| {scale:.3e}, max|diff| {float(diff.max()):.3e}, "
        f"bitwise={np.array_equal(a[ok], b[ok])}")
    return rel, n_over


def _bitwise_equal(a, b):
    """Exact equality with ``NaN == NaN``.

    ONLY for pure index copies.  Every call site must independently
    assert that the compared region contains real numbers, because this
    predicate is TRUE for two all-NaN arrays -- which is precisely the
    vacuous comparison the campaign has been bitten by.
    """
    return np.array_equal(np.asarray(a, dtype=np.float64),
                          np.asarray(b, dtype=np.float64), equal_nan=True)


def _assert_real(arr, name):
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
            f"includes cells the phase never writes (three of the five "
            f"allflux slots are NaN by construction on this lane), or a "
            f"dead branch is leaking a NaN into the gradient (R1b)")
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
    assert r <= tol, (
        f"{name}: adjoint residual {r:.3e} > {tol:.3e} "
        f"(<J v, w>={lhs:.12e}, <v, J^T w>={rhs:.12e}) -- MEASURED "
        f"value is {r:.3e}")
    return r


def _counted(fn):
    """(traced-call counter, wrapper) for the retrace assertions.

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


# The strict COMPUTE square, is..ie x js..je, in storage indices: the
# NumPy lane puts Fortran (isd, jsd) = (1-ng, 1-ng) at storage [0, 0],
# so Fortran i = is = 1 sits at index ng.  Used for every gradient
# OBJECTIVE, because the halo legitimately carries fills and NaN
# tripwires and an objective that touches them is not differentiable.
# Exclude by WINDOW, never by a nan-aware reduction.
_CS = slice(NG, NG + N)


def _win(a):
    """Compute square of a face-stacked ``(6, i, j, km)`` array."""
    return a[:, _CS, _CS, :]


def _deepcopy_faces(per_face: list) -> list:
    """Six per-face dicts with every array COPIED.

    Required, not tidiness: the NumPy transport phase MUTATES
    ``csw_outs[t]["uc"]``/``["vc"]``/``["divg_d"]`` (the post-p_grad_c
    exchanges) and mutates its ``flux_cap`` in place, so calling it on
    a module-scoped fixture would silently corrupt every later test.
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
    (``fv_arrays.F90:1512``).  ``use_ext_bundle=True`` because
    ``build_jax_duo_stepper_context`` implements the faithful exchange
    lane only, and the exchanges at ``dyn_core.F90:652``/``:655`` are
    the first thing this phase does.
    """
    return npstep.build_six_face_duo_context(
        N, NG, use_ext_bundle=True, oracle_conventions=True)


@pytest.fixture(scope="module")
def jctx(ctx):
    return build_jax_duo_stepper_context(ctx)


def _seeded_state(km, seed=0, hydrostatic=True):
    """A PHYSICAL 3-D column, distinct per face AND per level.

    Magnitudes matter and are not decoration: ``delp ~ 1e4 Pa`` and
    ``pt ~ 280 K`` keep the transported thickness positive through the
    ``d_sw2`` update, and a strictly positive ``delp`` everywhere --
    halo included -- is what keeps ``d_sw1``'s mass-weighted ``pt``
    transport away from a zero divisor.

    Distinct values per face and per level so that a face mix-up or a
    level mix-up cannot hide behind symmetry.  ``w`` is given real
    structure because on the NH arm it is the slot the barrier-1
    exclusion gate reads, and a uniform ``w`` would make that gate a
    comparison of one constant with itself.
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
            face["w"][:, :, k] = (
                -0.5 + 0.1 * t + 0.2 * k
                + 0.4 * rng.standard_normal((MA, MA)))
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
    return state_3d_to_jax(state_np)


@pytest.fixture(scope="module")
def csw_np(ctx, state_np):
    """The NumPy lane's ``c_sw`` phase output -- the SHARED input to
    both lanes' transport phases, so those gates are one-variable.

    The C-grid PRESSURE phase (``geopk`` -> ``p_grad_c``) is NOT run:
    it is the sibling unit and is gated in ``test_fv3_cgrid_phase_3d``,
    and all this phase needs from upstream is a realistic, finite,
    per-face-distinct ``uc``/``vc``/``divg_d``.  Running it would add a
    second routine to the fixture without changing what is under test.
    Built on the HYDROSTATIC arm; the NH fixture is separate so the two
    are not changed at once.
    """
    return npcg.csw_phase_3d(ctx, _deepcopy_faces(state_np), dt2=DT2,
                             km=KM, nord=2, duogrid=True)


@pytest.fixture(scope="module")
def csw_np_nh(ctx, state_np):
    """``c_sw`` on the NH arm -- the input to every NH transport gate,
    and in particular to the barrier-1 slot-exclusion gate, which needs
    allflux slot 2 (``w``) to be WRITTEN and finite."""
    return npcg.csw_phase_3d(ctx, _deepcopy_faces(state_np), dt2=DT2,
                             km=KM, nord=2, duogrid=True,
                             hydrostatic=False)


@pytest.fixture(scope="module")
def jcsw(csw_np):
    return _stack_np(csw_np)


@pytest.fixture(scope="module")
def jcsw_nh(csw_np_nh):
    return _stack_np(csw_np_nh)


def _cap_np(seed=77):
    """Six per-face capacitor dicts, PRE-CHARGED.

    Zeros would make "did the accumulation happen" indistinguishable
    from "was the capacitor overwritten": both give ``cap = flux``.
    Pre-charging with distinct values makes the two outcomes different
    numbers, which is the only way that gate can fail.
    """
    rng = np.random.default_rng(seed)
    caps = alloc_flux_capacitors(N, NG, KM)
    for t, cap in enumerate(caps):
        for name, arr in cap.items():
            arr[...] = (1.0 + 0.3 * t) * rng.standard_normal(arr.shape)
    return caps


def _stack_cap(caps: list) -> dict:
    """Six per-face capacitor dicts -> the face-stacked JAX container."""
    return {k: jnp.asarray(np.stack([np.asarray(c[k], dtype=np.float64)
                                     for c in caps]))
            for k in jdsw.CAPACITOR_FIELDS}


# =====================================================================
# A. layout, tables and the oracle contract
# =====================================================================

def test_duo_deck_is_built_from_the_spec_not_restated():
    """C7: :data:`DSW_DUO_DECK` must BE the spec's deck.

    Not a similarity check -- every knob the spec names is asserted
    equal, and ``SWConfig.from_mapping`` is what makes a NEW spec knob
    raise at import instead of being silently dropped.
    """
    assert npdsw.DUO_DECK_CFG, "the spec's deck dict is empty"
    for key, value in npdsw.DUO_DECK_CFG.items():
        assert hasattr(jdsw.DSW_DUO_DECK, key), (
            f"deck knob {key!r} is not an SWConfig field; "
            f"SWConfig.from_mapping should have raised at import")
        assert getattr(jdsw.DSW_DUO_DECK, key) == value, (
            f"deck knob {key!r}: JAX lane has "
            f"{getattr(jdsw.DSW_DUO_DECK, key)!r}, spec has {value!r}")
    # the deck is the shipped duo one, not SWConfig's defaults
    assert jdsw.DSW_DUO_DECK != SWConfig(), (
        "DSW_DUO_DECK equals the bare SWConfig defaults -- either the "
        "spec's deck stopped overriding anything, or from_mapping is "
        "dropping every key")


def test_barrier1_slot_table_matches_the_oracle_condition(jctx):
    """``dyn_core.F90:856`` -- ``iq==1 .or. iq==4 .or. iq>4``.

    Asserted against the table the phase actually uses, not against a
    restatement: ``tab.allflux_slots`` is what
    ``average_allflux_shared_edges`` selects with.
    """
    sel = sorted(np.asarray(jctx.tab.allflux_slots).tolist())
    assert sel == sorted(_SLOTS_AVERAGED), (
        f"tab.allflux_slots = {sel}, oracle averages "
        f"{sorted(_SLOTS_AVERAGED)} (0-based iq-1 for "
        f"iq in 1, 4, >4). nq={jctx.tab.nq}")
    for s in _SLOTS_EXCLUDED:
        assert s not in sel, f"slot index {s} must NOT be averaged"
    assert 4 + int(jctx.tab.nq) == _NSLOT


def test_eval_shape_declares_the_transport_layout(jctx, jstate, jcsw):
    """Shapes and dtypes BEFORE running anything expensive.

    Also pins convention C1's level-axis placement: ``km`` at axis 2,
    so an ``allflux`` stack is ``(6, i, j, km, slot)`` -- the oracle's
    own ``allflux_x(i,j,k,iq)`` (``dyn_core.F90:860``).
    """
    out = jax.eval_shape(
        lambda s, c: jdsw.dsw_transport_phase_3d(jctx, s, c, DT, KM),
        jstate, jcsw)
    assert out["allflux_x"].shape == (6, NPX, N, KM, _NSLOT)
    assert out["allflux_y"].shape == (6, N, NPX, KM, _NSLOT)
    assert out["allflux_x_prebarrier"].shape == out["allflux_x"].shape
    assert out["allflux_y_prebarrier"].shape == out["allflux_y"].shape
    assert out["delp"].shape == (6, MA, MA, KM)
    assert out["pt"].shape == (6, MA, MA, KM)
    assert out["uc"].shape == (6, MA + 1, MA, KM)
    assert out["vc"].shape == (6, MA, MA + 1, KM)
    assert out["divg_d"].shape == (6, MA + 1, MA + 1, KM)
    for name in ("w", "dw") + jdsw.CAPACITOR_FIELDS:
        assert name not in out, (
            f"{name!r} must be ABSENT on the hydrostatic, no-capacitor "
            f"call -- an absent key is an ABSENCE, and materialising it "
            f"would be a plausible-looking value where the spec has "
            f"nothing")
    for name, leaf in out.items():
        assert leaf.dtype == jnp.float64, (name, leaf.dtype)


def test_nh_and_capacitor_arms_add_exactly_their_own_keys(jctx, jstate,
                                                          jcsw_nh):
    caps = _stack_cap(_cap_np())
    out = jax.eval_shape(
        lambda s, c, f: jdsw.dsw_transport_phase_3d(
            jctx, s, c, DT, KM, hydrostatic=False, flux_cap=f),
        jstate, jcsw_nh, caps)
    assert out["w"].shape == (6, MA, MA, KM)
    assert out["dw"].shape == (6, N, N, KM)
    for name in jdsw.CAPACITOR_FIELDS:
        assert out[name].shape == caps[name].shape, name


def test_exchange_output_shapes(jctx, jcsw):
    out = jax.eval_shape(
        lambda c: jdsw.exchange_post_pgrad_3d(jctx, c, KM, nord=2), jcsw)
    assert set(out) == {"divg_d", "uc", "vc"}
    assert out["uc"].shape == (6, MA + 1, MA, KM)
    assert out["vc"].shape == (6, MA, MA + 1, KM)
    assert out["divg_d"].shape == (6, MA + 1, MA + 1, KM)


# =====================================================================
# B. parity vs the NumPy lane (tier 1)
# =====================================================================

def test_exchange_post_pgrad_3d_matches_numpy_lane(ctx, jctx, csw_np,
                                                   jcsw):
    """gate 1 for the ``:652``/``:655`` exchanges, per level.

    The spec's helper is private, which is fine to reach into from a
    TEST -- ``tests/test_no_private_cross_imports.py`` scans production
    roots only and says so ("white-box tests legitimately reach into
    internals").  Reaching for it directly makes this one-variable: the
    alternative (reading the mutated ``csw_outs`` after the whole
    transport phase) would fold ``d_sw1`` into the comparison.
    """
    ref = _deepcopy_faces(csw_np)
    npdsw._exchange_post_pgrad(ctx, ref, KM, nord=2)
    got = jdsw.exchange_post_pgrad_3d(jctx, jcsw, KM, nord=2)
    for name in ("uc", "vc", "divg_d"):
        for t in range(6):
            # TOL-PENDING: provisional bound; the orchestrator's
            # measurement job will replace this with `measured X,
            # bound = measured x N`.  DO NOT SHIP.
            # [class: accumulating (k2e / Lagrange weighted sums)]
            _cmp(got[name][t], ref[t][name],
                 f"exchange.{name}[face {t + 1}]", 1e-12)


def test_the_exchange_actually_changes_the_halo(jctx, jcsw):
    """Non-vacuity for the gate above: if the exchanges were a no-op,
    the parity comparison would compare two copies of the input and
    could not fail."""
    got = jdsw.exchange_post_pgrad_3d(jctx, jcsw, KM, nord=2)
    for name in ("uc", "vc", "divg_d"):
        before = np.asarray(jcsw[name], dtype=np.float64)
        after = np.asarray(got[name], dtype=np.float64)
        moved = np.isfinite(before) & np.isfinite(after) & (before != after)
        assert moved.any(), (
            f"{name}: the exchange changed nothing, so the parity gate "
            f"above is comparing the input with itself")


def test_divgd_exchange_is_gated_on_nord(jctx, jcsw):
    """``dyn_core.F90:652`` gates ``ext_scalar(divgd, ...)`` on
    ``flagstruct%nord > 0``.  At ``nord = 0`` ``divg_d`` must come back
    UNTOUCHED while ``uc``/``vc`` still move (``:655`` is ungated)."""
    at0 = jdsw.exchange_post_pgrad_3d(jctx, jcsw, KM, nord=0)
    _assert_real(jcsw["divg_d"], "divg_d input")
    assert _bitwise_equal(at0["divg_d"], jcsw["divg_d"]), (
        "nord=0 must leave divg_d byte-identical (dyn_core.F90:652)")
    before = np.asarray(jcsw["uc"], dtype=np.float64)
    after = np.asarray(at0["uc"], dtype=np.float64)
    moved = np.isfinite(before) & np.isfinite(after) & (before != after)
    assert moved.any(), (
        "the uc/vc exchange at :655 is UNGATED and must still run at "
        "nord=0; if it did not, the assertion above would be vacuous")


@pytest.mark.parametrize("hydrostatic", [True, False],
                         ids=["hydro", "nh"])
def test_dsw_transport_phase_3d_matches_numpy_lane(ctx, jctx, state_np,
                                                   jstate, csw_np,
                                                   csw_np_nh, jcsw,
                                                   jcsw_nh, hydrostatic):
    """gate 1 -- byte-identical inputs, per output, whole padded array.

    The comparison covers the HALO as well as the compute window: the
    fills and NaN tripwires a lane carries there are part of its
    contract, and ``_cmp``'s mask checks are what make that assertable
    instead of vacuous.
    """
    src = csw_np if hydrostatic else csw_np_nh
    jsrc = jcsw if hydrostatic else jcsw_nh
    ref = npdsw.dsw_transport_phase_3d(
        ctx, _deepcopy_faces(state_np), _deepcopy_faces(src), DT, KM,
        cfg=dict(npdsw.DUO_DECK_CFG), nq=1, hydrostatic=hydrostatic)
    got = jdsw.dsw_transport_phase_3d(jctx, jstate, jsrc, DT, KM,
                                      hydrostatic=hydrostatic)

    for t in range(6):
        for name in ("delp", "pt") + (() if hydrostatic else ("w",)):
            # TOL-PENDING: provisional bound; the orchestrator's
            # measurement job will replace this with `measured X,
            # bound = measured x N`.  DO NOT SHIP.
            # [class: accumulating (d_sw2 flux divergence update)]
            _cmp(got[name][t], ref[t][name],
                 f"dsw.{name}[face {t + 1}]", 1e-12)
        for name in ("allflux_x", "allflux_y"):
            # TOL-PENDING: provisional bound; the orchestrator's
            # measurement job will replace this with `measured X,
            # bound = measured x N`.  DO NOT SHIP.
            # [class: branch-switching (xppm/yppm limiters in d_sw1)]
            _cmp(got[name][t], ref[t][name],
                 f"dsw.{name}[face {t + 1}]", 1e-12)
        # the d_sw1 stage outputs the spec carries per level
        for name in _DSW1_COMPARED:
            want = np.stack([ref[t]["levels"][k][name]
                             for k in range(KM)], axis=2)
            # TOL-PENDING: provisional bound; the orchestrator's
            # measurement job will replace this with `measured X,
            # bound = measured x N`.  DO NOT SHIP.
            # [class: branch-switching (d_sw1 upwind + panel-edge
            #  selects)]
            _cmp(got[name][t], want, f"dsw1.{name}[face {t + 1}]", 1e-12)
        if not hydrostatic:
            want_dw = np.stack([ref[t]["levels"][k]["dw"]
                                for k in range(KM)], axis=2)
            # TOL-PENDING: provisional bound; the orchestrator's
            # measurement job will replace this with `measured X,
            # bound = measured x N`.  DO NOT SHIP.
            # [class: accumulating (del-6 dw increment)]
            _cmp(got["dw"][t], want_dw, f"dsw2.dw[face {t + 1}]", 1e-12)


def test_capacitors_match_numpy_and_accumulate(ctx, jctx, state_np,
                                               jstate, csw_np, jcsw):
    """The ``mfx``/``mfy``/``cx``/``cy`` capacitors (D4).

    Two claims in one gate, both required:
    (a) parity with the NumPy lane's IN-PLACE mutation of the same
        pre-charged capacitors;
    (b) they ACCUMULATE rather than overwrite -- asserted by starting
        from a non-zero charge and requiring the result to differ from
        both the charge and the bare flux.
    """
    caps_np = _cap_np()
    charge = _deepcopy_faces(caps_np)
    ref = npdsw.dsw_transport_phase_3d(
        ctx, _deepcopy_faces(state_np), _deepcopy_faces(csw_np), DT, KM,
        cfg=dict(npdsw.DUO_DECK_CFG), nq=1, flux_cap=caps_np)
    assert ref is not None
    got = jdsw.dsw_transport_phase_3d(jctx, jstate, jcsw, DT, KM,
                                      flux_cap=_stack_cap(charge))
    for name in jdsw.CAPACITOR_FIELDS:
        for t in range(6):
            # TOL-PENDING: provisional bound; the orchestrator's
            # measurement job will replace this with `measured X,
            # bound = measured x N`.  DO NOT SHIP.
            # [class: accumulating (substep flux capacitor)]
            _cmp(got[name][t], caps_np[t][name],
                 f"capacitor.{name}[face {t + 1}]", 1e-12)
        a = _assert_real(got[name], f"capacitor {name}")
        b = np.asarray(_stack_cap(charge)[name], dtype=np.float64)
        assert not np.array_equal(a, b), (
            f"{name}: the capacitor came back equal to its input charge, "
            f"so nothing was accumulated into it")


def test_capacitor_window_outside_the_kernel_dummy_is_untouched(
        jctx, jstate, jcsw):
    """D4, the slice.  ``alloc_flux_capacitors`` allocates ``mfx``
    ``(npx, m_a, km)`` while ``d_sw1_duo``'s ``xflux`` dummy is
    ``(npx, n)``; the columns beyond ``je`` "exist only because the
    caller-side arrays are allocated rectangular -- they stay zero"
    (the allocator's own docstring).  If the write-back slice were
    wrong, those columns would be overwritten."""
    caps = alloc_flux_capacitors(N, NG, KM)     # zeros, on purpose here
    got = jdsw.dsw_transport_phase_3d(jctx, jstate, jcsw, DT, KM,
                                      flux_cap=_stack_cap(caps))
    mfx = np.asarray(got["mfx"], dtype=np.float64)
    mfy = np.asarray(got["mfy"], dtype=np.float64)
    _assert_real(mfx[:, :, :N, :], "mfx kernel window")
    _assert_real(mfy[:, :N, :, :], "mfy kernel window")
    assert np.array_equal(mfx[:, :, N:, :], np.zeros_like(mfx[:, :, N:, :])), (
        "mfx columns beyond the kernel window were written; the D4 slice "
        "is wrong")
    assert np.array_equal(mfy[:, N:, :, :], np.zeros_like(mfy[:, N:, :, :])), (
        "mfy rows beyond the kernel window were written; the D4 slice is "
        "wrong")


def test_km1_level_assembly_is_a_pure_index_copy(ctx, jctx):
    """The assembler must add CADENCE, not change math.

    BITWISE, and EAGER on both sides: at ``km = 1`` the returned
    per-level output is ``jnp.stack`` of the very array a direct
    ``d_sw1_duo`` call returns -- a pure index copy of identical
    values, with no ``x*y + z`` between them for XLA to contract.  This
    is the ONLY bitwise gate in this file, and it is bitwise for that
    structural reason rather than an empirical one.

    ``allflux_x`` is compared PRE-barrier, because post-barrier it is
    (correctly) no longer the kernel's own output.
    """
    st = _seeded_state(1, seed=3)
    jst = state_3d_to_jax(st)
    csw = _stack_np(npcg.csw_phase_3d(ctx, _deepcopy_faces(st), dt2=DT2,
                                      km=1, nord=2, duogrid=True))
    c = jdsw.DSW_DUO_DECK
    # the SAME exchanged winds the phase feeds d_sw1 -- comparing
    # against un-exchanged uc/vc would make this a test of the
    # exchange, not of the assembly
    ex = jdsw.exchange_post_pgrad_3d(jctx, csw, 1, nord=c.nord)
    out = jdsw.dsw_transport_phase_3d(jctx, jst, csw, DT, 1)
    zx = jnp.zeros((NPX, N), dtype=jnp.float64)
    zy = jnp.zeros((N, NPX), dtype=jnp.float64)
    zcx = jnp.zeros((NPX, MA), dtype=jnp.float64)
    zcy = jnp.zeros((MA, NPX), dtype=jnp.float64)
    for t in range(6):
        direct = d_sw1_duo(
            jst["delp"][t][:, :, 0], jst["pt"][t][:, :, 0],
            jst["w"][t][:, :, 0], ex["uc"][t][:, :, 0],
            ex["vc"][t][:, :, 0], zx, zy, zcx, zcy,
            jctx.gs6[t], jctx.flags6[t], jctx.bd, NPX, NPX, dt=DT,
            hord_tr=c.hord_tr, hord_vt=c.hord_vt, hord_tm=c.hord_tm,
            hord_dp=c.hord_dp, nord_v=c.nord_v, nord_t=0,
            damp_v=c.damp_v, damp_t=0.0, workspace_sentinel=0.0)
        pairs = [(name, name) for name in _DSW1_COMPARED]
        pairs += [("allflux_x", "allflux_x_prebarrier"),
                  ("allflux_y", "allflux_y_prebarrier")]
        for kernel_name, out_name in pairs:
            got = np.asarray(out[out_name][t][:, :, 0])
            want = np.asarray(direct[kernel_name])
            _assert_real(want, f"face {t + 1} {kernel_name} reference")
            assert _bitwise_equal(got, want), (
                f"face {t + 1} {kernel_name}: the 3-D assembly is not a "
                f"pure index copy of the certified 2-D kernel's output "
                f"at km=1")


# =====================================================================
# C. BARRIER 1 -- the slot exclusion, and where the barrier SITS
# =====================================================================

def test_barrier1_leaves_w_and_qcon_slots_bit_identical(jctx, jstate,
                                                        jcsw_nh):
    """★ THE gate this module exists for.  ``dyn_core.F90:856``.

    Run on the NH arm ON PURPOSE: slot 2 (``w``) is written only there
    (``d_sw1``'s ``fv_tp_2d(w)``), and on the hydrostatic arm it is
    NaN, where a "bitwise unchanged" assertion is satisfied by
    ``NaN == NaN`` and proves nothing.  So the exclusion is asserted
    where it can actually fail, and the finiteness precondition is
    asserted first.

    Slot 3 (``q_con``) is never written on this lane at all
    (``inline_q=False``), so its CONTRACT -- entirely non-finite, both
    before and after -- is asserted instead of routing it through a
    comparison that cannot fail.  Same for slot 5 (tracer), which IS in
    the averaged set but whose blend is ``0.5*(NaN + NaN)``.
    """
    out = jdsw.dsw_transport_phase_3d(jctx, jstate, jcsw_nh, DT, KM,
                                      hydrostatic=False)
    for ax in ("x", "y"):
        pre = np.asarray(out[f"allflux_{ax}_prebarrier"], dtype=np.float64)
        post = np.asarray(out[f"allflux_{ax}"], dtype=np.float64)

        # --- the load-bearing exclusion: slot 2 (w), FINITE -----------
        w_pre, w_post = pre[..., _SLOT_W], post[..., _SLOT_W]
        assert np.isfinite(w_pre).all(), (
            f"allflux_{ax} slot {_SLOT_W + 1} (w) is not finite on the "
            f"NH arm -- the exclusion assertion below would be vacuous")
        _assert_real(w_pre, f"allflux_{ax} slot {_SLOT_W + 1} pre")
        assert _bitwise_equal(w_post, w_pre), (
            f"allflux_{ax} slot {_SLOT_W + 1} (w) MOVED across barrier "
            f"1; dyn_core.F90:856 averages iq in 1, 4, >4 only, so w "
            f"must come back byte-identical")

        # --- the never-written slots: assert the CONTRACT ------------
        for s in _SLOTS_NEVER_WRITTEN_NH:
            assert not np.isfinite(pre[..., s]).any(), (
                f"allflux_{ax} slot {s + 1} is written on this lane "
                f"after all (inline_q=False should leave it NaN); the "
                f"slot map in this file is wrong")
            assert not np.isfinite(post[..., s]).any(), (
                f"allflux_{ax} slot {s + 1} became finite across the "
                f"barrier")

        # --- non-vacuity: the averaged, WRITTEN slots must move ------
        for s in (_SLOT_DELP, _SLOT_TEMP):
            a, b = pre[..., s], post[..., s]
            assert np.isfinite(a).all() and np.isfinite(b).all(), (
                f"allflux_{ax} slot {s + 1} is not finite")
            assert np.any(a != b), (
                f"allflux_{ax} slot {s + 1} did not move across barrier "
                f"1 -- either the barrier is a no-op (in which case the "
                f"exclusion assertions above prove nothing) or the "
                f"fixture has no seam gradient")


def test_barrier1_exclusion_gate_fails_when_the_exclusion_is_removed(
        jctx, jstate, jcsw_nh, monkeypatch):
    """The MUTATION control for the gate above.

    Re-include slot 2 (``w``) in ``tab.allflux_slots`` -- the exact
    mutation this campaign's census injected at km=1 -- and the ``w``
    slot must then MOVE, i.e. the gate above would go red.  Without
    this, "w is unchanged" could be true because the barrier never
    touches anything.

    ``monkeypatch.setattr`` on the table object, so the module-scoped
    context is restored afterwards; the tables are ``__slots__``-based
    and assignable.
    """
    base = jdsw.dsw_transport_phase_3d(jctx, jstate, jcsw_nh, DT, KM,
                                       hydrostatic=False)
    monkeypatch.setattr(jctx.tab, "allflux_slots",
                        np.arange(_NSLOT, dtype=np.int32))
    mutated = jdsw.dsw_transport_phase_3d(jctx, jstate, jcsw_nh, DT, KM,
                                          hydrostatic=False)
    moved_any = False
    for ax in ("x", "y"):
        pre = np.asarray(mutated[f"allflux_{ax}_prebarrier"],
                         dtype=np.float64)[..., _SLOT_W]
        post = np.asarray(mutated[f"allflux_{ax}"],
                          dtype=np.float64)[..., _SLOT_W]
        assert np.isfinite(pre).all()
        moved_any |= bool(np.any(pre != post))
    assert moved_any, (
        "with slot 2 re-included in tab.allflux_slots the w flux STILL "
        "did not change, so the exclusion gate is not sensitive to the "
        "mutation it is supposed to catch")
    # and the delp slot must be unaffected by the mutation, so the
    # control isolates the one slot it claims to
    for ax in ("x", "y"):
        assert _bitwise_equal(
            np.asarray(mutated[f"allflux_{ax}"])[..., _SLOT_DELP],
            np.asarray(base[f"allflux_{ax}"])[..., _SLOT_DELP]), (
            f"allflux_{ax} slot 1 changed when only slot 2 was added to "
            f"the averaged set -- the mutation is not isolated")


def test_barrier1_only_touches_the_panel_seam(jctx, jstate, jcsw_nh):
    """``dyn_core.F90:877-885`` blends only ``fyy`` at
    ``j in {js, je+1}`` and ``fxx`` at ``i in {is, ie+1}``.

    Everything strictly interior must survive the barrier byte-
    identical, which is what makes the barrier an EDGE operator rather
    than a smoother.  This also pins that the 3-D lift did not widen
    the extent -- the trap that separates barrier 1 (``is..ie``) from
    barrier 2 (``is..ie+1``).
    """
    out = jdsw.dsw_transport_phase_3d(jctx, jstate, jcsw_nh, DT, KM,
                                      hydrostatic=False)
    s = _SLOT_DELP
    fx_pre = np.asarray(out["allflux_x_prebarrier"])[..., s]
    fx_post = np.asarray(out["allflux_x"])[..., s]
    fy_pre = np.asarray(out["allflux_y_prebarrier"])[..., s]
    fy_post = np.asarray(out["allflux_y"])[..., s]

    # x fluxes: only i = 0 and i = npx-1 (Fortran is, ie+1) may move
    interior_x = slice(1, NPX - 1)
    assert _bitwise_equal(fx_post[:, interior_x], fx_pre[:, interior_x]), (
        "barrier 1 moved an INTERIOR x flux; it blends i in {is, ie+1} "
        "only (dyn_core.F90:883-884)")
    assert np.any(fx_post[:, 0] != fx_pre[:, 0]) or \
        np.any(fx_post[:, NPX - 1] != fx_pre[:, NPX - 1]), (
        "no x seam column moved -- the interior assertion above is "
        "vacuous")
    # y fluxes: only j = 0 and j = npx-1 (Fortran js, je+1) may move
    interior_y = slice(1, NPX - 1)
    assert _bitwise_equal(fy_post[:, :, interior_y],
                          fy_pre[:, :, interior_y]), (
        "barrier 1 moved an INTERIOR y flux; it blends j in {js, je+1} "
        "only (dyn_core.F90:879-880)")
    assert np.any(fy_post[:, :, 0] != fy_pre[:, :, 0]) or \
        np.any(fy_post[:, :, NPX - 1] != fy_pre[:, :, NPX - 1]), (
        "no y seam row moved -- the interior assertion above is vacuous")


def test_dsw2_consumes_the_AVERAGED_fluxes_not_the_raw_ones(
        jctx, jstate, jcsw, monkeypatch):
    """The ordering that is the whole reason ``d_sw`` is split up.

    ``d_sw1`` computes fluxes, BARRIER 1 averages them
    (``dyn_core.F90:872``), and only then does ``d_sw2`` apply them
    (``:950``).  If the phase applied the RAW fluxes instead, ``delp``
    would be insensitive to whether the barrier ran.

    The lever is DROPPING SLOT 1 (delp) from the averaged set, so the
    ``delp`` flux is left raw while everything else is unchanged.  Not
    an EMPTY set: ``average_allflux_shared_edges`` reshapes by
    ``sel.size`` and a zero-length selection is a different failure,
    not a no-op.  And not a hand-rolled ``d_sw2`` update formula:
    re-deriving the kernel's arithmetic here would test my
    transcription of it rather than the phase.
    """
    base = jdsw.dsw_transport_phase_3d(jctx, jstate, jcsw, DT, KM)
    pre = np.asarray(base["allflux_x_prebarrier"])[..., _SLOT_DELP]
    post = np.asarray(base["allflux_x"])[..., _SLOT_DELP]
    assert np.any(pre != post), (
        "the barrier changed no delp flux at all, so 'averaged vs raw' "
        "is not distinguishable on this fixture and the gate below "
        "would be vacuous")

    monkeypatch.setattr(
        jctx.tab, "allflux_slots",
        np.asarray([_SLOT_TEMP, _SLOT_TRACER], dtype=np.int32))
    raw = jdsw.dsw_transport_phase_3d(jctx, jstate, jcsw, DT, KM)
    assert _bitwise_equal(
        np.asarray(raw["allflux_x"])[..., _SLOT_DELP], pre), (
        "with slot 1 dropped from the averaged set its flux must come "
        "back exactly as d_sw1 produced it")
    assert not _bitwise_equal(raw["delp"], base["delp"]), (
        "delp is IDENTICAL whether or not its flux was averaged, so "
        "d_sw2 is not consuming the barrier's output -- the barrier is "
        "bypassed")


def test_barrier_couples_faces_but_dsw1_does_not(jctx, jstate, jcsw):
    """Cadence: perturbing one face MUST reach another, and the only
    routes are the exchanges (``:652``/``:655``) and barrier 1
    (``:872``).  ``d_sw1``/``d_sw2`` themselves are face-local.

    This is the gate that separates this phase from the C-grid phase,
    where the correct assertion is the opposite one.
    """
    base = jdsw.dsw_transport_phase_3d(jctx, jstate, jcsw, DT, KM)
    pert = dict(jstate)
    pert["delp"] = jstate["delp"].at[2].add(500.0)
    got = jdsw.dsw_transport_phase_3d(jctx, pert, jcsw, DT, KM)
    d_own = np.abs(np.asarray(_win(got["delp"])[2])
                   - np.asarray(_win(base["delp"])[2])).max()
    assert d_own > 1e-9, "the perturbed face must respond"
    reached = [t for t in range(6)
               if t != 2 and np.abs(
                   np.asarray(got["allflux_x"][t][..., _SLOT_DELP])
                   - np.asarray(base["allflux_x"][t][..., _SLOT_DELP])
               ).max() > 0.0]
    assert reached, (
        "no other face saw the perturbation; barrier 1 is supposed to "
        "make the seam flux single-valued, which is a cross-face "
        "coupling")


def test_each_level_is_computed_from_its_own_level_only(jctx, jstate,
                                                        jcsw):
    """THE cadence gate. Perturb ONE level of the input; only that level
    of the output may move.  ``d_sw1``/``d_sw2`` dummies are 2-D
    (sw_core.F90:516, :1000-1012), the exchanges are horizontal and the
    barrier blends within a level, so nothing in this phase couples
    levels."""
    base = jdsw.dsw_transport_phase_3d(jctx, jstate, jcsw, DT, KM)
    pert = dict(jstate)
    pert["pt"] = jstate["pt"].at[0, :, :, 1].add(5.0)
    got = jdsw.dsw_transport_phase_3d(jctx, pert, jcsw, DT, KM)
    d = np.abs(np.asarray(_win(got["pt"])[0])
               - np.asarray(_win(base["pt"])[0]))
    assert d[:, :, 1].max() > 1e-9, "level 1 must respond"
    assert d[:, :, 0].max() == 0.0, "level 0 must be untouched"
    assert d[:, :, 2].max() == 0.0, "level 2 must be untouched"


# =====================================================================
# D. jit vs eager, and the retrace policy (tier 2)
# =====================================================================

def test_exchange_jit_matches_eager(jctx, jcsw):
    """NOT bitwise, and the reason is structural rather than empirical:
    ``ext_scalar``/``ext_vector`` route through ``k2e_remap_halo_rings``,
    ``corner_lagrange_fill``, ``c2l_ord2_cgrid_face`` and
    ``a2c_project``, every one of which evaluates ``Sum w*v`` --
    contraction sites XLA fuses into FMAs when jitted and not when
    eager."""
    fn = jdsw.make_exchange_post_pgrad_3d_jit()
    eager = jdsw.exchange_post_pgrad_3d(jctx, jcsw, KM, nord=2)
    fast = fn(jctx, jcsw, KM, nord=2)
    for name in eager:
        # TOL-PENDING: provisional bound; the orchestrator's measurement
        # job will replace this with `measured X, bound = measured x N`.
        # DO NOT SHIP.  [class: FMA contraction on a stencil sum]
        _cmp(fast[name], eager[name], f"jit-vs-eager exchange.{name}",
             1e-12)


@pytest.mark.parametrize("hydrostatic", [True, False],
                         ids=["hydro", "nh"])
def test_dsw_transport_phase_3d_jit_matches_eager(jctx, jstate, jcsw,
                                                  jcsw_nh, hydrostatic):
    """gate 3 -- an ASSERTION, not a comment.

    Not bitwise anywhere: ``d_sw1``'s flux divergences and ``d_sw2``'s
    update are ``Sum w*v``, so a few-ULP jit/eager gap is correct
    behaviour.  The one bitwise claim in this file is
    :func:`test_km1_level_assembly_is_a_pure_index_copy`, which
    compares two EAGER results.
    """
    src = jcsw if hydrostatic else jcsw_nh
    fn = jdsw.make_dsw_transport_phase_3d_jit()
    eager = jdsw.dsw_transport_phase_3d(jctx, jstate, src, DT, KM,
                                        hydrostatic=hydrostatic)
    fast = fn(jctx, jstate, src, DT, KM, hydrostatic=hydrostatic)
    assert set(fast) == set(eager)
    for name in eager:
        # TOL-PENDING: provisional bound; the orchestrator's measurement
        # job will replace this with `measured X, bound = measured x N`.
        # DO NOT SHIP.  [class: FMA contraction on a flux sum]
        _cmp(fast[name], eager[name], f"jit-vs-eager dsw.{name}", 1e-12)


def test_a_new_dt_does_not_retrace_but_a_new_cfg_does(jctx, jstate,
                                                      jcsw):
    """C3, asserted BOTH ways.

    ``dt`` dynamic is the whole reason a production time loop does not
    recompile; the non-vacuity half is that the counter DOES move when
    a genuinely static argument changes, otherwise the first assertion
    could be satisfied by a counter that never increments.

    ``cfg`` is the lever rather than ``km`` ON PURPOSE: a different
    ``km`` also changes every array SHAPE, so a retrace there would be
    attributable to the shapes and would prove nothing about the static
    split.  A different ``cfg`` leaves every shape identical.
    """
    box, wrapped = _counted(jdsw.dsw_transport_phase_3d)
    fn = jdsw.make_dsw_transport_phase_3d_jit(wrapped)
    c = jdsw.DSW_DUO_DECK
    fn(jctx, jstate, jcsw, 5.0, KM, cfg=c)
    assert box["n"] == 1
    fn(jctx, jstate, jcsw, 9.0, KM, cfg=c)
    assert box["n"] == 1, "a new dt must NOT retrace (C3)"
    fn(jctx, jstate, jcsw, 9.0, KM, cfg=c._replace(hord_tm=8))
    assert box["n"] == 2, "a new cfg MUST retrace (it is static)"


# =====================================================================
# E. guards -- each asserted AND shown non-vacuous
# =====================================================================

def test_km_above_the_remap_window_is_refused(jctx, jcsw):
    st = state_3d_to_jax(_seeded_state(4, seed=5))
    with pytest.raises(ValueError, match="fv_mapz"):
        jdsw.dsw_transport_phase_3d(jctx, st, jcsw, DT, 8)


def test_km_must_be_a_python_int(jctx, jstate, jcsw):
    with pytest.raises(TypeError, match="Python int"):
        jdsw.dsw_transport_phase_3d(jctx, jstate, jcsw, DT,
                                    jnp.asarray(KM))


def test_remap_follows_must_be_a_bool(jctx, jstate, jcsw):
    with pytest.raises(TypeError, match="remap_follows"):
        jdsw.dsw_transport_phase_3d(jctx, jstate, jcsw, DT, KM,
                                    remap_follows=1)


def test_hydrostatic_must_be_a_bool(jctx, jstate, jcsw):
    """A truthy non-bool would silently choose whether allflux slot 2
    is written at all, i.e. what the barrier is handed."""
    with pytest.raises(TypeError, match="hydrostatic"):
        jdsw.dsw_transport_phase_3d(jctx, jstate, jcsw, DT, KM,
                                    hydrostatic=1)


def test_bool_guard_is_non_vacuous(jctx, jstate, jcsw, monkeypatch):
    """Neutered, the ``1`` reaches ``d_sw1_duo``, which has its own copy
    of the same refusal -- so the call still raises, but from the
    KERNEL.  The discriminator is therefore the routine NAME in the
    message, not its wording: both read "hydrostatic must be a bool",
    and asserting on the wording would make this test pass whichever
    guard fired.
    """
    monkeypatch.setattr(jdsw, "require_bool",
                        lambda *a, **k: None)
    with pytest.raises(Exception) as ei:
        jdsw.dsw_transport_phase_3d(jctx, jstate, jcsw, DT, KM,
                                    hydrostatic=1)
    assert "dsw_transport_phase_3d:" not in str(ei.value), (
        f"the 3-D lane's guard still fired after being neutered: "
        f"{ei.value}")


def test_nord_must_be_integral(jctx, jcsw):
    with pytest.raises(ValueError, match="integral damping order"):
        jdsw.exchange_post_pgrad_3d(jctx, jcsw, KM, nord=2.7)


def test_nord_guard_is_non_vacuous(jctx, jcsw, monkeypatch):
    """Neutered, 2.7 reaches ``exchange_post_pgrad_sixface`` -- which
    has its own copy of the same refusal, so the call still raises but
    with the km=1 lane's message.  That is the proof the guard is what
    produces the 3-D lane's clean, named error."""
    monkeypatch.setattr(jdsw, "require_nord",
                        lambda fname, name, v: v)
    with pytest.raises(ValueError, match="integral damping order"):
        jdsw.exchange_post_pgrad_3d(jctx, jcsw, KM, nord=2.7)


def test_cfg_dict_is_refused_not_silently_converted(jctx, jstate, jcsw):
    """Dispatch hardening on a STATIC value.  A dict is unhashable, so
    accepting it eagerly would make the eager and compiled lanes take
    different types."""
    with pytest.raises(TypeError, match="SWConfig"):
        jdsw.dsw_transport_phase_3d(jctx, jstate, jcsw, DT, KM,
                                    cfg=dict(npdsw.DUO_DECK_CFG))


def test_cfg_from_mapping_raises_on_an_unknown_knob():
    with pytest.raises(ValueError, match="unknown knobs"):
        SWConfig.from_mapping({"hord_tm": 6, "hord_typo": 8})


def test_states_must_be_the_face_stacked_dict(jctx, state_np, jcsw):
    with pytest.raises(TypeError, match="face-stacked"):
        jdsw.dsw_transport_phase_3d(jctx, state_np, jcsw, DT, KM)


def test_csw_outs_missing_a_field_names_it(jctx, jstate, jcsw):
    bad = {k: v for k, v in jcsw.items() if k != "divg_d"}
    with pytest.raises(KeyError, match="divg_d"):
        jdsw.dsw_transport_phase_3d(jctx, jstate, bad, DT, KM)


def test_a_stagger_slip_raises_instead_of_broadcasting(jctx, jstate,
                                                       jcsw):
    """``uc`` is ``(m_b, m_a)`` and ``vc`` is ``(m_a, m_b)``; swapping
    them BROADCASTS in a later arithmetic op instead of raising."""
    bad = dict(jcsw)
    bad["uc"] = jnp.swapaxes(jcsw["uc"], 1, 2)
    with pytest.raises(ValueError, match="expected"):
        jdsw.dsw_transport_phase_3d(jctx, jstate, bad, DT, KM)


def test_shape_gate_is_non_vacuous(jctx, jstate, jcsw, monkeypatch):
    """Neutered, a stagger slip does NOT raise -- it changes the answer.

    ⛔ CORRECTED (job 9411351 measured it).  This gate used to demand
    that the neutered call raise SOMETHING deeper, on the assumption
    that a ``(m_b, m_a)`` operand handed to a ``(m_a, m_b)`` slot must
    eventually trip a shape error.  It does not: every downstream read
    is a windowed slice, and both staggers are large enough to serve
    every window, so the phase runs to completion on the wrong cells.
    ``DID NOT RAISE`` was the correct answer to the wrong question.

    That makes the gate MORE load-bearing, not less, so the assertion is
    now the strong one: with the gate off the call SUCCEEDS and the
    result DIFFERS.  Silently wrong is precisely what ``validate_stacked``
    is the only thing standing between this phase and.
    """
    good = jdsw.dsw_transport_phase_3d(jctx, jstate, jcsw, DT, KM)

    monkeypatch.setattr(jdsw, "validate_stacked", lambda *a, **k: None)
    bad_in = dict(jcsw)
    bad_in["uc"] = jnp.swapaxes(jcsw["uc"], 1, 2)
    bad = jdsw.dsw_transport_phase_3d(jctx, jstate, bad_in, DT, KM)

    # `equal_nan=True`: these stacks carry NaN scratch by construction
    # (the halo lane's tripwire fill), and plain array_equal calls two
    # identical NaN arrays UNEQUAL -- which would make `moved` non-empty
    # for every key and this gate unable to fail.
    moved = [k for k in good
             if k in bad and (good[k].shape != bad[k].shape
                              or not np.array_equal(np.asarray(good[k]),
                                                    np.asarray(bad[k]),
                                                    equal_nan=True))]
    assert moved, (
        "a transposed uc changed NOTHING in the phase output, so the "
        "shape gate it is protected by would be certifying nothing -- "
        "either the operand is unused on this arm or the fixture is "
        "symmetric under the swap")


def test_f64_gate_refuses_a_float32_state(jctx, jstate, jcsw):
    bad = dict(jstate)
    bad["pt"] = jstate["pt"].astype(jnp.float32)
    with pytest.raises((TypeError, ValueError)):
        jdsw.dsw_transport_phase_3d(jctx, bad, jcsw, DT, KM)


def test_f64_gate_is_non_vacuous(jctx, jstate, jcsw, monkeypatch):
    """Neutered, the f32 leaf travels on until ``d_sw1_duo``'s own copy
    of the same gate stops it.

    So the discriminator is the routine NAME, not the wording: every
    copy of this gate in the campaign says "must be float64", and
    asserting on the wording would pass whichever one fired.  What this
    proves is that the CLEAN, early refusal comes from the 3-D lane's
    gate rather than from luck three frames deeper.
    """
    monkeypatch.setattr(jdsw, "require_f64_jax", lambda *a, **k: None)
    bad = dict(jstate)
    bad["pt"] = jstate["pt"].astype(jnp.float32)
    with pytest.raises(Exception) as ei:
        jdsw.dsw_transport_phase_3d(jctx, bad, jcsw, DT, KM)
    assert "dsw_transport_phase_3d:" not in str(ei.value), (
        f"the 3-D lane's f64 gate still fired after being neutered: "
        f"{ei.value}")


def test_flux_cap_missing_a_capacitor_names_it(jctx, jstate, jcsw):
    caps = _stack_cap(_cap_np())
    del caps["cy"]
    with pytest.raises(KeyError, match="cy"):
        jdsw.dsw_transport_phase_3d(jctx, jstate, jcsw, DT, KM,
                                    flux_cap=caps)


def test_flux_cap_wrong_shape_is_refused(jctx, jstate, jcsw):
    caps = _stack_cap(_cap_np())
    caps["mfx"] = caps["mfx"][:, :, :N, :]
    with pytest.raises(ValueError, match="alloc_flux_capacitors"):
        jdsw.dsw_transport_phase_3d(jctx, jstate, jcsw, DT, KM,
                                    flux_cap=caps)


def test_flux_cap_guard_is_non_vacuous(jctx, jstate, jcsw, monkeypatch):
    """Neutered, the short capacitor reaches ``d_sw1_duo``, whose own
    ``_check_shape`` objects -- with a different message."""
    monkeypatch.setattr(jdsw, "_validate_flux_cap", lambda *a, **k: None)
    caps = _stack_cap(_cap_np())
    caps["cx"] = caps["cx"][:, :, :N, :]
    with pytest.raises(Exception) as ei:
        jdsw.dsw_transport_phase_3d(jctx, jstate, jcsw, DT, KM,
                                    flux_cap=caps)
    assert "alloc_flux_capacitors" not in str(ei.value), (
        "the capacitor guard still fired after being neutered")


def test_barrier_nq_mismatch_is_named(jctx, jstate, jcsw, monkeypatch):
    """The join this module owns: ``d_sw1``'s slot count and the halo
    table's ``4+nq`` must agree, and the message must name both."""
    monkeypatch.setattr(jctx.tab, "nq", 2)
    with pytest.raises(ValueError, match="4\\+nq"):
        jdsw.dsw_transport_phase_3d(jctx, jstate, jcsw, DT, KM)


def test_out_of_bounds_scatter_drops_the_update():
    """The PREMISE the barrier-layout guard rests on, MEASURED.

    The guard is load-bearing because a mis-shaped barrier operand is
    mis-blended SILENTLY rather than raising.  Which silent behaviour it
    is was stated wrongly at first: the guard said the index CLAMPS to
    the last valid one, which is JAX's GATHER rule.  For a SCATTER under
    the default ``promise_in_bounds`` mode the out-of-bounds update is
    DROPPED (``jax/_src/numpy/array_methods.py:728-731``).  This gate
    measured that and failed, which is exactly what a premise test is
    for; the guard's docstring now says DROPS.

    Both halves are asserted, so the test still fails if a future JAX
    starts raising (the guard would remain right, its reason would not):
    the array keeps its shape, and the last valid cell is UNTOUCHED.
    """
    x = jnp.zeros(3, dtype=jnp.float64)
    y = x.at[jnp.asarray([10])].set(1.0)
    assert np.asarray(y).shape == (3,)
    assert float(np.asarray(y)[2]) == 0.0, (
        "an out-of-bounds scatter wrote into the last index, i.e. it "
        "CLAMPED; the barrier-layout guard's stated justification needs "
        "updating (the guard itself is still required)")
    # ... and nothing else moved either: a dropped update is a no-op.
    assert not np.asarray(y).any(), (
        "an out-of-bounds scatter wrote somewhere; it is documented as "
        "dropped, so the whole array must be unchanged")
    # The CONTROL: the same scatter in bounds must land, or the two
    # assertions above would pass on a scatter that never works.
    z = x.at[jnp.asarray([2])].set(1.0)
    assert float(np.asarray(z)[2]) == 1.0


def test_barrier_layout_guard_refuses_a_wrong_leading_shape(jctx):
    """Exercised DIRECTLY, because the operand it protects is produced
    inside the phase and cannot be mis-shaped from outside."""
    good_x = jnp.zeros((6, NPX, N, KM, _NSLOT), dtype=jnp.float64)
    good_y = jnp.zeros((6, N, NPX, KM, _NSLOT), dtype=jnp.float64)
    # the right shapes must NOT raise, or the test below proves nothing
    jdsw._require_barrier_layout("t", jctx, good_x, good_y, KM)
    with pytest.raises(ValueError, match="DROPS the update"):
        jdsw._require_barrier_layout("t", jctx, good_y, good_y, KM)
    with pytest.raises(ValueError, match="DROPS the update"):
        jdsw._require_barrier_layout("t", jctx, good_x, good_x, KM)


def test_barrier_layout_guard_is_on_the_executed_path(jctx, jstate,
                                                      jcsw, monkeypatch):
    """A guard that is correct but never called is decoration.  Counted
    rather than assumed: the phase must invoke it exactly once."""
    calls = {"n": 0}

    def _spy(*a, **k):
        calls["n"] += 1

    monkeypatch.setattr(jdsw, "_require_barrier_layout", _spy)
    jdsw.dsw_transport_phase_3d(jctx, jstate, jcsw, DT, KM)
    assert calls["n"] == 1, (
        f"the barrier layout guard ran {calls['n']} times, expected 1")


# =====================================================================
# F. gradients
# =====================================================================

def test_exchange_adjoint_identity(jctx, jcsw):
    """gate 5, PRIMARY -- the ``:652``/``:655`` exchange chain."""
    def f(divgd, uc, vc):
        out = jdsw.exchange_post_pgrad_3d(
            jctx, {"divg_d": divgd, "uc": uc, "vc": vc}, KM, nord=2)
        return {"divg_d": _win(out["divg_d"]),
                "uc": _win(out["uc"]), "vc": _win(out["vc"])}

    # TOL-PENDING: provisional bound; the orchestrator's measurement job
    # will replace this with `measured X, bound = measured x N`.
    # DO NOT SHIP.  [class: adjoint identity, roundoff only]
    _check_adjoint("exchange d(divgd,uc,vc)", f,
                   (jcsw["divg_d"], jcsw["uc"], jcsw["vc"]), 1e-10)


def test_dsw_transport_adjoint_identity_scalars(jctx, jstate, jcsw):
    """gate 5, PRIMARY -- operand group (delp, pt).

    Tolerance-free: ``J v`` from ``jax.jvp``, ``J^T w`` from
    ``jax.vjp``, so the identity is exact in exact arithmetic and the
    residual is pure roundoff.  Run ON the state, including on the
    limiter switching surfaces ``check_grads`` cannot see.

    The objective takes the two allflux slots that are WRITTEN on the
    hydrostatic arm; including a NaN slot would make ``J v`` non-finite
    and the gate would fail for a reason that has nothing to do with
    the Jacobian.
    """
    w0, u0, v0 = jstate["w"], jstate["u"], jstate["v"]

    def f(delp, pt):
        out = jdsw.dsw_transport_phase_3d(
            jctx, {"delp": delp, "pt": pt, "w": w0, "u": u0, "v": v0},
            jcsw, DT, KM)
        return {"delp": _win(out["delp"]), "pt": _win(out["pt"]),
                "afx": out["allflux_x"][..., list(_SLOTS_WRITTEN_HYDRO)],
                "afy": out["allflux_y"][..., list(_SLOTS_WRITTEN_HYDRO)]}

    # TOL-PENDING: provisional bound; the orchestrator's measurement job
    # will replace this with `measured X, bound = measured x N`.
    # DO NOT SHIP.  [class: adjoint identity, roundoff only]
    _check_adjoint("dsw d(delp,pt)", f,
                   (jstate["delp"], jstate["pt"]), 1e-10)


def test_dsw_transport_adjoint_identity_cgrid_winds(jctx, jstate, jcsw):
    """gate 5, PRIMARY -- operand group (uc, vc), i.e. the winds the
    transport actually advects with.  Differentiating with respect to
    ``csw_outs`` and not the state is what exercises the exchange ->
    ``d_sw1`` join."""
    d0, p0, w0 = jstate["delp"], jstate["pt"], jstate["w"]
    u0, v0 = jstate["u"], jstate["v"]
    dg0 = jcsw["divg_d"]
    st = {"delp": d0, "pt": p0, "w": w0, "u": u0, "v": v0}

    def f(uc, vc):
        out = jdsw.dsw_transport_phase_3d(
            jctx, st, {"uc": uc, "vc": vc, "divg_d": dg0}, DT, KM)
        return {"delp": _win(out["delp"]),
                "afx": out["allflux_x"][..., list(_SLOTS_WRITTEN_HYDRO)]}

    # TOL-PENDING: provisional bound; the orchestrator's measurement job
    # will replace this with `measured X, bound = measured x N`.
    # DO NOT SHIP.  [class: adjoint identity, roundoff only]
    _check_adjoint("dsw d(uc,vc)", f, (jcsw["uc"], jcsw["vc"]), 1e-10)


def test_dsw_transport_adjoint_identity_nh_w(jctx, jstate, jcsw_nh):
    """gate 5, PRIMARY -- the NH arm's ``w``, which is the slot the
    barrier deliberately does NOT average.  A gradient that leaked
    through the excluded slot would show up here and nowhere else."""
    d0, p0 = jstate["delp"], jstate["pt"]
    u0, v0 = jstate["u"], jstate["v"]

    def f(w):
        out = jdsw.dsw_transport_phase_3d(
            jctx, {"delp": d0, "pt": p0, "w": w, "u": u0, "v": v0},
            jcsw_nh, DT, KM, hydrostatic=False)
        return {"w": _win(out["w"]), "dw": out["dw"],
                "afx": out["allflux_x"][..., list(_SLOTS_WRITTEN_NH)]}

    # TOL-PENDING: provisional bound; the orchestrator's measurement job
    # will replace this with `measured X, bound = measured x N`.
    # DO NOT SHIP.  [class: adjoint identity, roundoff only]
    _check_adjoint("dsw NH d(w)", f, (jstate["w"],), 1e-10)


def test_capacitor_adjoint_identity(jctx, jstate, jcsw):
    """gate 5, PRIMARY -- the capacitors are an ACCUMULATOR the caller
    threads across substeps, so a broken adjoint there breaks every
    multi-substep gradient.  ``mfx``/``mfy`` additionally go through the
    D4 window slice, which is where an index error would hide."""
    caps = _stack_cap(_cap_np())
    cx0, cy0 = caps["cx"], caps["cy"]

    def f(mfx, mfy):
        out = jdsw.dsw_transport_phase_3d(
            jctx, jstate, jcsw, DT, KM,
            flux_cap={"mfx": mfx, "mfy": mfy, "cx": cx0, "cy": cy0})
        return {"mfx": out["mfx"], "mfy": out["mfy"]}

    # TOL-PENDING: provisional bound; the orchestrator's measurement job
    # will replace this with `measured X, bound = measured x N`.
    # DO NOT SHIP.  [class: adjoint identity, roundoff only]
    _check_adjoint("dsw d(mfx,mfy)", f, (caps["mfx"], caps["mfy"]), 1e-10)


@pytest.mark.parametrize("order", [1, 2])
def test_dsw_transport_check_grads(jctx, jstate, jcsw, order):
    """gate 5, SUPPLEMENT -- smooth-region finite differences.

    ``order=1`` and ``order=2`` are SEPARATE parametrised IDs so an
    FD-resolution failure at order 2 can never be confused with a wrong
    Jacobian (STATE lesson 12).

    Differentiated with respect to two SCALAR multipliers rather than
    the fields themselves: ``delp`` spans 1e3 to 1e4 across the column,
    and finite-differencing a 3-D field with that spread leaves only a
    few significant digits in the difference -- an FD failure that says
    nothing about the Jacobian.  Scaling keeps the FD well conditioned
    while exercising the same directional derivative.

    This gate is EXPECTED to be the fragile one: ``d_sw1`` runs the
    ``xppm``/``yppm`` limiter pipelines, and a finite-difference step
    that crosses a limiter switch produces a discrepancy that is not
    rounding-scale.  The adjoint identity above is the gate that
    carries the claim; if this one goes red the measurement job should
    report which, not silently widen the bound.
    """
    one = jnp.asarray(1.0)
    d0, p0 = jstate["delp"], jstate["pt"]
    w0, u0, v0 = jstate["w"], jstate["u"], jstate["v"]

    def f(a, b):
        out = jdsw.dsw_transport_phase_3d(
            jctx, {"delp": a * d0, "pt": b * p0, "w": w0, "u": u0,
                   "v": v0}, jcsw, DT, KM)
        return (jnp.sum(_win(out["delp"]) ** 2)
                + jnp.sum(_win(out["pt"]) ** 2))

    # TOL-PENDING: provisional bounds; the orchestrator's measurement
    # job will replace these with `measured X, bound = measured x N`.
    # DO NOT SHIP.  [class: smooth-region finite differences]
    check_grads(f, (one, one), order=order, modes=("fwd", "rev"),
                atol=2e-2, rtol=2e-2, eps=1e-4)
