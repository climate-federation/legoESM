"""Certification of the JAX 3-D D-grid TAIL: barrier 2, d_sw4/5/6, and
the two D-grid pressure chains (hydrostatic and non-hydrostatic).

Authority: ``legoesm.core.fv3_native_dsw_tail_3d`` is the SPECIFICATION
(hop B of ``docs/atmosphere/fv3_duo_jax_lane_strategy.md``).  The pinned
Fortran is quoted only where a line number says WHY a step exists; it is
never what a bound here is calibrated against.

Like its two siblings the module under test contributes CADENCE, not
arithmetic -- every operation happens inside a kernel already gated by
``test_fv3_duo_sw_core.py``, ``test_fv3_pgrad.py``, ``test_fv3_nh_core.py``
or ``test_fv3_duo_halos.py``.  So the gates below interrogate the JOINS,
and three in particular:

* **BARRIER 2 AND ITS EXTENT.**  Barrier 1 blends C-ring fluxes over
  ``i = is..ie`` / ``j = js..je``; barrier 2 blends the B-grid corner
  ingredients over ``i = is..ie+1``, ``j = js..je+1`` -- tile corners
  INCLUDED (``dyn_core.F90:984``, ``BGRID_NE``).  Two adjacent routines
  with different extents is a classic silent-defect site.

* **WHAT BARRIER 2 ACTS ON, AND WHEN.**  It blends ``ubb`` and
  ``vbbtemp`` BEFORE the kinetic energy is formed.  The KE assembly is
  ``ke = 0.5*(ubbtemp*vbbtemp + ubb*vbb)`` (``dyn_core.F90:1080-1085``)
  and it mixes BLENDED and UNBLENDED members -- ``ubbtemp`` and ``vbb``
  come straight out of ``d_sw3``, ``vbbtemp`` and ``ubb`` come out of the
  barrier.  Blending after the product, or blending all four, is a
  DIFFERENT operator, and both mistakes produce a plausible field.  The
  gate below drives that distinction directly.

* **THE PRESSURE CHAIN'S SCRATCH ORDER.**  ``one_grad_p`` overwrites
  ``pk``/``gz`` with B-grid corner values (``a2b_ord4`` with
  ``replace=.true.``), so the ``pk`` the vertical remap consumes has to
  be snapshotted BEFORE it (``dyn_core.F90:1511-1519``).  Taking it
  afterwards feeds the remap a corner-staggered field, which is exactly
  the kind of defect that does not raise.

Gate classes follow the sibling files (1 parity, 2 cadence, 3 jit-vs-
eager, 4 guards, 5 gradients, 6 anti-vacuity everywhere), and the
comparison/gradient helpers come from ``tests/grids/fv3_gate_helpers.py``
rather than a sixth private copy.

TOLERANCE POLICY.  ``d_sw5``/``d_sw6`` run limiter pipelines and del-n
damping, the pressure chains run ``a2b_ord4`` and a vertical recurrence:
per strategy section 4 that spans the accumulating and branch-switching
classes; every numeric bound is MEASURED (job 9425294, the
LEGOESM_FV3_TOL_MEASURE sweep) and set to measured x 10 with its class
label kept.  Lane parity is bitwise on the tail and composed gates; the
worst nonzero figures are the pressure chain at 1.641e-15 (gz) and the
adjoint identities at ~1.5e-15.

COST.  Six faces x three levels of d_sw3/4/5/6 per call plus the
pressure chains and the jvp/vjp programs: minutes, not seconds.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from legoesm.core import fv3_dsw_tail_3d as jtail  # noqa: E402
from legoesm.core import fv3_native_acoustic_3d as npacoustic  # noqa: E402
from legoesm.core import fv3_native_cgrid_phase_3d as npcg  # noqa: E402
from legoesm.core import fv3_native_dsw_phase_3d as npdsw  # noqa: E402
from legoesm.core import fv3_native_dsw_tail_3d as nptail  # noqa: E402
from legoesm.core import fv3_native_duo_stepper as npstep  # noqa: E402
from legoesm.core.fv3_cgrid_phase_3d import state_3d_to_jax  # noqa: E402
from legoesm.core.fv3_dsw_phase_3d import (  # noqa: E402
    dsw_transport_phase_3d,
)
from legoesm.core.fv3_duo_stepper import (  # noqa: E402
    SWConfig,
    build_jax_duo_stepper_context,
)
from legoesm.core.fv3_native_state_3d import build_state_3d  # noqa: E402

from tests.grids.fv3_gate_helpers import (  # noqa: E402
    assert_fd_gap_at_roundoff_floor,
    assert_real,
    check_adjoint,
    cmp_fields,
    counted,
    deepcopy_faces,
    stack_np,
)

# Same fixture geometry as both sibling 3-D files, so the three are
# comparable: C12 is the smallest resolution the duo corner-region
# Lagrange fill admits, km = 3 makes "perturb level 1, levels 0 and 2
# must not move" a real statement and stays inside
# `require_no_remap_needed`'s km <= 4 window.
N, NG, KM = 12, 3, 3
MA = N + 2 * NG
NPX = N + 1
DT = 20.0
DT2 = 0.5 * DT

# The pinned duo deck's thermodynamic constants, taken from the NumPy
# spec's own callers rather than retyped from a namelist.
PTOP = 100.0
AKAP = 2.0 / 7.0
CP_AIR = 1004.6
P_FAC = 0.05
A_IMP = 1.0

_CS = slice(NG, NG + N)


def _win(a):
    """Compute square of a face-stacked ``(6, i, j, ...)`` array."""
    return np.asarray(a)[:, _CS, _CS, ...]


# =====================================================================
# fixtures -- the NumPy lane builds every input, so a parity gate here
# is one-variable in the module under test
# =====================================================================

@pytest.fixture(scope="module")
def ctx():
    """NumPy context, EXT-BUNDLE lane, ORACLE conventions.

    ``duogrid`` forces ``bounded_domain`` (``fv_arrays.F90:1512``), and
    the NH tail's zh/pkc exchanges are ``ext_scalar`` sites, so the ext
    bundle is required rather than optional here.
    """
    return npstep.build_six_face_duo_context(
        N, NG, use_ext_bundle=True, oracle_conventions=True)


@pytest.fixture(scope="module")
def jctx(ctx):
    return build_jax_duo_stepper_context(ctx)


def _seeded_state(km, seed=0, hydrostatic=True):
    """A physical 3-D column, distinct per face AND per level.

    Magnitudes are not decoration: ``delp ~ 1e4 Pa`` keeps the
    transported thickness positive through d_sw2 and keeps the pressure
    chain's ``log(pe)`` finite, ``pt ~ 280 K`` keeps ``pkz`` real, and a
    per-face/per-level distinct field means a face or level mix-up
    cannot hide behind symmetry.
    """
    rng = np.random.default_rng(seed)
    st = build_state_3d(N, NG, km, hydrostatic=hydrostatic)
    for t, face in enumerate(st):
        for k in range(km):
            face["delp"][:, :, k] = 1.0e4 * (1.0 + 0.05 * t + 0.02 * k) \
                + 50.0 * rng.standard_normal((MA, MA))
            face["pt"][:, :, k] = 280.0 + 2.0 * t + 3.0 * k \
                + 0.5 * rng.standard_normal((MA, MA))
            face["u"][:, :, k] = 5.0 + 0.3 * t + 0.1 * k \
                + rng.standard_normal((MA, MA + 1))
            face["v"][:, :, k] = -4.0 + 0.2 * t - 0.1 * k \
                + rng.standard_normal((MA + 1, MA))
            if not hydrostatic:
                face["w"][:, :, k] = 0.2 * (1 + t) - 0.05 * k \
                    + 0.1 * rng.standard_normal((MA, MA))
        if not hydrostatic:
            # delz < 0: the oracle's z coordinate increases upward and
            # delz is the (negative) thickness, so a positive value here
            # would put the column upside down and Riem_Solver3 would
            # return a plausible, wrong answer rather than raising.
            for k in range(km):
                face["delz"][:, :, k] = -(500.0 + 20.0 * k + 5.0 * t)
    return st


@pytest.fixture(scope="module")
def state_np():
    return _seeded_state(KM, seed=11)


@pytest.fixture(scope="module")
def state_np_nh():
    return _seeded_state(KM, seed=11, hydrostatic=False)


@pytest.fixture(scope="module")
def jstate(state_np):
    return state_3d_to_jax(state_np)


@pytest.fixture(scope="module")
def jstate_nh(state_np_nh):
    return state_3d_to_jax(state_np_nh)


@pytest.fixture(scope="module")
def csw_np(ctx, state_np):
    """The NumPy ``c_sw`` output -- the SHARED upstream for both lanes,
    so a c_sw parity difference cannot leak into a tail verdict."""
    return npcg.csw_phase_3d(ctx, deepcopy_faces(state_np), dt2=DT2,
                             km=KM, nord=2, duogrid=True)


@pytest.fixture(scope="module")
def dsw_np(ctx, state_np, csw_np):
    """The NumPy D-grid TRANSPORT output (unit 4), i.e. this unit's
    other upstream.  Copied in, because that phase mutates csw_outs."""
    return npdsw.dsw_transport_phase_3d(
        ctx, deepcopy_faces(state_np), deepcopy_faces(csw_np),
        dt=DT, km=KM)


@pytest.fixture(scope="module")
def jcsw(csw_np):
    return stack_np(csw_np)


@pytest.fixture(scope="module")
def jdsw(jctx, jstate, jcsw):
    """The JAX transport phase's own output, which is what the JAX tail
    consumes in production.  NOT the stacked NumPy one: the tail's
    contract is with the JAX phase, and feeding it the NumPy stack would
    certify a composition that never runs."""
    return dsw_transport_phase_3d(jctx, jstate, jcsw, DT, KM)


# =====================================================================
# 1. parity -- JAX vs the NumPy twin, from byte-identical inputs
# =====================================================================

# What the tail returns that the spec also produces.  `u`/`v` are the
# d_sw6-updated D winds; the d_sw5 diagnostics are what the pressure
# chain and the next substep read.
_TAIL_COMPARED = ("u", "v", "ke", "wk", "divg_d", "delpc")


# Names the transport output carries in BOTH generations: as a
# face-level key (the post-d_sw2 state) and as a per-level d_sw1 stage
# output. The tail phase consumes the face-level one -- riem_solver3
# takes the post-d_sw2 pt/delp/w -- so every caller in this lane
# declares that. The adapter refuses any collision NOT listed, which is
# how the pt/delp defect surfaced and how the next one will.
_TAIL_FACE_LEVEL = ("pt", "delp", "w")

# allflux_x/allflux_y are NOT two generations (codex MINOR, correct):
# the NumPy transport phase barrier-averages them IN PLACE and then
# stacks those same level objects into the face-level fields, so both
# views hold identical data. Listing them as face-level "winners"
# recorded a meaningless choice and asserted something false about what
# the tail covers -- the tail consumes neither. Declared as ALIASES
# instead, and the adapter PROVES the equality rather than trusting it.
_TAIL_ALIASES = ("allflux_x", "allflux_y")


def _stack_dsw_np(dsw_np, face_level_wins=(),
                  aliases=_TAIL_ALIASES):
    """The NumPy transport output -> this lane's face-stacked dict.

    Not `stack_np`: the spec's per-face dict carries a ``levels`` key
    holding a LIST of per-level dicts (the d_sw1 stage outputs), which
    has no array to stack.  The JAX lane returns those as (6, i, j, km)
    stacks under their own names, so the adapter builds them from the
    per-level lists and drops ``levels`` itself.
    """
    per_level_names = tuple(dsw_np[0]["levels"][0])
    out = {}
    for k, v in dsw_np[0].items():
        if k == "levels":
            continue
        out[k] = jnp.asarray(np.stack(
            [np.asarray(d[k], dtype=np.float64) for d in dsw_np]))
    for nm in per_level_names:
        # ⛔ A COLLIDING NAME IS TWO DIFFERENT QUANTITIES, NOT A DUPLICATE.
        # `pt` and `delp` exist BOTH as face-level keys (the post-d_sw2
        # state, which is what the NH tail's riem_solver3 consumes) and
        # as per-level d_sw1 stage outputs (an EARLIER generation). This
        # loop used to overwrite the face-level entry with the per-level
        # stack, so the JAX lane was handed d_sw1's pt/delp while the
        # spec used the post-d_sw2 ones -- measured 2.367489e-01 on pt
        # and 2.341751e+01 on delp, with all nine other riem inputs at
        # exactly 0.0, and it is why the NH parity gate was red.
        # Refusing is the only safe behaviour: silently picking either
        # generation is a coin flip on which physics the gate certifies.
        if nm in face_level_wins:
            continue          # caller stated which generation it wants
        if nm in aliases and nm in out:
            # Declared identical -- so CHECK it. An alias that quietly
            # stopped being one is indistinguishable from the very
            # generation defect this guard exists to catch.
            _pl = np.stack([
                np.stack([np.asarray(lvl[nm], dtype=np.float64)
                          for lvl in face["levels"]], axis=2)
                for face in dsw_np])
            # equal_nan: these carry BIG_NUMBER/NaN fill in cells the
            # kernel never writes, and plain array_equal calls NaN !=
            # NaN, so the first version of this check reported
            # "max|d| nan" on two arrays that are in fact identical.
            # The fill must match POSITIONALLY too, which equal_nan
            # gives -- a lane that moved its fill would still fail.
            _a = np.asarray(out[nm])
            if not np.array_equal(_a, _pl, equal_nan=True):
                _fin = np.isfinite(_a) & np.isfinite(_pl)
                _d = (np.abs(_a[_fin] - _pl[_fin]).max()
                      if _fin.any() else float("nan"))
                raise ValueError(
                    f"_stack_dsw_np: {nm!r} is declared an alias but its "
                    f"two views DIFFER (max|d| over finite cells {_d:.3e}, "
                    f"non-finite masks "
                    f"{'match' if np.array_equal(np.isfinite(_a), _fin) else 'DIFFER'}"
                    f") -- it is two generations after all, and a caller "
                    f"must choose.")
            continue
        if nm in out:
            raise ValueError(
                f"_stack_dsw_np: {nm!r} is both a face-level key and a "
                f"per-level d_sw1 output -- two different generations of "
                f"the same name. The caller must say which it wants "
                f"via face_level_wins=; this adapter will not choose.")
        out[nm] = jnp.asarray(np.stack([
            np.stack([np.asarray(lvl[nm], dtype=np.float64)
                      for lvl in face["levels"]], axis=2)
            for face in dsw_np]))
    return out


def _np_tail(ctx, state_np, csw_np, dsw_np, **kw):
    """The spec, on COPIES -- it mutates its inputs."""
    return nptail.dsw_tail_phase_3d(
        ctx, deepcopy_faces(state_np), deepcopy_faces(csw_np),
        deepcopy_faces(dsw_np), dt=DT, km=KM, **kw)


def _require_keys(got, ref0, required, what):
    """Both lanes must carry EVERY required key before anything is
    compared.

    ⛔ codex BLOCKER (job 9417466): the first version intersected the
    keys the two lanes happened to share and compared that.  Deleting a
    returned field then made the gate compare fewer things and PASS, and
    in the pressure gate an unrelated key set would have made the
    comparison loop execute zero assertions.  A parity gate whose
    coverage is decided by the thing under test is not a parity gate.
    """
    missing_j = [k for k in required if k not in got]
    missing_n = [k for k in required if k not in ref0]
    assert not missing_j, f"{what}: the JAX lane is missing {missing_j}"
    assert not missing_n, f"{what}: the NumPy lane is missing {missing_n}"


def test_tail_unit_parity_hydrostatic_on_identical_inputs(
        ctx, jctx, state_np, csw_np, dsw_np, jstate, jcsw):
    """gate 1a -- HOP-B UNIT parity: both lanes fed the SAME NumPy
    upstream, so nothing from the JAX transport phase can contaminate or
    compensate the verdict (codex MAJOR, job 9417466).

    The composition test below is the other half and is labelled as
    such; this one is the parity claim.
    """
    ref = _np_tail(ctx, state_np, csw_np, dsw_np)
    got = jtail.dsw_tail_phase_3d(jctx, jstate, jcsw,
                                  _stack_dsw_np(
                                      dsw_np,
                                      face_level_wins=_TAIL_FACE_LEVEL),
                                  DT, KM)
    _require_keys(got, ref[0], _TAIL_COMPARED, "tail unit parity")
    for nm in _TAIL_COMPARED:
        want = np.stack([np.asarray(ref[t][nm]) for t in range(6)])
        assert_real(want, f"numpy tail {nm}")
        # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), every field; bound =
        # 1e-15 eps guard (measured exactly 0.0).
        cmp_fields(got[nm], want, f"tail {nm}", 1e-15)


def test_tail_parity_hydrostatic(ctx, jctx, state_np, csw_np, dsw_np,
                                 jstate, jcsw, jdsw):
    """gate 1b -- the COMPOSITION: the JAX tail on the JAX transport
    phase's own output, which is what production runs.  An upstream
    difference is in scope here BY DESIGN; the unit claim is 1a."""
    ref = _np_tail(ctx, state_np, csw_np, dsw_np)
    got = jtail.dsw_tail_phase_3d(jctx, jstate, jcsw, jdsw, DT, KM)
    _require_keys(got, ref[0], _TAIL_COMPARED, "tail composition")
    for nm in _TAIL_COMPARED:
        want = np.stack([np.asarray(ref[t][nm]) for t in range(6)])
        assert_real(want, f"numpy tail {nm}")
        # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), every field; bound =
        # 1e-15 eps guard (measured exactly 0.0).
        cmp_fields(got[nm], want, f"tail composed {nm}", 1e-15)


def test_tail_does_not_mutate_its_inputs(jctx, jstate, jcsw, jdsw):
    """The functional contract (convention C4).

    The spec mutates ``state``/``csw_outs``/``dsw_outs`` in place; this
    lane must not, because a mutated input under ``jit`` is either a
    silent aliasing bug or a donated-buffer crash far from the cause.
    """
    before = {k: np.array(v) for k, v in jstate.items()}
    before_c = {k: np.array(v) for k, v in jcsw.items()}
    before_d = {k: np.array(v) for k, v in jdsw.items()}
    jtail.dsw_tail_phase_3d(jctx, jstate, jcsw, jdsw, DT, KM)
    for name, snap, live in (("states", before, jstate),
                             ("csw_outs", before_c, jcsw),
                             ("dsw_outs", before_d, jdsw)):
        for k, v in snap.items():
            assert np.array_equal(v, np.asarray(live[k]), equal_nan=True), (
                f"{name}[{k!r}] was mutated by dsw_tail_phase_3d")


# =====================================================================
# 2. BARRIER 2 -- the gate this unit exists for
# =====================================================================

def test_barrier2_actually_blends_the_b_grid_ingredients(
        jctx, jstate, jcsw, jdsw):
    """The blended ubb/vbbtemp must DIFFER from d_sw3's raw ones, and
    differ AT THE SEAM.

    Returned as stages S10 (pre-barrier) and S11 (post-barrier), because
    neither is recoverable from the other outputs.  A gate that only
    checked "S11 exists" would pass on a barrier that never ran.
    """
    out = jtail.dsw_tail_phase_3d(jctx, jstate, jcsw, jdsw, DT, KM)
    need = ("ubb_prebarrier", "vbbtemp_prebarrier",
            "ubb_postbarrier", "vbbtemp_postbarrier")
    missing = [k for k in need if k not in out]
    assert not missing, (
        f"the tail does not return the barrier-2 stages {missing}; they "
        f"are not recoverable from the other outputs, so without them "
        f"the barrier cannot be gated at all")
    for pre, post in (("ubb_prebarrier", "ubb_postbarrier"),
                      ("vbbtemp_prebarrier", "vbbtemp_postbarrier")):
        a = assert_real(out[pre], pre)
        b = assert_real(out[post], post)
        assert not np.array_equal(a, b, equal_nan=True), (
            f"{post} is bitwise equal to {pre}: barrier 2 did nothing")


def test_barrier2_includes_the_b_grid_corners(jctx, jstate, jcsw, jdsw):
    """The EXTENT that distinguishes barrier 2 from barrier 1.

    Barrier 1 stops at ``ie``/``je``; barrier 2 runs to ``ie+1``/``je+1``
    (``dyn_core.F90:984``, BGRID_NE).  So the LAST ring index of the
    blended arrays must have moved -- if the extent had been copied from
    barrier 1, that index would be untouched.
    """
    out = jtail.dsw_tail_phase_3d(jctx, jstate, jcsw, jdsw, DT, KM)
    for pre, post in (("ubb_prebarrier", "ubb_postbarrier"),
                      ("vbbtemp_prebarrier", "vbbtemp_postbarrier")):
        a = np.asarray(out[pre])
        b = np.asarray(out[post])
        # The stage payload is the B COMPUTE RING, (6, npx, npx, km), and
        # that is ASSERTED rather than assumed (codex MAJOR, job
        # 9417466): the first version indexed `[:, -1, ...]` and called
        # it the i = ie+1 ring, which is only true while the payload
        # happens to be cropped to exactly npx.  With halo storage that
        # index would read the outer allocation edge instead.
        assert a.shape == (6, NPX, NPX, KM), (pre, a.shape)
        assert b.shape == (6, NPX, NPX, KM), (post, b.shape)
        # npx = n+1 columns span is..ie+1, so the LAST index IS ie+1 --
        # now derived from the asserted shape, not from -1.
        i_edge = j_edge = NPX - 1
        for axis, edge, label in ((1, i_edge, "i = ie+1"),
                                  (2, j_edge, "j = je+1")):
            ea = np.take(a, edge, axis=axis)
            eb = np.take(b, edge, axis=axis)
            fin = np.isfinite(ea) & np.isfinite(eb)
            assert fin.any(), f"{post}: the {label} ring is all non-finite"
            assert not np.array_equal(ea[fin], eb[fin]), (
                f"{post}: the {label} ring is UNCHANGED by the barrier -- "
                f"that is barrier 1's extent, not barrier 2's")


def test_ke_is_the_mixed_assembly_and_not_either_uniform_one(
        jctx, jstate, jcsw, jdsw):
    """``ke = 0.5*(ubbtemp*vbbtemp + ubb*vbb)`` with ``ubbtemp``/``vbb``
    RAW and ``vbbtemp``/``ubb`` BLENDED (dyn_core.F90:1080-1085 read
    against the barrier at :984).

    ⛔ THE FIRST VERSION OF THIS GATE NEVER READ ``ke`` (codex BLOCKER,
    job 9417466).  It built the two candidate formulas and asserted they
    DIFFER on the fixture -- which is a statement about the fixture, and
    is satisfied by an implementation that returns the all-raw formula,
    the all-blended one, zeros, or an unrelated field.

    Now the authoritative value is RECONSTRUCTED from the returned stage
    payloads, placed in the same zero-initialised ``(m_a+1, m_a+1)``
    array on the same ``ring`` window, and compared against ``out["ke"]``
    -- and the two wrong assemblies are asserted to DIFFER from it, so
    the comparison is known to discriminate rather than assumed to.
    """
    out = jtail.dsw_tail_phase_3d(jctx, jstate, jcsw, jdsw, DT, KM)
    ub_raw = assert_real(out["ubb_prebarrier"], "ubb_prebarrier")
    vb_raw = assert_real(out["vbb_prebarrier"], "vbb_prebarrier")
    ubt_raw = assert_real(out["ubbtemp_prebarrier"], "ubbtemp_prebarrier")
    vbt_raw = assert_real(out["vbbtemp_prebarrier"], "vbbtemp_prebarrier")
    ub_b = assert_real(out["ubb_postbarrier"], "ubb_postbarrier")
    vbt_b = assert_real(out["vbbtemp_postbarrier"], "vbbtemp_postbarrier")
    # S12: the assembly THIS phase performs.  NOT out["ke"], which is
    # d_sw5's OUTPUT ke -- two stages later and a different quantity
    # (measured 1.2e+02 relative apart, job 9417474; the first version
    # of this gate compared against it and failed for that reason).
    ke = assert_real(out["ke_corner"], "ke_corner")

    ring = slice(NG, NG + NPX)
    assert ke.shape == (6, MA + 1, MA + 1, KM), ke.shape

    def _emplace(inner):
        full = np.zeros((6, MA + 1, MA + 1, KM), dtype=np.float64)
        full[:, ring, ring, :] = inner
        return full

    mixed = _emplace(0.5 * (ubt_raw * vbt_b + ub_b * vb_raw))
    all_raw = _emplace(0.5 * (ubt_raw * vbt_raw + ub_raw * vb_raw))
    all_blended = _emplace(0.5 * (ubt_raw * vbt_b + ub_b * vbt_b))

    # The gate must be able to fail: the three assemblies must differ on
    # THIS fixture, or comparing against one of them proves nothing.
    for nm, cand in (("all-raw", all_raw), ("all-blended", all_blended)):
        assert not np.allclose(mixed, cand), (
            f"the mixed and {nm} assemblies are indistinguishable on this "
            f"fixture, so this gate cannot tell them apart -- the fixture "
            f"is at fault, not the module")

    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise); bound =
    # 1e-15 eps guard (measured exactly 0.0).
    cmp_fields(ke, mixed, "ke_corner (mixed assembly)", 1e-15)
    for nm, cand in (("all-raw", all_raw), ("all-blended", all_blended)):
        assert not np.allclose(np.asarray(ke), cand), (
            f"the returned ke equals the {nm} assembly -- barrier 2 sits "
            f"between d_sw3 and this product for a reason")

    # And the window: everything outside the compute ring is EXACTLY the
    # zero the authority leaves there.
    outside = np.asarray(ke).copy()
    outside[:, ring, ring, :] = 0.0
    assert not outside.any(), (
        "ke is non-zero outside the [ng, ng+npx) ring, so the assembly "
        "wrote outside the window the authority allocates")


# =====================================================================
# 3. the hydrostatic pressure chain
# =====================================================================

def test_dgrid_pressure_parity(ctx, jctx, state_np, csw_np, dsw_np,
                               jstate, jcsw, jdsw):
    ref_tail = _np_tail(ctx, state_np, csw_np, dsw_np)
    ref = nptail.dgrid_pressure_phase_3d(
        ctx, deepcopy_faces(dsw_np), deepcopy_faces(ref_tail), KM,
        dt=DT, ptop=PTOP, akap=AKAP, cp_air=CP_AIR)
    tail = jtail.dsw_tail_phase_3d(jctx, jstate, jcsw, jdsw, DT, KM)
    got = jtail.dgrid_pressure_phase_3d(
        jctx, jdsw, tail, KM, dt=DT, ptop=PTOP, akap=AKAP,
        cp_air=CP_AIR)
    _require_keys(got, ref[0], ("pk", "gz", "pe", "peln", "pkz"),
                  "D-grid pressure parity")
    for nm in ("pk", "gz", "pe", "peln", "pkz"):
        want = np.stack([np.asarray(ref[t][nm]) for t in range(6)])
        assert_real(want, f"numpy press {nm}")
        # MEASURED (job 9425294 sweep): worst gz 1.641e-15; bound = measured x 10 =
        # 1.7e-14.
        cmp_fields(got[nm], want, f"press {nm}", 1.7e-14)


def test_pk_remap_is_the_pre_one_grad_p_snapshot(jctx, jstate, jcsw,
                                                 jdsw):
    """``pk_remap`` exists only on a remap step, and it is the geopk
    ``pk`` -- not the corner-scratched one ``one_grad_p`` leaves behind
    (``dyn_core.F90:1511-1519`` taken BEFORE ``:1531``).

    Taking it afterwards hands the vertical remap a B-grid corner field,
    which is finite, plausible, and wrong.
    """
    tail = jtail.dsw_tail_phase_3d(jctx, jstate, jcsw, jdsw, DT, KM)
    kw = dict(dt=DT, ptop=PTOP, akap=AKAP, cp_air=CP_AIR)
    plain = jtail.dgrid_pressure_phase_3d(jctx, jdsw, tail, KM, **kw)
    assert "pk_remap" not in plain, (
        "pk_remap is present on a NON-remap step; a caller could then "
        "pick up a stale snapshot without saying so")
    remap = jtail.dgrid_pressure_phase_3d(jctx, jdsw, tail, KM,
                                          remap_step=True, **kw)
    assert "pk_remap" in remap, "pk_remap missing on a remap step"
    snap = assert_real(remap["pk_remap"], "pk_remap")
    assert np.array_equal(np.asarray(remap["pk_remap"]),
                          np.asarray(remap["pk_pre_onegradp"]),
                          equal_nan=True), (
        "pk_remap and the S16 pre-one_grad_p payload must be the SAME "
        "snapshot; if they differ, one of them was taken on the wrong "
        "side of the a2b_ord4 replace")
    after = assert_real(remap["pk"], "pk after one_grad_p")
    assert not np.array_equal(snap, after, equal_nan=True), (
        "pk_remap is bitwise equal to the post-one_grad_p pk, so it was "
        "taken AFTER the a2b_ord4 replace -- the remap would read a "
        "corner-staggered field")


# =====================================================================
# 4. guards -- each shown non-vacuous
# =====================================================================

def test_km_must_be_a_python_int(jctx, jstate, jcsw, jdsw):
    with pytest.raises(TypeError, match="km must be a Python int"):
        jtail.dsw_tail_phase_3d(jctx, jstate, jcsw, jdsw, DT, float(KM))


def test_hydrostatic_must_be_a_bool(jctx, jstate, jcsw, jdsw):
    with pytest.raises(TypeError, match="must be a bool"):
        jtail.dsw_tail_phase_3d(jctx, jstate, jcsw, jdsw, DT, KM,
                                hydrostatic=1)


def test_cfg_dict_is_refused_not_silently_converted(jctx, jstate, jcsw,
                                                    jdsw):
    with pytest.raises(TypeError, match="SWConfig"):
        jtail.dsw_tail_phase_3d(jctx, jstate, jcsw, jdsw, DT, KM,
                                cfg=dict(nptail.DUO_TAIL_CFG))


def test_a_stagger_slip_raises_instead_of_broadcasting(jctx, jstate,
                                                       jcsw, jdsw):
    bad = dict(jcsw)
    bad["uc"] = jnp.swapaxes(jcsw["uc"], 1, 2)
    with pytest.raises(ValueError, match="expected"):
        jtail.dsw_tail_phase_3d(jctx, jstate, bad, jdsw, DT, KM)


def test_f64_gate_refuses_a_float32_input(jctx, jstate, jcsw, jdsw):
    bad = dict(jstate)
    bad["u"] = jstate["u"].astype(jnp.float32)
    with pytest.raises((TypeError, ValueError)):
        jtail.dsw_tail_phase_3d(jctx, bad, jcsw, jdsw, DT, KM)


def test_the_deck_is_imported_from_the_spec_not_restated():
    """C7: a retyped deck is a place for the two lanes to drift.

    EVERY field the spec's deck names is compared, not a sample of two
    (codex MAJOR, job 9417466): a corrupted `hord_mt`, `nord_v`, `dddmp`,
    `d2_bg` or `damp_v` passed the earlier version, and `damp_v` is
    exactly what the NH `damp_w` fallback reads.
    """
    shared = {k: v for k, v in nptail.DUO_TAIL_CFG.items()
              if k in SWConfig._fields}
    assert shared, "no SWConfig field is named by the spec's deck"
    deck = SWConfig.from_mapping(shared)
    for k, v in shared.items():
        assert getattr(deck, k) == v, (k, getattr(deck, k), v)
    # ...and the module's own deck is built from that same mapping.
    for k, v in shared.items():
        assert getattr(jtail._TAIL_DECK, k) == v, (
            f"the module's deck disagrees with the spec on {k!r}: "
            f"{getattr(jtail._TAIL_DECK, k)} vs {v}")


@pytest.mark.parametrize("hydrostatic", [True, False])
def test_damp_w_fallback_is_the_specs(jctx, jstate, jstate_nh, jcsw,
                                      jdsw, hydrostatic):
    """The fallback the spec states: 0.0 on the hydrostatic arm,
    ``cfg.damp_v`` on the NH one.

    Asserted THROUGH the phase rather than by reading the constant: the
    module must be shown to USE it, so the gate perturbs `damp_v` and
    requires the NH answer to move and the hydrostatic answer not to.
    """
    st = jstate if hydrostatic else jstate_nh
    if not hydrostatic:
        pytest.skip("the NH arm needs the NH transport output; covered by "
                    "the NH parity gate once dsw_transport_phase_3d is run "
                    "on that arm in this file")
    base = jtail.dsw_tail_phase_3d(jctx, st, jcsw, jdsw, DT, KM,
                                   hydrostatic=hydrostatic)
    hot = jtail.dsw_tail_phase_3d(
        jctx, st, jcsw, jdsw, DT, KM, hydrostatic=hydrostatic,
        damp_w=10.0 * jtail._TAIL_DECK.damp_v)
    same = np.array_equal(np.asarray(base["u"]), np.asarray(hot["u"]),
                          equal_nan=True)
    assert same, (
        "damp_w moved the hydrostatic answer; the spec forces it to 0.0 "
        "on that arm, so it must be inert there")


# =====================================================================
# 5. jit vs eager -- an ASSERTION, and NOT bitwise on kernel output
# =====================================================================

def test_tail_jit_equals_eager_and_does_not_retrace(jctx, jstate, jcsw,
                                                   jdsw):
    """Nothing that flows through a kernel is asserted bitwise across
    the jit boundary: d_sw5's del-n damping and d_sw6's update are
    ``Sum w*v`` contraction sites XLA fuses when jitted and not when
    eager (the campaign's cancellation finding).  A measured bound, and
    a retrace count.
    """
    eager = jtail.dsw_tail_phase_3d(jctx, jstate, jcsw, jdsw, DT, KM)
    # Through the module's OWN factory, which is where the static/dynamic
    # split is declared: ctx and km are baked in, dt stays traced.  A test
    # that re-derives that split with its own jax.jit would be certifying
    # the test's opinion instead of the module's contract -- and the first
    # version of this gate did exactly that, handing jit the ctx OBJECT as
    # a traced argument.
    box, wrapped = counted(jtail.dsw_tail_phase_3d)
    monkey = jtail.dsw_tail_phase_3d
    try:
        jtail.dsw_tail_phase_3d = wrapped
        fast = jtail.make_dsw_tail_phase_3d_jit(jctx, KM)
        got = fast(jstate, jcsw, jdsw, DT)
        got2 = fast(jstate, jcsw, jdsw, 0.5 * DT)
    finally:
        jtail.dsw_tail_phase_3d = monkey
    assert box["n"] == 1, (
        f"{box['n']} traces: dt must be DYNAMIC (convention C3), or a "
        f"new time step recompiles the whole phase")
    assert got2 is not None
    for nm in ("u", "v"):
        # MEASURED (job 9425294 sweep): worst u 4.021e-16 (FMA); bound = measured x 10 =
        # 4.1e-15.
        cmp_fields(got[nm], eager[nm], f"jit-vs-eager {nm}", 4.1e-15)


# =====================================================================
# 6. gradients -- the tolerance-free adjoint identity is PRIMARY
# =====================================================================

def test_tail_adjoint_identity_wind_group(jctx, jstate, jcsw, jdsw):
    """``<J v, w> == <v, J^T w>`` on the D winds, over the COMPUTE
    square only -- the halo carries fills and NaN tripwires and an
    objective that touches them is not differentiable."""
    def f(u, v):
        st = dict(jstate)
        st["u"], st["v"] = u, v
        out = jtail.dsw_tail_phase_3d(jctx, st, jcsw, jdsw, DT, KM)
        return (jnp.sum(out["u"][:, _CS, _CS, :] ** 2)
                + jnp.sum(out["v"][:, _CS, _CS, :] ** 2))

    # MEASURED (job 9425294 sweep): adjoint residual 1.542e-15; bound = measured x 10 =
    # 1.6e-14.
    check_adjoint("tail u/v", f, (jstate["u"], jstate["v"]), 1.6e-14)
    # NO FINITE DIFFERENCE ON THE PRODUCTION DECK, and that is measured
    # rather than conceded.  `check_grads` gave 6.6 % on this group (job
    # 9417462); the eps-SCALING ladder then gave gaps of 1.271e+12,
    # 3.484e+12, 9.969e+12 at eps = 1.6e-3, 8e-4, 4e-4 -- ratios 0.365
    # and 0.350, i.e. the gap GROWS as the step shrinks, and it sits
    # EIGHT DECADES above the 1.065e+04 roundoff floor (job 9417465).
    # That is the signature of a DISCONTINUITY inside every bracket, not
    # of roundoff and not of truncation: d_sw5's del-n damping and
    # d_sw6's limiter switch somewhere in each interval, so the
    # difference quotient carries a jump at every step size and no FD
    # ladder can certify this group.  The independent gradient evidence
    # for those kernels lives at the KERNEL level, in
    # test_fv3_duo_sw_core.py, where each is differentiated on its own
    # fixture; what this phase adds is the CADENCE, and a composition
    # error in the cadence is exactly what the adjoint identity does
    # catch.  The FD ladder runs on the LINEAR-ARM deck instead, below.


def test_tail_fd_ladder_on_the_linear_ppm_arm(jctx, jstate, jcsw, jdsw):
    """The INDEPENDENT gradient check, on a deck where a finite
    difference is meaningful.

    ``hord = 2`` is the oracle's PERFECTLY LINEAR PPM arm (tp_core.F90
    ``mord == 2``): no ``smt5``/``smt6`` selector, no ``copysign`` /
    ``min`` / ``max`` limiter, no ``pert_ppm``.  With the del-n damping
    knobs at zero as well, the remaining map is smooth along the wind
    direction, so the eps^2 ladder measures what it claims -- and a
    wrong reverse mode would leave the gap FLAT in eps, which no
    smoothness can imitate.  Same technique as
    ``test_fv3_nh_core``'s update_dz_d gate, and the same reason.

    This is the answer to "the adjoint identity cannot see a wrong
    Jacobian": it does not have to, alone.
    """
    # `nord` stays 1: d_sw5_duo REFUSES nord=0 ("oracle lane only,
    # nord in {1, 2}"), measured job 9417469.  What removes the del-n
    # branch is the COEFFICIENT, not the order -- fv_tp_2d gates the
    # block on `damp_c > 1e-4`, so zero damping skips it entirely while
    # the call stays on the certified lane.
    smooth = SWConfig(hord_tr=2, hord_vt=2, hord_tm=2, hord_dp=2,
                      hord_mt=2, nord_v=1, damp_v=0.0, dddmp=0.0,
                      d2_bg=0.0, d4_bg=0.0, nord=1)

    def f(u, v):
        st = dict(jstate)
        st["u"], st["v"] = u, v
        out = jtail.dsw_tail_phase_3d(jctx, st, jcsw, jdsw, DT, KM,
                                      cfg=smooth)
        return (jnp.sum(out["u"][:, _CS, _CS, :] ** 2)
                + jnp.sum(out["v"][:, _CS, _CS, :] ** 2))

    # ⛔ AND THE LADDER IS THE WRONG HALF OF THE INSTRUMENT HERE, which
    # the precondition caught rather than papering over (job 9417478):
    # gaps 2.172e+04, 5.741e+02, 1.429e+04 at eps = 1.6e-3, 8e-4, 4e-4
    # against a roundoff floor of 1.068e+04 -- erratic, and all AT the
    # floor.  On the linear arm the truncation term is below what fp64
    # can resolve at an output scale of ~1e14, so there is no eps^2
    # signal to scale and the ratios are noise.
    #
    # What remains assertable, and still compares the derivative against
    # the FUNCTION rather than against another transformation of the
    # same program, is that the gap sits AT that floor.  A wrong
    # Jacobian lifts it by the size of the error, which at these scales
    # is orders of magnitude.
    #
    # MEASURED gap/floor = 2.03 at eps = 1.6e-3; bound = 5x the floor.
    assert_fd_gap_at_roundoff_floor("tail u/v (linear arm)", f,
                                    (jstate["u"], jstate["v"]),
                                    margin=5.0, eps=1.6e-3)


def test_pressure_adjoint_identity_delp_group(jctx, jstate, jcsw, jdsw):
    def f(delp):
        d = dict(jdsw)
        d["delp"] = delp
        tail = jtail.dsw_tail_phase_3d(jctx, jstate, jcsw, d, DT, KM)
        press = jtail.dgrid_pressure_phase_3d(
            jctx, d, tail, KM, dt=DT, ptop=PTOP, akap=AKAP,
            cp_air=CP_AIR)
        return jnp.sum(press["gz"][:, _CS, _CS, :] ** 2)

    # MEASURED (job 9425294 sweep): adjoint residual 1.446e-15; bound = measured x 10 =
    # 1.5e-14.
    check_adjoint("press delp", f, (jdsw["delp"],), 1.5e-14)
    # The same independent instrument, and the same measured verdict as
    # the wind group: the ladder's precondition REFUSED this one too
    # (job 9417483) -- gaps 1.025e+04, 2.804e+04, 3.898e+04 at
    # eps = 1.6e-3, 8e-4, 4e-4 against a floor of 2.904e+03, i.e. the
    # gap GROWS as the step shrinks and tracks the floor's own 1/eps
    # within a factor of ~3.5 at every step.  `delp` does reach the
    # winds nonlinearly (geopk's vertical recurrence, one_grad_p's
    # 1/(wk+wk)), but that curvature is below what fp64 resolves here,
    # so there is no eps^2 signal to scale.
    #
    # MEASURED gap/floor = 3.53 at eps = 1.6e-3; bound = 8x the floor.
    assert_fd_gap_at_roundoff_floor("press delp", f, (jdsw["delp"],),
                                    margin=8.0, eps=1.6e-3)


# =====================================================================
# 7. THE NON-HYDROSTATIC D-GRID TAIL
#
# ⛔ codex BLOCKER (job 9417466): this file had NO test of
# `nh_exchanged_area6` or `dgrid_nh_pressure_phase_3d`.  The NH fixture
# was built and never consumed, so the entire chain -- update_dz_d,
# riem_solver3, the conditional halo order, both interface exchanges,
# the two-cell gz box, the pkc generations and every carry return --
# could have been deleted or reordered without failing anything, while
# the module docstring claimed both pressure chains were certified.
# =====================================================================

_NH_CARRY_KEYS = ("zh", "gz", "zs", "pk3", "pe", "pk", "peln", "ws")

# dp_ref: the reference thickness profile update_dz_d takes.  A LINEAR
# profile, so a level mix-up in the km+1 slot shows up as a wrong value
# rather than as the same number twice.
_DP0 = np.linspace(1.0e4, 1.2e4, KM)


def _nh_stack_carry(carry_np):
    """The spec's per-face lists -> the JAX lane's face-stacked dict."""
    ren = {"zh": "zh6", "gz": "gz6", "zs": "zs6", "pk3": "pk3_6",
           "pe": "pe6", "pk": "pk6", "peln": "peln6", "ws": "ws6"}
    return {k: jnp.asarray(np.stack([np.asarray(x, dtype=np.float64)
                                     for x in carry_np[ren[k]]]))
            for k in _NH_CARRY_KEYS}


@pytest.fixture(scope="module")
def nh_bundle(ctx, state_np_nh):
    """Everything the NH tail consumes, built by the NumPy lane.

    Both lanes are then fed THIS, so the gate is hop-B unit parity on
    the NH phase and nothing upstream can compensate.
    """
    hs6 = [np.zeros((MA, MA), dtype=np.float64) for _ in range(6)]
    carry = npacoustic.build_nh_carry(ctx, KM, hs6)
    # first-substep gz seed, the cadence the acoustic driver owns:
    # gz[..., km] = zs padded, then gz(k) = gz(k+1) - delz on the
    # compute window (dyn_core.F90:384-416).
    b = ctx["bd"]
    i0, j0 = b.is_ - b.isd, b.js - b.jsd
    for t in range(6):
        gz = carry["gz6"][t]
        gz[:, :, KM] = carry["zs6"][t]
        for k in range(KM - 1, -1, -1):
            gz[i0:i0 + N, j0:j0 + N, k] = (
                gz[i0:i0 + N, j0:j0 + N, k + 1]
                - state_np_nh[t]["delz"][:, :, k])
        carry["zh6"][t][:] = carry["gz6"][t]

    csw = npcg.csw_phase_3d(ctx, deepcopy_faces(state_np_nh), dt2=DT2,
                            km=KM, nord=2, duogrid=True,
                            hydrostatic=False)
    csw_press = npcg.cgrid_nh_pressure_phase_3d(
        ctx, deepcopy_faces(csw), carry["gz6"], carry["ws3_6"], KM,
        dt2=DT2, ptop=PTOP, akap=AKAP, cp_air=CP_AIR, p_fac=P_FAC,
        a_imp=A_IMP, dp0=_DP0, hs6=carry["hs6"], zs6=carry["zs6"])
    dsw = npdsw.dsw_transport_phase_3d(
        ctx, deepcopy_faces(state_np_nh), deepcopy_faces(csw),
        dt=DT, km=KM, hydrostatic=False)
    tail = nptail.dsw_tail_phase_3d(
        ctx, deepcopy_faces(state_np_nh), deepcopy_faces(csw),
        deepcopy_faces(dsw), dt=DT, km=KM, hydrostatic=False)
    return {"carry": carry, "csw": csw, "csw_press": csw_press,
            "dsw": dsw, "tail": tail}


def _run_nh(ctx_, bundle, lane, **kw):
    """One NH tail call on either lane, from COPIES of one fixture."""
    delz_np = [np.array(f["delz"], copy=True) for f in bundle["state"]]
    if lane == "numpy":
        # The spec MUTATES the carry it is handed, so the caller needs
        # the object back: `press` off the remap step is structurally
        # zero on this fixture (flat hs6 => zh(km) == zs => ws == 0),
        # and the members that actually move are the CARRY's.
        carry = {k: [np.array(x, copy=True) for x in v]
                 if isinstance(v, list) else v
                 for k, v in bundle["carry"].items()}
        out = nptail.dgrid_nh_pressure_phase_3d(
            ctx_, deepcopy_faces(bundle["csw_press"]),
            deepcopy_faces(bundle["dsw"]), deepcopy_faces(bundle["tail"]),
            carry, KM, dt=DT, ptop=PTOP, akap=AKAP, cp_air=CP_AIR,
            p_fac=P_FAC, a_imp=A_IMP, dp0=_DP0, delz6=delz_np, **kw)
        return out, carry
    return jtail.dgrid_nh_pressure_phase_3d(

        # `_stack_dsw_np`, not `stack_np`: the transport output carries a
        # "levels" list of per-level dicts, which has no array to stack
        # (job 9417488 hit exactly that here after the hydrostatic gates
        # had already been fixed for it).
        # riem_solver3 consumes the POST-d_sw2 pt/delp (the face-level
        # entries), not d_sw1's per-level stage outputs of the same name.
        ctx_, stack_np(bundle["csw_press"]),
        _stack_dsw_np(bundle["dsw"], face_level_wins=_TAIL_FACE_LEVEL),
        stack_np(bundle["tail"]), _nh_stack_carry(bundle["carry"]), KM,
        dt=DT, ptop=PTOP, akap=AKAP, cp_air=CP_AIR, p_fac=P_FAC,
        a_imp=A_IMP, dp0=jnp.asarray(_DP0),
        delz=jnp.asarray(np.stack(delz_np)), **kw)


def test_nh_exchanged_area_is_sentinel_free_and_matches_the_spec(
        ctx, jctx):
    """``nh_exchanged_area6``: the single-tile gridstruct leaves
    BIG_NUMBER in the corner-diagonal halos of ``area``, and
    ``update_dz_d``'s inner ``fv_tp_2d`` reads them."""
    ref = nptail.nh_exchanged_area6(dict(ctx))
    got = jtail.nh_exchanged_area6(jctx)
    want = np.stack([np.asarray(a, dtype=np.float64) for a in ref])
    a = np.asarray(got)
    assert a.shape == want.shape, (a.shape, want.shape)
    # The point of the routine: no sentinel survives.
    assert not np.isclose(np.abs(a), 1.0e8, rtol=1e-12).any(), (
        "a BIG_NUMBER sentinel survived the exchange")
    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise); bound =
    # 1e-15 eps guard (measured exactly 0.0).
    cmp_fields(a, want, "nh area", 1e-15)


@pytest.mark.parametrize("remap_step,use_logp,square_domain", [
    (False, False, True),      # the shipped duo cadence
    (True, False, True),       # the remap substep: pe_halo fires
    (False, True, True),       # use_logp: pln_halo INSTEAD of pk3_halo
    (False, False, False),     # no second pkc exchange
])
def test_nh_tail_parity(ctx, jctx, state_np_nh, nh_bundle,
                        remap_step, use_logp, square_domain):
    """Hop-B parity on every NH branch the spec can take.

    Parameterised over the three switches the chain's ORDER depends on,
    because each selects a different call: ``remap_step`` adds
    ``pe_halo``, ``use_logp`` swaps ``pk3_halo`` for ``pln_halo`` (an
    unconditional ``pk3_halo`` would overwrite log(p) halos with
    p**akap), and ``square_domain`` adds the second ``pkc`` exchange.

    WHAT IS COMPARED, and why it is not the obvious thing.  The
    ``press`` bundle (pe, pk, peln, ws) is written by the REMAP-STEP
    branch, and on this fixture ws is structurally zero as well -- flat
    ``hs6`` makes ``zh(km) == zs``, so ``ws = (zs - zh)/dt`` is exactly
    0.  Off the remap step every member of it is therefore zero in both
    lanes, and ``assert_real`` refused each in turn (jobs 9417616,
    9417626, 9417636, 9417640) rather than let a zero-vs-zero pass
    stand in for a parity claim.  What DOES move on every step is the
    carry -- zh from update_dz_d, gz from the zh*grav box, pk3 from the
    halo -- so that is the comparison off the remap step, and press
    joins it on.
    """
    bundle = dict(nh_bundle, state=state_np_nh)
    kw = dict(remap_step=remap_step, use_logp=use_logp,
              square_domain=square_domain)
    ref, ref_carry = _run_nh(dict(ctx), bundle, "numpy", **kw)
    got = _run_nh(jctx, bundle, "jax", **kw)

    ren = {"zh": "zh6", "gz": "gz6", "pk3": "pk3_6"}
    for nm, npnm in ren.items():
        want = np.stack([np.asarray(x) for x in ref_carry[npnm]])
        assert_real(want, f"numpy nh carry {nm}")
        # MEASURED by this gate's own machinery post edge_profile
        # unroll (job 9424782): 4.543e-14 per-element relative,
        # max|diff| 2.049e-08 on a ~1e5-magnitude field, ZERO cells over
        # 1e-13. Down from 3.498e-10 / 1.847e-04 before the fix -- the
        # old residual was the edge_profile scan-FMA defect, CONFIRMED
        # by its removal. Bound = measured x 10. What remains localises
        # wholly inside riem_solver3 (stage-split localiser, job
        # 9424741, ~5.8e-10 absolute per face). MEASURED, scan
        # hypothesis REFUTED (fv3_riem_solver3_scan_localiser.py, jobs
        # 9433880/9435602): running all eight of riem's lax.scan sites
        # as eager python loops over the production bodies left the
        # residual unchanged (zh 5.82e-10 on the worst face), while
        # eager jnp.exp differs from np.exp by ~1 ulp on identical
        # operand values (3.55e-15 on pk; a sim1-style composite
        # exp(gama*log(y)) on SAMPLED operands reproduces the 4.66e-10
        # magnitude). The XLA-CPU-exp-vs-libm-exp gap is the CANDIDATE
        # mechanism CONSISTENT with the residual -- not a proven
        # propagation (codex+GLM 2026-08-19: the primitive check fed
        # both exps the same pre-rounded product on constructed
        # operands; the closing experiment is a libm substitution at
        # the production exp sites, not yet run). What IS established:
        # unrolling cannot remove it, log is bitwise, exp is the one
        # divergent primitive found, and the bound is measured.
        # [class: transcendental-implementation candidate; the old
        #  "accumulating" tag was the retracted scan story]
        cmp_fields(got["nh"][nm], want, f"nh carry {nm}", 4.6e-13)

    if remap_step:
        # `ws` is EXCLUDED, not forgotten: the fixture has flat
        # orography, so zh(km) == zs and the surface vertical velocity
        # is identically 0.0 in both lanes. assert_real refused the
        # comparison, which is the guard working -- an equality over an
        # all-zero field passes at any tolerance and certifies nothing.
        # It gets a CONTRACT assertion instead, the same treatment `w`
        # gets on the hydrostatic arm.
        for nm in ("pe", "pk", "peln"):
            want = np.stack([np.asarray(ref[t][nm]) for t in range(6)])
            assert_real(want, f"numpy nh {nm}")
            cmp_fields(got["press"][nm], want, f"nh {nm}", 1e-12)
        for lane, arr in (("jax", np.asarray(got["press"]["ws"])),
                          ("numpy", np.stack([np.asarray(ref[t]["ws"])
                                              for t in range(6)]))):
            assert not arr.any(), (
                f"{lane}: ws is non-zero on a FLAT-orography fixture, "
                f"where zh(km) == zs forces it to 0")
    else:
        for nm in ("pe", "pk", "peln"):
            for lane, arr in (("jax", np.asarray(got["press"][nm])),
                              ("numpy", np.stack([np.asarray(ref[t][nm])
                                                  for t in range(6)]))):
                assert not arr.any(), (
                    f"{lane}: {nm} is non-zero off the remap step, "
                    f"where nothing writes it")


def test_nh_tail_returns_every_array_the_spec_mutates(ctx, jctx,
                                                      state_np_nh,
                                                      nh_bundle):
    """Convention C4, as a MANIFEST rather than a claim (codex MAJOR).

    The spec mutates the eight carry members, ``delz6``, the
    ``csw_press`` ``pkc`` storage and the D winds. Every one must come
    back, and each must have MOVED against the input it was built from
    -- a returned key that is bitwise the input is a stage that did not
    run.
    """
    bundle = dict(nh_bundle, state=state_np_nh)
    got = _run_nh(jctx, bundle, "jax")
    _require_keys(got["nh"], {k: 1 for k in _NH_CARRY_KEYS},
                  _NH_CARRY_KEYS, "NH carry")
    for nm in ("delz", "pkc", "u", "v", "w"):
        assert nm in got, f"the NH tail does not return {nm!r}"

    before = _nh_stack_carry(nh_bundle["carry"])
    moved = [k for k in ("zh", "gz", "pk3", "pe", "pk", "peln", "ws")
             if not np.array_equal(np.asarray(before[k]),
                                   np.asarray(got["nh"][k]),
                                   equal_nan=True)]
    assert set(moved) >= {"zh", "gz", "pk3"}, (
        f"only {moved} moved: update_dz_d writes zh, the gz = zh*grav "
        f"box writes gz, and pk3_halo writes pk3, so a carry member "
        f"that is bitwise its input means its stage did not run")
    # zs is the surface and is NOT written by this phase -- asserted so
    # a future edit that starts writing it is noticed.
    assert np.array_equal(np.asarray(before["zs"]),
                          np.asarray(got["nh"]["zs"]), equal_nan=True), (
        "zs moved; it is the surface elevation and nothing in this "
        "chain writes it")


def test_nh_tail_use_logp_selects_pln_halo_not_pk3_halo(ctx, jctx,
                                                        state_np_nh,
                                                        nh_bundle):
    """The exclusive branch, driven rather than read.

    ``dyn_core.F90:1444-1448``: ``pln_halo`` under ``use_logp``,
    ``pk3_halo`` otherwise. An unconditional ``pk3_halo`` would
    overwrite log(p) halos with p**akap -- finite, plausible, wrong. So
    the two settings must give DIFFERENT pk3, and each must match its
    own NumPy reference (the parity gate above does the matching).
    """
    bundle = dict(nh_bundle, state=state_np_nh)
    a = _run_nh(jctx, bundle, "jax", use_logp=False)
    b = _run_nh(jctx, bundle, "jax", use_logp=True)
    pa = assert_real(a["nh"]["pk3"], "pk3 (pk3_halo)")
    pb = assert_real(b["nh"]["pk3"], "pk3 (pln_halo)")
    assert not np.array_equal(pa, pb, equal_nan=True), (
        "use_logp did not change pk3, so the pln_halo/pk3_halo branch "
        "is not selected by it")


def test_nh_update_dz_d_twins_agree_on_this_fixture(ctx, jctx, nh_bundle,
                                                    state_np_nh):
    """`zh` is written by ONE kernel, so compare that kernel directly.

    The NH parity gate fails on `zh` at 2.465e-02 and nothing else in
    the phase writes it (``gz`` is ``zh*grav`` afterwards).  Its twins
    are separately gated in ``test_fv3_nh_core``, but on THAT file's
    fixture -- and a kernel can agree on one fixture and not another,
    which is the whole reason the phase-level gates exist.

    So: both ``update_dz_d`` twins, the same operands taken from this
    bundle, per face.  Agreement here moves the search into the phase
    (the exchanges after the kernel); disagreement moves it into the
    kernel and off this module entirely.
    """
    from legoesm.core.fv3_native_nh_core import (
        update_dz_d as np_udzd,
    )
    from legoesm.core.fv3_nh_core import update_dz_d as j_udzd

    carry = nh_bundle["carry"]
    dsw = nh_bundle["dsw"]
    bd = ctx["bd"]
    # The JAX twin takes a STATIC BOUNDS TUPLE, not the NumPy `bd`
    # object -- (is_, ie, js, je, ng), so a fresh-but-equal object
    # cannot retrace it. Passing `bd` gave "too many values to unpack
    # (expected 5)".
    bounds = (bd.is_, bd.ie, bd.js, bd.je, NG)
    npx = N + 1
    area6 = nptail.nh_exchanged_area6(dict(ctx))
    rarea6 = [1.0 / np.asarray(a) for a in area6]
    ndif = np.full(KM + 1, 2.0)
    damp = np.full(KM + 1, 0.12)
    rdt = 1.0 / DT

    for t in range(3):          # three faces is enough to localise
        crx = np.stack([dsw[t]["levels"][k]["crx_adv"] for k in range(KM)],
                       axis=2)
        cry = np.stack([dsw[t]["levels"][k]["cry_adv"] for k in range(KM)],
                       axis=2)
        xfx = np.stack([dsw[t]["levels"][k]["xfx_adv"] for k in range(KM)],
                       axis=2)
        yfx = np.stack([dsw[t]["levels"][k]["yfx_adv"] for k in range(KM)],
                       axis=2)
        gs = ctx["gs6"][t]
        gs_nh = dict(gs)
        gs_nh["area"] = area6[t]
        gs_nh["rarea"] = rarea6[t]

        zh_np = np.array(carry["zh6"][t], copy=True)
        ws_np = np.array(carry["ws6"][t], copy=True)
        np_udzd(ndif, damp, 6, bd, KM, npx, npx, area6[t], rarea6[t],
                _DP0, carry["zs6"][t], zh_np, np.array(crx, copy=True),
                np.array(cry, copy=True), np.array(xfx, copy=True),
                np.array(yfx, copy=True), ws_np, rdt, gs_nh, lim_fac=1.0)

        zh_j, _ws_j = j_udzd(
            (2.0,) * (KM + 1), (0.12,) * (KM + 1), 6, bounds, KM,
            npx, npx,
            jnp.asarray(area6[t]), jnp.asarray(rarea6[t]),
            jnp.asarray(_DP0), jnp.asarray(carry["zs6"][t]),
            jnp.asarray(carry["zh6"][t]), jnp.asarray(crx),
            jnp.asarray(cry), jnp.asarray(xfx), jnp.asarray(yfx),
            jnp.asarray(carry["ws6"][t]), rdt,
            jnp.asarray(gs["dxa"]), jnp.asarray(gs["dya"]),
            jnp.asarray(gs["del6_u"]), jnp.asarray(gs["del6_v"]),
            lim_fac=1.0,
            # The flags the NumPy kernel reads out of `gridstruct` and
            # hands to fv_tp_2d / del6_vt_flux.  Omitting them is what
            # this probe measured at 1.155e-04.
            bounded_domain=jctx.flags6[t].bounded_domain,
            grid_type=jctx.flags6[t].grid_type,
            sw_corner=jctx.flags6[t].sw_corner,
            se_corner=jctx.flags6[t].se_corner,
            nw_corner=jctx.flags6[t].nw_corner,
            ne_corner=jctx.flags6[t].ne_corner)

        assert_real(zh_np, f"numpy update_dz_d zh face {t + 1}")
        # RESOLVED (jobs 9421844 -> 9424569): the residual was
        # edge_profile's Thomas recurrences running inside lax.scan,
        # whose body is XLA-COMPILED even on eager calls -- the compiled
        # body's contracted multiply-adds perturbed the *_adv outputs at
        # ~1e-14 relative, and on a level where zh is horizontally
        # CONSTANT those ulps flipped two upwind selector bits and broke
        # exact free-stream preservation (2 cells, 2.9e-6, NumPy exact
        # to 17 digits). My first PLAUSIBLE mechanism -- divergence
        # association -- was REFUTED by the stage-split probe: fv_tp_2d,
        # xppm/yppm and del6_vt_flux were all bitwise identical; ONLY
        # the scan differed. The recurrences are now unrolled in python
        # over the static km (see edge_profile's docstring), and the
        # stage-split probe measured the full twins BITWISE identical on
        # this bundle afterwards.
        #
        # So the comparison is EXACT EQUALITY, not a tolerance: the
        # eager path is op-by-op primitive-for-primitive with the spec,
        # and any reappearing difference is a regression of exactly the
        # class this hunt just closed -- a tolerance would hide it for
        # months again. (Under jit only the documented ~1e-14 parity
        # holds; this gate runs eager.)
        zh_ja = np.asarray(zh_j)
        assert np.array_equal(zh_ja, zh_np, equal_nan=True), (
            f"update_dz_d zh face {t + 1}: twins no longer bitwise -- "
            f"max|d| {np.abs(np.where(np.isfinite(zh_ja) & np.isfinite(zh_np), zh_ja - zh_np, 0.0)).max():.3e}, "
            f"{int((zh_ja != zh_np).sum() - (~(np.isfinite(zh_ja)) & ~(np.isfinite(zh_np))).sum())} cells. "
            f"The eager edge_profile/update_dz_d path regressed from "
            f"op-by-op primitives (see the RESOLVED note above).")
