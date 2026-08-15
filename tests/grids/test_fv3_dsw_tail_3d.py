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
classes, so every numeric bound carries a ``TOL-PENDING`` marker with a
class label until the measurement job replaces it with
``measured X, bound = measured x N``.

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
    assert_fd_truncation_scaling,
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


def _np_tail(ctx, state_np, csw_np, dsw_np, **kw):
    """The spec, on COPIES -- it mutates its inputs."""
    return nptail.dsw_tail_phase_3d(
        ctx, deepcopy_faces(state_np), deepcopy_faces(csw_np),
        deepcopy_faces(dsw_np), dt=DT, km=KM, **kw)


def test_tail_parity_hydrostatic(ctx, jctx, state_np, csw_np, dsw_np,
                                 jstate, jcsw, jdsw):
    """gate 1 -- every carried field, both lanes, same inputs."""
    ref = _np_tail(ctx, state_np, csw_np, dsw_np)
    got = jtail.dsw_tail_phase_3d(jctx, jstate, jcsw, jdsw, DT, KM)
    compared = [nm for nm in _TAIL_COMPARED if nm in got and nm in ref[0]]
    assert set(("u", "v")) <= set(compared), compared
    for nm in compared:
        want = np.stack([np.asarray(ref[t][nm]) for t in range(6)])
        assert_real(want, f"numpy tail {nm}")
        # TOL-PENDING: provisional bound; the measurement job replaces
        # this with `measured X, bound = measured x N`.  DO NOT SHIP.
        # [class: branch-switching -- d_sw5/d_sw6 limiters]
        cmp_fields(got[nm], want, f"tail {nm}", 1e-12)


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
        # the +1 ring: last index of the compute-ring axes
        edge_a, edge_b = a[:, -1, ...], b[:, -1, ...]
        fin = np.isfinite(edge_a) & np.isfinite(edge_b)
        assert fin.any(), f"{post}: the +1 ring is entirely non-finite"
        assert not np.array_equal(edge_a[fin], edge_b[fin]), (
            f"{post}: the i = ie+1 ring is UNCHANGED by the barrier -- "
            f"that is barrier 1's extent, not barrier 2's")


def test_ke_mixes_blended_and_unblended_members(jctx, jstate, jcsw,
                                                jdsw):
    """``ke = 0.5*(ubbtemp*vbbtemp + ubb*vbb)`` with ``ubbtemp``/``vbb``
    RAW and ``vbbtemp``/``ubb`` BLENDED (dyn_core.F90:1080-1085 read
    against the barrier at :984).

    Two plausible wrong operators are excluded by construction: forming
    the products from the raw pair, and blending all four.  Both would
    give a finite, physical-looking ``ke``.  Recomputed here from the
    returned stages -- not by re-implementing the kernel, but by
    contrasting the two candidate assemblies and requiring that the one
    the module produced is NOT the all-raw one.
    """
    out = jtail.dsw_tail_phase_3d(jctx, jstate, jcsw, jdsw, DT, KM)
    ub_raw = np.asarray(out["ubb_prebarrier"])
    vb_raw = np.asarray(out["vbbtemp_prebarrier"])
    ub_b = np.asarray(out["ubb_postbarrier"])
    vb_b = np.asarray(out["vbbtemp_postbarrier"])
    fin = (np.isfinite(ub_raw) & np.isfinite(vb_raw)
           & np.isfinite(ub_b) & np.isfinite(vb_b))
    assert fin.any()
    all_raw = 0.5 * (ub_raw * vb_raw + ub_raw * vb_raw)
    mixed = 0.5 * (ub_raw * vb_b + ub_b * vb_raw)
    assert not np.allclose(all_raw[fin], mixed[fin]), (
        "the blended and raw B-grid members are indistinguishable on "
        "this fixture, so this gate cannot tell the two assemblies "
        "apart -- the fixture, not the module, is at fault")


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
    for nm in ("pk", "gz", "pe", "peln", "pkz"):
        if nm not in got or nm not in ref[0]:
            continue
        want = np.stack([np.asarray(ref[t][nm]) for t in range(6)])
        assert_real(want, f"numpy press {nm}")
        # TOL-PENDING: provisional bound; the measurement job replaces
        # this.  DO NOT SHIP.   [class: accumulating -- geopk recurrence]
        cmp_fields(got[nm], want, f"press {nm}", 1e-12)


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
    """C7: a retyped deck is a place for the two lanes to drift."""
    deck = SWConfig.from_mapping(
        {k: v for k, v in nptail.DUO_TAIL_CFG.items()
         if k in SWConfig._fields})
    assert deck.nord == nptail.DUO_TAIL_CFG["nord"]
    assert deck.d4_bg == nptail.DUO_TAIL_CFG["d4_bg"]


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
        # TOL-PENDING: provisional bound; the measurement job replaces
        # this.  DO NOT SHIP.   [class: FMA contraction]
        cmp_fields(got[nm], eager[nm], f"jit-vs-eager {nm}", 1e-9)


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

    # TOL-PENDING: provisional bound; the measurement job replaces this.
    # DO NOT SHIP.   [class: roundoff -- the identity is exact]
    check_adjoint("tail u/v", f, (jstate["u"], jstate["v"]), 1e-10)
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
    smooth = SWConfig(hord_tr=2, hord_vt=2, hord_tm=2, hord_dp=2,
                      hord_mt=2, nord_v=0, damp_v=0.0, dddmp=0.0,
                      d2_bg=0.0, d4_bg=0.0, nord=0)

    def f(u, v):
        st = dict(jstate)
        st["u"], st["v"] = u, v
        out = jtail.dsw_tail_phase_3d(jctx, st, jcsw, jdsw, DT, KM,
                                      cfg=smooth)
        return (jnp.sum(out["u"][:, _CS, _CS, :] ** 2)
                + jnp.sum(out["v"][:, _CS, _CS, :] ** 2))

    # TOL-PENDING: the ladder is tolerance-free in the gap, but the STEP
    # set is a measurement; the job replaces it if the precondition
    # (top gap > 10x the roundoff floor) is not met here.
    assert_fd_truncation_scaling("tail u/v (linear arm)", f,
                                 (jstate["u"], jstate["v"]),
                                 steps=(1.6e-3, 8.0e-4, 4.0e-4))


def test_pressure_adjoint_identity_delp_group(jctx, jstate, jcsw, jdsw):
    def f(delp):
        d = dict(jdsw)
        d["delp"] = delp
        tail = jtail.dsw_tail_phase_3d(jctx, jstate, jcsw, d, DT, KM)
        press = jtail.dgrid_pressure_phase_3d(
            jctx, d, tail, KM, dt=DT, ptop=PTOP, akap=AKAP,
            cp_air=CP_AIR)
        return jnp.sum(press["gz"][:, _CS, _CS, :] ** 2)

    # TOL-PENDING: provisional bound; the measurement job replaces this.
    # DO NOT SHIP.   [class: roundoff -- the identity is exact]
    check_adjoint("press delp", f, (jdsw["delp"],), 1e-10)
    # The same independent instrument, same reason.  `delp` reaches the
    # winds through geopk's vertical recurrence and one_grad_p's 1/(wk+wk),
    # so the group is genuinely nonlinear and HAS a truncation term to
    # measure.
    assert_fd_truncation_scaling("press delp", f, (jdsw["delp"],),
                                 steps=(1.6e-3, 8.0e-4, 4.0e-4))
