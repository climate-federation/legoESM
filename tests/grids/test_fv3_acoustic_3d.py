"""Certification of the JAX 3-D acoustic driver (module 4 of 6).

Authority: ``legoesm.core.fv3_native_acoustic_3d`` is the SPECIFICATION
(hop B of ``docs/atmosphere/fv3_duo_jax_lane_strategy.md``).

This module introduces no numerics at all -- every arithmetic operation
happens inside the three phase modules, each of which has its own gate
file.  What it contributes is WHICH phase runs at which sub-step, and
that is what the gates below interrogate:

* **THE THREE SUB-STEPS ARE THREE DIFFERENT PROGRAMS.**  ``it == 1``
  refreshes the SCALARS as well as the winds and, on the NH arm, seeds
  ``gz`` from ``zs`` and rebuilds it from ``delz``; ``it == n_split``
  takes the ``pk`` snapshot the vertical remap reads and calls
  ``pe_halo``; the middle ones are identical to each other.  A
  ``lax.scan`` body must be ONE program, so the loop is peel / scan /
  peel -- and the gates check the composition at ``n_split`` = 1, 2, 3
  and 4, which is the smallest set covering "no middle", "no scan" and
  "a scan of two".

* **THE gz <-> zh CADENCE** (the module's stated trap): the first
  sub-step exchanges ``gz`` and SAVES ``zh = gz``; every later one
  RESTORES ``gz = zh``.  Backwards, the run stays finite and plausible.

* **THE FUNCTIONAL CARRY.**  The spec mutates ``state``, the NH carry
  and the flux capacitors in place across sub-steps; this lane threads
  them through the scan carry and returns them.  A key that the spec
  writes and this lane drops is a silent divergence, so the returned
  set is asserted against a manifest rather than eyeballed.

TOLERANCE POLICY.  Everything here composes limiter-heavy kernels, so
per strategy section 4 every numeric bound carries a ``TOL-PENDING``
marker with a class label until the measurement job replaces it with
``measured X, bound = measured x N``.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from legoesm.core import fv3_acoustic_3d as jac  # noqa: E402
from legoesm.core import fv3_native_acoustic_3d as npac  # noqa: E402
from legoesm.core.fv3_cgrid_phase_3d import (  # noqa: E402
    state_3d_to_jax,
)
from legoesm.core.fv3_duo_stepper import (  # noqa: E402
    build_jax_duo_stepper_context,
)
from legoesm.core.fv3_native_duo_stepper import (  # noqa: E402
    build_six_face_duo_context,
)
from legoesm.core.fv3_native_state_3d import (  # noqa: E402
    STATE_FIELDS,
    build_state_3d,
)

from tests.grids.fv3_gate_helpers import (  # noqa: E402
    assert_real,
    cmp_fields,
    counted,
    deepcopy_faces,
)

# Same geometry as every other 3-D gate file, so the four are comparable.
N, NG, KM = 12, 3, 3
MA = N + 2 * NG
DT_ATMOS = 60.0
PTOP, AKAP, CP_AIR = 100.0, 2.0 / 7.0, 1004.6


@pytest.fixture(scope="module")
def ctx():
    return build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                      oracle_conventions=True)


@pytest.fixture(scope="module")
def jctx(ctx):
    return build_jax_duo_stepper_context(ctx)


def _seeded(km, seed=5):
    """A physical column, distinct per face AND per level."""
    rng = np.random.default_rng(seed)
    st = build_state_3d(N, NG, km, hydrostatic=True)
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
    return st


@pytest.fixture(scope="module")
def state_np():
    return _seeded(KM)


@pytest.fixture(scope="module")
def jstate(state_np):
    return state_3d_to_jax(state_np)


def _np_loop(ctx, state_np, n_split):
    """The spec's loop, on a COPY -- it mutates its state in place."""
    st = deepcopy_faces(state_np)
    npac.acoustic_loop_3d(ctx, st, DT_ATMOS, KM, n_split=n_split,
                          ptop=PTOP, akap=AKAP, cp_air=CP_AIR)
    return st


@pytest.mark.parametrize("n_split", [1, 2, 3, 4])
def test_loop_parity_across_the_peel_scan_peel_structure(
        ctx, jctx, state_np, jstate, n_split):
    """The smallest set of trip counts that covers the structure.

    ``n_split = 1`` is the peel alone (and IS the remap step);
    ``2`` is two peels and no scan; ``3`` scans one middle sub-step;
    ``4`` scans two.  A composition that works at one of these and not
    the others is exactly what a single-value test would miss.
    """
    ref = _np_loop(ctx, state_np, n_split)
    got = jac.acoustic_loop_3d(jctx, jstate, DT_ATMOS, KM,
                               n_split=n_split, ptop=PTOP, akap=AKAP,
                               cp_air=CP_AIR)
    state = got["state"] if isinstance(got, dict) else got
    # `w` is IDENTICALLY ZERO on the hydrostatic arm -- the anti-vacuity
    # guard in assert_real caught the first version comparing it (job
    # 9417521), which is the guard working: a relative comparison of
    # zero against zero passes at any tolerance and certifies nothing.
    # It gets a CONTRACT assertion instead, below.
    compared = [nm for nm in STATE_FIELDS if nm in state and nm != "w"]
    assert {"delp", "pt", "u", "v"} <= set(compared), compared
    for nm in compared:
        want = np.stack([np.asarray(ref[t][nm]) for t in range(6)])
        assert_real(want, f"numpy loop {nm} (n_split={n_split})")
        # TOL-PENDING: provisional bound; the measurement job replaces
        # this with `measured X, bound = measured x N`.  DO NOT SHIP.
        # [class: branch-switching, composed over n_split sub-steps]
        cmp_fields(state[nm], want, f"loop {nm} n_split={n_split}", 1e-12)
    if "w" in state:
        # The contract, asserted rather than compared: nothing on the
        # hydrostatic arm writes w, in EITHER lane.
        for lane, arr in (("jax", np.asarray(state["w"])),
                          ("numpy", np.stack([np.asarray(ref[t]["w"])
                                              for t in range(6)]))):
            assert not arr.any(), (
                f"{lane}: w is non-zero on the hydrostatic arm, where "
                f"nothing writes it")


def test_loop_does_not_mutate_its_input_state(jctx, jstate):
    """The spec mutates ``state`` in place across sub-steps; this lane
    threads it through the carry and returns it."""
    before = {k: np.array(v) for k, v in jstate.items()}
    jac.acoustic_loop_3d(jctx, jstate, DT_ATMOS, KM, n_split=2,
                         ptop=PTOP, akap=AKAP, cp_air=CP_AIR)
    for k, v in before.items():
        assert np.array_equal(v, np.asarray(jstate[k]), equal_nan=True), (
            f"acoustic_loop_3d mutated its input state[{k!r}]")


def test_a_second_substep_is_not_a_repeat_of_the_first(jctx, jstate):
    """``first_substep`` must CHANGE the program.

    It refreshes the scalar halos as well as the winds, so a lane that
    ignored the flag would give the same answer for both -- and the
    difference would only show up much later, as a stale-halo drift.
    """
    a = jac.acoustic_substep_3d(jctx, jstate, DT_ATMOS, KM,
                                first_substep=True, ptop=PTOP,
                                akap=AKAP, cp_air=CP_AIR)
    b = jac.acoustic_substep_3d(jctx, jstate, DT_ATMOS, KM,
                                first_substep=False, ptop=PTOP,
                                akap=AKAP, cp_air=CP_AIR)
    sa, sb = a["state"], b["state"]
    moved = [nm for nm in STATE_FIELDS
             if nm in sa and not np.array_equal(
                 np.asarray(sa[nm]), np.asarray(sb[nm]), equal_nan=True)]
    assert moved, (
        "first_substep=True and False gave a BITWISE identical state, so "
        "the flag selects nothing -- the entry exchange must refresh the "
        "scalars only on the first sub-step")


def test_n_split_must_be_at_least_one(jctx, jstate):
    with pytest.raises(ValueError, match="n_split"):
        jac.acoustic_loop_3d(jctx, jstate, DT_ATMOS, KM, n_split=0,
                             ptop=PTOP, akap=AKAP, cp_air=CP_AIR)


def test_km_must_be_a_python_int(jctx, jstate):
    with pytest.raises(TypeError, match="km must be a Python int"):
        jac.acoustic_substep_3d(jctx, jstate, DT_ATMOS, float(KM),
                                first_substep=True, ptop=PTOP,
                                akap=AKAP, cp_air=CP_AIR)


def test_nh_arm_refuses_a_missing_carry(jctx, jstate):
    """``hydrostatic=False`` without the NH carry and dp0 must RAISE,
    not run a hydrostatic step under an NH name."""
    with pytest.raises((ValueError, TypeError, KeyError)):
        jac.acoustic_substep_3d(jctx, jstate, DT_ATMOS, KM,
                                first_substep=True, ptop=PTOP,
                                akap=AKAP, cp_air=CP_AIR,
                                hydrostatic=False)


def test_loop_jit_equals_eager_and_dt_stays_dynamic(jctx, jstate):
    """``dt_atmos`` is DYNAMIC (convention C3): a new outer time step
    must not recompile the loop."""
    eager = jac.acoustic_loop_3d(jctx, jstate, DT_ATMOS, KM, n_split=2,
                                 ptop=PTOP, akap=AKAP, cp_air=CP_AIR)
    box, wrapped = counted(jac.acoustic_loop_3d)
    monkey = jac.acoustic_loop_3d
    try:
        jac.acoustic_loop_3d = wrapped
        fast = jac.make_acoustic_loop_3d_jit(
            jctx, KM, n_split=2, ptop=PTOP, akap=AKAP, cp_air=CP_AIR)
        got = fast(jstate, DT_ATMOS)
        got2 = fast(jstate, 0.5 * DT_ATMOS)
    finally:
        jac.acoustic_loop_3d = monkey
    assert box["n"] == 1, (
        f"{box['n']} traces: dt_atmos must be dynamic, or every new "
        f"outer step recompiles the whole acoustic loop")
    assert got2 is not None
    ge = eager["state"] if isinstance(eager, dict) else eager
    gg = got["state"] if isinstance(got, dict) else got
    for nm in ("u", "v"):
        # TOL-PENDING: provisional bound.  DO NOT SHIP.
        # [class: FMA contraction, composed over sub-steps]
        cmp_fields(gg[nm], ge[nm], f"loop jit-vs-eager {nm}", 1e-9)


# =====================================================================
# ONE SUB-STEP, which is the bisect between "the sub-step is wrong" and
# "the loop plumbing is wrong".  Added when the loop parity showed u
# off by 2.106e-02 on 1512 of 6156 cells while delp and pt matched
# (job 9417532) -- a discrepancy that could sit on either side.
# =====================================================================

@pytest.mark.parametrize("first_substep", [True, False])
def test_substep_parity(ctx, jctx, state_np, jstate, first_substep):
    """One ``it`` of the loop, both lanes, from the same state.

    ``remap_step`` is tied to ``first_substep`` here only to keep the
    two cases distinct; the loop gate covers the combinations.
    """
    st = deepcopy_faces(state_np)
    npac.acoustic_substep_3d(ctx, st, DT_ATMOS, KM,
                             first_substep=first_substep, ptop=PTOP,
                             akap=AKAP, cp_air=CP_AIR,
                             remap_step=first_substep)
    got = jac.acoustic_substep_3d(jctx, jstate, DT_ATMOS, KM,
                                  first_substep=first_substep,
                                  ptop=PTOP, akap=AKAP, cp_air=CP_AIR,
                                  remap_step=first_substep)
    state = got["state"]
    for nm in ("delp", "pt", "u", "v"):
        want = np.stack([np.asarray(st[t][nm]) for t in range(6)])
        assert_real(want, f"numpy substep {nm}")
        # TOL-PENDING: provisional bound; the measurement job replaces
        # this.  DO NOT SHIP.   [class: branch-switching, one sub-step]
        cmp_fields(state[nm], want, f"substep {nm} first={first_substep}",
                   1e-12)


def test_entry_exchange_parity(ctx, jctx, state_np, jstate):
    """The entry halo exchange alone, both lanes.

    The next bisect after the sub-step: ``u`` is off by 2.106e-02 inside
    ONE sub-step (job 9417545) while ``delp`` and ``pt`` match, and the
    entry exchange is the first thing that touches ``u``.  If this
    passes, the wind difference is made downstream; if it fails, it is
    made here, and the halo is where a 2 % wind error with a panel-edge
    footprint would come from.
    """
    st = deepcopy_faces(state_np)
    npac.exchange_state_halos_3d(ctx, st, KM, scalars=True, winds=True)
    got = jac.exchange_state_halos_3d(jctx, jstate, KM, scalars=True,
                                      winds=True)
    for nm in ("delp", "pt", "u", "v"):
        want = np.stack([np.asarray(st[t][nm]) for t in range(6)])
        assert_real(want, f"numpy entry-exchange {nm}")
        # TOL-PENDING: provisional bound.  DO NOT SHIP.
        # [class: exchange -- weighted stencil sums]
        cmp_fields(got[nm], want, f"entry exchange {nm}", 1e-12)


def test_substep_stage_bisect(ctx, jctx, state_np, jstate):
    """Which STAGE of the sub-step first disagrees.

    The sub-step gate says the wind error is made inside one sub-step
    and the entry-exchange gate says it is not made there, so the
    remaining candidates are the four phases the sub-step composes.
    Each is separately parity-gated on its OWN fixture, which means the
    defect is in what this module hands one of them -- and that is a
    statement about a specific stage boundary, so the bisect walks them
    in order and reports the FIRST one that moves.

    Ordered exactly as the sub-step runs them; the first failure names
    the stage and everything after it is consequence, so this test
    deliberately does not continue past it.
    """
    import legoesm.core.fv3_native_cgrid_phase_3d as npcg
    import legoesm.core.fv3_native_dsw_phase_3d as npdsw
    import legoesm.core.fv3_native_dsw_tail_3d as nptail
    from legoesm.core.fv3_cgrid_phase_3d import (
        cgrid_pressure_phase_3d,
        csw_phase_3d,
    )
    from legoesm.core.fv3_dsw_phase_3d import dsw_transport_phase_3d
    from legoesm.core.fv3_dsw_tail_3d import dsw_tail_phase_3d

    dt, dt2 = DT_ATMOS, 0.5 * DT_ATMOS

    # both lanes from the SAME exchanged state
    st = deepcopy_faces(state_np)
    npac.exchange_state_halos_3d(ctx, st, KM, scalars=True, winds=True)
    jst = jac.exchange_state_halos_3d(jctx, jstate, KM, scalars=True,
                                      winds=True)

    n_csw = npcg.csw_phase_3d(ctx, deepcopy_faces(st), dt2=dt2, km=KM,
                              nord=2, duogrid=True)
    j_csw = csw_phase_3d(jctx, jst, dt2, KM, nord=2)
    for nm in ("uc", "vc", "divg_d"):
        want = np.stack([np.asarray(n_csw[t][nm]) for t in range(6)])
        # TOL-PENDING: provisional bound.  DO NOT SHIP.  [class: stage]
        cmp_fields(j_csw[nm], want, f"stage c_sw {nm}", 1e-12)

    # The NumPy pressure phase's RETURN is not used: what this stage
    # contributes to the winds is its in-place mutation of n_csw's
    # uc/vc, which is the reference compared below.
    npcg.cgrid_pressure_phase_3d(ctx, n_csw, KM, dt2=dt2, ptop=PTOP,
                                 akap=AKAP, cp_air=CP_AIR)
    j_press = cgrid_pressure_phase_3d(jctx, j_csw, KM, dt2=dt2,
                                      ptop=PTOP, akap=AKAP,
                                      cp_air=CP_AIR)
    for nm in ("uc", "vc"):
        want = np.stack([np.asarray(n_csw[t][nm]) for t in range(6)])
        # p_grad_c mutates uc/vc in the NumPy lane, so the reference is
        # the MUTATED n_csw, not the returned bundle.
        cmp_fields(j_press[nm], want, f"stage p_grad_c {nm}", 1e-12)

    j_csw = {**j_csw, "uc": j_press["uc"], "vc": j_press["vc"]}
    n_dsw = npdsw.dsw_transport_phase_3d(ctx, deepcopy_faces(st),
                                         deepcopy_faces(n_csw), dt=dt,
                                         km=KM)
    j_dsw = dsw_transport_phase_3d(jctx, jst, j_csw, dt, KM)
    # `delp`/`pt` are the d_sw2 UPDATE and sit at the top level of the
    # spec's per-face dict; `ut`/`vt` live inside its "levels" list (the
    # per-level d_sw1 outputs), so they are stacked from there rather
    # than looked up at the top, which is what raised KeyError on the
    # first run of this bisect.
    for nm in ("delp", "pt"):
        want = np.stack([np.asarray(n_dsw[t][nm]) for t in range(6)])
        cmp_fields(j_dsw[nm], want, f"stage d_sw1/2 {nm}", 1e-12)
    for nm in ("ut", "vt"):
        want = np.stack([
            np.stack([np.asarray(lvl[nm]) for lvl in n_dsw[t]["levels"]],
                     axis=2) for t in range(6)])
        cmp_fields(j_dsw[nm], want, f"stage d_sw1 {nm}", 1e-12)

    n_tail = nptail.dsw_tail_phase_3d(ctx, deepcopy_faces(st),
                                      deepcopy_faces(n_csw),
                                      deepcopy_faces(n_dsw), dt=dt,
                                      km=KM)
    j_tail = dsw_tail_phase_3d(jctx, jst, j_csw, j_dsw, dt, KM)
    for nm in ("u", "v"):
        want = np.stack([np.asarray(n_tail[t][nm]) for t in range(6)])
        cmp_fields(j_tail[nm], want, f"stage d_sw3-6 {nm}", 1e-12)

    # --- the two stages the first version of this bisect stopped short
    # of, and therefore the only remaining candidates: the post-d_sw
    # delp/pt exchange (dyn_core.F90:1336-1337) and the D-grid pressure
    # chain (:1401 geopk, :1531 one_grad_p).  The stages above all
    # PASSED (job 9417559), so the wind error is made in one of these.
    import legoesm.grids.fv3_native_gridstruct as npgs  # noqa: F401
    from legoesm.core.fv3_dsw_tail_3d import dgrid_pressure_phase_3d

    for _nm in ("delp", "pt"):
        for k in range(KM):
            f6 = [n_dsw[t][_nm][:, :, k] for t in range(6)]
            npac._pad_scalars_6(f6, ctx, N, NG,
                                npgs.exchange_agrid_scalar_halos)
    j_dsw = {**j_dsw, "delp": jac._exchange_scalar_stack(
        j_dsw["delp"], jctx.tab, KM),
        "pt": jac._exchange_scalar_stack(j_dsw["pt"], jctx.tab, KM)}
    for nm in ("delp", "pt"):
        want = np.stack([np.asarray(n_dsw[t][nm]) for t in range(6)])
        cmp_fields(j_dsw[nm], want, f"stage post-d_sw exchange {nm}",
                   1e-12)

    # one_grad_p MUTATES the tail winds IN PLACE, so the copy handed to
    # it is the object that ends up holding the updated u/v -- keep it.
    # The first version passed `deepcopy_faces(n_tail)` inline and then
    # compared against `n_tail`, i.e. against the PRE-one_grad_p winds,
    # and duly reported rel = 1.000e+00 (job 9417607).  A comparison
    # against the wrong side of a mutation looks exactly like a defect.
    n_tail_prs = deepcopy_faces(n_tail)
    n_prs = nptail.dgrid_pressure_phase_3d(ctx, deepcopy_faces(n_dsw),
                                           n_tail_prs, KM,
                                           dt=dt, ptop=PTOP, akap=AKAP,
                                           cp_air=CP_AIR)
    j_prs = dgrid_pressure_phase_3d(jctx, j_dsw, j_tail, KM, dt=dt,
                                    ptop=PTOP, akap=AKAP, cp_air=CP_AIR)
    for nm in ("pk", "gz"):
        want = np.stack([np.asarray(n_prs[t][nm]) for t in range(6)])
        cmp_fields(j_prs[nm], want, f"stage D geopk {nm}", 1e-12)
    for nm in ("u", "v"):
        want = np.stack([np.asarray(n_tail_prs[t][nm]) for t in range(6)])
        cmp_fields(j_prs[nm], want, f"stage one_grad_p {nm}", 1e-12)
