"""Certification of the JAX km=1 duo STEPPER against the NumPy lane.

Authority: ``legoesm.core.fv3_native_duo_stepper`` is the SPECIFICATION
(hop B of ``docs/atmosphere/fv3_duo_jax_lane_strategy.md``).  The pinned
Fortran is quoted only where a line number says WHY a step exists; it is
never what a bound here is calibrated against.

This module contributes ORDER, not arithmetic -- every operation inside
it happens in a kernel gated by ``test_fv3_duo_sw_core.py``,
``test_fv3_pgrad.py`` or ``test_fv3_duo_halos.py``.  So the gates below
are chosen to interrogate the JOINS, which is precisely what per-kernel
gates cannot see (FESOM2-JAX §2.4's localization argument, strategy §2):

1. **parity** -- JAX vs the NumPy twin from a byte-identical initial
   state, per field, per stage, with a measured bound carrying a
   bound MEASURED in job 9425294 and set to measured x 10;
2. **tier 3, the multi-step replay** -- N steps on both lanes from one
   IC, per-field growth judged against the 1-step value.  This is the
   gate single-step tests cannot provide and the one that catches a
   defect at a join between kernels rather than inside one;
3. **jit vs eager** -- an ASSERTION, plus a trace counter proving no
   retrace on a new ``dt`` AND (non-vacuity) that the counter DOES move
   on a new ``d_ext``.  Bitwise is asserted ONLY where there is no
   floating-point sum for XLA to contract into an FMA -- here that is
   exactly the stale-halo carry-forward, which is a pure index copy;
4. **gradients** -- the PRIMARY gate is the tolerance-free adjoint
   identity ``<J v, w> == <v, J^T w>`` (``J v`` from ``jax.jvp``,
   ``J^T w`` from ``jax.vjp``), per operand group, with an explicit
   non-vacuity assert that ``<J v, w> != 0``.  ``check_grads`` is a
   SCOPED supplement on the module's own new leaves (the ``*_1lev``
   pressure adapters), with ``order=1`` and ``order=2`` as separate
   parametrised IDs so an FD-resolution failure can never be confused
   with a wrong Jacobian (STATE lesson 12);
5. **the two behavioural pins** -- the deliberately STALE-BY-ONE D-wind
   halo (``dyn_core.F90:1332-1338``) and the ``entry_ascalar`` ``it==1``
   gate (``:432``).  Each is asserted BOTH ways: the faithful behaviour,
   and a non-vacuity check that the alternative would have produced a
   different array.

TOLERANCE POLICY.  A full acoustic step is not a neighbour-reading
kernel: it runs the PPM limiters inside every transport call, and a
limiter flag ADDS or DROPS a whole flux term, so the result is
DISCONTINUOUS across a switching surface and a rounding-level lane
difference near one can produce a discrepancy far above 1e-15.  Every
numeric bound below is MEASURED (job 9425294, the LEGOESM_FV3_TOL_MEASURE
sweep) and set to measured x 10, keeping its class label.  The loosest
bounds in the file are the full-step jit-vs-eager wind gaps
(u 7.511e-07, v 7.456e-07; the localisers place them in the stage chain
at 5.646e-07), PLAUSIBLE-MECHANISM (see scope note) by probe job 9433881 as selector bit-flips
at 6 of 2052 cells -- see ``_JIT_EAGER_BOUND`` for the record and the
flip-cell census gate that now accompanies the magnitude bound.

COST.  These gates run a whole six-face acoustic step, so they are
minutes, not seconds: at C12 the module compiles the step for four
distinct static configurations plus the jvp and vjp programs.  That is
inherent to gating a composition and is stated here so the measurement
job's wall clock is not read as a hang.
"""
from __future__ import annotations

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (  # noqa: E402
    FV3DuoDynamicsModel,
    ORACLE_DAMPING,
)
from legoesm.core import fv3_duo_stepper as jstep_mod  # noqa: E402
from legoesm.core import fv3_native_duo_stepper as npstep  # noqa: E402
from legoesm.grids import fv3_duo_halos as jhalo  # noqa: E402

from tests.grids.fv3_gate_helpers import (  # noqa: E402
    gate_scalar,
    gated_check_grads,
)

# C12 is the smallest resolution the duo corner-region Lagrange fill
# admits (`build_jax_duo_halo_tables` refuses n < 4 because the X- and
# X+ abscissa windows would overlap the opposite wedge) and is the
# resolution every other JAX-lane test module uses, so the fixtures are
# comparable across the port.
N, NG = 12, 3
MA = N + 2 * NG
MB = N + 2 * NG + 1
NPX = N + 1

# A time step comfortably inside the C12 acoustic CFL.  Its magnitude is
# irrelevant to a one-step parity gate and load-bearing for the 4-step
# replay, which must not be measuring a blow-up.
DT = 450.0

# Every duo deck RESOLVES D_EXT = 0.0 (logfile.000000.out:406), so that
# is the value the parity gates run at.  D_EXT_ON exercises the
# external-mode filter branch, which is otherwise DEAD CODE in this
# module (`one_grad_p` has `if d_ext > 0.0`) -- a gate suite that never
# takes it would certify nothing about `divg2`.
D_EXT_OFF = 0.0
D_EXT_ON = 0.02

_STATE_KEYS = ("delp", "pt", "u", "v")


# =====================================================================
# fixtures
# =====================================================================

@pytest.fixture(scope="module")
def ctx():
    """The NumPy context, EXT-BUNDLE lane (the faithful one).

    ``oracle_conventions=True`` is the BOUNDED-conventions gridstruct
    the Zenodo duo runs execute; ``duogrid`` forces it
    (fv_arrays.F90:1512) and ``c_sw`` refuses the combination without
    it.
    """
    return npstep.build_six_face_duo_context(
        N, NG, use_ext_bundle=True, oracle_conventions=True)


@pytest.fixture(scope="module")
def jctx(ctx):
    return jstep_mod.build_jax_duo_stepper_context(ctx)


@pytest.fixture(scope="module")
def states0(ctx):
    """The balanced Williamson-2 six-face IC -- a PHYSICAL state.

    Not random noise: the limiters, the divergence damping and the
    pressure gradient all behave differently on a balanced field, and a
    gate run on noise would be certifying a regime the model never
    visits.
    """
    return npstep.w2_six_face_state(ctx)


@pytest.fixture(scope="module")
def jstates0(states0):
    return jstep_mod.states_to_jax(states0)


@pytest.fixture(scope="module")
def jstep(jctx):
    """The production jitted step, compiled ONCE for the whole module."""
    return jstep_mod.make_full_acoustic_step_sixface_jit()


# =====================================================================
# helpers
# =====================================================================

_SENTINEL_FLOOR = 1.0e20


# ⛔ KNOWN INSTRUMENT DEFECT IN THIS FILE, INHERITED AND NOT YET FIXED.
# A concurrent session RETRACTED the magnitude-based classification
# below in ``test_fv3_duo_sw_core.py`` (same job, 9404093): a cell is a
# workspace fill only if it holds ``1e30`` or ``1e25`` EXACTLY, and
# classifying by ``|x| >= 1e20`` wrongly swept up the sentinel-
# PROPAGATED cascade (1e22 … 1e111), which is ordinary arithmetic and
# therefore subject to FMA contraction.  That session also replaced the
# GLOBAL ``max|a-b| / max|b|`` metric with a PER-ELEMENT
# ``|a-b| / (|b| + median|b|)``, so one huge cell can no longer divide
# every real discrepancy to nothing.  Both corrections apply here.
#
# They are NOT applied in this round ON PURPOSE.  Swapping the metric
# now would change the tolerance definition at the same time as the
# three fixes this round is testing, and the next run has to be a
# ONE-VARIABLE test of those fixes -- a red gate would be unattributable
# between "the jit gap is still there" and "the metric moved".  Every
# number quoted in this file is therefore under the GLOBAL metric and is
# labelled as such.  FOLLOW-UP, next round: adopt the corrected helper
# (ideally as one shared ``tests/grids/conftest.py`` fixture rather than
# a third copy) and RE-MEASURE every bound under it.
#
# The stepper's exposure to the classification half is lower than
# sw_core's, and that is measured rather than assumed: this module runs
# ``d_sw1`` with ``workspace_sentinel=0.0`` (the NumPy stepper's own
# choice, fv3_native_duo_stepper.py:680), which is precisely what stops
# the 1e30 cascade from forming.  The SCALE half applies in full.


def _cmp(got, ref, name, tol):
    """Mask-aware relative comparison -- and it MUST be able to fail.

    Restates the contract of ``test_fv3_duo_sw_core._cmp`` (pytest
    modules are not an import surface, and importing one test module
    from another would execute its fixtures' module-level jax config).
    See the retraction banner above for what is already known to be
    wrong with this version and why it is being fixed in the next round
    rather than this one.

    Three defects it guards against, each earned in this campaign:

    * a mismatch in the NON-FINITE mask is a port bug on its own (a cell
      the oracle never writes must stay a tripwire in BOTH lanes), so it
      is checked before any value;
    * a mismatch in the SENTINEL mask is the same defect wearing a
      finite disguise -- ``1e30`` passes every ``isfinite`` guard;
    * a sentinel left in the comparison SET makes the relative bound
      VACUOUS, because ``max|ref|`` becomes ``1e30`` and every physical
      discrepancy divides to nothing.
    """
    a = np.asarray(got, dtype=np.float64)
    b = np.asarray(ref, dtype=np.float64)
    assert a.shape == b.shape, (name, a.shape, b.shape)

    na, nb = ~np.isfinite(a), ~np.isfinite(b)
    assert np.array_equal(na, nb), (
        f"{name}: non-finite masks differ (jax {int(na.sum())} vs numpy "
        f"{int(nb.sum())} cells of {a.size})")
    assert np.array_equal(a[na], b[nb], equal_nan=True), (
        f"{name}: non-finite VALUES differ")

    sa = np.isfinite(a) & (np.abs(a) >= _SENTINEL_FLOOR)
    sb = np.isfinite(b) & (np.abs(b) >= _SENTINEL_FLOOR)
    assert np.array_equal(sa, sb), (
        f"{name}: sentinel masks differ (jax {int(sa.sum())} vs numpy "
        f"{int(sb.sum())} cells of {a.size})")
    if sa.any():
        assert np.array_equal(a[sa], b[sa]), (
            f"{name}: the sentinel VALUES differ -- both lanes must "
            f"round-trip the identical workspace fill")

    ok = np.isfinite(a) & ~sa
    if not ok.any():
        # Every cell is a sentinel/tripwire: the exact mask checks above
        # ARE the whole gate for such slots (14 legitimate call paths;
        # the first version of the codex-MAJOR fix raised here and broke
        # them all -- measured, job 9431489). Codex's actual hole was
        # SILENCE: this used to return "0.0 measured" and vanish from
        # the harvest. Now it is loud in the harvest and explicit in
        # normal mode's logs, but not an error -- the mask equality was
        # asserted and passed.
        if os.environ.get("LEGOESM_FV3_TOL_MEASURE") == "1":
            print(f"TOLMEASURE {name!r}: mask-only (0 physical cells; "
                  f"the sentinel-mask equality is the gate)", flush=True)
        return 0.0
    scale = max(float(np.abs(b[ok]).max()), 1e-30)
    rel = float(np.abs(a[ok] - b[ok]).max()) / scale
    # MEASUREMENT MODE (LEGOESM_FV3_TOL_MEASURE=1): print and skip ONLY
    # the tolerance assert; the mask checks above still raise.
    if os.environ.get("LEGOESM_FV3_TOL_MEASURE") == "1":
        print(f"TOLMEASURE {name!r}: rel {rel:.3e} (bound {tol:.3e})",
              flush=True)
        return rel
    assert rel <= tol, (
        f"{name}: rel {rel:.3e} > {tol:.3e} MEASURED={rel:.3e} "
        f"over {int(ok.sum())} of {a.size} cells "
        f"(bitwise={np.array_equal(a[ok], b[ok])})")
    return rel


def _rel(got, ref) -> float:
    """Max relative difference over the finite, non-sentinel cells.

    The measurement half of :func:`_cmp` with no assertion, used by the
    tier-3 replay, which compares GROWTH rather than a fixed bound.

    It uses the SAME metric as :func:`_cmp` by construction, and that is
    load-bearing rather than tidiness: it is what let job 9404093's
    replay ``rel(1)`` for ``u`` be compared digit-for-digit against the
    jit-vs-eager gap and found IDENTICAL, which is the evidence that the
    two are one number.  If one of the pair is ever migrated to the
    corrected per-element metric, the other must migrate in the same
    commit or that comparison silently stops meaning anything.
    """
    a = np.asarray(got, dtype=np.float64)
    b = np.asarray(ref, dtype=np.float64)
    ok = (np.isfinite(a) & np.isfinite(b)
          & (np.abs(b) < _SENTINEL_FLOOR))
    assert ok.any(), "no comparable cells -- the replay metric is vacuous"
    scale = max(float(np.abs(b[ok]).max()), 1e-30)
    return float(np.abs(a[ok] - b[ok]).max()) / scale


def _stack_np(per_face: list) -> dict:
    """Six NumPy per-face dicts -> the JAX lane's face-stacked container."""
    keys = tuple(per_face[0])
    return {k: jnp.asarray(np.stack([np.asarray(d[k], dtype=np.float64)
                                     for d in per_face]))
            for k in keys}


def _deepcopy_faces(per_face: list) -> list:
    """Six per-face dicts with every array COPIED.

    Required, not tidiness: the NumPy ``dsw12_step_sixface`` passes its
    ``csw_outs[t]["divg_d"]`` straight into ``exchange_post_pgrad_sixface``,
    which mutates IN PLACE -- so calling it on a module-scoped fixture
    would silently corrupt every later test that reads the same fixture.
    """
    return [{k: np.array(v, copy=True) for k, v in d.items()}
            for d in per_face]


# Per-field COMPUTE windows: what the step actually writes.  Used
# wherever a difference has to be reduced -- the halo carries cells the
# lane deliberately leaves as NaN/sentinel tripwires, and reducing over
# them with `nanmax` would HIDE a real NaN instead of excluding a known
# one.  Exclude by WINDOW, never by a nan-aware reduction.
_WINDOWS = {
    "delp": (slice(NG, NG + N), slice(NG, NG + N)),
    "pt": (slice(NG, NG + N), slice(NG, NG + N)),
    "u": (slice(NG, NG + N), slice(NG, NG + N + 1)),
    "v": (slice(NG, NG + N + 1), slice(NG, NG + N)),
}


def _window(a, field):
    i, j = _WINDOWS[field]
    return np.asarray(a)[:, i, j]


def _max_window_diff(a, b) -> float:
    """max |a - b| over the four prognostics' compute windows."""
    return max(float(np.max(np.abs(_window(a[k], k) - _window(b[k], k))))
               for k in _STATE_KEYS)


def _tree_dot(a, b) -> float:
    """Plain inner product -- NO ``nan_to_num``.

    Sanitising here would silently repair a NaN the adjoint identity is
    supposed to EXPOSE (an R1b dead-branch leak shows up as exactly
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

    NO finite differences: ``J v`` comes from ``jax.jvp`` and
    ``J^T w`` from ``jax.vjp``, so the identity is exact in exact
    arithmetic and the residual is pure floating-point roundoff.  Its
    power does not depend on an FD step, on operand scaling, or on the
    output's dynamic range -- which is why it, and not
    ``check_grads``, is the primary gradient gate for this port.
    """
    rng = np.random.default_rng(seed)
    primals = tuple(jnp.asarray(p) for p in primals)
    v = tuple(jnp.asarray(rng.standard_normal(p.shape)) for p in primals)
    _, jv = jax.jvp(f, primals, v)
    for i, leaf in enumerate(jax.tree_util.tree_leaves(jv)):
        assert np.isfinite(np.asarray(leaf)).all(), (
            f"J v leaf {i} is not finite -- the objective window "
            f"includes cells the step never writes, or a dead branch is "
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
    """(traced-call counter, wrapper) for the retrace assertions."""
    box = {"n": 0}

    def wrapper(*a, **k):
        box["n"] += 1
        return fn(*a, **k)

    return box, wrapper


def _perturb_halo(states, key="delp", amount=1.0):
    """Perturb ONE halo side-strip, leaving every compute cell alone.

    Fortran ``i = 0`` (numpy row ``ng-1``, the first halo ring), ``j``
    over the compute span: a cell the A-scalar strip exchange
    unconditionally rewrites from the neighbour face, and one every
    upwind stencil in ``c_sw``/``d_sw1`` reads.  That is what makes the
    ``entry_ascalar`` gate OBSERVABLE -- on an already-consistent state
    the exchange is a no-op and the gate would be untestable.
    """
    f = states[key]
    return {**states,
            key: f.at[:, NG - 1, NG:NG + N].add(amount)}


# =====================================================================
# bookkeeping -- the module's own claims about its restated constants
# =====================================================================

def test_sw_config_defaults_match_the_numpy_lane():
    """The restated stage defaults must equal the NumPy lane's.

    They are restated rather than imported because ``_SW_CFG_DEFAULT``
    is PRIVATE and a cross-module private import is banned by
    ``tests/test_no_private_cross_imports.py``; this test is what makes
    the duplication safe.  Attribute access on a private symbol from a
    TEST is not an import and is the established pattern in
    ``test_fv3_duo_sw_core.py``.
    """
    assert jstep_mod.SW_CFG_DEFAULT._asdict() == dict(
        npstep._SW_CFG_DEFAULT)
    # SW_CFG_CASE8 is PUBLIC on the NumPy side, so this one is a
    # cross-check against the real symbol, not a restatement
    assert jstep_mod.SW_CFG_CASE8._asdict() == dict(npstep.SW_CFG_CASE8)


def test_stepper_nq_matches_the_allflux_slot_count():
    """Barrier 1's ``do iq=1,4+nq`` and ``d_sw1``'s allocation must agree.

    The NumPy stepper passes ``nq = 1`` positionally
    (``average_allflux_shared_edges(afx6, afy6, 1, n, ng)``); the JAX
    barrier reads it off the halo table and REFUSES a mismatched slot
    axis, so a silent disagreement is impossible -- but a silent CHANGE
    on either side would break every stepper build, which is what this
    pins.
    """
    assert jstep_mod._STEPPER_NQ == 1


def test_no_donate_argnums_in_this_lane():
    """Strategy R4: buffer donation conflicts with reverse-mode AD, and
    this lane exists to be differentiated.

    The discriminator is the USE (``donate_argnums=``), not the word --
    the jit-policy comment names the ban in prose, so a bare substring
    test would be vacuous.
    """
    import inspect
    src = inspect.getsource(jstep_mod)
    assert "donate_argnums=" not in src
    assert "donate_argnums" in src, (
        "the doctrine comment naming the ban has gone missing")


# =====================================================================
# tier 0 -- the state layout contract
# =====================================================================

def test_state_layout_is_face_stacked(jstates0):
    """Contract §1/§2: a LEADING face axis of 6, per stagger.

    A ``(6, …)`` stack exists within one stagger because all six faces
    share a shape; ACROSS staggers it does not, which is why ``u`` and
    ``v`` are two arrays and never one.
    """
    # ``w`` is optional on INPUT (case-6 ICs carry it, W2 does not) and
    # never on output, so the admitted key set is stated as a range
    assert set(_STATE_KEYS) <= set(jstates0)
    assert set(jstates0) <= set(_STATE_KEYS) | {"w"}
    assert jstates0["delp"].shape == (6, MA, MA)
    assert jstates0["pt"].shape == (6, MA, MA)
    assert jstates0["u"].shape == (6, MA, MB)
    assert jstates0["v"].shape == (6, MB, MA)
    for k, a in jstates0.items():
        assert a.dtype == jnp.float64, (k, a.dtype)


def test_state_boundary_adapters_round_trip(states0, jstates0):
    """BITWISE -- the adapters are pure reshapes/copies with no
    floating-point sum for XLA to contract, so bitwise is the right
    assertion here (and would be wrong on any summing path)."""
    back = jstep_mod.states_to_numpy(jstates0)
    assert len(back) == 6
    for t in range(6):
        for k in _STATE_KEYS:
            assert np.array_equal(back[t][k],
                                  np.asarray(states0[t][k]))


def test_context_is_identity_hashable(ctx, jctx):
    """The context is a STATIC jit argument, hashed by identity.

    Consequence stated in its docstring and pinned here: two
    structurally identical bundles are two cache keys, so a caller that
    rebuilds it per step would recompile per step.
    """
    assert hash(jctx) == id(jctx)
    other = jstep_mod.build_jax_duo_stepper_context(ctx)
    assert other != jctx and hash(other) != hash(jctx)
    assert jctx == jctx


def test_context_carries_the_static_halves(ctx, jctx):
    assert (jctx.n, jctx.ng, jctx.npx, jctx.m_a) == (N, NG, NPX, MA)
    assert jctx.bd is ctx["bd"]
    assert len(jctx.gs6) == 6 and len(jctx.flags6) == 6
    for t in range(6):
        assert jctx.flags6[t].bounded_domain is True
        assert jctx.flags6[t].da_min_c == ctx["gs6"][t]["da_min_c"]
    assert jctx.hs6.shape == (6, MA, MA)
    assert jctx.tab.nq == jstep_mod._STEPPER_NQ


# =====================================================================
# tier 0 -- guards (dispatch hardening); each shown NON-VACUOUS by the
# faithful context building successfully in the same test
# =====================================================================

def test_context_refuses_the_non_ext_bundle_lane(ctx):
    bad = {**ctx, "use_ext_bundle": False}
    with pytest.raises(ValueError, match="EXT-BUNDLE lane only"):
        jstep_mod.build_jax_duo_stepper_context(bad)
    # non-vacuous: the same ctx WITH the bundle builds
    assert jstep_mod.build_jax_duo_stepper_context(ctx) is not None


def test_context_refuses_the_legacy_and_unported_options(ctx):
    with pytest.raises(ValueError, match="use_k2e_scalars"):
        jstep_mod.build_jax_duo_stepper_context(
            {**ctx, "use_k2e_scalars": True})
    with pytest.raises(ValueError, match="step_dump"):
        jstep_mod.build_jax_duo_stepper_context(
            {**ctx, "step_dump": lambda *a: None})


@pytest.mark.parametrize("fam", ["ascalar", "dvec", "divgd", "cvec"])
def test_context_refuses_a_step_time_ext_exclusion(ctx, fam):
    """The four STEP-TIME families each substitute a different exchange
    inside the step; the JAX lane implements the faithful path only."""
    with pytest.raises(ValueError, match="ext_exclude"):
        jstep_mod.build_jax_duo_stepper_context(
            {**ctx, "ext_exclude": (fam,)})


@pytest.mark.parametrize("fam", ["metrics", "f0"])
def test_context_accepts_a_build_time_ext_exclusion(ctx, fam):
    """``metrics``/``f0`` are consumed by ``build_six_face_duo_context``
    BEFORE the step, so excluding them changes the context, not the
    cadence -- and must not be refused (that would be a guard firing on
    the wrong set)."""
    assert jstep_mod.build_jax_duo_stepper_context(
        {**ctx, "ext_exclude": (fam,)}) is not None


@pytest.mark.parametrize(
    "var,val,match",
    [("LEGOESM_DUO_ENTRY_ASCALAR", "off", "entry_ascalar=False"),
     ("LEGOESM_DUO_PG_BVERTEX", "mean2", "bvertex"),
     ("LEGOESM_DUO_AVG_B_ENDPOINTS", "local", "skip_b_endpoints")])
def test_env_diagnostic_modes_are_refused(ctx, monkeypatch, var, val,
                                          match):
    """D2 -- a production lane whose numerics depend on the environment
    is a silent-divergence trap.  Each variable CHANGES what the NumPy
    stepper computes, so a JAX run made with one exported would
    disagree with the NumPy run it is compared against."""
    monkeypatch.setenv(var, val)
    with pytest.raises(ValueError, match=match):
        jstep_mod.build_jax_duo_stepper_context(ctx)
    monkeypatch.delenv(var)
    # non-vacuous: unset, the same context builds
    assert jstep_mod.build_jax_duo_stepper_context(ctx) is not None


def test_dtype_uniformity_and_model_boundary_guards(jctx, jstates0):
    """Uniform f32 is legal within a phase; mixed state is not.

    The model boundary separately refuses a uniform f32 state when the
    configured storage dtype is f64, before calling the compiled step.
    """
    f32_faces = [
        {k: np.zeros((MA, MA), np.float32) for k in _STATE_KEYS}
        for _ in range(6)
    ]
    uniform_f32 = jstep_mod.states_to_jax(f32_faces)
    assert all(a.dtype == jnp.float32 for a in uniform_f32.values())

    mixed = {**jstates0, "delp": jstates0["delp"].astype(jnp.float32)}
    with pytest.raises(TypeError, match="MIXED float dtypes"):
        jstep_mod.full_acoustic_step_sixface(jctx, mixed, DT,
                                             d_ext=D_EXT_OFF)

    # BOTH DIRECTIONS, on a REALLY CONSTRUCTED model (codex's version
    # used object.__new__ with _step_fn None, which cannot drift-check
    # against the constructor; GLM + Claude 2026-09-10).  The guard has
    # to refuse an f32 carry under f64 storage AND an f64 carry under
    # f32 storage -- the second is the silent-downcast direction.
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import FV3DuoConfig
    from legoesm.grids.factory import create_fv3_duo_grid
    grid24 = create_fv3_duo_grid(24)
    for storage, carry, other in (("float64", uniform_f32, jnp.float64),
                                  ("float32", jstates0, jnp.float32)):
        m = FV3DuoDynamicsModel(grid24, FV3DuoConfig(**ORACLE_DAMPING, km=5, hydrostatic=True,
                                                     n_split=1,
                                                     storage_dtype=storage))
        assert m._storage_dtype == np.dtype(other)
        with pytest.raises(TypeError, match="storage_dtype"):
            m.step({"state": carry, "press": None, "q": [], "omga": None,
                    "nh": None}, DT)


def test_sw_cfg_must_be_hashable(jctx, jstates0):
    """A plain dict cannot be a jit-static argument; refuse it at the
    door instead of failing inside ``jax.jit`` with a hashability
    error that names nothing."""
    with pytest.raises(TypeError, match="SWConfig.from_mapping"):
        jstep_mod.full_acoustic_step_sixface(
            jctx, jstates0, DT, d_ext=D_EXT_OFF,
            sw_cfg={"hord_tr": 8})
    # non-vacuous: the converted form is accepted
    cfg = jstep_mod.SWConfig.from_mapping({"hord_tr": 8})
    assert isinstance(cfg, jstep_mod.SWConfig) and cfg.hord_tr == 8


def test_sw_config_from_mapping_rejects_an_unknown_knob():
    """A knob that is silently ignored is a different run wearing the
    same name -- so an unknown key raises rather than being dropped."""
    with pytest.raises(ValueError, match="unknown knobs"):
        jstep_mod.SWConfig.from_mapping({"hord_zz": 8})


def test_exchange_post_pgrad_rejects_a_fractional_nord(jctx, jstates0):
    """``nord`` is a damping ORDER, integral by construction; the deck
    dicts mix ints and floats, so ``int()`` would round 2.7 to 2 without
    a word."""
    z = jnp.zeros((6, MB, MB), dtype=jnp.float64)
    uc = jnp.zeros((6, MB, MA), dtype=jnp.float64)
    vc = jnp.zeros((6, MA, MB), dtype=jnp.float64)
    with pytest.raises(ValueError, match="integral damping order"):
        jstep_mod.exchange_post_pgrad_sixface(jctx, z, uc, vc, nord=2.7)
    # non-vacuous: the integral value runs
    out = jstep_mod.exchange_post_pgrad_sixface(jctx, z, uc, vc, nord=2)
    assert len(out) == 3


def test_advance_and_run_reject_nonsense_counts(jctx, jstates0):
    with pytest.raises(ValueError, match="n_split must be >= 1"):
        jstep_mod.advance_duo_outer_step(jctx, jstates0, 900.0, 0)
    with pytest.raises(ValueError, match="nsteps must be >= 0"):
        jstep_mod.run_duo_sw(jctx, jstates0, DT, -1)


# =====================================================================
# gate 1 -- parity, leaf by leaf (the *_1lev pressure adapters)
# =====================================================================

@pytest.fixture(scope="module")
def csw_np(ctx, states0):
    """The NumPy ``c_sw`` outputs on the raw IC -- the shared operand
    for every leaf-level comparison below, so both lanes see BYTE-
    IDENTICAL inputs (the controlled-comparison requirement)."""
    return npstep.csw_step_sixface(ctx, states0, dt2=0.5 * DT)


def test_geopk_sw_1lev_parity(ctx, jctx, csw_np):
    """gate 1.  FIXTURE CLASS: accumulating (a k recurrence and a
    log/exp pair), no data branch."""
    bd = ctx["bd"]
    for t in range(6):
        hs = np.zeros_like(csw_np[t]["delpc"])
        pk_n, gz_n = npstep.geopk_sw_1lev(csw_np[t]["delpc"], hs, bd,
                                          pt=csw_np[t]["ptc"])
        pk_j, gz_j = jstep_mod.geopk_sw_1lev(
            jnp.asarray(csw_np[t]["delpc"]), jnp.asarray(hs), bd,
            pt=jnp.asarray(csw_np[t]["ptc"]))
        # MEASURED (job 9425294 sweep): worst 1.510e-16 (faces 2/5); bound = measured x 10 =
        # 1.6e-15.
        _cmp(pk_j, pk_n, f"geopk_sw_1lev.pk[face {t}]", 1.6e-15)
        # MEASURED (job 9425294 sweep): worst 1.510e-16 (faces 2/5); bound = measured x 10 =
        # 1.6e-15.
        _cmp(gz_j, gz_n, f"geopk_sw_1lev.gz[face {t}]", 1.6e-15)


def test_geopk_sw_1lev_d_widens_the_box(ctx, jctx, csw_np, states0):
    """gate 1 + a CONTROL: the D-grid call (``cg=.false.``) must write a
    WIDER box than the C-grid one (``is-2..ie+2`` vs ``is-1..ie+1``,
    dyn_core's geopk range predicate).  Without this the two adapters
    could be the same function under two names and the parity gate
    above would not notice."""
    bd = ctx["bd"]
    delp = np.asarray(states0[0]["delp"])
    pt = np.asarray(states0[0]["pt"])
    hs = np.zeros_like(delp)
    pk_n, gz_n = npstep.geopk_sw_1lev_d(delp, hs, bd, pt=pt)
    pk_j, gz_j = jstep_mod.geopk_sw_1lev_d(
        jnp.asarray(delp), jnp.asarray(hs), bd, pt=jnp.asarray(pt))
    # MEASURED (job 9425294 sweep): 1.239e-16; bound = measured x 10 =
    # 1.3e-15.
    _cmp(pk_j, pk_n, "geopk_sw_1lev_d.pk", 1.3e-15)
    # MEASURED (job 9425294 sweep): 1.239e-16; bound = measured x 10 =
    # 1.3e-15.
    _cmp(gz_j, gz_n, "geopk_sw_1lev_d.gz", 1.3e-15)
    pk_c, _ = jstep_mod.geopk_sw_1lev(jnp.asarray(delp), jnp.asarray(hs),
                                      bd, pt=jnp.asarray(pt))
    wide = np.asarray(pk_j)[:, :, 1] != 0.0      # unwritten_fill = 0.0
    narrow = np.asarray(pk_c)[:, :, 1] != 0.0
    assert wide.sum() > narrow.sum(), (
        "the D-grid geopk did not write a wider box than the C-grid one "
        f"({int(wide.sum())} vs {int(narrow.sum())} cells) -- the two "
        f"adapters are not selecting different cg branches")


def test_p_grad_c_1lev_parity(ctx, jctx, csw_np):
    """gate 1.  FIXTURE CLASS: pointwise/neighbour-reading, no branch."""
    bd = ctx["bd"]
    for t in range(6):
        hs = np.zeros_like(csw_np[t]["delpc"])
        pk_n, gz_n = npstep.geopk_sw_1lev(csw_np[t]["delpc"], hs, bd,
                                          pt=csw_np[t]["ptc"])
        uc_n = np.array(csw_np[t]["uc"], copy=True)
        vc_n = np.array(csw_np[t]["vc"], copy=True)
        npstep.p_grad_c_1lev(0.5 * DT, csw_np[t]["delpc"], pk_n, gz_n,
                             uc_n, vc_n, ctx["gs6"][t], bd)
        uc_j, vc_j = jstep_mod.p_grad_c_1lev(
            0.5 * DT, jnp.asarray(csw_np[t]["delpc"]), jnp.asarray(pk_n),
            jnp.asarray(gz_n), jnp.asarray(csw_np[t]["uc"]),
            jnp.asarray(csw_np[t]["vc"]), jctx.gs6[t], bd)
        # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), every face; bound =
        # 1e-15 eps guard (measured exactly 0.0).
        _cmp(uc_j, uc_n, f"p_grad_c_1lev.uc[face {t}]", 1e-15)
        # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), every face; bound =
        # 1e-15 eps guard (measured exactly 0.0).
        _cmp(vc_j, vc_n, f"p_grad_c_1lev.vc[face {t}]", 1e-15)


@pytest.mark.parametrize("d_ext", [D_EXT_OFF, D_EXT_ON])
def test_one_grad_p_1lev_parity(ctx, jctx, states0, d_ext):
    """gate 1, BOTH sides of the ``d_ext > 0`` branch.

    At ``d_ext = 0`` the external-mode filter increments are structurally
    zero; at 0.02 they are live.  A suite that ran only the deck value
    would leave ``wk1``/``wk2`` uncertified.
    """
    bd = ctx["bd"]
    for t in range(6):
        delp = np.asarray(states0[t]["delp"])
        pt = np.asarray(states0[t]["pt"])
        hs = np.zeros_like(delp)
        pk_n, gz_n = npstep.geopk_sw_1lev_d(delp, hs, bd, pt=pt)
        divg2 = np.full((NPX, NPX), 3.0e-4)
        u_n = np.array(states0[t]["u"], copy=True)
        v_n = np.array(states0[t]["v"], copy=True)
        npstep.one_grad_p_1lev(u_n, v_n, pk_n, gz_n, divg2,
                               ctx["gs6"][t], bd, NPX, NPX, dt=DT,
                               d_ext=d_ext)
        u_j, v_j = jstep_mod.one_grad_p_1lev(
            jnp.asarray(states0[t]["u"]), jnp.asarray(states0[t]["v"]),
            jnp.asarray(pk_n), jnp.asarray(gz_n), jnp.asarray(divg2),
            jctx.gs6[t], bd, NPX, NPX, dt=DT, d_ext=d_ext)
        # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), every face/d_ext; bound =
        # 1e-15 eps guard (measured exactly 0.0).
        _cmp(u_j, u_n, f"one_grad_p_1lev.u[face {t}, d_ext={d_ext}]",
             1e-15)
        # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), every face/d_ext; bound =
        # 1e-15 eps guard (measured exactly 0.0).
        _cmp(v_j, v_n, f"one_grad_p_1lev.v[face {t}, d_ext={d_ext}]",
             1e-15)


def test_one_grad_p_1lev_does_not_mutate_its_pressure_operands(ctx, jctx,
                                                               states0):
    """ALIASING CONTRACT.  The NumPy adapter has to COPY ``pkc``/``gz``
    on entry because the shared kernel is faithful to dyn_core and
    mutates them through ``a2b_ord4(replace=.true.)``; the JAX kernel
    returns new arrays instead, so the caller's operands must be
    untouched BY CONSTRUCTION.  Asserted, not argued -- the NumPy lane
    shipped a regression here once."""
    bd = ctx["bd"]
    delp = np.asarray(states0[0]["delp"])
    pt = np.asarray(states0[0]["pt"])
    pk, gz = jstep_mod.geopk_sw_1lev_d(jnp.asarray(delp),
                                       jnp.zeros_like(jnp.asarray(delp)),
                                       bd, pt=jnp.asarray(pt))
    pk_before = np.array(pk, copy=True)
    gz_before = np.array(gz, copy=True)
    jstep_mod.one_grad_p_1lev(
        jnp.asarray(states0[0]["u"]), jnp.asarray(states0[0]["v"]),
        pk, gz, jnp.zeros((NPX, NPX), dtype=jnp.float64), jctx.gs6[0],
        bd, NPX, NPX, dt=DT, d_ext=D_EXT_OFF)
    assert np.array_equal(np.asarray(pk), pk_before)
    assert np.array_equal(np.asarray(gz), gz_before)


def test_exchange_post_pgrad_parity(ctx, jctx, csw_np):
    """gate 1 -- the ``:652``/``:655`` pair, both faces of the ``nord``
    gate.  At ``nord = 0`` the divgd exchange must NOT fire (it is gated
    on the DIVERGENCE-damping order) while the uc/vc one still must."""
    for nord in (0, 1, 2):
        divgd_n = [np.array(o["divg_d"], copy=True) for o in csw_np]
        uc_n = [np.array(o["uc"], copy=True) for o in csw_np]
        vc_n = [np.array(o["vc"], copy=True) for o in csw_np]
        npstep.exchange_post_pgrad_sixface(ctx, divgd_n, uc_n, vc_n,
                                           nord=nord)
        d_j, u_j, v_j = jstep_mod.exchange_post_pgrad_sixface(
            jctx, _stack_np(csw_np)["divg_d"], _stack_np(csw_np)["uc"],
            _stack_np(csw_np)["vc"], nord=nord)
        # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), every nord; bound =
        # 1e-15 eps guard (measured exactly 0.0).
        _cmp(d_j, np.stack(divgd_n), f"post_pgrad.divgd[nord={nord}]",
             1e-15)
        # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), every nord; bound =
        # 1e-15 eps guard (measured exactly 0.0).
        _cmp(u_j, np.stack(uc_n), f"post_pgrad.uc[nord={nord}]", 1e-15)
        # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), every nord; bound =
        # 1e-15 eps guard (measured exactly 0.0).
        _cmp(v_j, np.stack(vc_n), f"post_pgrad.vc[nord={nord}]", 1e-15)


def test_exchange_post_pgrad_nord_gate_is_live(jctx, csw_np):
    """CONTROL for the test above: at ``nord = 0`` the divgd array must
    come back UNCHANGED, at ``nord > 0`` it must not.  Without this the
    parity gate would pass even if the exchange never ran."""
    d0 = _stack_np(csw_np)["divg_d"]
    uc = _stack_np(csw_np)["uc"]
    vc = _stack_np(csw_np)["vc"]
    off, _, _ = jstep_mod.exchange_post_pgrad_sixface(jctx, d0, uc, vc,
                                                      nord=0)
    on, _, _ = jstep_mod.exchange_post_pgrad_sixface(jctx, d0, uc, vc,
                                                     nord=1)
    assert np.array_equal(np.asarray(off), np.asarray(d0)), (
        "nord=0 changed divgd -- the :652 gate is not being applied")
    assert not np.array_equal(np.asarray(on), np.asarray(d0)), (
        "nord=1 left divgd unchanged -- the exchange did not run, so "
        "the parity gate above proves nothing")


# =====================================================================
# gate 1 -- parity, stage by stage
# =====================================================================

def test_csw_step_sixface_parity(ctx, jctx, states0, jstates0, csw_np):
    """gate 1.  FIXTURE CLASS: LIMITER-CROSSING (``c_sw`` runs the
    transport upwind selects), so this bound must not be shared with the
    pointwise gates."""
    got = jstep_mod.csw_step_sixface(jctx, jstates0, 0.5 * DT)
    ref = _stack_np(csw_np)
    assert set(got) == set(ref), (sorted(got), sorted(ref))
    for k in sorted(ref):
        # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), every key; bound =
        # 1e-15 eps guard (measured exactly 0.0).
        _cmp(got[k], ref[k], f"csw_step_sixface.{k}", 1e-15)


def test_dsw12_step_sixface_parity(ctx, jctx, states0, jstates0, csw_np):
    """gate 1, ISOLATED: both lanes are fed the SAME (NumPy) ``c_sw``
    output, so this measures ``dsw12``'s own residual rather than one
    accumulated through ``c_sw``.  The whole-step gates below measure
    the accumulation."""
    ref = npstep.dsw12_step_sixface(ctx, states0,
                                    _deepcopy_faces(csw_np), dt=DT)
    got = jstep_mod.dsw12_step_sixface(jctx, jstates0, _stack_np(csw_np),
                                       DT)
    ref_s = _stack_np(ref)
    assert set(got) == set(ref_s), (sorted(got), sorted(ref_s))
    for k in sorted(ref_s):
        # MEASURED (job 9425294 sweep): worst vc 1.181e-14; bound = measured x 10 =
        # 1.2e-13.
        _cmp(got[k], ref_s[k], f"dsw12_step_sixface.{k}", 1.2e-13)


def test_acoustic_step_sixface_parity(ctx, jctx, states0, jstates0):
    """gate 1 -- the whole stage chain including BOTH barriers, from a
    byte-identical IC.  Scores every carried field, not two headline
    numbers: a defect confined to (say) ``divg_d`` would leave
    ``delp``/``pt`` looking fine."""
    ref = _stack_np(npstep.acoustic_step_sixface(ctx, states0, DT))
    got = jstep_mod.acoustic_step_sixface(jctx, jstates0, DT)
    assert set(got) == set(ref), (sorted(got), sorted(ref))
    for k in sorted(ref):
        # MEASURED (job 9425294 sweep): worst ke 3.479e-16; bound = measured x 10 =
        # 3.5e-15.
        _cmp(got[k], ref[k], f"acoustic_step_sixface.{k}", 3.5e-15)


@pytest.mark.parametrize("d_ext", [D_EXT_OFF, D_EXT_ON])
def test_full_acoustic_step_parity(ctx, jctx, states0, jstates0, d_ext):
    """gate 1 -- THE gate this campaign converges on, at both settings
    of the external-mode filter.

    ``d_ext = 0`` is the value every duo deck resolves and is what any
    oracle comparison must use; ``d_ext = 0.02`` is here only to keep
    the ``divg2``/``wk1``/``wk2`` branch from being dead code in the
    suite.
    """
    ref = npstep.full_acoustic_step_sixface(ctx, states0, DT,
                                            d_ext=d_ext)
    got = jstep_mod.full_acoustic_step_sixface(jctx, jstates0, DT,
                                               d_ext=d_ext)
    ref_s = _stack_np(ref)
    assert set(got) == set(_STATE_KEYS), sorted(got)
    for k in _STATE_KEYS:
        # MEASURED (job 9425294 sweep): worst delp 4.958e-16 (both d_ext arms); bound = measured x 10 =
        # 5.0e-15.
        _cmp(got[k], ref_s[k],
             f"full_acoustic_step.{k}[d_ext={d_ext}]", 5.0e-15)


def test_full_acoustic_step_parity_case8_config(ctx, jctx, states0,
                                                jstates0):
    """gate 1 on the OTHER shipped configuration.

    ``SW_CFG_CASE8`` is what ``run_duo_stepper_case6.py`` runs and what
    the calibrated ``case6_duo_oracle_gate.py`` scores, so a lane
    certified only at the W2 defaults would be certified on a
    configuration the gate never uses: hords all 8, vorticity damping
    OFF, ``dddmp = 0``, ``nord = 2`` (a LONGER divergence-damping
    recurrence, i.e. a different unrolled program).
    """
    ref = npstep.full_acoustic_step_sixface(
        ctx, states0, DT, d_ext=D_EXT_OFF, sw_cfg=npstep.SW_CFG_CASE8)
    got = jstep_mod.full_acoustic_step_sixface(
        jctx, jstates0, DT, d_ext=D_EXT_OFF,
        sw_cfg=jstep_mod.SW_CFG_CASE8)
    ref_s = _stack_np(ref)
    for k in _STATE_KEYS:
        # MEASURED (job 9425294 sweep): worst u/v 3.680e-16; bound = measured x 10 =
        # 3.7e-15.
        _cmp(got[k], ref_s[k], f"full_step_case8.{k}", 3.7e-15)


def test_advance_duo_outer_step_parity(ctx, jctx, states0, jstates0):
    """gate 1 on the OUTER step -- two inner substeps, i.e. the first
    place the ``entry_ascalar`` cadence can differ between the lanes."""
    ref = _stack_np(npstep.advance_duo_outer_step(
        ctx, states0, 2.0 * DT, 2, d_ext=D_EXT_OFF))
    got = jstep_mod.advance_duo_outer_step(jctx, jstates0, 2.0 * DT, 2,
                                            d_ext=D_EXT_OFF)
    for k in _STATE_KEYS:
        # MEASURED (job 9425294 sweep): worst v 1.472e-14; bound = measured x 10 =
        # 1.5e-13.
        _cmp(got[k], ref[k], f"advance_duo_outer_step.{k}", 1.5e-13)


def test_run_duo_sw_parity(ctx, jctx, states0, jstates0):
    """gate 1 on the flat time loop.

    ``run_duo_sw`` and ``advance_duo_outer_step`` are DIFFERENT exchange
    schedules (flat = entry exchange every step; outer = once per
    block), so each needs its own gate -- one cannot stand in for the
    other.
    """
    ref = _stack_np(npstep.run_duo_sw(ctx, states0, DT, 2,
                                      d_ext=D_EXT_OFF))
    got = jstep_mod.run_duo_sw(jctx, jstates0, DT, 2, d_ext=D_EXT_OFF)
    for k in _STATE_KEYS:
        # MEASURED (job 9425294 sweep): worst v 1.472e-14; bound = measured x 10 =
        # 1.5e-13.
        _cmp(got[k], ref[k], f"run_duo_sw.{k}", 1.5e-13)


# NOTE on the pair above: the flat and outer schedules are NOT asserted
# to differ from each other.  They differ only in whether the SECOND
# substep re-runs the entry A-scalar exchange, and after the first
# step's tail refresh (dyn_core.F90:1336-1337) that exchange may be a
# no-op -- whether it is depends on what the k2e ring stencil reads,
# which has not been established here.  Asserting a difference would be
# asserting an unverified claim.  The cadence itself is gated, twice and
# decisively, by `test_entry_ascalar_gate_is_live` and
# `test_advance_duo_outer_step_fires_entry_only_on_the_first_substep`.


# =====================================================================
# gate 2 (tier 3) -- the MULTI-STEP REPLAY
# =====================================================================

# Allowed amplification of the per-field lane difference at N steps,
# relative to the ANCHOR below.  This is the gate that single-step tests
# cannot provide: a defect at a JOIN between kernels (rather than inside
# one) shows up as growth, not as a large first step.
#
# MEASURED, job 9404093 (C12, dt=450 s, d_ext=0, jitted JAX arm):
#   delp   rel(1) 1.810e-14   rel(2) 1.324e-07   rel(4) 2.053e-07
#   u      rel(1) 7.511e-07   rel(2) 8.285e-06   rel(4) <= 7.511e-05
#   pt, v  passed at the old bounds; values NOT recorded (see below)
# u's 1->2 ratio is 11.03; delp's is 7315, then only 1.55 from 2 to 4.
#
# THE ANCHOR IS NOT rel(1) ALONE, and that correction is the real fix
# here.  `delp`'s rel(1) of 1.8e-14 is not a floor -- it is `delp` not
# yet having felt the step-1 wind error, which reaches it one step later
# through the flux divergence -- so anchoring on it demanded that a
# LATER step stay within 10x of machine precision.  The strategy doc's
# own standing rider says exactly this ("if the measured maximum is
# exactly zero, multiplying pins the bound to zero").  The anchor is
# therefore max(rel(1), _REPLAY_FLOOR), with the floor set from the
# lane's actual measured one-step difference (u, 7.5e-07).
#
# MEASURED (job 9425294 sweep), the full table this time: growth ratio
# rn/anchor is u 8.285 (n=2) / 6.992 (n=4), v 2.447 / 2.126, delp 0.132
# / 0.205, pt ~4e-9 -- worst is u at each n; bounds = measured x 10.
# [class: N-step growth vs the one-step floor]
_REPLAY_AMP = {2: 8.3e1, 4: 7.0e1}

# MEASURED (job 9425294 sweep): the lane's one-step floor is 7.511e-07
# (u; v 7.456e-07 -- v now carries the gap too), still DOMINATED BY THE
# JIT-VS-EAGER GAP and therefore an UNEXPLAINED residual, not agreement
# -- see `test_full_step_jit_equals_eager`.  Rounded up to 1e-6.
# [class: one-step floor of the whole step]
_REPLAY_FLOOR = 1.0e-6

# How many steps the EAGER arm replays.  Two is enough to answer the
# only question it exists for -- is the growth a join defect or an
# artefact of the jitted lowering? -- and each eager step costs far more
# than a compiled one.
_REPLAY_EAGER_N = 2


class ReplayMeasurement(UserWarning):
    """Carries the replay table out of a PASSING test.

    A gate that reports only on failure cannot pin a tolerance: job
    9404093 left `pt`, `v` and `u`'s 4-step value unmeasured for exactly
    that reason.  pytest prints its warnings summary for passing tests
    too, so this is the emission channel that does not depend on `-s`.
    """


@pytest.fixture(scope="module")
def replay(ctx, jctx, states0, jstates0, jstep):
    """Three lanes, 1/2/4 steps from ONE initial state.

    * NumPy -- the authority;
    * JAX JITTED -- the lane the runners execute, and the growth gate's
      subject;
    * JAX EAGER, to ``_REPLAY_EAGER_N`` steps -- the DISCRIMINATOR.  If
      the eager arm tracks NumPy while the jitted arm does not, the
      growth is an artefact of the compiled lowering and not a defect at
      a join between kernels.  Those two have entirely different fixes,
      so the gate must not report growth without saying which it is.
    """
    out = {}
    s_np, s_jx, s_eg = states0, jstates0, jstates0
    for n in range(1, 5):
        s_np = npstep.full_acoustic_step_sixface(ctx, s_np, DT,
                                                 d_ext=D_EXT_OFF)
        s_jx = jstep(jctx, s_jx, DT, d_ext=D_EXT_OFF)
        if n <= _REPLAY_EAGER_N:
            s_eg = jstep_mod.full_acoustic_step_sixface(
                jctx, s_eg, DT, d_ext=D_EXT_OFF)
        if n in (1, 2, 4):
            out[n] = (_stack_np(s_np), s_jx,
                      s_eg if n <= _REPLAY_EAGER_N else None)

    rows = []
    for n in sorted(out):
        ref, got, eg = out[n]
        for k in _STATE_KEYS:
            eager = ("" if eg is None
                     else f" eager {_rel(eg[k], ref[k]):.4e}")
            rows.append(f"n={n} {k}: jit {_rel(got[k], ref[k]):.4e}"
                        f"{eager}")
    warnings.warn("TIER-3 REPLAY TABLE (vs NumPy, C12 dt=450 d_ext=0): "
                  + "; ".join(rows), ReplayMeasurement, stacklevel=1)
    return out


@pytest.mark.parametrize("field", _STATE_KEYS)
def test_tier3_replay_stays_finite(replay, field):
    """A replay whose state went non-finite would make every growth
    ratio meaningless, so this runs before any ratio.

    Finiteness is asserted on the COMPUTE WINDOW: the halo legitimately
    carries the lane's NaN tripwires (cells the oracle never writes), so
    demanding a finite full array would fail on correct code.  The halo
    is not skipped, though -- the non-finite MASKS must match between
    the lanes everywhere, which is the actual port check.
    """
    for n, (ref, got, _eg) in replay.items():
        assert np.isfinite(_window(ref[field], field)).all(), (
            field, n, "numpy compute window")
        assert np.isfinite(_window(got[field], field)).all(), (
            field, n, "jax compute window")
        assert np.array_equal(~np.isfinite(np.asarray(ref[field])),
                              ~np.isfinite(np.asarray(got[field]))), (
            f"{field} at n={n}: the non-finite masks differ between the "
            f"lanes -- a cell one lane leaves as a tripwire the other "
            f"filled with a number")


@pytest.mark.parametrize("field", _STATE_KEYS)
@pytest.mark.parametrize("nsteps", [2, 4])
def test_tier3_multistep_replay_growth(replay, field, nsteps):
    """TIER 3 -- the join gate.

    Because every kernel is verified on its own, a difference that
    appears only after several steps cannot originate inside a kernel
    and must arise where two kernels are joined (FESOM2-JAX §2.4).  The
    criterion is therefore GROWTH relative to the 1-step value, not an
    absolute bound: an absolute bound at N steps would be satisfied by a
    lane whose first step was already wrong.
    """
    ref1, got1, _ = replay[1]
    refn, gotn, _ = replay[nsteps]
    r1 = _rel(got1[field], ref1[field])
    rn = _rel(gotn[field], refn[field])
    anchor = max(r1, _REPLAY_FLOOR)
    bound = anchor * _REPLAY_AMP[nsteps]
    if os.environ.get("LEGOESM_FV3_TOL_MEASURE") == "1":
        print(f"TOLMEASURE 'tier3 growth {field} n={nsteps}': rel "
              f"{rn / anchor:.3e} (bound {_REPLAY_AMP[nsteps]:.3e}, "
              f"quantity n-step growth ratio rn/anchor; r1 {r1:.3e}, "
              f"rn {rn:.3e}, floor {_REPLAY_FLOOR:.3e})", flush=True)
        return
    assert rn <= bound, (
        f"{field}: {nsteps}-step lane difference {rn:.3e} > {bound:.3e} "
        f"(1-step {r1:.3e}, anchor {anchor:.3e}, allowed amplification "
        f"{_REPLAY_AMP[nsteps]:g}) -- MEASURED 1-step {r1:.3e}, "
        f"{nsteps}-step {rn:.3e}")


def test_tier3_eager_replay_isolates_the_jit_gap(replay):
    """THE DISCRIMINATOR for the growth gate above.

    Two mechanisms produce N-step growth and they have nothing in
    common: a defect at a JOIN between kernels, or a difference in the
    compiled lowering that the eager lane does not have.  Reporting
    growth without saying which would be reporting a symptom.

    So: the EAGER JAX arm is scored against NumPy over the same steps.
    Eager tracking NumPy while the jitted arm does not means the growth
    is carried by the jitted lowering -- there is no join defect, and
    the fix is in the compiled path, not in the composition.
    """
    for n in sorted(k for k in replay if replay[k][2] is not None):
        ref, got, eg = replay[n]
        for field in _STATE_KEYS:
            r_eager = _rel(eg[field], ref[field])
            r_jit = _rel(got[field], ref[field])
            # MEASURED (job 9425294 sweep): worst v n=2 1.472e-14 (eager lane vs NumPy); bound = measured x 10 =
            # 1.5e-13.
            gate_scalar(f"tier3 eager {field} n={n}", r_eager, 1.5e-13,
                        quantity="eager JAX vs NumPy n-step rel "
                                 f"(jit arm {r_jit:.3e})")


# =====================================================================
# gate 3 -- jit vs eager, and the retrace budget
# =====================================================================

# Per-field jit-vs-eager bounds, MEASURED in job 9404093; the wind
# entries PLAUSIBLE-MECHANISM (see scope note) by probe job 9433881
# (scripts/validate/fv3_duo_jit_gap_localiser.py).
#
# THE WIND GAP IS SELECTOR BIT-FLIPS BETWEEN TWO LEGAL COMPILATIONS,
# NOT A LOWERING DEFECT.  The probe's per-cell census: u and v each
# differ at exactly 6 of 2052 cells (all on face 3, rows j=3-4), each
# an O(local-field) jump (~20 abs on fields O(1e5-1e7)); EVERY other
# cell agrees at <= 2.1e-16.  The stage bisection places the injection
# in ``d_sw3``: jitting the d_sw3->d_sw6 tail ALONE on inputs
# BYTE-IDENTICAL to eager reproduces the full 5.646e-07 at the same 6
# cells, and the first differing array is d_sw3's PPM output
# ``ubbtemp`` -- a 6.29e-05 jump at THREE B-grid cells (face 3, j=1,
# i=2/6/10) while every operand feeding it agrees to <= 3e-14.  A
# nine-decade LOCAL amplification of an ulp-scale operand difference is
# only reachable through a DISCONTINUOUS branch; d_sw3's hord-6 PPM
# flux is exactly that (the smt5/smt6 limiter flags add or drop a whole
# flux term -- fv3_tp_core's own non-smoothness inventory, and the same
# anatomy as the fv3_nh_core edge_profile flips, here between two legal
# XLA contractions of one program rather than vs the spec).  The old
# ⛔ note's "3.4e9 ULP is not rounding" arithmetic assumed a SMOOTH
# path; the path is not smooth, so the magnitude is expected.
#
# GROWTH (same probe): jit-vs-eager over 1/2/4/8 full steps measures
# u 7.511e-07 / 8.285e-06 / 6.992e-06 / 5.021e-06 -- a ONE-TIME
# injection that spreads to more cells by advection (6 -> 150 above
# 1e-8) but SATURATES and decays in magnitude after step 2.  It does
# not compound; identical table for the n_split cadence.  The tier-3
# replay growth failures of job 9404093 are this, not a defect.
#
# A magnitude bound cannot distinguish 6 flipped cells from a
# whole-field lowering defect, so the winds ALSO carry the flip-cell
# census gate ``_FLIP_CELL_BOUND`` below: a systematic defect moves
# hundreds of cells and goes red there even inside the magnitude bound.
#
# The u-to-v SPREAD (job 9404093 had v <= 1e-12, u-only): the only
# functional change to the executed chain between the two jobs is
# 932e5b1cd (d_sw1's panel-edge selects -> ``_sel_div``), which
# re-lowered the program, and a legal re-compilation moving WHICH cells
# sit within an ulp of a switching surface is precisely this
# mechanism's behaviour.  PLAUSIBLE (not re-run at the old SHA); u's
# value staying bit-identical across it while v moved is consistent.
_JIT_EAGER_BOUND = {
    # MEASURED (job 9425294 sweep): delp 1.810e-14, pt 4.108e-15; bounds
    # = measured x 10.  [class: jit-vs-eager, rounding-scale]
    "delp": 1.9e-13, "pt": 4.2e-14,
    # MEASURED (job 9425294 sweep): u 7.511e-07, v 7.456e-07; bounds =
    # measured x 10.  [class: jit-vs-eager, selector bit-flips at 6 of
    # 2052 cells -- PLAUSIBLE-MECHANISM (see scope note), probe job 9433881; see block above]
    "u": 7.6e-06, "v": 7.5e-06,
}

# MEASURED (job 9425294 sweep): the localiser's per-key jit-vs-eager
# gaps -- u/v 5.646e-07, delp 1.215e-14, pt 4.108e-15; bounds =
# measured x 10.  The wind gap is ALREADY in the stage chain (before
# the D-grid tail): probe job 9433881 shows the SAME 6 cells as the
# full step, injected by d_sw3's PPM selector flips -- see the
# _JIT_EAGER_BOUND block.  [class: jit-vs-eager localiser,
# selector bit-flips PLAUSIBLE-MECHANISM (see scope note)]
_ACOUSTIC_JIT_BOUND = {
    "delp": 1.3e-13, "pt": 4.2e-14, "u": 5.7e-06, "v": 5.7e-06,
}

# MEASURED (probe job 9433881): 6 of 2052 wind cells above 1e-10 under
# _cmp's global metric, for BOTH u and v, in BOTH the full step and the
# stage chain; bound = measured x 10.  This is the census half of the
# wind gates: selector bit-flips touch a handful of cells, a systematic
# lowering defect touches hundreds.
# SCOPE NOTE (codex BLOCKER + GLM MAJOR, 2026-08-18, both reviews of
# the first version of these comments): the selector-flip mechanism is
# PLAUSIBLE, NOT observed -- the probe compares stage outputs, never the
# smt5/smt6 predicate itself, and "large discontinuous jump + inventory
# of discontinuities" names a candidate, not a culprit. Upgrading to
# CONFIRMED needs the predicate exported from the kernel under BOTH
# compilations and a forced-flip reproduction (deferred; production
# debug surface). "Saturates/does not compound" is measured ONLY at
# C12 / Williamson-2 / DT=450 / CPU / 8 steps -- the growth table is
# equally consistent with steady per-step re-injection diluted by
# advection, and NO claim is made for other resolutions, states, decks,
# backends, or horizons. The census below is a SPARSITY REGRESSION
# GUARD for this fixture, not a benign-vs-defect classifier: a
# systematic regression touching 7-60 cells, or the same 6 cells at
# <10x magnitude, passes it (the magnitude bound gates the latter only
# beyond its own x10 headroom).
_FLIP_CELL_BOUND = 60
_FLIP_CELL_THRESH = 1e-10


def _flip_cell_count(got, ref) -> int:
    """Cells above ``_FLIP_CELL_THRESH`` under ``_cmp``'s GLOBAL metric.

    Same scale (max|ref| over finite non-sentinel cells) as ``_cmp`` so
    the count composes with the magnitude bounds it accompanies.
    """
    a = np.asarray(got, dtype=np.float64)
    b = np.asarray(ref, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b) \
        & (np.abs(a) < _SENTINEL_FLOOR) & (np.abs(b) < _SENTINEL_FLOOR)
    if not ok.any():
        return 0
    scale = max(float(np.abs(b[ok]).max()), 1e-30)
    rel = np.where(ok, np.abs(a - b), 0.0) / scale
    return int((rel > _FLIP_CELL_THRESH).sum())


def _gate_flip_cells(got, ref, name: str) -> None:
    """The census gate for a wind field, measure-mode aware like _cmp."""
    n_flip = _flip_cell_count(got, ref)
    if os.environ.get("LEGOESM_FV3_TOL_MEASURE") == "1":
        print(f"TOLMEASURE {name!r}: flip_cells {n_flip} "
              f"(bound {_FLIP_CELL_BOUND})", flush=True)
        return
    assert n_flip <= _FLIP_CELL_BOUND, (
        f"{name}: {n_flip} cells above {_FLIP_CELL_THRESH:.0e} "
        f"(bound {_FLIP_CELL_BOUND}) -- selector bit-flips touch a "
        f"handful of cells (measured 6, probe job 9433881); hundreds+ "
        f"means a SYSTEMATIC jit-vs-eager difference, which the "
        f"magnitude bound alone cannot see")


def test_full_step_jit_equals_eager(jctx, jstates0, jstep):
    """gate 2 -- magnitude bound + flip-cell census, see the block above.

    NOT bitwise: the step is one long chain of ``x*y + z`` and the
    jitted lowering contracts those into FMAs, so a few-ULP gap is
    expected everywhere -- and where those ulps cross the PPM limiter's
    switching surface (d_sw3, hord 6), a finite O(local-field) jump at
    a FEW cells is expected too (probe job 9433881: 6 of 2052 cells,
    the rest at <= 2.1e-16).  The census gate is what separates that
    characterised pattern from a systematic lowering defect.
    """
    eager = jstep_mod.full_acoustic_step_sixface(jctx, jstates0, DT,
                                                 d_ext=D_EXT_OFF)
    got = jstep(jctx, jstates0, DT, d_ext=D_EXT_OFF)
    for k in _STATE_KEYS:
        _cmp(got[k], eager[k], f"full_step jit.{k}",
             _JIT_EAGER_BOUND[k])
        if k in ("u", "v"):
            _gate_flip_cells(got[k], eager[k],
                             f"full_step jit.{k} flip_cells")


def test_jit_gap_localiser_stage_chain(jctx, jstates0):
    """LOCALISER 1 of 3 -- is the `u` gap in the stage chain or the tail?

    ``acoustic_step_sixface`` is everything BEFORE the D-grid tail.  If
    its `u` is rounding-scale here while the full step's is 7.5e-07, the
    gap lives in the tail (geopk_d -> divg2 -> one_grad_p) and the next
    two localisers name the kernel.  If it already shows here, the tail
    is exonerated and the SW chain is the subject.

    ANSWERED (probe job 9433881, fv3_duo_jit_gap_localiser.py): it
    shows HERE, at the SAME 6 cells as the full step -- the tail is
    exonerated (its two kernel localisers below measured bitwise /
    1e-16 independently).  Within the chain the injection is d_sw3's
    PPM output ``ubbtemp`` (3 B-grid cells, face 3, j=1, i=2/6/10;
    operands agree to <= 3e-14): limiter selector bit-flips between two
    legal compilations.  See the _JIT_EAGER_BOUND block for the full
    record; this test now carries the same flip-cell census gate.
    """
    eager = jstep_mod.acoustic_step_sixface(jctx, jstates0, DT)
    fn = jstep_mod.make_acoustic_step_sixface_jit()
    got = fn(jctx, jstates0, DT)
    for k in ("delp", "pt", "u", "v"):
        # MEASURED (job 9425294 sweep): u and v 5.646e-07, delp 1.215e-14, pt 4.108e-15
        # -- the wind jit-vs-eager gap ALREADY shows in the stage chain
        # (selector bit-flips in d_sw3, probe job 9433881); per-key
        # bounds = measured x 10 via _ACOUSTIC_JIT_BOUND.
        _cmp(got[k], eager[k], f"acoustic_step jit.{k}",
             _ACOUSTIC_JIT_BOUND[k])
        if k in ("u", "v"):
            _gate_flip_cells(got[k], eager[k],
                             f"acoustic_step jit.{k} flip_cells")


def test_jit_gap_localiser_geopk_d(ctx, jctx, states0):
    """LOCALISER 2 of 3 -- the D-grid ``geopk``.

    Runs on face 0 only: the gap is not face-specific (all four fields
    are scored across all six faces above and only `u` moves), and one
    face is enough to name a kernel.
    """
    bd = ctx["bd"]
    delp = jnp.asarray(states0[0]["delp"])
    pt = jnp.asarray(states0[0]["pt"])
    hs = jnp.zeros_like(delp)
    pk_e, gz_e = jstep_mod.geopk_sw_1lev_d(delp, hs, bd, pt=pt)
    fn = jstep_mod.make_geopk_sw_1lev_d_jit()
    pk_j, gz_j = fn(delp, hs, bd, pt=pt)
    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise); bound =
    # 1e-15 eps guard (measured exactly 0.0).
    _cmp(pk_j, pk_e, "geopk_sw_1lev_d jit.pk", 1e-15)
    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise); bound =
    # 1e-15 eps guard (measured exactly 0.0).
    _cmp(gz_j, gz_e, "geopk_sw_1lev_d jit.gz", 1e-15)


def test_jit_gap_localiser_one_grad_p(ctx, jctx, states0):
    """LOCALISER 3 of 3 -- ``one_grad_p``, the prime suspect.

    It is the only routine in the step that writes `u` last, and at
    km=1 SW its pressure bracket is a difference of two nearly equal
    products (``gz`` and ``pk`` both reduce to ``delp`` there), i.e. a
    cancellation whose conditioning could in principle amplify a
    rounding-level input difference.  "Could in principle" is why this
    is a MEASUREMENT and not the explanation: the amplification factor
    needed is ~1e9 and nothing here has shown one.
    """
    bd = ctx["bd"]
    delp = jnp.asarray(states0[0]["delp"])
    pt = jnp.asarray(states0[0]["pt"])
    pk, gz = jstep_mod.geopk_sw_1lev_d(delp, jnp.zeros_like(delp), bd,
                                       pt=pt)
    divg2 = jnp.zeros((NPX, NPX), dtype=jnp.float64)
    u0 = jnp.asarray(states0[0]["u"])
    v0 = jnp.asarray(states0[0]["v"])
    u_e, v_e = jstep_mod.one_grad_p_1lev(u0, v0, pk, gz, divg2,
                                         jctx.gs6[0], bd, NPX, NPX,
                                         dt=DT, d_ext=D_EXT_OFF)
    fn = jstep_mod.make_one_grad_p_1lev_jit()
    u_j, v_j = fn(u0, v0, pk, gz, divg2, jctx.gs6[0], bd, NPX, NPX,
                  dt=DT, d_ext=D_EXT_OFF)
    # MEASURED (job 9425294 sweep): 1.240e-16; bound = measured x 10 =
    # 1.3e-15.
    _cmp(u_j, u_e, "one_grad_p_1lev jit.u", 1.3e-15)
    # MEASURED (job 9425294 sweep): 2.539e-16; bound = measured x 10 =
    # 2.6e-15.
    _cmp(v_j, v_e, "one_grad_p_1lev jit.v", 2.6e-15)


def test_full_step_no_retrace_on_dt(jctx, jstates0):
    """gate 2 mechanics -- ``dt`` is proved DYNAMIC.

    ``fv3_pgrad``'s own jit factories pin the time step STATIC, which
    would recompile the whole step on every new ``dt``; the stepper
    calls the raw kernels for exactly this reason (module deviation D3),
    and this is the assertion that says so.
    """
    box, wrapped = _counted(jstep_mod.full_acoustic_step_sixface)
    fn = jstep_mod.make_full_acoustic_step_sixface_jit(wrapped)
    fn(jctx, jstates0, DT, d_ext=D_EXT_OFF)
    fn(jctx, jstates0, 2.0 * DT, d_ext=D_EXT_OFF)
    fn(jctx, jstates0, 0.5 * DT, d_ext=D_EXT_OFF)
    assert box["n"] == 1, f"retraced on a new dt: {box['n']} traces"


def test_full_step_retraces_on_a_new_static_argument(jctx, jstates0):
    """NON-VACUITY for the counter above.

    A trace counter that can never move would make the no-retrace test
    unfalsifiable.  ``d_ext`` and ``entry_ascalar`` are STATIC by
    design -- ``one_grad_p`` branches on ``d_ext`` in Python and
    ``entry_ascalar`` selects whether two exchanges exist at all -- so
    each new value MUST produce a new trace.
    """
    box, wrapped = _counted(jstep_mod.full_acoustic_step_sixface)
    fn = jstep_mod.make_full_acoustic_step_sixface_jit(wrapped)
    fn(jctx, jstates0, DT, d_ext=D_EXT_OFF)
    fn(jctx, jstates0, DT, d_ext=D_EXT_ON)
    assert box["n"] == 2, (
        f"d_ext did not retrace ({box['n']} traces) -- it is static by "
        f"design; a shared trace would mean one of the two runs used "
        f"the other's branch")
    fn(jctx, jstates0, DT, d_ext=D_EXT_OFF, entry_ascalar=False)
    assert box["n"] == 3, (
        f"entry_ascalar did not retrace ({box['n']} traces)")


def test_d_ext_is_static_when_passed_positionally(jctx, jstates0):
    """``d_ext`` is listed in BOTH ``static_argnums`` and
    ``static_argnames`` so it is static either way a caller passes it.
    Without the argnums entry a positional call would hand
    ``one_grad_p`` a TRACER and its ``if d_ext > 0.0`` would raise."""
    fn = jstep_mod.make_full_acoustic_step_sixface_jit()
    out = fn(jctx, jstates0, DT, D_EXT_ON)      # positional d_ext
    assert np.isfinite(np.asarray(out["delp"])).all()


# =====================================================================
# gate 4 -- gradients
# =====================================================================

def _objective_windows(out):
    """The compute windows of the four prognostics.

    Restricted to what the step WRITES: the halo strips carry cells no
    stage touches, and including them would make the inner products
    insensitive to most of the operator (or, where the lane keeps a
    NaN tripwire, poison them outright).  ``one_grad_p`` writes ``u``
    over Fortran i=is..ie, j=js..je+1 and ``v`` over i=is..ie+1,
    j=js..je, which is the cell x node / node x cell pairing below.
    """
    cs = slice(NG, NG + N)
    bs = slice(NG, NG + N + 1)
    return {"delp": out["delp"][:, cs, cs], "pt": out["pt"][:, cs, cs],
            "u": out["u"][:, cs, bs], "v": out["v"][:, bs, cs]}


def test_full_step_adjoint_identity_scalars(jctx, jstates0, jstep):
    """gate 4, PRIMARY -- operand group (delp, pt).

    Tolerance-free: ``J v`` from ``jax.jvp``, ``J^T w`` from
    ``jax.vjp``, so the identity is exact in exact arithmetic and the
    residual is pure roundoff.  Neither an FD step nor the array's
    dynamic range can make it fail spuriously, which is what sank
    ``check_grads(order=2)`` on this port's first run.
    """
    u0, v0 = jstates0["u"], jstates0["v"]

    def f(delp, pt):
        return _objective_windows(jstep(
            jctx, {"delp": delp, "pt": pt, "u": u0, "v": v0}, DT,
            d_ext=D_EXT_OFF))

    # MEASURED (job 9425294 sweep): adjoint residual 1.333e-16; bound = measured x 10 =
    # 1.4e-15.
    _check_adjoint("full_step d(delp,pt)", f,
                   (jstates0["delp"], jstates0["pt"]), 1.4e-15)


def test_full_step_adjoint_identity_winds(jctx, jstates0, jstep):
    """gate 4, PRIMARY -- operand group (u, v).

    Split from the scalars deliberately: a Jacobian block that is
    identically zero (a wind the step never propagates into ``delp``,
    say) would still satisfy the identity when summed with a healthy
    block, so the groups are certified separately and each carries its
    own ``<J v, w> != 0`` non-vacuity assert.
    """
    delp0, pt0 = jstates0["delp"], jstates0["pt"]

    def f(u, v):
        return _objective_windows(jstep(
            jctx, {"delp": delp0, "pt": pt0, "u": u, "v": v}, DT,
            d_ext=D_EXT_OFF))

    # MEASURED (job 9425294 sweep): adjoint residual 5.724e-16; bound = measured x 10 =
    # 5.8e-15.
    _check_adjoint("full_step d(u,v)", f,
                   (jstates0["u"], jstates0["v"]), 5.8e-15)


@pytest.mark.parametrize("order", [1, 2])
def test_p_grad_c_1lev_check_grads(ctx, jctx, csw_np, order):
    """gate 4, SUPPLEMENT -- scoped to this module's OWN new leaves.

    ``order=1`` and ``order=2`` are separate parametrised IDs so an
    FD-resolution failure at order 2 can never be confused with a wrong
    Jacobian (STATE lesson 12).  The scope is the ``*_1lev`` adapters
    rather than the whole step for two reasons: they are the only new
    arithmetic-carrying code here, and they contain NO limiter, so an
    FD ball around this state is inside one branch -- which is the
    precondition ``check_grads`` needs and the full step cannot offer.
    """
    bd = ctx["bd"]
    hs = np.zeros_like(csw_np[0]["delpc"])
    pk, gz = jstep_mod.geopk_sw_1lev(jnp.asarray(csw_np[0]["delpc"]),
                                     jnp.asarray(hs), bd,
                                     pt=jnp.asarray(csw_np[0]["ptc"]))
    gs, uc0 = jctx.gs6[0], jnp.asarray(csw_np[0]["uc"])
    vc0 = jnp.asarray(csw_np[0]["vc"])
    delpc = jnp.asarray(csw_np[0]["delpc"])
    cs = slice(NG, NG + N)
    one = jnp.asarray(1.0)

    # DIFFERENTIATED VARIABLES ARE O(1) SCALE FACTORS, not the fields.
    # A finite-difference step on `pk` itself (~3e4 here) would be a
    # relative perturbation of ~1e-9 at any usable absolute eps, leaving
    # only a handful of significant digits in the difference -- an FD
    # failure that says nothing about the Jacobian.  Scaling instead
    # makes the FD well conditioned while still exercising the same
    # directional derivative.
    def f(a, b):
        uc, vc = jstep_mod.p_grad_c_1lev(0.5 * DT, delpc, a * pk, b * gz,
                                         uc0, vc0, gs, bd)
        return jnp.sum(uc[cs, cs] ** 2) + jnp.sum(vc[cs, cs] ** 2)

    # MEASURED (job 9425294 sweep): smallest-passing check_grads atol=rtol 1e-3
    # (order=2; order=1 passed at 1e-8); bound = one decade up = 1e-2.
    gated_check_grads(f"p_grad_c_1lev check_grads order={order}", f,
                      (one, one), order=order, modes=("fwd", "rev"),
                      atol=1e-2, rtol=1e-2, eps=1e-4)


@pytest.mark.parametrize("order", [1, 2])
def test_one_grad_p_1lev_check_grads(ctx, jctx, states0, order):
    """gate 4, SUPPLEMENT -- the second of the module's own leaves.

    Run with ``d_ext`` ON so the external-mode increment is inside the
    differentiated region; at the deck's 0.0 those terms are structural
    zeros and the gate would certify a smaller operator than the one
    the research configuration runs.
    """
    bd = ctx["bd"]
    delp = jnp.asarray(states0[0]["delp"])
    pt = jnp.asarray(states0[0]["pt"])
    pk, gz = jstep_mod.geopk_sw_1lev_d(delp, jnp.zeros_like(delp), bd,
                                       pt=pt)
    gs = jctx.gs6[0]
    u0 = jnp.asarray(states0[0]["u"])
    v0 = jnp.asarray(states0[0]["v"])
    divg2 = jnp.full((NPX, NPX), 3.0e-4, dtype=jnp.float64)
    cs = slice(NG, NG + N)
    bs = slice(NG, NG + N + 1)
    one = jnp.asarray(1.0)

    # O(1) scale factors, for the FD-conditioning reason stated on the
    # p_grad_c gate above.
    def f(a, b, c):
        u, v = jstep_mod.one_grad_p_1lev(u0, v0, a * pk, b * gz,
                                         c * divg2, gs, bd, NPX, NPX,
                                         dt=DT, d_ext=D_EXT_ON)
        return jnp.sum(u[cs, bs] ** 2) + jnp.sum(v[bs, cs] ** 2)

    # MEASURED (job 9425294 sweep): smallest-passing check_grads atol=rtol 1e-7
    # (order=2; order=1 passed at 1e-11); bound = one decade up = 1e-6.
    gated_check_grads(f"one_grad_p_1lev check_grads order={order}", f,
                      (one, one, one), order=order, modes=("fwd", "rev"),
                      atol=1e-6, rtol=1e-6, eps=1e-4)


# =====================================================================
# gate 5 -- the two behavioural pins
# =====================================================================

def test_returned_d_wind_halo_is_stale_by_one(jctx, jstates0):
    """THE pin.  ``dyn_core.F90:1332-1338`` refreshes ONLY ``delp`` and
    ``pt`` after the step; the D-wind ``ext_vector`` runs at the NEXT
    step's entry (``:471``).  So the returned D-wind halos still carry
    what THIS step's entry exchange left there, and a twin that
    refreshes them here is a DIVERGENCE that looks more correct.

    Three assertions, because none alone is decisive:

    1. the outermost ``u`` row (numpy 0 == Fortran ``i = 1-ng``) is
       BITWISE the value THIS step's entry ``ext_vector`` wrote.
       Bitwise is legitimate here and only here: no stage writes that
       row -- both ``d_sw6`` (sw_core.F90:1934-1944, window
       ``is..ie``) and ``one_grad_p`` (same window) stop at the compute
       box -- so the cell reaches the output by pure carry-forward with
       no floating-point sum for XLA to contract into an FMA;
    2. the COMPUTE window did change, so a refresh computed from the
       post-step winds could NOT have reproduced the entry value by
       coincidence.  Without this, assertion 1 would also pass on a
       step that did nothing at all;
    3. NON-VACUITY: applying ``ext_vector`` to the RETURNED state DOES
       change that row.  If someone adds the "helpful" post-step
       refresh, this is what goes red.
    """
    tab = jctx.tab
    u_entry, v_entry = jhalo.ext_vector_dgrid_sixface(
        jstates0["u"], jstates0["v"], tab)
    out = jstep_mod.full_acoustic_step_sixface(jctx, jstates0, DT,
                                               d_ext=D_EXT_OFF)

    assert np.array_equal(np.asarray(out["u"])[:, 0, :],
                          np.asarray(u_entry)[:, 0, :]), (
        "the outermost u halo row is NOT the entry-exchange value -- "
        "either a post-step vector refresh was added (a divergence from "
        "dyn_core.F90:1332-1338) or a stage grew its write window")
    assert np.array_equal(np.asarray(out["v"])[:, :, 0],
                          np.asarray(v_entry)[:, :, 0]), (
        "the outermost v halo column is NOT the entry-exchange value")

    moved = _max_window_diff(out, {k: jstates0[k] for k in _STATE_KEYS})
    assert moved > 0.0, (
        "the step left the compute window unchanged, so assertion 1 "
        "above is satisfied trivially -- a post-step refresh would have "
        "reproduced the entry value")

    u_ref, v_ref = jhalo.ext_vector_dgrid_sixface(out["u"], out["v"], tab)
    du = float(np.max(np.abs(np.asarray(u_ref)[:, 0, :]
                             - np.asarray(out["u"])[:, 0, :])))
    dv = float(np.max(np.abs(np.asarray(v_ref)[:, :, 0]
                             - np.asarray(out["v"])[:, :, 0])))
    assert du > 0.0 and dv > 0.0, (
        f"refreshing the returned winds changed nothing (du={du:.3e}, "
        f"dv={dv:.3e}) -- the stale-by-one assertion above is then "
        f"vacuous, because a refreshed lane would pass it too")


def test_delp_and_pt_are_refreshed_after_the_step(jctx, jstates0):
    """The OTHER half of ``dyn_core.F90:1332-1338``: the A-scalars ARE
    refreshed, and only in the halo.

    Without this, "we do not refresh the winds" could be satisfied by a
    lane that refreshes nothing at all.  Compared against the stage
    output (``acoustic_step_sixface``), which is the state the tail
    refresh receives, so the two assertions are exactly what
    ``ext_scalar`` is contracted to do: the compute box carried through
    BITWISE (it writes halo slots only -- a pure index/gather path with
    no sum), and the halo CHANGED.

    ⛔ RUN ON A NON-CONSTANT ``pt``, and that is not tidiness.  Job
    9404093 failed this gate on `pt` with the halo difference EXACTLY
    0.0, and the cause was the FIXTURE, not the code: ``w2_six_face_state``
    sets ``pt = ones_like(delp)`` (fv3_native_duo_stepper.py:1091), so
    ``pt`` is identically 1 over the whole array and an exchange of a
    constant field cannot change a cell by construction.  `delp` passed
    the same assertion in the same run, which is what proves the tail
    refresh runs.  A diagnostic that can only return zero is not a
    gate -- so the fixture is perturbed into a smooth non-constant
    ``pt`` and the fixture's own non-constancy is asserted first.
    """
    dp = jstates0["delp"]
    pt_var = jstates0["pt"] * (1.0 + 1.0e-3 * dp / jnp.max(jnp.abs(dp)))
    assert float(jnp.max(pt_var) - jnp.min(pt_var)) > 0.0, (
        "the perturbed pt is still constant -- this gate would be "
        "vacuous again")
    st = {**jstates0, "pt": pt_var}

    stage = jstep_mod.acoustic_step_sixface(jctx, st, DT)
    out = jstep_mod.full_acoustic_step_sixface(jctx, st, DT,
                                               d_ext=D_EXT_OFF)
    cs = slice(NG, NG + N)
    for k in ("delp", "pt"):
        assert np.array_equal(np.asarray(out[k])[:, cs, cs],
                              np.asarray(stage[k])[:, cs, cs]), (
            f"the tail ext_scalar changed a COMPUTE cell of {k} -- it is "
            f"contracted to write halo slots only")
        halo = float(np.max(np.abs(np.asarray(out[k])[:, 0, :]
                                   - np.asarray(stage[k])[:, 0, :])))
        assert halo > 0.0, (
            f"the outermost {k} halo row is unchanged by the tail "
            f"refresh -- dyn_core.F90:1336-1337 did not run")


def test_w2_pt_is_identically_one(states0):
    """Pins the fact that made the gate above vacuous, so the next
    reader does not re-derive it from a failure.

    ``w2_six_face_state`` sets ``pt = ones_like(delp)``: on this IC any
    ``pt``-only diagnostic that looks for a CHANGE is measuring nothing.
    """
    for t in range(6):
        assert np.array_equal(np.asarray(states0[t]["pt"]),
                              np.ones_like(np.asarray(states0[t]["pt"])))


def test_entry_ascalar_gate_is_live(jctx, jstates0):
    """``dyn_core.F90:432`` ``if ( it==1 )`` -- the ENTRY A-scalar
    exchange.

    Observable only on a state whose ``delp`` halo DISAGREES with its
    neighbours: on a consistent state the exchange is a no-op and the
    gate would be untestable, so the fixture perturbs one halo strip and
    leaves every compute cell alone.

    * ``entry_ascalar=True`` must ERASE the perturbation (the exchange
      overwrites the halo from the neighbour face), so the result must
      match the unperturbed run;
    * ``entry_ascalar=False`` must NOT, so the result must differ.
    """
    base = jstep_mod.full_acoustic_step_sixface(jctx, jstates0, DT,
                                                d_ext=D_EXT_OFF)
    pert = _perturb_halo(jstates0)
    on = jstep_mod.full_acoustic_step_sixface(jctx, pert, DT,
                                              d_ext=D_EXT_OFF,
                                              entry_ascalar=True)
    off = jstep_mod.full_acoustic_step_sixface(jctx, pert, DT,
                                               d_ext=D_EXT_OFF,
                                               entry_ascalar=False)
    for k in _STATE_KEYS:
        # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), every field; bound =
        # 1e-15 eps guard (measured exactly 0.0).
        _cmp(on[k], base[k], f"entry_ascalar erased the halo perturbation"
                             f" ({k})", 1e-15)
    diff = _max_window_diff(off, base)
    assert diff > 0.0, (
        "entry_ascalar=False produced the same state as the exchanged "
        "run -- the halo perturbation never reached the answer, so the "
        "erasure assertion above is vacuous")


def test_step_fn_seam_is_used_and_carries_the_it1_cadence(jctx,
                                                          jstates0):
    """The ``step_fn`` seam must be HONOURED and must receive the
    ``it == 1`` cadence.

    This is the cheapest possible test of ``dyn_core.F90:432``: a
    recording fake in place of the step, asserting the exact
    ``entry_ascalar`` sequence and the exact ``dt``.  It costs no model
    evaluation and it fails on a cadence bug that a state comparison
    could only detect indirectly.  The seam exists so the runners get a
    jitted step WITHOUT re-implementing that cadence.
    """
    seen = []

    def fake(ctx_, states_, dt_, *, d_ext, sw_cfg, entry_ascalar):
        seen.append((float(dt_), entry_ascalar, d_ext))
        return states_

    out = jstep_mod.advance_duo_outer_step(
        jctx, jstates0, 7.0 * DT, 7, d_ext=D_EXT_OFF, step_fn=fake)
    assert out is jstates0, "step_fn was bypassed -- the seam is dead"
    assert [s[1] for s in seen] == [True] + [False] * 6, (
        f"entry_ascalar sequence {[s[1] for s in seen]} != True then six "
        f"False -- dyn_core.F90:432 gates the entry A-scalar exchange on "
        f"`it == 1`")
    assert all(abs(s[0] - DT) < 1e-12 for s in seen), (
        f"dt_atmos was not split n_split ways: {[s[0] for s in seen]}")
    assert all(s[2] == D_EXT_OFF for s in seen)

    flat = []

    def fake2(ctx_, states_, dt_, *, d_ext, sw_cfg):
        flat.append(float(dt_))
        return states_

    jstep_mod.run_duo_sw(jctx, jstates0, DT, 3, d_ext=D_EXT_OFF,
                         step_fn=fake2)
    assert len(flat) == 3, (
        "run_duo_sw did not use step_fn three times")


def test_advance_duo_outer_step_fires_entry_only_on_the_first_substep(
        jctx, jstates0):
    """``advance_duo_outer_step`` must compose as
    ``entry=True`` then ``entry=False`` (``dyn_core.F90:432``,
    ``if ( it==1 )``), and NOT as two ``entry=True`` steps.

    Run from a HALO-PERTURBED state, because that is the only state on
    which the two schedules differ: on a consistent state the tail
    refresh (``:1336-1337``) makes the second entry exchange idempotent
    and every schedule agrees.
    """
    pert = _perturb_halo(jstates0)
    composed = jstep_mod.advance_duo_outer_step(jctx, pert, 2.0 * DT, 2,
                                                d_ext=D_EXT_OFF)
    manual = jstep_mod.full_acoustic_step_sixface(
        jctx, pert, DT, d_ext=D_EXT_OFF, entry_ascalar=True)
    manual = jstep_mod.full_acoustic_step_sixface(
        jctx, manual, DT, d_ext=D_EXT_OFF, entry_ascalar=False)
    for k in _STATE_KEYS:
        # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), every field; bound =
        # 1e-15 eps guard (measured exactly 0.0).
        _cmp(composed[k], manual[k], f"outer step composition ({k})",
             1e-15)

    wrong = jstep_mod.full_acoustic_step_sixface(
        jctx, pert, DT, d_ext=D_EXT_OFF, entry_ascalar=False)
    wrong = jstep_mod.full_acoustic_step_sixface(
        jctx, wrong, DT, d_ext=D_EXT_OFF, entry_ascalar=False)
    diff = _max_window_diff(composed, wrong)
    assert diff > 0.0, (
        "the outer step matched a schedule that NEVER runs the entry "
        "exchange -- `it == 0` is not selecting entry_ascalar=True on "
        "the first substep")
