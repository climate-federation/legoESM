"""Certification of the JAX tracer transport (module 5 of 6).

Authority: ``legoesm.core.fv3_native_tracer2d`` is the SPECIFICATION
(hop B of ``docs/atmosphere/fv3_duo_jax_lane_strategy.md``).  A
difference between the two lanes is a port defect in this module by
definition, never a finding about the oracle.

WHAT MAKES THIS MODULE DIFFERENT FROM THE OTHER FIVE.  Everywhere else
the control flow is fixed by the deck; here the trip count is DERIVED
FROM THE DATA -- ``nsplt = int(1 + cmax)`` per level
(fv_tracer2d.F90:247), where ``cmax`` is a max-reduce over the
Courant-number capacitors.  A ``lax.scan`` needs a static trip count, so
this lane runs a fixed ``NSPLT_MAX``-long scan and masks the inactive
iterations.  That substitution is the single node in the whole port that
is not mechanical, and it fails in three distinct ways:

1. **The mask leaks** -- an "inactive" iteration that is not exactly a
   no-op.  Caught here by a TOLERANCE-INDEPENDENT invariant rather than
   a norm: tracer advection is per-level, so a level's answer may not
   depend on what schedule the OTHER levels resolved.  The same level is
   run inside an all-``nsplt``-1 fixture and inside a mixed one and the
   two must agree EXACTLY (``array_equal``, not a tolerance).
2. **The schedule is wrong on one side of a transition** -- ``int(1 +
   cmax)`` is a step function, and a port that is right at ``cmax =
   0.4`` can be wrong at ``0.6``.  Every gate that claims a schedule
   ASSERTS THE MEASURED ONE (from the returned ``nsplt``) instead of
   predicting it, and the transition gate walks both sides of a
   crossing and requires the resolved count to actually change -- a
   sweep in which it never changes tests one branch twice.
3. **The cap is silently exceeded** -- the failure mode this campaign
   cares about most, because ``NSPLT_MAX`` is a CFL assumption and a
   truncated trip count is a plausible wrong answer, not a crash.  Per
   C5 the lane refuses loudly; the gate drives ``cmax`` past the cap and
   requires the raise.

TOLERANCE POLICY.  The advection kernels are limiter-heavy, so per
strategy section 4 numeric bounds carry ``TOL-PENDING`` until the
measurement job replaces them.  The three structural gates above carry
no tolerance at all and are already final.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from legoesm.core import fv3_native_tracer2d as nptr  # noqa: E402
from legoesm.core import fv3_tracer2d as jtr  # noqa: E402
from legoesm.core.fv3_duo_stepper import (  # noqa: E402
    build_jax_duo_stepper_context,
)
from legoesm.core.fv3_native_duo_stepper import (  # noqa: E402
    build_six_face_duo_context,
)

from tests.grids.fv3_gate_helpers import (  # noqa: E402
    assert_real,
    cmp_fields,
)

# Same geometry as the other 3-D gate files, so all six are comparable.
N, NG, KM = 12, 3, 3
MA = N + 2 * NG
NQ = 2
HORD_TR = 8
DT = 60.0  # signature parity only; the body never reads it.

CAP_KEYS = ("mfx", "mfy", "cx", "cy")


@pytest.fixture(scope="module")
def ctx():
    return build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                      oracle_conventions=True)


@pytest.fixture(scope="module")
def jctx(ctx):
    return build_jax_duo_stepper_context(ctx)


def _tracers(seed=11):
    """Two tracers with structure in every axis (face, tracer, level).

    A constant tracer is advected exactly by any scheme, including a
    broken one, so the field has to vary -- and it varies per LEVEL as
    well, because the whole point of this module is that different
    levels take different numbers of sub-steps.
    """
    rng = np.random.default_rng(seed)
    q_np = [[np.zeros((MA, MA, KM)) for _ in range(NQ)] for _ in range(6)]
    for t in range(6):
        for iq in range(NQ):
            for k in range(KM):
                q_np[t][iq][:, :, k] = (1.0 + 0.3 * t + 0.7 * iq + 0.5 * k
                                        + 0.2 * rng.standard_normal((MA, MA)))
    q_j = jnp.asarray(np.stack([np.stack(q_np[t]) for t in range(6)]))
    return q_np, q_j


def _dp1(seed=12):
    rng = np.random.default_rng(seed)
    dp_np = [1.0e4 + 50.0 * rng.standard_normal((MA, MA, KM))
             for _ in range(6)]
    return dp_np, jnp.asarray(np.stack(dp_np))


def _caps(level_amp, seed=13):
    """Capacitors whose per-level Courant amplitude is ``level_amp[k]``.

    ``cmax`` is a max over |cx|,|cy| (plus a metric term), so scaling
    the level slab scales the resolved ``nsplt`` -- which is how every
    schedule in this file is dialled.  The gates never TRUST this
    mapping: they assert the schedule the lane actually returned.
    """
    rng = np.random.default_rng(seed)
    npx = N + 1
    shp = {"mfx": (npx, MA, KM), "mfy": (MA, npx, KM),
           "cx": (npx, MA, KM), "cy": (MA, npx, KM)}
    cap_np = []
    for _t in range(6):
        d = {}
        for nm, s in shp.items():
            a = rng.standard_normal(s)
            for k in range(KM):
                a[:, :, k] *= level_amp[k]
            d[nm] = a
        cap_np.append(d)
    cap_j = {nm: jnp.asarray(np.stack([cap_np[t][nm] for t in range(6)]))
             for nm in shp}
    return cap_np, cap_j


def _run_np(ctx, level_amp):
    """The spec, on copies -- it mutates q, dp1 and the capacitors."""
    q_np, _ = _tracers()
    dp_np, _ = _dp1()
    cap_np, _ = _caps(level_amp)
    nptr.tracer_2d_1l_sixface(ctx, q_np, dp_np, cap_np, km=KM, nq=NQ,
                              hord_tr=HORD_TR, dt=DT)
    return q_np, dp_np, cap_np


def _run_jax(jctx, level_amp, *, check=True):
    q_j = _tracers()[1]
    dp_j = _dp1()[1]
    cap_j = _caps(level_amp)[1]
    out = jtr.tracer_2d_1l_sixface(jctx, q_j, dp_j, cap_j, km=KM, nq=NQ,
                                   hord_tr=HORD_TR, dt=DT)
    if check:
        jtr.check_nsplt_schedule(out)
    return out


def _schedule(out):
    return np.asarray(out["nsplt"]).astype(int).tolist()


# --------------------------------------------------------------------
# 1.  Parity, at schedules chosen to sit on both sides of a transition
# --------------------------------------------------------------------

# `nsplt = int(1 + cmax)`: the first raises no sub-cycle at all, the
# second sub-cycles every level, the third mixes the two WITHIN one call
# (which is the case a per-level bug survives).  The resolved counts are
# asserted, not assumed.
_AMPS = {"all_single": (0.10, 0.12, 0.14),
         "all_split": (1.40, 1.60, 1.90),
         "mixed": (0.10, 1.40, 2.60)}


@pytest.mark.parametrize("label", sorted(_AMPS))
def test_parity_against_the_spec(ctx, jctx, label):
    amp = _AMPS[label]
    q_ref, dp_ref, cap_ref = _run_np(ctx, amp)
    got = _run_jax(jctx, amp)

    sched = _schedule(got)
    assert len(sched) == KM, sched
    if label == "all_single":
        assert set(sched) == {1}, sched
    elif label == "all_split":
        assert min(sched) > 1, sched
    else:
        assert len(set(sched)) > 1 and min(sched) == 1 and max(sched) > 1, \
            f"the mixed fixture did not mix: {sched}"

    want_q = np.stack([np.stack(q_ref[t]) for t in range(6)])
    assert_real(want_q, f"numpy q ({label})")
    # TOL-PENDING (limiter-heavy advection): provisional bound.
    cmp_fields(np.asarray(got["q"]), want_q, f"q ({label}, nsplt={sched})",
               rtol=1e-11)

    want_dp = np.stack([np.asarray(dp_ref[t]) for t in range(6)])
    assert_real(want_dp, f"numpy dp1 ({label})")
    cmp_fields(np.asarray(got["dp1"]), want_dp,
               f"dp1 ({label}, nsplt={sched})", rtol=1e-11)


@pytest.mark.parametrize("label", sorted(_AMPS))
def test_capacitors_come_back_frac_rescaled(ctx, jctx, label):
    """The spec divides the capacitors by nsplt IN PLACE (:262-291).

    This lane is functional, so the rescale has to be RETURNED; a lane
    that computed it correctly and dropped it would still pass every q
    comparison above, because q is advected before the caller ever looks
    at the capacitors again.
    """
    amp = _AMPS[label]
    _, _, cap_ref = _run_np(ctx, amp)
    got = _run_jax(jctx, amp)
    for nm in CAP_KEYS:
        want = np.stack([cap_ref[t][nm] for t in range(6)])
        assert_real(want, f"numpy {nm} ({label})")
        cmp_fields(np.asarray(got[nm]), want, f"{nm} ({label})", rtol=1e-13)


# --------------------------------------------------------------------
# 2.  The masked scan: tolerance-independent structural gates
# --------------------------------------------------------------------

def test_a_levels_answer_does_not_depend_on_other_levels_schedules(jctx):
    """The mask is exact, or this fails.

    Tracer advection is per-level; nothing in ``tracer_2d_1L`` couples
    them except the shared trip count that this lane replaced with a
    mask.  So level 0, resolved at nsplt = 1, must come out BIT-IDENTICAL
    whether the other two levels resolved to 1 or to 3.  Compared with
    ``array_equal``: a leaking mask shows up as a tiny difference, which
    is exactly what a tolerance would forgive.
    """
    single = _run_jax(jctx, (0.10, 0.12, 0.14))
    mixed = _run_jax(jctx, (0.10, 1.40, 2.60))
    assert _schedule(single)[0] == 1 and _schedule(mixed)[0] == 1
    assert _schedule(mixed)[1:] != [1, 1], _schedule(mixed)

    a = np.asarray(single["q"])[..., 0]
    b = np.asarray(mixed["q"])[..., 0]
    assert_real(a, "level-0 q (all-single fixture)")
    assert np.array_equal(a, b), (
        "level 0 changed when OTHER levels sub-cycled: max|d| "
        f"{np.abs(a - b).max():.6e} -- the inactive-iteration mask is "
        "not a no-op")


def test_both_sides_of_a_schedule_transition(ctx, jctx):
    """Walk a crossing and require the trip count to actually change.

    ``int(1 + cmax)`` is a step function.  A sweep that never crosses a
    step tests one branch twice and reports full coverage, so the gate
    asserts the schedule MOVED before it believes the parity result.
    """
    seen = {}
    for amp in (0.80, 0.92, 1.05, 1.20):
        got = _run_jax(jctx, (amp, amp, amp))
        seen[amp] = _schedule(got)[0]
    assert len(set(seen.values())) > 1, (
        f"no transition crossed, every amplitude gave nsplt={seen}")

    # Parity on the two amplitudes that straddle the crossing.
    lo = min(a for a in seen if seen[a] == min(seen.values()))
    hi = min(a for a in seen if seen[a] > min(seen.values()))
    for amp in (lo, hi):
        q_ref, _, _ = _run_np(ctx, (amp, amp, amp))
        got = _run_jax(jctx, (amp, amp, amp))
        want = np.stack([np.stack(q_ref[t]) for t in range(6)])
        assert_real(want, f"numpy q (amp={amp})")
        # TOL-PENDING (limiter-heavy advection): provisional bound.
        cmp_fields(np.asarray(got["q"]), want,
                   f"q at amp={amp} (nsplt={seen[amp]})", rtol=1e-11)


def test_exceeding_the_cap_raises_instead_of_truncating(jctx):
    """C5: a resolved nsplt above NSPLT_MAX is refused, never clipped.

    A truncated trip count under-advects and returns a finite, plausible
    field -- the failure class this campaign treats as the worst kind.
    """
    out = _run_jax(jctx, (12.0, 12.0, 12.0), check=False)
    assert bool(np.asarray(out["nsplt_exceeded"])), (
        f"cap not flagged; resolved nsplt={_schedule(out)} vs NSPLT_MAX="
        f"{jtr.NSPLT_MAX}")
    with pytest.raises(ValueError, match="NSPLT_MAX"):
        jtr.check_nsplt_schedule(out)


def test_the_cap_check_is_quiet_on_an_in_envelope_schedule(jctx):
    """The other half of the guard: it must not fire on a normal run.

    A check that raises always is not a check.
    """
    out = _run_jax(jctx, (0.10, 1.40, 2.60), check=False)
    assert not bool(np.asarray(out["nsplt_exceeded"])), _schedule(out)
    jtr.check_nsplt_schedule(out)  # must not raise


# --------------------------------------------------------------------
# 3.  Lane refusals, jit parity, differentiability
# --------------------------------------------------------------------

@pytest.mark.parametrize("kw", [{"z_tracer": False}, {"q_split": 2},
                                {"nord_tr": 1}, {"trdm": 1.0},
                                {"inline_q": True}])
def test_out_of_lane_options_are_refused_at_entry(jctx, kw):
    """Every option the ported lane does not implement raises.

    A silently ignored switch is the dispatch-hardening failure mode:
    the caller asks for different physics and gets the default.
    """
    q_j, dp_j, cap_j = _tracers()[1], _dp1()[1], _caps((0.1,) * KM)[1]
    with pytest.raises((ValueError, NotImplementedError)):
        jtr.tracer_2d_1l_sixface(jctx, q_j, dp_j, cap_j, km=KM, nq=NQ,
                                 hord_tr=HORD_TR, dt=DT, **kw)


def test_jit_matches_eager(jctx):
    """Divergence here means a tracer bug (a Python branch on data).

    Given this module's whole difficulty is a data-derived trip count,
    that is the specific thing worth checking.
    """
    amp = _AMPS["mixed"]
    eager = _run_jax(jctx, amp)
    fn = jax.jit(
        lambda q, dp, cap: jtr.tracer_2d_1l_sixface(
            jctx, q, dp, cap, km=KM, nq=NQ, hord_tr=HORD_TR, dt=DT),
    )
    jitted = fn(_tracers()[1], _dp1()[1], _caps(amp)[1])
    assert _schedule(jitted) == _schedule(eager)
    for nm in ("q", "dp1") + CAP_KEYS:
        a, b = np.asarray(eager[nm]), np.asarray(jitted[nm])
        assert_real(a, f"eager {nm}")
        # TOL-PENDING: jit reassociates; bound to be measured.
        cmp_fields(b, a, f"jit vs eager {nm}", rtol=1e-12)


def test_gradient_is_finite_and_carries_no_term_through_the_trip_count(jctx):
    """The schedule is stop_gradient'ed, so d(out)/d(cmax) is piecewise.

    What must hold is that the derivative through the ADVECTION is real
    and finite; what must NOT happen is a NaN from differentiating the
    integer-valued schedule.  Checked on the mixed fixture so at least
    one level is genuinely sub-cycling.
    """
    dp_j, cap_j = _dp1()[1], _caps(_AMPS["mixed"])[1]

    def loss(q):
        out = jtr.tracer_2d_1l_sixface(jctx, q, dp_j, cap_j, km=KM, nq=NQ,
                                       hord_tr=HORD_TR, dt=DT)
        return jnp.sum(out["q"] ** 2)

    g = np.asarray(jax.grad(loss)(_tracers()[1]))
    assert np.isfinite(g).all(), (
        f"{int((~np.isfinite(g)).sum())} non-finite gradient entries")
    assert np.abs(g).max() > 0.0, "gradient is identically zero (vacuous)"
