"""Certification of the JAX ``fv_dynamics`` driver (module 6 of 6).

Authority: ``legoesm.core.fv3_native_dynamics`` is the SPECIFICATION
(hop B of ``docs/atmosphere/fv3_duo_jax_lane_strategy.md``).

This is the top of the port -- the routine an integration actually
calls -- so its gates are about COMPOSITION, not arithmetic.  Every
kernel below it is certified in its own gate file; what is new here is:

* **THE OUTER k_split LOOP AND THE pt ROUND TRIP.**  ``pt`` enters as
  TEMPERATURE, is converted to ``theta_v`` at :396-408, lives that way
  through the acoustic loop, and is converted back inside the remap
  (fv_mapz.F90:209-217).  The two halves are in different modules, so a
  lane that drops either one returns a plausible field in the wrong
  variable -- roughly a factor of ``1/pkz`` off, finite and smooth.
  Gated by an explicit CONTRACT assertion on the returned range, not
  only by the parity comparison, because a parity comparison against a
  spec that had the same bug would pass.

* **THE TRACERS RIDE THE STEP.**  ``tracer_2d_1L`` runs between the
  acoustic loop and the remap on ``dp1`` = delp as it was BEFORE
  dyn_core, and the remap then carries q with it.  A driver that
  advected nothing still returns a perfectly good dynamical state, so
  ``q`` is asserted to have MOVED as well as to match.

* **THE LANE REFUSALS ARE PART OF THE PORT.**  ``zvir != 0`` and
  ``consv_te != 0`` reach unported code (the tracers stop being
  passengers; the energy fixer does not exist here).  Per C5 those raise
  rather than run something adjacent, and the raise is gated -- an
  unported arm that silently proceeds is the defect class this campaign
  ranks worst, because it returns numbers.

TOLERANCE POLICY.  A full step composes every limiter in the model, so
per strategy section 4 the numeric bounds carry ``TOL-PENDING`` until
the measurement job replaces them with ``measured X, bound = measured x
N``.  The contract and movement gates carry no tolerance and are final.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from legoesm.core import fv3_dynamics as jdyn  # noqa: E402
from legoesm.core import fv3_native_dynamics as npdyn  # noqa: E402
from legoesm.core.fv3_cgrid_phase_3d import (  # noqa: E402
    state_3d_to_jax,
)
from legoesm.core.fv3_duo_stepper import (  # noqa: E402
    build_jax_duo_stepper_context,
)
from legoesm.core.fv3_native_duo_stepper import (  # noqa: E402
    build_six_face_duo_context,
)
from legoesm.core.fv3_native_eta import set_eta_analytic  # noqa: E402
from legoesm.core.fv3_native_state_3d import (  # noqa: E402
    build_state_3d,
)

from tests.grids.fv3_gate_helpers import (  # noqa: E402
    assert_real,
    cmp_fields,
    deepcopy_faces,
)

# N and NG match the other 3-D gate files; KM does NOT, and cannot.
# set_eta_analytic implements fv_eta.F90's `case (5,10)` and RAISES for
# anything else -- km < 5 has no table in the oracle at all. The first
# version of this file used KM = 3 for consistency with its siblings and
# every test in it errored at fixture construction, which is how the
# module's return-contract defect survived to the review. 5 is the
# smallest legal value, and it is also > REMAP_MIN_NPZ = 4, so the remap
# actually runs -- at km <= 4 the oracle skips it and pt would come back
# as theta_v, making the round-trip gate below vacuous.
N, NG, KM = 12, 3, 5
MA = N + 2 * NG
NQ = 2
BDT = 60.0
AKAP, CP_AIR = 2.0 / 7.0, 1004.6
KORD = 9

# A temperature field is ~250-320 K; theta_v on this deck is several
# hundred K higher, so "did the round trip close" is answerable by
# RANGE alone -- which is the point of the contract gate below.
T_LO, T_HI = 150.0, 400.0


@pytest.fixture(scope="module")
def ctx():
    return build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                      oracle_conventions=True)


@pytest.fixture(scope="module")
def jctx(ctx):
    return build_jax_duo_stepper_context(ctx)


@pytest.fixture(scope="module")
def eta():
    ak, bk, ptop, _ks = set_eta_analytic(KM)
    return ak, bk, float(ptop)


def _state(hydrostatic, seed=21):
    rng = np.random.default_rng(seed)
    st = build_state_3d(N, NG, KM, hydrostatic=hydrostatic)
    for t, face in enumerate(st):
        for k in range(KM):
            face["delp"][:, :, k] = 1.0e4 * (1.0 + 0.05 * t + 0.02 * k) \
                + 50.0 * rng.standard_normal((MA, MA))
            face["pt"][:, :, k] = 280.0 + 2.0 * t + 3.0 * k \
                + 0.5 * rng.standard_normal((MA, MA))
            face["u"][:, :, k] = 5.0 + 0.3 * t + 0.1 * k \
                + rng.standard_normal((MA, MA + 1))
            face["v"][:, :, k] = -4.0 + 0.2 * t - 0.1 * k \
                + rng.standard_normal((MA + 1, MA))
            if not hydrostatic:
                face["w"][:, :, k] = 0.2 * rng.standard_normal((MA, MA))
                face["delz"][:, :, k] = -(500.0 + 20.0 * k)
    return st


def _tracers(seed=22):
    rng = np.random.default_rng(seed)
    return [[1.0 + 0.3 * t + 0.7 * iq
             + 0.2 * rng.standard_normal((MA, MA, KM))
             for iq in range(NQ)] for t in range(6)]


def _press_np(state, ptop):
    return [npdyn.p_var_hydrostatic(state[t]["delp"], ptop=ptop, akap=AKAP,
                                    n=N, ng=NG, km=KM) for t in range(6)]


def _press_jax(jstate, ptop):
    return jdyn.p_var_hydrostatic(jstate["delp"], ptop=ptop, akap=AKAP,
                                  n=N, ng=NG, km=KM)


def _common(ptop, ak, bk, hydrostatic, k_split, n_split):
    return dict(bdt=BDT, km=KM, k_split=k_split, n_split=n_split, ptop=ptop,
                ak=ak, bk=bk, akap=AKAP, cp_air=CP_AIR, kord_mt=KORD,
                kord_tm=KORD, kord_tr=KORD, hydrostatic=hydrostatic)


def _run_np(ctx, eta, *, hydrostatic, k_split=1, n_split=2):
    ak, bk, ptop = eta
    st = _state(hydrostatic)
    q = _tracers()
    press = _press_np(st, ptop)
    npdyn.fv_dynamics_step(ctx, st, press,
                           q=q, **_common(ptop, ak, bk, hydrostatic,
                                          k_split, n_split))
    return st, press, q


def _run_jax(jctx, eta, *, hydrostatic, k_split=1, n_split=2):
    ak, bk, ptop = eta
    jst = state_3d_to_jax(_state(hydrostatic))
    q = jnp.asarray(np.stack([np.stack(f) for f in _tracers()]))
    press = _press_jax(jst, ptop)
    return jdyn.fv_dynamics_step(jctx, jst, press, q=q,
                                 **_common(ptop, ak, bk, hydrostatic,
                                           k_split, n_split))


def _out_state(got):
    return got["state"] if "state" in got else got


# --------------------------------------------------------------------
# 1.  Parity, over the composition axes that change the program
# --------------------------------------------------------------------

@pytest.mark.parametrize("hydrostatic", [True, False])
@pytest.mark.parametrize("k_split", [1, 2])
def test_full_step_parity_against_the_spec(ctx, jctx, eta, hydrostatic,
                                           k_split):
    """``k_split`` = 1 and 2 separate "one remap" from "a remap loop".

    With ``k_split = 2`` the acoustic loop runs twice against a dp1 that
    is re-snapshotted per outer iteration; a lane that hoisted that
    snapshot out of the loop is correct at 1 and wrong at 2.
    """
    ref_state, ref_press, ref_q = _run_np(
        ctx, eta, hydrostatic=hydrostatic, k_split=k_split)
    got = _run_jax(jctx, eta, hydrostatic=hydrostatic, k_split=k_split)
    state = _out_state(got)

    names = ["delp", "pt", "u", "v"] + ([] if hydrostatic else ["w", "delz"])
    for nm in names:
        want = np.stack([np.asarray(ref_state[t][nm]) for t in range(6)])
        # `w` is identically zero on the hydrostatic arm, which is why it
        # is excluded there rather than compared: assert_real refuses a
        # comparison that cannot fail.
        assert_real(want, f"numpy {nm} (hydro={hydrostatic}, k={k_split})")
        # TOL-PENDING (a full step composes every limiter in the model).
        cmp_fields(np.asarray(state[nm]), want,
                   f"{nm} (hydro={hydrostatic}, k_split={k_split})",
                   rtol=1e-10)

    want_q = np.stack([np.stack(ref_q[t]) for t in range(6)])
    assert_real(want_q, "numpy q")
    cmp_fields(np.asarray(got["q"]), want_q,
               f"q (hydro={hydrostatic}, k_split={k_split})", rtol=1e-10)


@pytest.mark.parametrize("hydrostatic", [True, False])
def test_pressure_diagnostics_come_back_matching(ctx, jctx, eta,
                                                 hydrostatic):
    """``press`` is intent(inout) in the Fortran and returned here.

    The remap rebuilds ps/pe/peln/pk/pkz; a lane that advanced the
    prognostics but left the diagnostics stale would pass every gate
    above and desynchronise on the NEXT call.
    """
    _, ref_press, _ = _run_np(ctx, eta, hydrostatic=hydrostatic)
    got = _run_jax(jctx, eta, hydrostatic=hydrostatic)
    press = got["press"]
    for nm in ("ps", "pe", "peln", "pk", "pkz"):
        want = np.stack([np.asarray(ref_press[t][nm]) for t in range(6)])
        assert_real(want, f"numpy {nm}")
        # TOL-PENDING.
        cmp_fields(np.asarray(press[nm]), want, f"{nm} (hydro={hydrostatic})",
                   rtol=1e-11)


# --------------------------------------------------------------------
# 2.  Contract gates -- these would survive a spec with the same bug
# --------------------------------------------------------------------

@pytest.mark.parametrize("hydrostatic", [True, False])
def test_pt_leaves_as_temperature_not_theta_v(jctx, eta, hydrostatic):
    """The round trip closes, checked by RANGE.

    :396-408 converts pt -> theta_v and fv_mapz.F90:209-217 converts it
    back.  Dropping the second half returns theta_v: finite, smooth,
    several hundred K too warm.  A parity test cannot see that if both
    lanes did it; a range assertion can.
    """
    # ANTI-VACUITY FIRST: the gate is only a gate if theta_v on THIS
    # fixture actually lands outside the window. theta_v = pt*(1+dp1)/pkz
    # with dp1 = 0 (zvir = 0 is refused otherwise), so it is computable
    # here -- and asserting the range without checking this would be a
    # test that cannot fail, which is the failure mode this campaign
    # ranks worst in an instrument.
    ak, bk, ptop = eta
    st0 = _state(hydrostatic)
    pkz = np.stack([np.asarray(p["pkz"]) for p in _press_np(st0, ptop)])
    pt0 = np.stack([f["pt"][NG:-NG, NG:-NG, :] for f in st0])
    theta_v = pt0 / pkz
    assert not ((theta_v > T_LO) & (theta_v < T_HI)).all(), (
        f"theta_v on this fixture is inside [{T_LO}, {T_HI}] "
        f"([{theta_v.min():.1f}, {theta_v.max():.1f}] K), so the range "
        "assertion below cannot detect a dropped conversion")

    got = _out_state(_run_jax(jctx, eta, hydrostatic=hydrostatic))
    pt = np.asarray(got["pt"])[:, NG:-NG, NG:-NG, :]
    assert_real(pt, "returned pt")
    lo, hi = float(pt.min()), float(pt.max())
    assert T_LO < lo and hi < T_HI, (
        f"pt is outside the temperature range [{T_LO}, {T_HI}]: "
        f"[{lo:.1f}, {hi:.1f}] K -- the pt <-> theta_v round trip did "
        "not close")


def test_the_tracers_actually_moved(jctx, eta):
    """q must both MATCH the spec and DIFFER from its input.

    A driver that never called tracer_2d_1L returns the input q, which
    matches a spec bug of the same shape and looks entirely healthy.
    """
    q_in = np.stack([np.stack(f) for f in _tracers()])
    got = _run_jax(jctx, eta, hydrostatic=True)
    q_out = np.asarray(got["q"])
    assert_real(q_out, "returned q")
    live = (slice(None), slice(None), slice(NG, -NG), slice(NG, -NG))
    moved = np.abs(q_out[live] - q_in[live]).max()
    assert moved > 0.0, "q is unchanged: the tracers were never advected"


def test_mass_drift_parity_not_conservation(ctx, jctx, eta):
    """A relative statement, deliberately -- NOT a conservation claim.

    Whether this dycore conserves dry mass exactly on the duo seam is a
    question about FV3, not about this port, so the gate compares the
    two lanes' mass CHANGE instead of asserting either is zero -- the
    port is what is under test here.
    """
    live = (slice(NG, -NG), slice(NG, -NG), slice(None))
    st0 = _state(True)
    m0 = sum(float(np.asarray(st0[t]["delp"])[live].sum()) for t in range(6))

    ref_state, _, _ = _run_np(ctx, eta, hydrostatic=True)
    m_np = sum(float(np.asarray(ref_state[t]["delp"])[live].sum())
               for t in range(6))
    got = _out_state(_run_jax(jctx, eta, hydrostatic=True))
    m_j = float(np.asarray(got["delp"])[:, NG:-NG, NG:-NG, :].sum())

    d_np, d_j = m_np - m0, m_j - m0
    scale = max(abs(d_np), 1e-6 * abs(m0))
    # TOL-PENDING: bound on the AGREEMENT of the two drifts.
    assert abs(d_j - d_np) <= 1e-9 * scale, (
        f"mass change disagrees: numpy {d_np:.6e}, jax {d_j:.6e} "
        f"(initial {m0:.6e})")


# --------------------------------------------------------------------
# 3.  Lane refusals, jit parity, differentiability
# --------------------------------------------------------------------

@pytest.mark.parametrize("kw,exc", [
    ({"zvir": 0.61}, NotImplementedError),
    ({"consv_te": 1.0}, NotImplementedError),
    ({"k_split": 0}, ValueError),
    ({"n_split": 0}, ValueError),
    ({"n_sponge": 2}, (ValueError, NotImplementedError)),
    ({"tau": 5.0}, (ValueError, NotImplementedError)),
])
def test_unported_arms_raise_instead_of_running_something_adjacent(
        jctx, eta, kw, exc):
    """Each of these reaches code that is not in this port.

    ``zvir != 0`` makes the tracers stop being passengers; ``consv_te !=
    0`` calls an energy fixer that does not exist here; the sponge
    arguments select non-uniform damping.  Running the default instead
    would return numbers, which is the worst available outcome.
    """
    ak, bk, ptop = eta
    jst = state_3d_to_jax(_state(True))
    q = jnp.asarray(np.stack([np.stack(f) for f in _tracers()]))
    press = _press_jax(jst, ptop)
    args = _common(ptop, ak, bk, True, 1, 2)
    args.update(kw)
    with pytest.raises(exc):
        jdyn.fv_dynamics_step(jctx, jst, press, q=q, **args)


def test_jit_matches_eager(jctx, eta):
    """Divergence means a Python branch on a traced value."""
    ak, bk, ptop = eta
    jst = state_3d_to_jax(_state(True))
    q = jnp.asarray(np.stack([np.stack(f) for f in _tracers()]))
    press = _press_jax(jst, ptop)
    kw = _common(ptop, ak, bk, True, 1, 2)

    eager = _out_state(jdyn.fv_dynamics_step(jctx, jst, press, q=q, **kw))
    fn = jax.jit(lambda s, p, qq: jdyn.fv_dynamics_step(
        jctx, s, p, q=qq, **kw))
    jitted = _out_state(fn(jst, press, q))
    for nm in ("delp", "pt", "u", "v"):
        a = np.asarray(eager[nm])
        assert_real(a, f"eager {nm}")
        # TOL-PENDING: jit reassociates; bound to be measured.
        cmp_fields(np.asarray(jitted[nm]), a, f"jit vs eager {nm}",
                   rtol=1e-12)


def test_gradient_through_a_whole_step_is_finite(jctx, eta):
    """End-to-end differentiability is the reason this lane exists.

    Every kernel has its own gradient gate; what this one adds is that
    the COMPOSITION does not introduce a NaN -- the usual sources being
    a 0*inf in a limiter and a sqrt at zero.
    """
    ak, bk, ptop = eta
    jst = state_3d_to_jax(_state(True))
    q = jnp.asarray(np.stack([np.stack(f) for f in _tracers()]))
    press = _press_jax(jst, ptop)
    kw = _common(ptop, ak, bk, True, 1, 2)

    def loss(pt):
        st = {**jst, "pt": pt}
        out = _out_state(jdyn.fv_dynamics_step(jctx, st, press, q=q, **kw))
        return jnp.sum(out["pt"] ** 2) + jnp.sum(out["u"] ** 2)

    g = np.asarray(jax.grad(loss)(jst["pt"]))
    assert np.isfinite(g).all(), (
        f"{int((~np.isfinite(g)).sum())} non-finite gradient entries")
    assert np.abs(g).max() > 0.0, "gradient is identically zero (vacuous)"
