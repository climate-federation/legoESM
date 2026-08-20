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

* **THE LANE REFUSALS ARE PART OF THE PORT.**  Moist coupling
  (``zvir != 0``) IS enabled on BOTH arms and gated below; on the NH
  arm ``pkz`` is RECOMPUTED here with the ``(1+dp1)`` factor inside its
  log (``fv_dynamics.F90:299-322``) rather than taken from the caller.
  What stays refused is NEGATIVE ``consv_te`` (a different Fortran
  branch entirely) and moist x consv (no oracle deck), not the fixer
  exist here).  Per C5 those raise rather
  than run something adjacent, and the raise is gated -- an unported
  arm that silently proceeds is the defect class this campaign ranks
  worst, because it returns numbers.  ``sphum_index`` is VALIDATED, not
  defaulted: a guessed index couples the wrong species and still
  returns numbers.

TOLERANCE POLICY.  Every numeric bound in this file is MEASURED (job
9425079, the LEGOESM_FV3_TOL_MEASURE sweep) and set to measured x 10,
EXCEPT any bound still carrying a ``TOL-PENDING`` marker -- those are
provisional and are not certification limits.
The worst composed-chain figure is w at 3.126e-11 relative (NH
k_split=2); pressure diagnostics sit at 1e-16..5e-15.  The contract and
movement gates carry no tolerance and are final.
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

from legoesm.grids.fv3_native_gridstruct import (  # noqa: E402
    FV3_GRAV as _FV3_GRAV,
    FV3_RDGAS as _FV3_RDGAS,
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
# The deck's triple, read from the oracle-parity runner -- NOT one
# value for all three. kord_tm is NEGATIVE: a positive kord_tm selects a
# different operator (remaps theta_v in linear p, needs pkez and the te
# array, converts pt at fv_mapz.F90:495-501 instead of :209-217), and
# the lane refuses it rather than running the wrong arm.
KORD_MT, KORD_TM, KORD_TR = 9, -9, 9

# A temperature field is ~250-320 K; theta_v on this deck is several
# hundred K higher, so "did the round trip close" is answerable by
# RANGE alone -- which is the point of the contract gate below.
T_LO, T_HI = 150.0, 400.0


@pytest.fixture(autouse=True)
def _drop_compiled_graphs():
    """Free each test's compiled executables before the next one.

    A full fv_dynamics step unrolls six faces of vertical remap per
    k_split iteration, and XLA keeps every compiled executable alive in
    the process cache. Run individually all four parity cases pass (126
    s, 127 s, 313 s, 522 s); run in one process the fourth ABORTS inside
    backend_compile_and_load -- and it aborts identically at 32G and at
    180G, so it is the retained graphs, not the working set. Measured,
    after I had already claimed memory was the cause and was wrong.
    """
    yield
    jax.clear_caches()


@pytest.fixture(scope="module")
def ctx():
    c = build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                   oracle_conventions=True)
    # The NH carry derives zs = phis/grav (dyn_core.F90:262-278) and the
    # spec refuses hydrostatic=False without it. Flat orography, matching
    # the oracle-parity runner's own NH setup -- and the tail gates rely
    # on that flatness elsewhere (it is what makes ws identically zero).
    c["hs6"] = [np.zeros((MA, MA), dtype=np.float64) for _ in range(6)]
    return c


@pytest.fixture(scope="module")
def jctx(ctx):
    return build_jax_duo_stepper_context(ctx)


@pytest.fixture(scope="module")
def eta():
    ak, bk, ptop, _ks = set_eta_analytic(KM)
    return ak, bk, float(ptop)


def _state(hydrostatic, seed=21):
    rng = np.random.default_rng(seed)
    # remap_follows=True is a PROMISE, and it is true here: km=5 clears
    # fv_dynamics.F90:568's `npz > 4`, so fv_dynamics_step runs
    # Lagrangian_to_Eulerian and pt comes back in K. Without it the
    # builder refuses km > 4 -- correctly, since an unremapped state at
    # this km leaves delp a deformed Lagrangian thickness.
    st = build_state_3d(N, NG, KM, hydrostatic=hydrostatic,
                        remap_follows=True)
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


def _tracers(seed=22, scale=1.0):
    """``scale`` exists for the MOIST arm only.

    The default tracers sit at O(1), which is fine while they are
    passengers; once ``zvir != 0`` they enter ``pt*(1+zvir*q)`` and an
    O(1) q would double the temperature.  The moist gates pass
    ``scale=0.01`` so dp1 lands at a physical few per mil.  Both lanes
    call this with the same seed and scale, so q is identical by
    construction rather than by copying.
    """
    rng = np.random.default_rng(seed)
    return [[scale * (1.0 + 0.3 * t + 0.7 * iq
                      + 0.2 * rng.standard_normal((MA, MA, KM)))
             for iq in range(NQ)] for t in range(6)]


def _press_np(state, ptop):
    return [npdyn.p_var_hydrostatic(state[t]["delp"], ptop=ptop, akap=AKAP,
                                    n=N, ng=NG, km=KM) for t in range(6)]


def _press_jax(jstate, ptop):
    return jdyn.p_var_hydrostatic(jstate["delp"], ptop=ptop, akap=AKAP,
                                  n=N, ng=NG, km=KM)


def _common(ptop, ak, bk, hydrostatic, k_split, n_split):
    return dict(bdt=BDT, km=KM, k_split=k_split, n_split=n_split, ptop=ptop,
                ak=ak, bk=bk, akap=AKAP, cp_air=CP_AIR, kord_mt=KORD_MT,
                kord_tm=KORD_TM, kord_tr=KORD_TR, hydrostatic=hydrostatic,
                # The resolved NH deck runs W_LIMITER=T (fv_mapz.F90:368)
                # and the lane refuses hydrostatic=False without an
                # explicit choice rather than defaulting one.
                w_limiter=not hydrostatic)


def _moist(zvir, sphum_index):
    return {} if zvir == 0.0 else dict(zvir=zvir, sphum_index=sphum_index)


def _run_np(ctx, eta, *, hydrostatic, k_split=1, n_split=2, zvir=0.0,
            sphum_index=None, q_scale=1.0):
    ak, bk, ptop = eta
    st = _state(hydrostatic)
    q = _tracers(scale=q_scale)
    press = _press_np(st, ptop)
    npdyn.fv_dynamics_step(ctx, st, press,
                           q=q, **_common(ptop, ak, bk, hydrostatic,
                                          k_split, n_split),
                           **_moist(zvir, sphum_index))
    return st, press, q


def _run_jax(jctx, eta, *, hydrostatic, k_split=1, n_split=2, zvir=0.0,
             sphum_index=None, q_scale=1.0):
    ak, bk, ptop = eta
    jst = state_3d_to_jax(_state(hydrostatic))
    # Tracer-major, matching this module's contract (nq entries, each
    # face-stacked) -- NOT module 5's single (6, nq, ...) stack.
    _t = _tracers(scale=q_scale)
    q = [jnp.asarray(np.stack([_t[t][iq] for t in range(6)]))
         for iq in range(NQ)]
    press = _press_jax(jst, ptop)
    return jdyn.fv_dynamics_step(jctx, jst, press, q=q,
                                 **_common(ptop, ak, bk, hydrostatic,
                                           k_split, n_split),
                                 **_moist(zvir, sphum_index))


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
        # MEASURED (job 9425079, LEGOESM_FV3_TOL_MEASURE sweep): worst field w at
        # 3.126e-11 (NH k_split=2, the deepest chain: w rides
        # riem_solver3 twice plus the remap); next worst pt 2.247e-13,
        # winds < 8e-14. Bound = worst measured x 10.
        cmp_fields(np.asarray(state[nm]), want,
                   f"{nm} (hydro={hydrostatic}, k_split={k_split})",
                   tol=3.2e-10)

    # Tracer-major on BOTH sides: the module returns nq face-stacked
    # arrays, so building the reference face-major would compare
    # transposed axes and report a difference that is pure layout.
    want_q = np.stack([np.stack([ref_q[t][iq] for t in range(6)])
                       for iq in range(NQ)])
    assert_real(want_q, "numpy q")
    # MEASURED (job 9425079, LEGOESM_FV3_TOL_MEASURE sweep): worst 3.448e-14; bound =
    # measured x 10.
    cmp_fields(np.asarray(got["q"]), want_q,
               f"q (hydro={hydrostatic}, k_split={k_split})", tol=3.5e-13)


ZVIR = 0.6077338443  # rvgas/rdgas - 1, the pinned deck's constant

Q_SCALE = 0.01  # physical specific humidity; see _tracers


@pytest.mark.parametrize("hydrostatic", [True, False])
@pytest.mark.parametrize("k_split", [1, 2])
def test_moist_parity_against_the_spec(ctx, jctx, eta, hydrostatic,
                                       k_split):
    """The zvir arm, port against spec, over the same composition axis.

    ``zvir != 0`` makes the tracers stop being passengers twice over:
    dp1 = zvir*q(sphum) enters ``pt = pt*(1.+dp1)/pkz``
    (fv_dynamics.F90:291/:402) at entry, and ``r_vir`` enters the
    closing ``pt/(1+r_vir*q)`` (fv_mapz.F90:975) at exit.  Both hops are
    on this path; a lane that wired only one would still return numbers.
    """
    ref_state, _ref_press, ref_q = _run_np(
        ctx, eta, hydrostatic=hydrostatic, k_split=k_split, zvir=ZVIR,
        sphum_index=0, q_scale=Q_SCALE)
    got = _run_jax(jctx, eta, hydrostatic=hydrostatic, k_split=k_split,
                   zvir=ZVIR, sphum_index=0, q_scale=Q_SCALE)
    state = _out_state(got)
    tag = f"hydro={hydrostatic}, k_split={k_split}"
    names = ["delp", "pt", "u", "v"] + ([] if hydrostatic
                                        else ["w", "delz"])
    for nm in names:
        want = np.stack([np.asarray(ref_state[t][nm]) for t in range(6)])
        assert_real(want, f"numpy moist {nm} ({tag})")
        # TOL-PENDING(moist-parity)
        cmp_fields(np.asarray(state[nm]), want, f"moist {nm} ({tag})",
                   tol=3.2e-10)
    want_q = np.stack([np.stack([ref_q[t][iq] for t in range(6)])
                       for iq in range(NQ)])
    assert_real(want_q, "numpy moist q")
    # TOL-PENDING(moist-parity-q)
    cmp_fields(np.asarray(got["q"]), want_q, f"moist q ({tag})",
               tol=3.5e-13)


def _run_jax_press(jctx, eta, press, *, hydrostatic, zvir, sphum_index,
                   k_split=1):
    """One moist step on a CALLER-SUPPLIED pressure bundle."""
    ak, bk, ptop = eta
    jst = state_3d_to_jax(_state(hydrostatic))
    _t = _tracers(scale=Q_SCALE)
    q = [jnp.asarray(np.stack([_t[t][iq] for t in range(6)]))
         for iq in range(NQ)]
    return _out_state(jdyn.fv_dynamics_step(
        jctx, jst, press, q=q,
        **_common(ptop, ak, bk, hydrostatic, k_split, 2),
        **_moist(zvir, sphum_index)))


@pytest.mark.parametrize("k_split", [1, 2])
def test_nh_moist_pkz_is_recomputed_not_trusted(jctx, eta, k_split):
    """fv_dynamics.F90:299-322 OVERWRITES pkz on every NH call.

    A caller's pkz is dry -- ``p_var_nonhydrostatic`` builds it dry, and
    a previous step's remap returns it dry (fv_mapz.F90:479-481 carries
    no ``(1+dp1)``).  A lane that trusted it would run the whole moist
    NH step against a pkz missing the virtual factor and still return
    numbers.  So: hand the SAME moist NH step two DIFFERENT pkz values
    and require the answer not to move.

    The hydrostatic arm is the control.  It does NOT recompute -- the
    oracle's hydrostatic branch (:281-294) only forms dp1 -- so there
    the same substitution MUST move the answer.  Without that half this
    gate could pass on a lane that ignored pkz everywhere.

    WHAT THIS TEST DOES NOT PIN: that the recompute is MOIST.  A lane
    that recomputed with ``dp1=None`` -- the dry form -- stays green
    here, because the answer still stops depending on the caller's pkz.
    Moistness is pinned by
    :func:`test_nh_moist_pkz_matches_the_oracle_expression`'s
    anti-vacuity assert.  NOT by the parity gate: that one is
    wet-port-vs-wet-spec and cannot see a SYMMETRIC drop in both lanes,
    and unlike the hydrostatic arm the NH arm has no oracle deck to
    catch it either (GLM MINOR, job 9442724).  ``k_split`` is
    parametrized because a consumer of the caller's pkz on a
    non-last-step path would escape a single-iteration run.
    """
    if k_split != 1:
        # MEASURED, not assumed (jobs 9443826 and 9443895): this case
        # dies in XLA with "LLVM compilation error: Cannot allocate
        # memory" after a 3m27s `jit_scan` compile inside sim1_solver.
        # Both discriminators were run and both REFUTED the easy
        # explanations -- it fails ALONE at 600G, and it fails alone in
        # a FRESH process at 400G, so it is neither the job's memory
        # limit nor code memory accumulated across the module's other
        # graphs. It is a compile-scale blowup in eagerly-executed
        # lax.scan, in a path none of the moist work touches.
        #
        # THE MODEL CONFIGURATION IS NOT IN DOUBT: moist NH at
        # k_split=2 runs and is asserted by
        # test_moist_parity_against_the_spec[False-2], which passes in
        # the same suite. What is lost is specifically GLM's k_split>1
        # question for THIS gate -- whether a consumer of the caller's
        # pkz sits on a non-last-step path -- and parity cannot answer
        # it, because both lanes would consume it alike. OPEN, and
        # recorded in the STATE doc rather than dropped.
        pytest.skip("XLA cannot compile this case; see the comment -- "
                    "resource and accumulation both refuted (9443826, "
                    "9443895)")
    ak, bk, ptop = eta
    for hydrostatic in (False, True):
        jst = state_3d_to_jax(_state(hydrostatic))
        press_a = _press_jax(jst, ptop)
        # The second bundle is the first with pkz PERTURBED, not an NH
        # p_var: build_state_3d allocates delz only on the NH arm, so
        # the hydrostatic control cannot ask for an NH pressure bundle
        # (it raised KeyError: 'delz' in job 9442478). A scaled pkz is
        # the cleaner substitution anyway -- the claim under test is
        # only "a DIFFERENT pkz", and this way both arms are perturbed
        # identically.
        press_b = dict(press_a)
        press_b["pkz"] = press_a["pkz"] * 1.001
        # anti-vacuity: the two bundles must actually differ in pkz, or
        # "the answer did not move" is trivially true.
        assert (np.asarray(press_a["pkz"]).tobytes()
                != np.asarray(press_b["pkz"]).tobytes())
        a = _run_jax_press(jctx, eta, press_a, hydrostatic=hydrostatic,
                           zvir=ZVIR, sphum_index=0, k_split=k_split)
        b = _run_jax_press(jctx, eta, press_b, hydrostatic=hydrostatic,
                           zvir=ZVIR, sphum_index=0, k_split=k_split)
        # EVERY returned field, not just pt: a consumer of the caller's
        # pkz sitting in the w or delz path would escape a pt-only
        # comparison and be caught only later, indirectly, at tolerance
        # level (GLM MINOR, job 9442483).
        names = ["delp", "pt", "u", "v"] + ([] if hydrostatic
                                            else ["w", "delz"])
        same = all(np.asarray(a[nm]).tobytes()
                   == np.asarray(b[nm]).tobytes() for nm in names)
        if hydrostatic:
            assert not same, ("the hydrostatic arm must CONSUME the "
                              "caller's pkz; if it does not, the NH half "
                              "of this gate proves nothing")
        else:
            assert same, ("the NH moist arm must RECOMPUTE pkz "
                          "(fv_dynamics.F90:299-322), not consume the "
                          "caller's dry one")


def test_nh_moist_pkz_matches_the_oracle_expression(eta):
    """The association is gated BELOW the transcendentals.

    ``pkz = exp(kappa*log(arg))``, and XLA's ``exp`` differs from
    libm's by ~1 ulp, so a cross-lane bitwise check on ``pkz`` is not
    available.  The previous version of this test therefore compared
    ``pkz`` under a 10x-of-the-dry-residual window -- and codex showed
    that window ADMITS the exact mistake the docstring claimed to
    reject (job 9442717): pre-scaling ``pt`` at the call site,
    ``rdg*delp*(pt*(1+dp1))/delz``, is algebraically identical and
    differs only by multiplication rounding, comfortably inside any
    exp/log-sized tolerance.  The comment claiming a misassociation
    "would blow up" was false.

    ``nh_pkz_log_arg`` exists so this gate can be exact: below the
    transcendentals the trees differ by real bits.
    """
    faces = _state(False)
    jst = state_3d_to_jax(_state(False))
    rdg = -_FV3_RDGAS / _FV3_GRAV
    dp1 = 0.0077 * np.ones((6, N, N, KM))
    cs = slice(NG, NG + N)

    for t in range(6):                       # ALL SIX FACES
        dpw = faces[t]["delp"][cs, cs, :]
        ptw = faces[t]["pt"][cs, cs, :]
        dz = faces[t]["delz"]
        want = npdyn.nh_pkz_log_arg(rdg, dpw, ptw, dz, dp1[t])
        got = np.asarray(jdyn.nh_pkz_log_arg(
            rdg, jnp.asarray(dpw), jnp.asarray(ptw), jnp.asarray(dz),
            jnp.asarray(dp1[t])))
        assert got.tobytes() == want.tobytes(), \
            f"face {t}: port and spec disagree on the log argument"

        # THE MUTATION THIS GATE EXISTS FOR: the call-site formulation.
        call_site = rdg * dpw * (ptw * (1.0 + dp1[t])) / dz
        assert call_site.tobytes() != want.tobytes(), \
            (f"face {t}: the two multiplication trees agree bitwise on "
             f"this fixture, so the gate above cannot see the mistake "
             f"it exists to catch")
        # and the factor OUTSIDE the log, the other named mistake
        outside = rdg * dpw * ptw / dz * (1.0 + dp1[t])
        assert outside.tobytes() != want.tobytes()

        # ANTI-VACUITY: dp1 must change the argument at all.
        assert want.tobytes() != npdyn.nh_pkz_log_arg(
            rdg, dpw, ptw, dz).tobytes()

    # and the composed pkz still agrees across lanes at the exp/log
    # floor -- the association gate above says nothing about the
    # transcendental chain, so both layers are asserted.
    kw = dict(ptop=eta[2], akap=AKAP, n=N, ng=NG, km=KM)
    got_pkz = np.asarray(jdyn.p_var_nonhydrostatic(
        jst["delp"], jst["delz"], jst["pt"],
        dp1=jnp.asarray(dp1), **kw)["pkz"])
    for t in range(6):
        want_pkz = npdyn.p_var_nonhydrostatic(
            faces[t]["delp"], faces[t]["delz"], faces[t]["pt"],
            dp1=dp1[t], **kw)["pkz"]
        # MEASURED: 1 ulp of fp64 on an exp/log chain.
        # TOL-PENDING(nh-moist-pkz)
        assert np.max(np.abs(got_pkz[t] - want_pkz)
                      / np.abs(want_pkz)) <= 1e-14


def test_the_moist_arm_actually_changed_the_answer(ctx, jctx, eta):
    """Anti-vacuity: without this, a zvir that was silently dropped on
    the port side would pass the parity gate above by matching a spec
    lane that had dropped it too.  Both lanes must MOVE, and move by
    the same amount, relative to their own dry run.

    dp1 ~ zvir*q ~ 6e-3 here, so pt moves in the third digit -- far
    above any parity residual, which is why a plain magnitude assert is
    enough and no tolerance is needed.
    """
    dry = _out_state(_run_jax(jctx, eta, hydrostatic=True, q_scale=Q_SCALE))
    wet = _out_state(_run_jax(jctx, eta, hydrostatic=True, zvir=ZVIR,
                              sphum_index=0, q_scale=Q_SCALE))
    resp_port = np.asarray(wet["pt"]) - np.asarray(dry["pt"])
    assert np.max(np.abs(resp_port)) > 1.0e-3, \
        f"zvir moved pt by only {np.max(np.abs(resp_port)):.3e} K"

    dry_np, _, _ = _run_np(ctx, eta, hydrostatic=True, q_scale=Q_SCALE)
    wet_np, _, _ = _run_np(ctx, eta, hydrostatic=True, zvir=ZVIR,
                           sphum_index=0, q_scale=Q_SCALE)
    resp_spec = (np.stack([wet_np[t]["pt"] for t in range(6)])
                 - np.stack([dry_np[t]["pt"] for t in range(6)]))
    # THE WHOLE RESPONSE FIELD, not its maximum. Two different fields
    # can share a max magnitude, so a port that coupled the wrong cell,
    # face, axis or tracer would pass a scalar comparison (codex MAJOR,
    # job 9442422). The response is a DIFFERENCE of two ~300 K fields,
    # so its own scale is ~1 K and the parity tolerance applies to it
    # directly.
    assert_real(resp_spec, "spec moist response")
    # TOL-PENDING(moist-response)
    cmp_fields(resp_port, resp_spec, "moist pt response (wet - dry)",
               tol=3.2e-10)


def test_dry_branch_is_not_a_multiply_by_one(jctx, eta):
    """The dry branch exists for the ORACLE's structure, not for rounding.

    THIS TEST USED TO ASSERT THE OPPOSITE AND WAS WRONG (job 9442478).
    Both lanes' comments said a zeros array "rounds twice" because
    ``(1.0+dp1)/pkz`` forms a reciprocal first -- true of the OLD
    association ``pt *= (1.0+dp1)/pkz``, and FALSE since the association
    was corrected to the oracle's ``(pt*(1+dp1))/pkz``: ``1.0+0.0`` is
    exactly 1.0 and ``win*1.0`` is exact in IEEE, so the zeros array is
    now bit-identical to the dry branch.  (Codex confirmed the
    retraction holds for every finite input including subnormals and
    both signed zeros, job 9442717.)

    What the branch is actually for, and what is asserted here: the
    oracle forms no ``dp1`` AT ALL when ``zvir = 0``
    (fv_dynamics.F90:281-294 is the moist branch), so ``dp1=None`` is
    the faithful shape, and it avoids materialising and multiplying a
    whole zeros field.  Equality is the CONTRACT -- if these two ever
    diverge, the dry lane's certified 1.1866e-09 has silently moved.
    """
    jst = state_3d_to_jax(_state(True))
    pkz = _press_jax(jst, eta[2])["pkz"]
    ref = jdyn.pt_to_theta_v(jst["pt"], pkz, n=N, ng=NG)
    wet0 = jdyn.pt_to_theta_v(jst["pt"], pkz, n=N, ng=NG,
                              dp1=jnp.zeros_like(pkz))
    assert np.asarray(ref).tobytes() == np.asarray(wet0).tobytes(), \
        ("dp1=zeros is no longer bit-identical to dp1=None; the moist "
         "association changed and the dry certification has moved")

    # ANTI-VACUITY: a NON-zero dp1 must move it, or the equality above
    # would be satisfied by a lane that ignored dp1 entirely.
    wet = jdyn.pt_to_theta_v(jst["pt"], pkz, n=N, ng=NG,
                             dp1=jnp.full_like(pkz, 0.0077))
    assert np.asarray(ref).tobytes() != np.asarray(wet).tobytes()


@pytest.mark.skipif(jax.default_backend() != "cpu",
                    reason="a BITWISE association claim is a compiler "
                           "contract, and this lane's is the fp64 CPU "
                           "backend; XLA on GPU/TPU may contract the "
                           "multiply-add or reassociate (codex MAJOR, "
                           "job 9442422)")
def test_moist_association_matches_the_spec_bitwise(eta):
    """``pt*(1.+dp1)/pkz`` (:402) associates left to right.

    The claim under test is NOT "Fortran guarantees two roundings" --
    that is the compiler's business.  It is the one this port is
    actually held to: the JAX lane must agree with the NumPy SPEC,
    which is the authority, and numpy evaluates the same source
    expression without reassociating.  Comparing the port against the
    SPEC rather than against a locally re-typed expression is also what
    makes this a cross-lane check instead of a self-comparison.
    """
    jst = state_3d_to_jax(_state(True))
    pkz = _press_jax(jst, eta[2])["pkz"]
    dp1 = 0.0077 * np.ones(np.asarray(pkz).shape)

    got = np.asarray(jdyn.pt_to_theta_v(jst["pt"], pkz, n=N, ng=NG,
                                        dp1=jnp.asarray(dp1)))
    ref = np.stack([f["pt"] for f in _state(True)])
    for t in range(6):
        npdyn.pt_to_theta_v(ref[t], np.asarray(pkz)[t], n=N, ng=NG,
                            dp1=dp1[t])
    assert got.tobytes() == ref.tobytes(), \
        "port and spec disagree bitwise on the moist conversion"

    # ANTI-VACUITY: the wrong association must actually differ on this
    # data, or the assert above would pass with either spelling.
    win = np.asarray(jst["pt"])[:, NG:NG + N, NG:NG + N, :]
    pkzn = np.asarray(pkz)
    bad = win * ((1.0 + dp1) / pkzn)
    assert bad.tobytes() != ref[:, NG:NG + N, NG:NG + N, :].tobytes()


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
        # MEASURED (job 9425079, LEGOESM_FV3_TOL_MEASURE sweep): worst pkz 4.689e-15 (the
        # kappa power is the only transcendental); ps/pe/peln < 2e-16.
        # Bound = worst measured x 10.
        cmp_fields(np.asarray(press[nm]), want, f"{nm} (hydro={hydrostatic})",
                   tol=4.7e-14)


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
    _t = _tracers()
    q_in = np.stack([np.stack([_t[t][iq] for t in range(6)])
                     for iq in range(NQ)])
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
    # Bound DERIVED from the measured delp parity (6.034e-14 rel,
    # job 9425079): the two lanes' summed-mass drifts can differ by at
    # most n_cells x max|delp diff|, far inside 1e-9 x scale. Never
    # itself failed; kept as drift-parity (codex: not a conservation
    # claim), the name says so.
    assert abs(d_j - d_np) <= 1e-9 * scale, (
        f"mass change disagrees: numpy {d_np:.6e}, jax {d_j:.6e} "
        f"(initial {m0:.6e})")


# --------------------------------------------------------------------
# 3.  Lane refusals, jit parity, differentiability
# --------------------------------------------------------------------

@pytest.mark.parametrize("kw,exc", [
    # zvir is ENABLED on BOTH arms and so is the energy fixer; what
    # stays refused is NEGATIVE consv_te and moist x consv,
    # and what stays VALIDATED is the sphum index and the NH arm's
    # need for delz (the fixture here is a HYDROSTATIC state, which
    # carries none, so hydrostatic=False must be caught rather than
    # indexed into).
    # match= is REQUIRED here: deleting the delz guard would otherwise
    # leave this green via the NH hs6/w_limiter ValueErrors, and the
    # "caught rather than indexed into" claim would silently change
    # referent (GLM MINOR, job 9442483).
    ({"zvir": 0.61, "sphum_index": 0, "hydrostatic": False},
     (ValueError, "delz")),
    ({"zvir": 0.61}, ValueError),                       # sphum_index None
    ({"zvir": 0.61, "sphum_index": -1}, ValueError),    # wrap-around
    ({"zvir": 0.61, "sphum_index": NQ}, ValueError),    # out of range
    ({"zvir": 0.61, "sphum_index": True}, ValueError),  # bool is not int
    # positive consv_te RUNS now; these are the arms that do not.
    ({"consv_te": -1.0}, NotImplementedError),   # prescribed-flux branch
    ({"consv_te": 1.0, "zvir": 0.61, "sphum_index": 0},
     NotImplementedError),                       # moist x consv, unscored
    ({"consv_te": 1.0, "hydrostatic": False}, NotImplementedError),
    ({"k_split": 0}, ValueError),
    ({"n_split": 0}, ValueError),
    ({"n_sponge": 2}, (ValueError, NotImplementedError)),
    ({"tau": 5.0}, (ValueError, NotImplementedError)),
])
def test_unported_arms_raise_instead_of_running_something_adjacent(
        jctx, eta, kw, exc):
    """Each of these is either unported or invalid input.

    a NEGATIVE ``consv_te`` selects a prescribed-flux branch that is
    not ported, and moist x consv has no oracle deck;
    the sponge arguments select non-uniform damping; and a bad
    ``sphum_index`` would couple an arbitrary tracer into theta_v.
    Running the default instead would return numbers, which is the
    worst available outcome.  (Hydrostatic ``zvir != 0`` is NOT here --
    it is supported, and its gates are above.)
    """
    ak, bk, ptop = eta
    jst = state_3d_to_jax(_state(True))
    # Tracer-major, matching this module's contract (nq entries, each
    # face-stacked) -- NOT module 5's single (6, nq, ...) stack.
    _t = _tracers()
    q = [jnp.asarray(np.stack([_t[t][iq] for t in range(6)]))
         for iq in range(NQ)]
    press = _press_jax(jst, ptop)
    args = _common(ptop, ak, bk, True, 1, 2)
    args.update(kw)
    if isinstance(exc, tuple) and len(exc) == 2 and isinstance(exc[1], str):
        exc, match = exc
    else:
        match = None
    with pytest.raises(exc, match=match):
        jdyn.fv_dynamics_step(jctx, jst, press, q=q, **args)


def test_jit_matches_eager(jctx, eta):
    """Divergence means a Python branch on a traced value."""
    ak, bk, ptop = eta
    jst = state_3d_to_jax(_state(True))
    # Tracer-major, matching this module's contract (nq entries, each
    # face-stacked) -- NOT module 5's single (6, nq, ...) stack.
    _t = _tracers()
    q = [jnp.asarray(np.stack([_t[t][iq] for t in range(6)]))
         for iq in range(NQ)]
    press = _press_jax(jst, ptop)
    kw = _common(ptop, ak, bk, True, 1, 2)

    eager = _out_state(jdyn.fv_dynamics_step(jctx, jst, press, q=q, **kw))
    # The MODULE's factory, not a hand-rolled jax.jit: it is what a
    # caller would use, and it is where the static/dynamic split is
    # declared. Jitting fv_dynamics_step directly cannot work -- the
    # return carries pt_units, a str -- so a hand-rolled jit here would
    # have tested a path nobody runs.
    fn = jdyn.make_fv_dynamics_step_jit(
        jctx, KM, k_split=1, n_split=2, ptop=ptop, ak=ak, bk=bk,
        akap=AKAP, cp_air=CP_AIR, kord_mt=KORD_MT,
        kord_tm=KORD_TM, kord_tr=KORD_TR, hydrostatic=True)
    jitted = _out_state(fn(jst, press, q, BDT))
    for nm in ("delp", "pt", "u", "v"):
        a = np.asarray(eager[nm])
        assert_real(a, f"eager {nm}")
        # MEASURED (job 9425079, LEGOESM_FV3_TOL_MEASURE sweep): worst u 4.217e-14; bound =
        # measured x 10.
        cmp_fields(np.asarray(jitted[nm]), a, f"jit vs eager {nm}",
                   tol=4.3e-13)


def test_gradient_through_a_whole_step_is_finite(jctx, eta):
    """End-to-end differentiability is the reason this lane exists.

    Every kernel has its own gradient gate; what this one adds is that
    the COMPOSITION does not introduce a NaN -- the usual sources being
    a 0*inf in a limiter and a sqrt at zero.
    """
    ak, bk, ptop = eta
    jst = state_3d_to_jax(_state(True))
    # Tracer-major, matching this module's contract (nq entries, each
    # face-stacked) -- NOT module 5's single (6, nq, ...) stack.
    _t = _tracers()
    q = [jnp.asarray(np.stack([_t[t][iq] for t in range(6)]))
         for iq in range(NQ)]
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
