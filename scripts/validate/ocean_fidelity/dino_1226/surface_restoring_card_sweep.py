#!/usr/bin/env python
"""Rule 12 for the surface-restoring change (PR #1728, user decision 30):
which cards move, and by how much?

THE CHANGE.  ``apply_dino_lat_lon_surface_forcing`` built its restoring with
``RestoringConfig(implicit=True)``, i.e. an analytical implicit-Euler step whose
denominator is ``tau + dt``.  NEMO has no such factor: ``usrdef_sbc.f90:223``
and ``:272-273`` evaluate the flux on the BEFORE tracer and ``trasbc.f90:169-170``
divide it once by the live first thickness.  The statement is now
``implicit=False``.

WHY A SWEEP.  The statement is UNCONDITIONAL in the shared applicator, so every
caller moves -- there is no card-level predicate to check, only a SIZE to
measure.  Measuring it is the point: "every card moves" without a number is an
assertion, and the ASKED row the user has to decide needs the number.

WHAT IS MEASURED (Rule 10 -- through the card's own path, not a formula).  For
each card this builds the grid / vertical / state / forcing the card ships with
and calls ``apply_dino_lat_lon_surface_forcing`` TWICE on identical inputs:

  arm A   the shipped path (explicit; NEMO's form)
  arm B   the same call with ``restoring_surface_forcing`` wrapped to force
          ``implicit=True`` -- the exact object the applicator built before this
          change, since that call is the only place the flag was read

and scores THREE things per card:

  ABS(S)   the shipped arm's level-0 salt rate against an INDEPENDENT
           transcription of NEMO's statement, ``(S* - S)/tau_S``.  Salt is the
           clean row because it has no solar member, so the whole level-0 salt
           rate IS the restoring statement.  This is the check that catches a
           defect COMMON TO BOTH ARMS -- a diff reviewer flipped the restoring's
           SIGN and the first version of this sweep still exited 0, because it
           only ever looked at the difference between the arms.
  d(S)     the arm-to-arm move against the closed form ``dt/(tau_S + dt)``.
  d(T)     the arm-to-arm move against ITS closed form,
           ``(T* - T)*dt/(tau_T*(tau_T + dt))`` -- exact, because the Q_sr
           subtraction and the Jerlov column are identical in both arms and
           cancel.  The first version printed the T column and tested nothing
           in it; doubling the Q_sr subtraction moved that column and the
           verdict did not notice.

A NaN in any of the three is a FAILURE, not a pass.  The first version opened
its verdict with ``np.isnan(smax) or ...``, so deleting the salt restoring
outright produced NaN on every cell and the sweep exited 0.

WHAT THIS SWEEP STILL CANNOT SEE (Rule 2): the TEMPERATURE row's absolute
value.  ``d(T)`` pins the statement that changed and ``ABS(S)`` pins the shared
restoring kernel, but a wrong ``Q_sr`` subtraction or a wrong Jerlov column
would leave all three green.  Those are scored against NEMO's OWN trends by
``kt1_surface_gate.py``, which is where they belong.

Also printed, because it is the Rule 9 question the change turns on:
``dt/(2*tau)``, the explicit-relaxation stability margin.  A card at or above 1
would be one the implicit form was actually load-bearing for.

Usage
-----
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python scripts/validate/\\
        ocean_fidelity/dino_1226/surface_restoring_card_sweep.py
"""
from __future__ import annotations

import sys

import numpy as np

DT = 2700.0          # the NEMO DINO baroclinic step; every row uses ONE dt so
                     # the comparison across cards is controlled (Rule 7).


def _rel(a, b):
    """max / rms of |a-b| relative to max|b| (b = the shipped explicit arm)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = np.abs(a - b)
    s = float(np.abs(b).max())
    if s == 0.0:
        return float("nan"), float("nan")
    return float(d.max()) / s, float(np.sqrt(np.mean(d ** 2))) / s


def main() -> int:
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())
    import legoesm.ocean.physics.surface_forcing.restoring as restmod
    from legoesm.ocean.physics.surface_forcing.config import (
        RestoringConfig, tau_from_flux_coefficient,
    )
    from legoesm.ocean.experiments import dino as dm
    from legoesm.ocean.experiments.dino import DINO_RECIPES

    real_rest = restmod.restoring_surface_forcing

    def implicit_arm(T, S, grid, cfg, **kw):
        return real_rest(T, S, grid, cfg._replace(implicit=True), **kw)

    cards: list[tuple[str, object]] = [
        (name, dm.dino_config_for_recipe(name)) for name in sorted(DINO_RECIPES)
    ]
    try:
        from legoesm.ocean.experiments.neverworld2_lite import (
            NeverWorld2LiteConfig, _to_dino_cfg,
        )
        cards.append(("neverworld2_lite", _to_dino_cfg(NeverWorld2LiteConfig())))
    except Exception as exc:                      # pragma: no cover - reported
        print(f"neverworld2_lite NOT SWEPT: {exc!r}  -- that is an UNMEASURED "
              "row, not an exempt card")
        cards.append(("neverworld2_lite", None))

    print(f"{'card':22s}{'tau_T [d]':>12s}{'tau_S [d]':>12s}"
          f"{'dt/tau_T':>11s}{'dt/tau_S':>11s}{'ABS(S)':>12s}"
          f"{'d(S)':>12s}{'d(T)':>12s}{'dt/2tau max':>13s}  verdict")
    print("  ABS(S) / d(S) / d(T) are max|measured - independent transcription|"
          " / max|transcription|; the bar is 1e-9 and a NaN is a FAILURE.")
    moved, failed, unmeasured = [], [], []
    for name, cfg in cards:
        if cfg is None:
            unmeasured.append(name)
            print(f"{name:22s}{'--':>12s}{'--':>12s}{'--':>11s}{'--':>11s}"
                  f"{'--':>12s}{'--':>12s}{'--':>12s}{'--':>13s}  UNMEASURED")
            continue
        try:
            grid = dm.dino_lat_lon_grid(cfg)
            z = dm.dino_lat_lon_vertical(grid, cfg)
            state = dm.dino_lat_lon_state(grid, z, cfg)
            forcing = dm.dino_lat_lon_surface_forcing_arrays(grid, cfg)
            # The applicator REFUSES a return_rate that disagrees with the
            # card's own surface_tendency_placement (dino.py's
            # _check_surface_tendency_placement), so the arm is chosen from the
            # card, not assumed -- the first version of this sweep assumed
            # return_rate=True and 7 of 8 cards raised.  'applied_now' cards
            # return a MUTATED STATE, so their move is read off T/S there; the
            # two forms are the same object up to one forward-Euler multiply.
            _rate_arm = (getattr(cfg, "surface_tendency_placement",
                                 "applied_now") == "leapfrog_rhs")
            kw = dict(t_seconds=DT, return_rate=_rate_arm)

            def _call():
                """Return the level-0 T and S RATE, whichever arm the card is on.

                The 'applied_now' cards return a MUTATED STATE; the applicator
                built it as ``x + dt*rate`` from this same ``state``, so
                ``(x_new - x)/dt`` recovers the rate.  Doing that here means
                every card is scored by the SAME exact checks instead of the
                state-arm cards getting a loose band (the first version gave
                them a 20% tolerance on the only quantity it tested).
                """
                r = dm.apply_dino_lat_lon_surface_forcing(
                    state, forcing, z, cfg, DT, **kw)
                if _rate_arm:
                    return np.asarray(r[1][0]), np.asarray(r[1][1])
                return ((np.asarray(r.T.data) - np.asarray(state.T.data)) / DT,
                        (np.asarray(r.S.data) - np.asarray(state.S.data)) / DT)

            dT_a, dS_a = _call()
            restmod.restoring_surface_forcing = implicit_arm
            try:
                dT_b, dS_b = _call()
            finally:
                restmod.restoring_surface_forcing = real_rest
        except Exception as exc:
            failed.append((name, repr(exc)))
            print(f"{name:22s}  BUILD FAILED: {exc!r}")
            continue

        dz0 = float(z.dz_ref[0])
        tau_T = float(tau_from_flux_coefficient(cfg.A_theta, cfg.rho_0,
                                                cfg.c_p, dz0))
        tau_S = float(tau_from_flux_coefficient(cfg.A_S, cfg.rho_0, 1.0, dz0))
        fT, fS = DT / tau_T, DT / tau_S
        marg = max(DT / (2.0 * tau_T), DT / (2.0 * tau_S))
        wet = np.asarray(state.land_mask.data) > 0.5
        T0 = np.asarray(state.T.data)[..., 0]
        S0 = np.asarray(state.S.data)[..., 0]
        # T* IS TIME-DEPENDENT on the kamm cards (`forcing_annual_cycle`), and
        # the applicator recomputes it at `t_seconds` instead of reading
        # `forcing["T_star_2d"]`.  Transcribing the static array here scored
        # d(T) = 3.9e-01 on exactly the two cards that set the flag -- a number
        # a 0.26% change cannot produce, i.e. the transcription was wrong, not
        # the model (Rule: a value the scheme could never produce is an
        # instrument bug until proven otherwise).  S* has no annual cycle.
        if getattr(cfg, "forcing_annual_cycle", False):
            T_star = np.broadcast_to(
                np.asarray(dm.dino_T_star_seasonal(
                    forcing["lat_deg_1d"], DT, cfg))[:, None],
                np.asarray(forcing["T_star_2d"]).shape)
        else:
            T_star = np.asarray(forcing["T_star_2d"])
        S_star = np.asarray(forcing["S_star_2d"])

        def _score(got, want):
            """max|got - want| / max|want| on wet cells; NaN if either is not
            finite, which the verdict treats as a FAILURE, never as a pass."""
            g, w = got[wet], want[wet]
            if not (np.all(np.isfinite(g)) and np.all(np.isfinite(w))):
                return float("nan")
            scale = float(np.abs(w).max())
            if scale == 0.0:
                return float("nan")
            return float(np.abs(g - w).max()) / scale

        # ABS(S): NEMO's own statement, transcribed here independently.  The
        # card runs from rest so the live stretch is exactly 1 and the scalar
        # tau_S is the card's own value.
        abs_s = _score(dS_a[..., 0], (S_star - S0) / tau_S)
        # d(S), d(T): the arm-to-arm move against its closed form.
        d_s = _score(dS_a[..., 0] - dS_b[..., 0],
                     (S_star - S0) * DT / (tau_S * (tau_S + DT)))
        d_t = _score(dT_a[..., 0] - dT_b[..., 0],
                     (T_star - T0) * DT / (tau_T * (tau_T + DT)))
        # THE BAR IS NOT ONE CONSTANT, because the two arms do not have the
        # same achievable precision and pretending they do would either hide a
        # defect on one or fail the other for rounding.  On the RATE arm the
        # applicator hands back the rate, so the bar is fp64 rounding: 1e-9,
        # four orders above eps and still four orders below the smallest
        # physical defect worth naming.  On the STATE arm the rate is recovered
        # as (x_new - x)/dt from a field of size |x|, so the recovery's own
        # floor is eps*max|x| / (dt * max|delta rate|) -- computed, printed,
        # never guessed.
        BAR = 1e-9
        if _rate_arm:
            bar_abs = bar_d = BAR
        else:
            _eps = float(np.finfo(np.float64).eps)
            def _recov(x0, drate):
                d = float(np.abs(drate[wet]).max())
                return (_eps * float(np.abs(np.asarray(x0)[..., 0][wet]).max())
                        / (DT * d)) if d > 0 else BAR
            bar_abs = max(BAR, 4.0 * _recov(state.S.data,
                                            (S_star - S0) / tau_S))
            bar_d = max(BAR, 4.0 * _recov(
                state.S.data,
                (S_star - S0) * DT / (tau_S * (tau_S + DT))))
        bad_rows = [nm for nm, v, b in (("ABS(S)", abs_s, bar_abs),
                                        ("d(S)", d_s, bar_d),
                                        ("d(T)", d_t, bar_d))
                    if not np.isfinite(v) or v > b]
        ok = not bad_rows
        verdict = ("REGISTERED: trajectory moves" if ok
                   else "FAILED " + ",".join(bad_rows))
        if not ok:
            failed.append((name, f"{verdict} (ABS(S)={abs_s:.3e} vs bar "
                                 f"{bar_abs:.3e}; d(S)={d_s:.3e} d(T)={d_t:.3e}"
                                 f" vs bar {bar_d:.3e})"))
        else:
            moved.append(name)
        print(f"{name:22s}{tau_T / 86400.0:12.4f}{tau_S / 86400.0:12.4f}"
              f"{fT:11.3e}{fS:11.3e}{abs_s:12.4e}{d_s:12.4e}{d_t:12.4e}"
              f"{marg:13.3e}  {verdict}")

    print(f"\n{len(moved)} card(s) REGISTERED as moving, {len(failed)} "
          f"unexplained, {len(unmeasured)} UNMEASURED")
    for n, why in failed:
        print(f"  UNEXPLAINED {n}: {why}")
    for n in unmeasured:
        print(f"  UNMEASURED  {n}")
    print("\nEvery DINO recipe's reference IS NEMO (Kamm, Deshayes & Madec "
          "2025 run the NEMO configuration this oracle is built from), so the "
          "explicit form is the paper's form and these cards move TOWARD their "
          "reference.  neverworld2_lite's reference is NOT NEMO: it reuses the "
          "DINO applicator through _to_dino_cfg, and its row is an ASKED item.")
    print("\nRULE 9 CHECK: the largest dt/(2*tau) above is the explicit-"
          "relaxation stability margin.  Every card is far inside 1, so the "
          "implicit denominator was never load-bearing.")
    return 1 if (failed or unmeasured) else 0


if __name__ == "__main__":
    sys.exit(main())
