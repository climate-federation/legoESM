#!/usr/bin/env python
"""The FROM-REST spread floor: is the year-long twin gap bigger than chaos?

Preregistered in ``PREREG_verdict360_fromrest.md``.  READ IT FIRST -- in
particular §1, which says under what measured condition this harness is worth
running at all and refuses to proceed otherwise.

THE QUESTION.  legoESM and NEMO, both started from rest on the same DINO card,
differ by a 3-D temperature rms that grows ~2.1e-3 K at day 30 to ~4.3e-3 K at
day 360.  Is that evidence of a remaining operator mismatch, or is it inside
the spread each model generates on its own from an infinitesimally perturbed
initial state?

WHY THIS IS NOT verdict360.py.  ``verdict360.py`` scores a twin started from a
NEMO RESTART at day 180 of an equilibrated run.  Its floor is measured in a
fully eddying regime.  A from-rest year-1 DINO is dominated by deterministic
adjustment for part of that window, so the floor is a DIFFERENT quantity and
has to be measured, not inherited.  A claim review flagged exactly this:
between day 90 and day 360 of the restart-seeded ensemble the ACC spread grew
by a factor ~2200 while the from-rest gap grew by ~1.8, and a gap that is flat
while chaos grows is the signature of a BOUNDED DETERMINISTIC OFFSET.  If that
is what is happening, a day-360 yes/no verdict is a coin-flip on when the
growing floor overtakes a flat gap, and it says nothing about fidelity.

SO THIS HARNESS RUNS IN TWO PHASES AND THE FIRST ONE CAN CANCEL THE SECOND.

  PHASE 0   legoESM ONLY.  Four members, from rest, 360 days, the NEMO
            perturbation applied to the initial temperature.  Costs no NEMO
            time at all.  Reports the ensemble spread at days 30/90/180/360 in
            the SAME 3-D T rms metric as the gap, and classifies against the
            preregistered window.  Justification for using it as a floor
            estimate: on the restart-seeded record NEMO's own spread is 9-16600x
            TIGHTER than legoESM's, so 77-99% of the combined
            ``sqrt(spread_lego^2 + spread_nemo^2)`` IS legoESM's spread.

  PHASE 1   the NEMO members, acquired by ``nemo_dino_fromrest_members/run.sh``,
            and the two-sample comparison.  Refuses to run unless PHASE 0 has
            been run and recorded a floor inside the window.

THE PERTURBATION IS NEMO'S OWN, to the statement.  DINO already carries it:
``usrdef_istate.F90:177-183`` (compiled at ``BLD/ppsrc/nemo/usrdef_istate.f90``
``:188-194``) adds, under ``nn_pert_seed /= 0`` (declared
``usrdef_nam.F90:81`` and read from ``namusr_def``, ``:126``):

    1.e-10 * SIN( REAL( NINT(pdept)*73 + NINT(gphit*1000)*179
                        + nn_pert_seed*997 ) ) * tmask

so the NEMO members need NO source patch -- one namelist line each -- and
legoESM must reproduce that expression EXACTLY rather than invent its own
nudge.  Two things about it matter and are easy to get wrong: it is 1e-10
ABSOLUTE (not relative), and its argument uses only depth and latitude, so the
perturbation is ZONALLY UNIFORM.  A different amplitude or a different spatial
projection is a different experiment.

THE SCORING IS PAIRWISE, NOT A DIFFERENCE OF MEANS.  ``|mean_A - mean_B|``
against a spread is meaningful for a signed scalar such as a transport; for a
positive-definite field norm it is a category error.  So the field rows use the
two-sample energy/distance statistic on the pairwise rms distances
``d(A_i, B_j)`` versus ``d(A_i, A_j)`` and ``d(B_i, B_j)``.  The scalar rows
(ACC, the density contrasts) keep the ``verdict360`` mean-difference form,
which IS valid for them.

AND THE ANSWER IS A CROSSING DAY, NOT A BINARY.  The output is the first day at
which the gap falls inside 2x the floor, or "never within the year".
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

#: NEMO's own constants, from usrdef_istate.F90:180.  Named, not inlined, so a
#: change here is visible as a change of experiment.
PERT_AMPLITUDE_K = 1.0e-10
PERT_DEPTH_MULT = 73
PERT_LAT_MULT = 179
PERT_SEED_MULT = 997
PERT_LAT_SCALE = 1000.0

#: THE DECIDING FACTOR, derived rather than chosen.  A diff reviewer showed
#: that the first draft's "window" [2e-3, 1e-2] K was 98% pre-determined and
#: that its refusal branch was BACKWARDS.  The correct statement is one line of
#: algebra, so it is written as algebra:
#:
#:   PHASE 1 can only ever INCREASE the floor, because the combined floor is
#:   sqrt(spread_lego^2 + spread_nemo^2) >= spread_lego.  So:
#:
#:     gap <= 2*floor_lego   ->  INDISTINGUISHABLE is ALREADY established by
#:                               PHASE 0 alone and PHASE 1 cannot overturn it.
#:                               Spending the NEMO members buys nothing.
#:     gap >  2*floor_lego   ->  distinguishable so far, and PHASE 1 can flip
#:                               it ONLY IF NEMO's own from-rest spread is at
#:                               least  sqrt((gap/(2*floor_lego))^2 - 1)
#:                               times legoESM's.  That REQUIRED FACTOR is
#:                               printed, and compared against what is known:
#:                               on the restart-seeded record NEMO is 9-16600x
#:                               TIGHTER, so a required factor above ~1 makes
#:                               PHASE 1 dead on arrival.
#:
#: There is no free constant here to tune.
NEMO_TIGHTER_RANGE = (9.0, 16600.0)   # restart-seeded record, PR #1728

#: The measured from-rest gap this floor is read against, WITH ITS PROVENANCE.
#: Hardcoding five numbers with no run behind them is how a comparison drifts
#: onto another run; the stamp is checked by --score-phase0 against the maps
#: sidecars when they are present.
GAP_PROVENANCE = {
    "run_dir": "/data/abyssal/dbalwada/dino_fromrest_y1/lego_trueframe_r3t",
    "metric": "wet-masked 3-D temperature rms difference [K], "
              "twin_nemo_ts_maps.py --run-dino-dir, field row 'T3D'",
    "nemo_days_30_180": "cfgs/DINO/RUN_TRAJ kt 960/1920/2880/5760",
    "nemo_day_360": "cfgs/DINO/RUN_FROMREST_Y1 kt 11520",
    "commit": "PR #1728, the Kmm-stretch round",
    "day360_snapshot_sha256":
        "0ef0c080f1626edf9a9703221c65d779631161dfa778a5f2cf99a075704a8f2c",
}
GAP_K = {30: 2.0388e-3, 60: 2.1567e-3, 90: 2.3044e-3,
         180: 4.5899e-3, 360: 3.9243e-3}

N_MEMBERS = 4


def _nint(x):
    """Fortran ``NINT``: round half AWAY FROM ZERO.

    ``np.round`` is round-half-to-EVEN, which disagrees with Fortran on every
    exact .5 -- and ``gphit*1000`` lands on exact halves on a regular grid, so
    this is not a pedantic difference.
    """
    x = np.asarray(x, dtype=np.float64)
    return np.trunc(x + np.copysign(0.5, x))


def nemo_istate_perturbation(t_depth_m, lat_deg, tmask, seed: int):
    """NEMO's ``nn_pert_seed`` temperature perturbation, statement for statement.

    ``usrdef_istate.F90:180`` / ppsrc ``usrdef_istate.f90:191``::

        pts(:,:,jk,jp_tem) = pts(:,:,jk,jp_tem)
           + 1.e-10_wp * SIN( REAL( NINT(pdept(:,:,jk))*73
                                  + NINT(gphit(:,:)*1000._wp)*179
                                  + nn_pert_seed*997, wp ) ) * ptmask(:,:,jk)

    Parameters are the card's own depth / latitude / mask arrays, broadcast to
    ``(n_lat, n_lon, nlev)``.  ``seed = 0`` returns exactly zero, as NEMO's
    ``IF( nn_pert_seed /= 0 )`` guard does.
    """
    if seed == 0:
        return np.zeros(np.broadcast(t_depth_m, lat_deg, tmask).shape)
    arg = (_nint(t_depth_m) * PERT_DEPTH_MULT
           + _nint(np.asarray(lat_deg) * PERT_LAT_SCALE) * PERT_LAT_MULT
           + seed * PERT_SEED_MULT)
    return PERT_AMPLITUDE_K * np.sin(arg) * np.asarray(tmask)


def _rms(a, wet):
    a = np.asarray(a)
    return float(np.sqrt(np.mean(a[wet] ** 2)))


def _pairwise(A, B, wet):
    """rms distances within A, within B, and across -- the two-sample statistic.

    Returns ``(within, across)`` where ``within`` pools d(A_i,A_j) and
    d(B_i,B_j) (the FLOOR) and ``across`` is d(A_i,B_j) (the GAP).  No mean
    field is ever formed, which is the point: an rms is positive-definite and
    the difference of two rms values is not a distance.
    """
    within, across = [], []
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            within.append(_rms(A[i] - A[j], wet))
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            within.append(_rms(B[i] - B[j], wet))
    for x in A:
        for y in B:
            across.append(_rms(x - y, wet))
    return np.array(within), np.array(across)


def _nemo_istate_operands():
    """NEMO's OWN ``pdept`` and ``gphit``, read from the mesh, not look-alikes.

    ``usr_def_istate`` is called from ``istate.F90`` with ``pdept =
    gdept(:,:,:,Kbb)``, which under DINO's ``ln_zco_nam = .true.``
    (``RUN_FROMREST_KT1/namelist_cfg:70``) is ``gdept_0`` -- verified here to
    be constant over (i,j) rather than assumed -- and ``gphit`` is latitude in
    DEGREES.

    The first draft of this harness used ``z_coord.t_depth_ref`` and
    ``grid.lat``.  BOTH ARE WRONG and neither would have raised:
      * ``grid.lat`` is in RADIANS (-1.219 .. 1.219), so ``NINT(lat*1000)``
        would have been a different integer at every row -- a different
        experiment wearing NEMO's formula;
      * ``t_depth_ref`` is legoESM's cell-CENTRE depth and NEMO's ``gdept_1d``
        is the analytic stretching value; they agree to k=25 and then part
        company by up to 104.97 m, which crosses whole metres and therefore
        changes ``NINT``.
    """
    # Rule 1c: fp64 EXPLICITLY.  The phase-0 rewrite dropped the policy call
    # and it was harmless only because the mesh reader returns f64 anyway; a
    # 1e-10 K perturbation is far below f32 resolution on a ~10 K field, so a
    # policy slip here would silently make every member the control.
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())
    from legoesm.ocean.fidelity import nemo_dino_mesh as ndm
    m = ndm.nemo_dino_mesh()
    dep3 = np.asarray(m.gdept_0, dtype=np.float64)
    spread = float(np.max(dep3.max(axis=(0, 1)) - dep3.min(axis=(0, 1))))
    if spread != 0.0:
        raise SystemExit(
            f"gdept_0 varies by {spread:.3e} m within a level, so this mesh is "
            "not ln_zco and NEMO's perturbation would not be zonally uniform; "
            "the harness's own uniformity assertion would then be wrong, not "
            "the perturbation.")
    lat3 = np.broadcast_to(
        np.asarray(m.gphit, dtype=np.float64)[:, :, None], dep3.shape)
    mask3 = np.asarray(m.tmask > 0.5).astype(np.float64)
    return dep3, lat3, mask3


def phase0(out_root: str, days: int, snap: int, gpu_note: str) -> int:
    """Four legoESM members from rest; the floor, and the window verdict."""
    dep3, lat3, mask3 = _nemo_istate_operands()

    # NINT is a step function, so an operand sitting on a half-integer is the
    # one place fp64 noise could change the perturbation.  Measured, not hoped.
    for nm, arr in (("NINT(pdept)", dep3[0, 0, :]),
                    ("NINT(gphit*1000)", lat3[:, 0, 0] * PERT_LAT_SCALE)):
        d = float(np.min(np.abs(np.abs(arr - np.floor(arr)) - 0.5)))
        print(f"  {nm}: closest operand to a half-integer is {d:.4f} away "
              f"(fp64 noise here is ~1e-13)")
        if d < 1e-6:
            print("  ^^ an operand sits ON a rounding boundary; NINT's "
                  "round-half-away-from-zero and the oracle's could disagree")
            return 1

    # The perturbation must be REAL and TINY: printed, not assumed.
    p1 = nemo_istate_perturbation(dep3, lat3, mask3, 1)
    print(f"PHASE 0 -- {N_MEMBERS} legoESM members from rest, {days} days")
    print(f"  perturbation (seed 1): max|dT| = {np.abs(p1).max():.3e} K on "
          f"{int((p1 != 0).sum())} cells; seed 0 gives "
          f"{np.abs(nemo_istate_perturbation(dep3, lat3, mask3, 0)).max():.3e}")
    # NEMO's argument has no longitude, so the perturbation is constant along
    # a row -- ON WET CELLS.  The mask itself varies with longitude, and a
    # first version of this check measured the MASK and reported 1e-10, which
    # is the whole amplitude.  Measured on wet cells only, as intended.
    _w = mask3 > 0.5
    _pm = np.where(_w, p1, np.nan)
    import warnings
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _zs = float(np.nanmax(np.nanmax(_pm, axis=1) - np.nanmin(_pm, axis=1)))
    print(f"  zonal structure (WET cells only): max over lon of (max-min) "
          f"within a row = {_zs:.3e} K -- NEMO's argument has no longitude, "
          "so this must be 0")
    if _zs != 0.0:
        print("  ^^ the perturbation is NOT zonally uniform, so it is not "
              "NEMO's (usrdef_istate.F90:180)")
        return 1
    # Distinct members are the whole experiment: two seeds that collided would
    # give a floor of zero and an "indistinguishable" verdict for free.
    fields = {s: nemo_istate_perturbation(dep3, lat3, mask3, s)
              for s in range(1, N_MEMBERS + 1)}
    for a in range(1, N_MEMBERS + 1):
        for b in range(a + 1, N_MEMBERS + 1):
            sep = float(np.abs(fields[a] - fields[b]).max())
            if sep == 0.0:
                print(f"  seeds {a} and {b} give the SAME field; the floor "
                      "would be measured on a duplicated member")
                return 1
    print(f"  seed separation: min over pairs of max|dT_i - dT_j| = "
          f"{min(float(np.abs(fields[a] - fields[b]).max()) for a in range(1, N_MEMBERS + 1) for b in range(a + 1, N_MEMBERS + 1)):.3e} K")
    for s in range(1, N_MEMBERS + 1):
        d = os.path.join(out_root, f"member_{s:02d}")
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "perturbation_T.npy"), fields[s])
    print(f"  per-member perturbation fields written under {out_root}/")
    print("\n  THE MEMBERS ARE NOT RUN BY THIS SCRIPT.  Run them with:\n"
          f"{gpu_note}")
    print("  then re-run with --score-phase0 to get the floor.")
    return 0


def run_member(out_root: str, seed: int, days: int, snap: int,
               config: str) -> int:
    """Run ONE member THROUGH ``run_dino.py``'s OWN ``main()``.

    Rule 10, taken literally: the member is not a re-implementation of the
    production loop, it IS the production loop.  ``scripts/run/run_dino.py``
    is imported as a module, its ``dino_lat_lon_state`` is wrapped so the
    returned state carries the perturbation NEMO's ``usr_def_istate`` would
    have added, ``sys.argv`` is set to the card's own command line, and
    ``main()`` runs.  Every other statement of the driver -- the fp64 policy,
    the recipe overlay, the leap-frog/surface-placement guard, the forcing
    call, ``model.step``, the snapshot writer -- executes unchanged.

    This is why no ``--initial-T-perturbation`` production flag is added: the
    ensemble machinery stays out of the driver that every DINO run uses.
    """
    import importlib.util

    d = os.path.join(out_root, f"member_{seed:02d}")
    pfile = os.path.join(d, "perturbation_T.npy")
    if not os.path.exists(pfile):
        raise SystemExit(
            f"no {pfile}: run --phase0 first.  The member must use the SAME "
            "perturbation bytes the phase-0 assertions were run against, not "
            "a recomputed look-alike.")
    pert = np.load(pfile)

    repo = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
    driver = os.path.join(repo, "scripts", "run", "run_dino.py")
    spec = importlib.util.spec_from_file_location("run_dino_member", driver)
    rd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rd)

    orig = rd.dino_lat_lon_state
    calls = []

    def perturbed(grid, z_coord, cfg=None, land_mask_override=None):
        st = orig(grid, z_coord, cfg, land_mask_override=land_mask_override)
        base = np.asarray(st.T.data)
        if base.shape != pert.shape:
            raise SystemExit(
                f"perturbation shape {pert.shape} != state T {base.shape}")
        if base.dtype != np.float64:
            raise SystemExit(
                f"state T is {base.dtype}, not float64 -- a 1e-10 K "
                "perturbation is below f32 resolution on a ~10 K field, so "
                "this member would be the control")
        out = st._replace(T=st.T.replace(data=st.T.data + pert))
        moved = np.asarray(out.T.data) - base
        calls.append((float(np.abs(moved).max()),
                      int(np.count_nonzero(moved))))
        return out

    rd.dino_lat_lon_state = perturbed
    sys.argv = [driver, "--config", config, "--days", str(days),
                "--snapshot-every-days", str(snap), "--output-dir", d]
    print(f"MEMBER seed={seed} -> {d}")
    print(f"  argv: {' '.join(sys.argv[1:])}")
    rd.main()
    if not calls:
        raise SystemExit(
            "the wrapped state builder was NEVER CALLED, so this member ran "
            "the UNPERTURBED initial state and is a duplicate of the control")
    mx, n = calls[0]
    print(f"  perturbation landed in the state that ran: max|dT| = {mx:.3e} K "
          f"on {n} cells ({len(calls)} state build(s))")
    return 0


def score_phase0(out_root: str, snap: int) -> int:
    from legoesm.ocean.fidelity import nemo_dino_mesh as ndm
    wet = np.asarray(ndm.nemo_dino_mesh().tmask > 0.5)
    dirs = sorted(glob.glob(os.path.join(out_root, "member_*", "snapshots")))
    if len(dirs) < N_MEMBERS:
        raise SystemExit(
            f"only {len(dirs)} member snapshot directories under {out_root}; "
            f"{N_MEMBERS} are needed.  Run the members first (--phase0 prints "
            "the command).")
    per_day: dict[int, list[np.ndarray]] = {}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "snapshot_*.npz"))):
            k = int(os.path.basename(f).split("_")[1].split(".")[0])
            per_day.setdefault(k * snap, []).append(np.load(f)["T"])
    # THE MEMBERS MUST ACTUALLY DIFFER AT DAY 0.  If a member ran the
    # unperturbed state -- the one way the no-knob path can fail silently --
    # its day-0 distances collapse and the floor is manufactured downward,
    # which is exactly the direction that would fake "INDISTINGUISHABLE".
    if 0 in per_day and len(per_day[0]) >= 2:
        w0, _ = _pairwise(per_day[0], per_day[0][:0], wet)
        print(f"\n  day-0 pairwise rms spread: min {w0.min():.4e} K, "
              f"max {w0.max():.4e} K (NEMO's perturbation amplitude is "
              f"{PERT_AMPLITUDE_K:.1e} K, so a pair at 0 means a member ran "
              "UNPERTURBED)")
        if float(w0.min()) == 0.0:
            raise SystemExit(
                "two members are IDENTICAL at day 0: at least one ran the "
                "unperturbed initial state, so the ensemble floor below "
                "would be measured on a duplicated member")
    print(f"\n  {'day':>6s}{'members':>9s}{'spread rms [K]':>16s}"
          f"{'gap [K]':>12s}{'gap/2*floor':>13s}")
    floor = {}
    cross = None
    for day in sorted(per_day):
        M = per_day[day]
        if len(M) < 2:
            continue
        w, _ = _pairwise(M, M[:0], wet)
        f = float(np.sqrt(np.mean(w ** 2)))
        floor[day] = f
        gap = GAP_K.get(day)
        r = (gap / (2 * f)) if (gap and f > 0) else float("nan")
        if gap and f > 0 and r <= 1.0 and cross is None:
            cross = day
        print(f"  {day:>6d}{len(M):>9d}{f:>16.4e}"
              f"{(gap if gap else float('nan')):>12.4e}{r:>13.3f}")
    if 360 not in floor:
        print("\n  no day-360 snapshot; the verdict needs one")
        return 1
    # ---- THE PREREGISTERED §1 CLASSIFICATION, which outranks the binary ---
    # PREREG §1: "a gap that is FLAT while chaos GROWS is the signature of a
    # bounded deterministic offset, not of a diverging trajectory.  If that is
    # what is happening, a day-360 yes/no verdict is a coin-flip on when a
    # growing floor overtakes a flat gap and says nothing about fidelity."
    # So the growth RATIOS are printed before the binary, not after it.
    d0 = min(d for d in floor if d >= 30 and d in GAP_K)
    fg = floor[360] / floor[d0] if floor[d0] > 0 else float("inf")
    gg = GAP_K[360] / GAP_K[d0]
    print(f"\n  PREREG §1 -- day {d0} to day 360: the FLOOR grows {fg:.1f}x "
          f"({floor[d0]:.4e} -> {floor[360]:.4e} K) while the GAP grows "
          f"{gg:.2f}x ({GAP_K[d0]:.4e} -> {GAP_K[360]:.4e} K).")
    ratios = {d: GAP_K[d] / (2 * floor[d]) for d in sorted(GAP_K)
              if d in floor and floor[d] > 0}
    print("    gap/(2*floor) by day: "
          + "  ".join(f"d{d}:{r:.2f}" for d, r in ratios.items()))
    if fg > 10.0 * gg:
        print("    CLASSIFICATION: BOUNDED DETERMINISTIC OFFSET, not chaos. "
              "The gap is inside the floor at day 360 only because the floor "
              "OVERTOOK it; early in the year, when both trajectories are "
              "still deterministic, the gap is "
              f"{ratios[d0]:.0f}x the floor. A day-360 binary is therefore "
              "NOT a fidelity statement, and the next measurement is an "
              "OPERATOR, not an ensemble.")
    else:
        print("    CLASSIFICATION: the gap and the floor grow together, so "
              "the day-360 comparison is a genuine chaos test.")

    f360, gap360 = floor[360], GAP_K[360]
    r = gap360 / (2.0 * f360) if f360 > 0 else float("inf")
    need = float(np.sqrt(max(r * r - 1.0, 0.0))) if np.isfinite(r) else np.inf
    print(f"\n  day-360 floor {f360:.4e} K, gap {gap360:.4e} K, "
          f"gap/(2*floor) = {r:.3f}")
    lo_t, hi_t = NEMO_TIGHTER_RANGE
    proceed = (r > 1.0) and (need <= 1.0 / lo_t)
    json.dump({"floor_K": floor, "gap_K": GAP_K,
               "gap_provenance": GAP_PROVENANCE, "crossing_day": cross,
               "ratio_360": r, "required_nemo_spread_factor": need,
               "phase1_worth_running": bool(proceed)},
              open(os.path.join(out_root, "phase0_floor.json"), "w"), indent=1)
    if r <= 1.0:
        print("  VERDICT: INDISTINGUISHABLE at day 360, established by PHASE 0 "
              "ALONE.  The combined floor can only be LARGER than legoESM's, "
              "so the NEMO members cannot overturn this. DO NOT spend PHASE 1."
              f"  First day within 2*floor: {cross}")
        return 0
    print(f"  distinguishable on legoESM's floor alone. For PHASE 1 to flip "
          f"it, NEMO's own from-rest spread must be at least {need:.3f}x "
          "legoESM's.")
    print(f"  On the restart-seeded record NEMO is {lo_t:.0f}-{hi_t:.0f}x "
          f"TIGHTER, i.e. a factor of at most {1.0 / lo_t:.4f}.")
    if not proceed:
        print("  VERDICT: PHASE 1 is DEAD ON ARRIVAL -- NEMO would have to be "
              "LOOSER than legoESM by a factor its own record contradicts. "
              "The gap is a bounded deterministic offset (it FALLS from "
              f"{GAP_K[180]:.4e} K at day 180 to {gap360:.4e} K at day 360, "
              "and a trajectory diverging from a 1e-10 K seed does not "
              "shrink), so the next measurement is an OPERATOR, not an "
              "ensemble.")
        return 1
    print("  VERDICT: PHASE 1 can change the answer -- worth its NEMO time.")
    return 0


def phase1(out_root: str, nemo_root: str, snap: int) -> int:
    if not os.path.exists(os.path.join(out_root, "phase0_floor.json")):
        raise SystemExit(
            "REFUSING: PHASE 0 has not been scored.  Its whole purpose is to "
            "say whether PHASE 1 can produce an informative answer, and "
            "running PHASE 1 first would discard that.")
    raise SystemExit(
        "PHASE 1 needs the NEMO members.  Acquire them with\n"
        "  scripts/validate/ocean_fidelity/dino_1226/"
        "nemo_dino_fromrest_members/run.sh\n"
        "then this phase reads them from --nemo-root.  Not implemented "
        "against a record that does not exist yet -- a reader written "
        "without its record is a reader nobody has ever seen run.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root",
                    default="/data/abyssal/dbalwada/dino_fromrest_y1/"
                            "verdict360_fromrest")
    ap.add_argument("--nemo-root", default=None)
    ap.add_argument("--days", type=int, default=360)
    ap.add_argument("--snapshot-every-days", type=int, default=30)
    ap.add_argument("--phase0", action="store_true")
    ap.add_argument("--run-member", type=int, default=None,
                    help="run ONE member (seed 1..N) through run_dino.main()")
    ap.add_argument("--config",
                    default="scripts/experiment/dino/"
                            "nemo_faithful_kamm_mlf.yaml")
    ap.add_argument("--score-phase0", action="store_true")
    ap.add_argument("--phase1", action="store_true")
    a = ap.parse_args()
    note = (
        "    for s in 1 2 3 4; do\n"
        "      CUDA_VISIBLE_DEVICES=<uuid> JAX_PLATFORMS=cuda "
        "JAX_ENABLE_X64=1 \\\n"
        "        python scripts/validate/ocean_fidelity/dino_1226/"
        "verdict360_fromrest.py \\\n"
        f"          --run-member $s --days {a.days} "
        f"--snapshot-every-days {a.snapshot_every_days} \\\n"
        f"          --out-root {a.out_root}\n"
        "    done\n"
        "  NO PRODUCTION FLAG IS ADDED.  --run-member imports\n"
        "  scripts/run/run_dino.py and calls its OWN main() with only the\n"
        "  initial-state builder wrapped, so the member runs the production\n"
        "  driver statement for statement and the ensemble machinery stays\n"
        "  out of the driver every DINO run uses.")
    if a.phase0:
        return phase0(a.out_root, a.days, a.snapshot_every_days, note)
    if a.run_member is not None:
        if not 1 <= a.run_member <= N_MEMBERS:
            ap.error(f"--run-member must be in 1..{N_MEMBERS}")
        return run_member(a.out_root, a.run_member, a.days,
                          a.snapshot_every_days, a.config)
    if a.score_phase0:
        return score_phase0(a.out_root, a.snapshot_every_days)
    if a.phase1:
        return phase1(a.out_root, a.nemo_root, a.snapshot_every_days)
    ap.error("choose --phase0, --run-member, --score-phase0 or --phase1")


if __name__ == "__main__":
    sys.exit(main())
