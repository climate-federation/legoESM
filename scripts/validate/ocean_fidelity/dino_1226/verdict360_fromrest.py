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


def phase0(out_root: str, days: int, snap: int, gpu_note: str) -> int:
    """Four legoESM members from rest; the floor, and the window verdict."""
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())
    from legoesm.ocean.experiments import dino as dm
    from legoesm.ocean.fidelity import nemo_dino_mesh as ndm

    cfg = dm.nemo_faithful_dino_config(
        base=dm.dino_config_for_recipe("nemo_dino_kamm_mlf"))
    grid = dm.dino_lat_lon_grid(cfg)
    z = dm.dino_lat_lon_vertical(grid, cfg)
    base = dm.dino_lat_lon_state(grid, z, cfg)
    lat = np.asarray(grid.lat)
    lat3 = np.broadcast_to(lat[:, None, None], base.T.data.shape)
    dep = np.asarray(z.t_depth_ref)
    dep3 = np.broadcast_to(dep[None, None, :], base.T.data.shape)
    mask3 = np.asarray(ndm.nemo_dino_mesh().tmask > 0.5).astype(float)

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
    for s in range(1, N_MEMBERS + 1):
        d = os.path.join(out_root, f"member_{s:02d}")
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "perturbation_T.npy"),
                nemo_istate_perturbation(dep3, lat3, mask3, s))
    print(f"  per-member perturbation fields written under {out_root}/")
    print("\n  THE MEMBERS ARE NOT RUN BY THIS SCRIPT.  Run them with:\n"
          f"{gpu_note}")
    print("  then re-run with --score-phase0 to get the floor.")
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
    ap.add_argument("--score-phase0", action="store_true")
    ap.add_argument("--phase1", action="store_true")
    a = ap.parse_args()
    note = (
        "    for s in 1 2 3 4; do\n"
        "      CUDA_VISIBLE_DEVICES=<uuid> JAX_PLATFORMS=cuda "
        "JAX_ENABLE_X64=1 \\\n"
        "        python scripts/run/run_dino.py \\\n"
        "          --config scripts/experiment/dino/"
        "nemo_faithful_kamm_mlf.yaml \\\n"
        f"          --days {a.days} --snapshot-every-days "
        f"{a.snapshot_every_days} \\\n"
        f"          --output-dir {a.out_root}/member_0$s \\\n"
        "          --initial-T-perturbation "
        f"{a.out_root}/member_0$s/perturbation_T.npy\n"
        "    done\n"
        "  NOTE: --initial-T-perturbation DOES NOT EXIST YET.  It is a new\n"
        "  production knob and is listed in the round's ASKED table with a\n"
        "  recommendation AGAINST adding it; the alternative, which needs no\n"
        "  knob, is for this harness to build and perturb the state itself\n"
        "  and call the model directly.  Nothing here is run until that is\n"
        "  decided.")
    if a.phase0:
        return phase0(a.out_root, a.days, a.snapshot_every_days, note)
    if a.score_phase0:
        return score_phase0(a.out_root, a.snapshot_every_days)
    if a.phase1:
        return phase1(a.out_root, a.nemo_root, a.snapshot_every_days)
    ap.error("choose --phase0, --score-phase0 or --phase1")


if __name__ == "__main__":
    sys.exit(main())
