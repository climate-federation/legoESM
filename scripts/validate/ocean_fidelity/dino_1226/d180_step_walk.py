#!/usr/bin/env python
"""#1455 DAY-180 ordered single-step divergence walk (the twin's own start).

WHY THIS EXISTS
---------------
The 90-day twin's ACC sits ~+1.87 Sv above NEMO's, FLAT (no growth), and the
ACC noise floor of that comparison is 0.091 Sv -- the gap is ~20x the floor.
A flat, systematic offset that large cannot be built out of roundoff, so it
must be owned by an operator that differs at the twin's OWN starting point.
Every previous ordered seam walk was run at the YEAR-20 restart
(RUN_SEQDUMP_Y20_1R, kt=230401), which is a different ocean state; a term
that sits at the bar at year 20 is NOT thereby at the bar at day 180.  This
module is the walk at day 180 (kt=5761), the state the twin actually starts
from.

PHASES
------
PHASE 0  NEMO side.  Verify the instrumented (SEQ-DUMP) binary reproduces the
         certified trajectory at day 180, so its dumps describe the twin's own
         run and not a perturbed one.
PHASE 1  FREE-RUNNING walk.  Bridge NEMO's day-180 state into legoESM, take
         ONE step, and compare legoESM's state after each stage against NEMO's
         dump for that stage, in NEMO execution order.  Report the FIRST stage
         over its class bar.
PHASE 2  RE-SEEDED walk.  Seed each stage's INPUT with NEMO's own pre-stage
         state, run only that stage, compare its output.  Produces the ranked
         defect table, reconciled against fidelity_bar_gate.py's year-20 rows.

PRE-REGISTERED PASS CRITERIA (written before the first run; do not soften)
-------------------------------------------------------------------------
PHASE 0 PASS  ==  every spatial variable of the step-5764 restart agrees to
    max|diff| EXACTLY 0.0 between the SEQ-DUMP binary's 1-rank run
    (RUN_SEQDUMP_D180_1R) and the certified binary's 16-rank run
    (RUN_ACC_CERT4), which the oracle repo's .binary_provenance.txt records as
    bit-identical to the recorded RUN_90D_TWIN trajectory.  ANY non-zero
    difference FAILS -- there is no tolerance here, because the two runs are
    the same arithmetic and a difference would mean the instrumentation is not
    dump-only or the decomposition is not neutral.  Both controls (binary
    neutrality AND decomposition neutrality) ride on this one comparison
    because the two runs differ in BOTH; a PASS proves both, a FAIL would need
    them separated before anything else is believed.
    A variable present on one side and absent on the other FAILS.
    A NaN anywhere FAILS.

PHASE 1 PASS  ==  no stage exceeds its class bar (fidelity_bar_gate.CLASS_BAR:
    POINTWISE <= 1e-15, ACCUMULATING <= 1e-12, CONDITIONED = mechanism proof).
    A FAIL is the DELIVERABLE, not an error: the first failing stage is the
    answer this instrument was built to produce.

PHASE 2 PASS  ==  every re-seeded stage at its class bar.  Expected to FAIL;
    the ranked table is the deliverable.

IRREDUCIBILITY RULE (user directive, non-negotiable)
----------------------------------------------------
NOTHING measured here may be labelled irreducible / roundoff / ceiling without
BOTH of:
  (a) OPERAND IDENTITY -- the two sides consume identical operands and differ
      only in reduction ORDER, shown with the file:line of both sides;
  (b) GROWTH NULLITY -- the difference is sign-random and spatially
      incoherent, and its magnitude is consistent with eps-scaling (state the
      eps multiple explicitly).
Failing either test the divergence is DEBT (fixable class), full stop.  The
ACC gap is 20x the ACC noise floor, so roundoff is quantitatively excluded as
its owner; a mislabelled "essential" divergence is exactly the failure this
rule exists to prevent.

This module prints measurements and PASS/FAIL against the pre-registered
criteria above.  It does NOT print an interpretation, a mechanism, or a
verdict about the +1.87 Sv -- that belongs in the analysis, after the controls
pass, never baked into the tool where it gets echoed back as evidence.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_OCEAN_FIDELITY = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _SCRIPTS_OCEAN_FIDELITY):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DINO_CFG = os.environ.get(
    "DINO_ORACLE_ROOT",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO")
# The day-180 SEQ-DUMP run built by this task: instrumented binary md5
# ea0c113c (BLD/bin/nemo.exe.seqdump_ea0c113c), 1 MPI rank, nn_it000=5761,
# nn_itend=5764, restart = RUN_TRAJ/DINO_00005760_restart.nc (adatrj=180.0).
RUN_D180 = os.environ.get("DINO_NEMO_RUN_SEQDUMP_D180",
                          os.path.join(DINO_CFG, "RUN_SEQDUMP_D180_1R"))
# The certified binary's own 4-step run from the SAME restart, 16 MPI ranks.
RUN_CERT4 = os.environ.get("DINO_NEMO_RUN_ACC_CERT4",
                           os.path.join(DINO_CFG, "RUN_ACC_CERT4"))
RUN_TRAJ = os.environ.get(
    "DINO_NEMO_RUN_TRAJ", os.path.join(DINO_CFG, "RUN_TRAJ"))

IC_STEP = 5760          # the twin's day-0 restart (adatrj = 180.0)
KT_FIRST = IC_STEP + 1  # 5761, the step the seam dumps describe
KT_LAST = IC_STEP + 4   # 5764, the control's restart step

JPI, JPJ, HLS = 56, 203, 2            # 1-rank haloed global, nn_hls=2
NI, NJ = JPI - 2 * HLS, JPJ - 2 * HLS  # 52 x 199 interior (== mesh_mask)


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def provenance() -> None:
    """Stamp git SHA, e3t mode and seasonal clock; REFUSE a dirty tree.

    Same gate as ``kamm_twin_90d.provenance_gate`` (#1455 a009c6812: two runs
    of a harness at byte-identical committed source differed by 2.5 Sv and the
    difference was unrecoverable because nothing stamped the tree state), plus
    the two knobs that silently change what is being measured here: the e3t
    ladder mode (#1226's 12.9% analytic-vs-true-ladder trap) and the seasonal
    clock offset (#1455's antiphase-forcing confound).
    """
    from kamm_twin_90d import provenance_gate, seasonal_t0_seconds
    provenance_gate()
    e3t = os.environ.get("LEGOESM_NEMO_E3T")
    print(f"PROVENANCE: LEGOESM_NEMO_E3T={e3t!r}")
    restart = os.path.join(RUN_TRAJ, f"DINO_{IC_STEP:08d}_restart.nc")
    t0 = seasonal_t0_seconds(restart)
    print(f"PROVENANCE: clock t0 = {t0:.0f} s = {t0/86400.0:.3f} d "
          f"(from {os.path.basename(restart)})")
    for tag, d in (("RUN_D180", RUN_D180), ("RUN_CERT4", RUN_CERT4)):
        sha = subprocess.run(["git", "-C", DINO_CFG, "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        print(f"PROVENANCE: {tag}={d} oracle_HEAD={sha}")


def _fatal_if_nan(a: np.ndarray, what: str) -> np.ndarray:
    if not np.isfinite(a).all():
        raise SystemExit(f"NaN/Inf in {what} -- fatal (see module docstring)")
    return a


# ---------------------------------------------------------------------------
# PHASE 0
# ---------------------------------------------------------------------------
def phase0() -> int:
    """Restart bit-identity control.  Returns 0 on PASS, 1 on FAIL."""
    import netCDF4 as nc
    from rebuild_nemo_restart import rebuild

    one_path = os.path.join(RUN_D180, f"DINO_{KT_LAST:08d}_restart.nc")
    tiles = os.path.join(RUN_CERT4, f"DINO_{KT_LAST:08d}_restart_00*.nc")
    print(f"PHASE 0  seqdump-1rank : {one_path}")
    print(f"PHASE 0  certified-16r : {tiles}")

    one = nc.Dataset(one_path)
    names = [k for k, v in one.variables.items()
             if v.ndim >= 3 and "x" in v.dimensions and "y" in v.dimensions]
    stitched = rebuild(tiles, names)

    missing = [k for k in names if k not in stitched]
    rows, n_diff = [], 0
    for k in names:
        if k in missing:
            continue
        a = _fatal_if_nan(np.squeeze(np.asarray(one.variables[k][:])),
                          f"1-rank {k}")
        b = _fatal_if_nan(np.squeeze(stitched[k]), f"stitched {k}")
        if a.shape != b.shape:
            rows.append((np.inf, k, f"shape {a.shape} vs {b.shape}"))
            n_diff += 1
            continue
        d = float(np.max(np.abs(a - b)))
        rows.append((d, k, ""))
        if d != 0.0:
            n_diff += 1
    one.close()
    rows.sort(reverse=True, key=lambda r: r[0])

    print(f"PHASE 0  spatial variables compared = {len(rows)}"
          f"  differing = {n_diff}  missing-on-one-side = {len(missing)}")
    for d, k, note in rows[:5]:
        print(f"    {k:24s} max|diff| = {d:.6e} {note}")
    ok = (n_diff == 0) and not missing and rows
    print("PHASE 0  PASS" if ok else "PHASE 0  FAIL")
    if missing:
        print(f"    missing: {missing}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# PHASE 1 -- free-running walk
# ---------------------------------------------------------------------------
# The per-step reference restarts this arm compares against, one NEMO step
# apart, produced by the CERTIFIED binary at 1 rank from the day-180 restart
# (RUN_D180_STEP1_1R, nn_stock=1) and exposed under the per-rank filename the
# replay's stitcher globs for (RUN_D180_STEP1, single-tile symlinks).  Its
# step-5764 restart is bit-identical to RUN_SEQDUMP_D180_1R's, so this
# trajectory is the same one Phase 0 certified.
RUN_D180_STEP1 = os.environ.get(
    "DINO_NEMO_RUN_D180_STEP1", os.path.join(DINO_CFG, "RUN_D180_STEP1"))
RUN_Y20_STEP1 = os.environ.get(
    "DINO_NEMO_RUN_TWIN_STEP1_Y20", os.path.join(DINO_CFG, "RUN_TWIN_STEP1"))

_ARMS = {
    # arm -> (IC step, per-step reference restart dir)
    "d180": (5760, RUN_D180_STEP1),
    "y20": (230400, RUN_Y20_STEP1),
}

_FIELDS = ("max_deta_now", "max_du_now", "max_dv_now", "max_dT_now",
           "max_dS_now", "max_de3t", "max_du_before", "max_dv_before",
           "max_dT_before", "max_dS_before", "max_den")


def _run_arm(arm: str, n_steps: int) -> dict:
    """One free-running replay arm.  ``IC_STEP`` is read at import time, so it
    must be set BEFORE ``multistep_replay`` is imported -- run each arm in its
    own subprocess rather than re-importing, which is why this is spawned."""
    ic, run_dir = _ARMS[arm]
    env = dict(os.environ)
    env["DINO_1226_IC_STEP"] = str(ic)
    env["DINO_NEMO_RUN_TWIN_STEP1"] = run_dir
    code = (
        "import json, sys, numpy as np;"
        "sys.path.insert(0, %r);"
        "import multistep_replay as m;"
        "m.provenance('phase1');"
        "d = m.run_replay(%d);"
        "print('@@JSON@@' + json.dumps({k: np.asarray(v).tolist() "
        "for k, v in d.items()}))" % (_THIS_DIR, n_steps))
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True)
    tail = out.stdout.strip().splitlines()
    for line in tail:
        if not line.startswith("@@JSON@@"):
            print(f"  [{arm}] {line}")
    hit = [l for l in tail if l.startswith("@@JSON@@")]
    if not hit:
        print(out.stderr[-4000:])
        raise SystemExit(f"arm {arm!r} produced no result")
    import json
    return json.loads(hit[-1][len("@@JSON@@"):])


def phase1(n_steps: int = 4, arms: tuple[str, ...] = ("d180", "y20")) -> int:
    """Free-running walk: ONE (then N) legoESM steps from the bridged state,
    compared against NEMO's OWN restart at each step.

    Both arms run the SAME code with the SAME step count and the SAME
    seasonal-clock rule; the ONLY variable is the initial condition (and the
    reference restarts that go with it).  That is what makes a day-180 number
    comparable with a year-20 number here.
    """
    res = {}
    for arm in arms:
        print(f"\nPHASE 1  arm={arm}  IC_STEP={_ARMS[arm][0]}  "
              f"ref={_ARMS[arm][1]}")
        res[arm] = _run_arm(arm, n_steps)
        for f in _FIELDS:
            _fatal_if_nan(np.asarray(res[arm][f]), f"{arm}:{f}")

    print("\nPHASE 1  free-running divergence, max|legoESM - NEMO| on wet cells")
    hdr = "  " + "field".ljust(16)
    for arm in arms:
        hdr += "".join(f"{arm}:k{k+1}".rjust(14) for k in range(n_steps))
    print(hdr)
    for f in _FIELDS:
        line = "  " + f.replace("max_", "").ljust(16)
        for arm in arms:
            line += "".join(f"{v:14.4e}" for v in res[arm][f])
        print(line)
    print("\nPHASE 1  reported (a FAIL here is the deliverable, not an error)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", default="0", choices=["0", "1"],
                    help="0 = NEMO-side control, 1 = free-running walk")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--arms", default="d180,y20")
    args = ap.parse_args(argv)
    provenance()
    if args.phase == "0":
        return phase0()
    return phase1(n_steps=args.steps, arms=tuple(args.arms.split(",")))


if __name__ == "__main__":
    raise SystemExit(main())
