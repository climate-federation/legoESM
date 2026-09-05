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
PHASE 1  FREE-RUNNING walk.  Bridge NEMO's day-180 state into legoESM,
         integrate, and compare the WHOLE committed state against NEMO's own
         restart at each step.  This is a whole-step state replay, NOT a
         per-stage walk: legoESM's step exposes no per-stage intermediate, so
         there is nothing to compare stage by stage from inside it.
PHASE 1b WHERE the one-step velocity difference lives (depth-mean vs shear,
         and its vertical profile).
PHASE 1c Whether the per-step ACC difference is of a size that could build the
         90-day gap.  See the RETRACTION on phase1c: it is not, and the
         measurement refutes its own premise.

The PER-STAGE (re-seeded) walk is NOT in this module.  It is the ~25 existing
#1226 per-operator probes re-pointed at the day-180 reference run through
``dump_lane``; each one seeds its own operator with NEMO's own pre-stage input
and compares that operator's output.  Run them with ``DINO_1226_LANE=d180``.
Nothing here imports ``fidelity_bar_gate.CLASS_BAR`` -- the class bars are
quoted below as the standard those probes are read against, and each probe
prints its own numbers.

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

PHASE 1 PASS  ==  the committed state after each step matches NEMO's own
    restart at that step to the class bars the #1226 sweep is read against
    (fidelity_bar_gate: POINTWISE <= 1e-15, ACCUMULATING <= 1e-12, CONDITIONED
    = mechanism proof only).  A FAIL is the DELIVERABLE, not an error.
    MEASURED: FAIL by ~13 orders -- see the recorded results below.

PER-STAGE PASS  ==  every re-seeded operator at its class bar, measured by its
    own probe on the d180 lane.  Expected to FAIL; the ranked table is the
    deliverable.

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

RETRACTIONS (2026-08-20, from the dual adversarial review, before the walk's
numbers were carried anywhere)
------------------------------------------------------------------------
R1.  "the one-step divergence is 99% SHEAR" IS A PROPERTY OF A GRID THE
     PRODUCTION TWIN DOES NOT RUN.  Phase 1b was first recorded with
     LEGOESM_NEMO_E3T=both.  On the SHIPPED grid ("off") the depth-mean part
     is far from negligible:

       one step, day 180, wet faces      u rms      depth-mean   share
       LEGOESM_NEMO_E3T=both           3.0895e-04   2.4382e-06   0.008
       LEGOESM_NEMO_E3T=off (SHIPPED)  3.4395e-04   4.5876e-05   0.133
       v, both                         9.1752e-05   8.4998e-06   0.093
       v, off (SHIPPED)                1.2720e-04   5.5722e-05   0.438

     So on the configuration whose ACC gap is being explained the barotropic
     part is 19x larger and carries 13% of the u difference and 44% of the v
     difference.  The surface concentration survives on both grids (top-level
     rms 1.377e-3 vs 1.367e-3), the "99% shear" framing does not.

R2.  PHASE 1c's PREMISE IS REFUTED BY ITS OWN DATA.  A 4-step ACC rate carries
     no information about 960 steps here.  NEMO's ACC FALLS over the first 4
     steps (-3.3e-3 Sv/step) and RISES monotonically over 90 days (+3.42 Sv
     total), so the reference model's own short-window rate has the wrong sign
     and ~3x the wrong magnitude.  Extrapolating the arms fails the same way:
     the 90-day outcomes are +1.659 and -0.516 Sv where their 4-step rates
     predict +4.8 and -204 Sv.  The per-step deltas also alternate in sign
     (+1.28e-2, -3.5e-5, +7.2e-3, -3.8e-4), i.e. a leapfrog two-step mode
     rather than a rate.  Phase 1c is retained as a measurement of the
     first-step ACC increment; ANY statement of the form "this rate is of the
     right size to build the 90-day gap" is withdrawn.

R3.  THE STEP-COMPOSITION REFUTATION IS NOT EVIDENCE.  See the note at the
     ablation itself: it moves two fields, and _nemo_mlf_step is documented as
     identical to _leapfrog_step except in how the dissipation pass is
     composed, so the comparison is legoESM against legoESM and says nothing
     about whether either matches NEMO's composition.  The surface-stress
     refutation (i) survives as a verdict, but its "50x worse in the depth
     mean" figure was also measured on the non-production grid; on the shipped
     grid the depth-mean degradation is 2.9x, not 50x, and the overall u rms
     gets slightly BETTER, not slightly worse.

R4.  CLASS-BAR LANGUAGE IN THE SWEEP SUMMARY WAS TOO SOFT.  Describing the
     isoneutral-slope residual as "grow about 10x (still 1e-10)" and the
     hydrostatic v-component as a small mover understates them: 1e-10 is 100x
     the ACCUMULATING bar and 2e-5 is ten orders over the POINTWISE bar.
     Neither was given the operand-identity + growth-nullity proof the
     irreducibility rule requires, so neither may be called at-bar.

R5.  THE off-ARM NON-REPRODUCTION IS BROADER THAN "0.21 Sv OF ACC".  This
     tree's shipped-grid arm scores PASS 2 / FAIL 3, where the recorded
     baseline scores PASS 4 / FAIL 1 at the same 5x level.  Density metrics
     moved too, so whatever differs is not ACC-only.

RECORDED RESULT (2026-08-20) -- the vertical-ladder arms
--------------------------------------------------------
Phase 1c showed the four-step ACC divergence at the twin's own starting point
is 43x larger under the shipped 1-D thickness ladder than under NEMO's own, so
the four ladder modes were run out to 90 days.  These are NOT run by this
module (they need no new code -- they are the existing twin runner and the
existing acceptance gate, one environment variable apart), so the exact
commands are recorded here instead:

    for E in off e3t_only gdept_only both; do
      CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=$E \
        python kamm_twin_90d.py nemo_dino_kamm_mlf out_$E.npz \
        --days 90 --bridge-before --save-3d
      CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=$E \
        python acceptance_gate_90d.py out_$E.npz --level 5
    done

All four arms are bit-identical at day 0 (same land mask, max|d u3d| =
max|d T3d| = 0.0) and all four report STABLE over 2880 steps.

  ladders (thickness, T-depth)   ACC gap vs NEMO d30/d60/d90   ACC gate
  1-D,  1-D    (SHIPPED DEFAULT)   +1.619  +1.525  +1.659       FAIL
  NEMO, 1-D                        +1.030  +0.739  +0.865       FAIL
  1-D,  NEMO                       +0.272  +0.220  +0.164       PASS
  NEMO, NEMO                       -0.308  -0.557  -0.516       FAIL
  (gate threshold 0.455 Sv; ACC noise floor 0.091 Sv; NEMO d90 = 65.369 Sv)

NEMO's own T-point depths with the thickness ladder left alone cut the ACC gap
by 90% and are the only one of the four arms that passes the ACC metric.  The
deep density contrast improves 14x on that arm as well (3.98e-06 -> 2.87e-07).
The surface density metrics do not improve on any arm.

The two ladders are IDENTICAL above ~1000 m and differ only over levels 25-34,
by up to 14.8% in thickness (of e3t_1d; 12.9% of e3t_0) and 3.9% in T-point
depth -- a DEEP geometry difference, and
therefore a different thing from the top-two-level shear divergence Phase 1b
measures.

RETRACTION (2026-08-20, same day, before anything was built on it).  The line
above -- "NEMO's own T-point depths ... are the only one of the four arms that
passes the ACC metric" -- is TRUE OF THE RECORDED METRIC AND MISLEADING AS
PHYSICS.  ``acc_full`` integrates the zonal transport across ALL 199 rows of the
section, including the CLOSED-BASIN latitudes where gyre recirculation lives.
``acc_thermal_wind.acc_band`` restricts the same integral to the re-entrant
channel (rows 14..48, the rows that are wet at every longitude) and weights with
the true partial-cell thickness -- i.e. it is the circumpolar transport.  The
two rank the four arms almost OPPOSITELY:

  arm          acc_full gap   acc_band gap   baroclinic gap   barotropic gap
  1-D,  1-D          +1.659         +3.483           +0.490          +2.717
  NEMO, 1-D          +0.865         +0.690           +0.582          -0.005
  1-D,  NEMO         +0.164         +2.893           +0.097          +2.775
  NEMO, NEMO         -0.516         +0.099           +0.076          +0.058

The depth-only arm wins the recorded metric while leaving a +2.775 Sv
BAROTROPIC excess in the actual channel.  On the channel band it is NEMO's BOTH
ladders that close the gap -- +0.099 Sv against the shipped +3.483 Sv, a 97%
cut, and the only arm where both components are near zero.

The two halves fix DIFFERENT components, which is why neither alone is the
answer: the THICKNESS ladder fixes the BAROTROPIC (bottom-referenced) transport
(+2.717 -> -0.005) and the DEPTH ladder fixes the BAROCLINIC (thermal-wind)
transport (+0.490 -> +0.097).  Only the depth half has the equation-of-state
path traced below; the thickness half's barotropic path is NOT traced here and
is OPEN.

WHAT SURVIVES BOTH METRICS: the SHIPPED ladder pair is the WORST of the four on
both, and the vertical geometry dominates the gap on both.  What does NOT
survive is any claim that one half alone is the fix.

Why the recorded metric rewards an arm whose channel transport is wrong is
UNEXPLAINED and is the next thing to settle; do not pick a ladder on the
strength of ``acc_full`` alone.

METRIC-WEIGHT ROBUSTNESS (the obvious attack on the table above: the recorded
ACC metric weights every arm with e3t_1d, which is the model's OWN thickness
only on the two 1-D-thickness arms).  Re-scored with NEMO's true partial-cell
e3t_0 as the vertical weight, applied identically to both sides:

  arm            gap (e3t_1d weight)   gap (e3t_0 weight)
  1-D,  1-D               +1.659               +1.666
  NEMO, 1-D               +0.865               +0.787
  1-D,  NEMO              +0.164               +0.118
  NEMO, NEMO              -0.516               -0.638

The ranking and the magnitudes survive; the depth-ladder arm stays inside the
gate threshold under both weights.  The result is not an artifact of the
metric's vertical weight.

MECHANISM PATH, traced in code rather than inferred: the bridge's T-depth
ladder becomes ``z_coord.t_depth_ref`` (nemo_state_bridge.py:174,228,275,535),
which is passed as ``eos_geometric_depth_1d`` into the density/pressure
computation (ocean_pe_latlon_cgrid.py:1279-1299).  So NEMO's own T-depths
change the pressure depth the equation of state sees, hence in-situ density,
hence the thermal-wind shear the transport integrates.  That the deep density
contrast improves 14x on exactly that arm is consistent with this path; it is
not by itself proof that no other path contributes.

CAVEATS, both real:
  * my 1-D/1-D arm gives +1.659 Sv where the recorded #1455 baseline gives
    +1.872335 Sv.
    RESOLVED 2026-08-21: it was PRECISION, and the four arms above are an fp32
    measurement.  kamm_twin_90d.py sets no precision policy, so it inherits the
    fp32 control dtype unless it is wrapped in run_fp64.py -- the recorded
    baseline was, and these arms were not (their logs carry the bridge's
    NON-fp64 warning).  Re-running the shipped-grid arm at HEAD UNDER fp64
    returns a state BIT-IDENTICAL to that baseline in EVERY SAVED FIELD
    (max|d u3d| at days 30 and 90, max|d T3d| and max|d eta3d| at day 90 all
    exactly 0.0; artifacts /tmp/dino_clock_baseline/twin90_corrected.npz and
    /tmp/dino_fp64_arms/twin90_off_fp64.npz, logs alongside each) and an ACC gap
    of +1.8723.  So the model diff between the two commits is inert ON THOSE
    FIELDS -- which is what this comparison scores -- and the precision policy
    owns 100% of the 0.21 Sv.
    The "real and UNEXPLAINED" above is WITHDRAWN.
    Consequence for the table above: every number in it is fp32.  The two arms
    re-measured at fp64 both moved -- acc_full +1.659 -> +1.8723 ("off") and
    -0.516 -> -0.5965 ("both"), channel band +3.483 -> +2.9319 and +0.099 ->
    +0.2874.  The ranking, the signs, and the shipped-to-NEMO-ladder channel-band
    cut (+2.9319 -> +0.2874, 90%) all survive; the reading that the "both"
    channel band sits AT the 0.091 Sv floor does not -- at fp64 it is 3.2x that
    floor.  The two half-ladder arms have NOT been re-run, so anything resting
    on them -- including which half fixes which transport component -- is still
    an fp32 result.
  * nemo_state_bridge carries a comment stating legoESM is unstable on NEMO's
    ladders (max|u| 0.66 -> 2.2 m/s in 20 days) and citing that as the reason
    the wrong ladder is the default.  It did not reproduce here: day-90 max|u|
    is 0.6332 / 0.6341 / 0.6344 / 0.6347 across the four arms.  That comment
    does not record the state or configuration it was measured on, so this is a
    non-reproduction under ONE configuration, not proof it was never true.

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
    ladder mode (#1226's analytic-vs-true-ladder trap: up to 70.4 m at the
    deepest wet level, 12.9% of e3t_0 / 14.8% of e3t_1d) and the seasonal
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


def _require_clean(proc, arm: str) -> None:
    """A non-zero exit is FATAL even if the run already printed results.

    Without this a subprocess that emits its JSON and then dies -- on a
    teardown assertion, an OOM, a late guard -- reports clean numbers, which
    is the failure mode the campaign's own tool-status rule exists to stop.
    """
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-4000:])
        raise SystemExit(
            f"arm {arm!r} exited {proc.returncode}: refusing to report its "
            "numbers (see the tail above)")


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
    _require_clean(out, arm)
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


# ---------------------------------------------------------------------------
# PHASE 1b -- WHERE the one-step velocity divergence lives
# ---------------------------------------------------------------------------
def phase1b(arm: str = "d180", *, stress_implicit: bool | None = None) -> int:
    """Split the ONE-step u/v divergence into its depth-mean (barotropic) and
    shear (baroclinic) parts, and profile it by level.

    Phase 1 reports that ONE step from a bit-exact bridged state already
    differs by ~1e-2 m/s, while every per-term momentum row in the #1226 sweep
    sits within ~1e-5 of NEMO.  Those two facts cannot both be about the same
    thing, so this asks WHICH PART of the velocity carries the difference:

      * depth-mean-dominated  -> the barotropic solve / its reconciliation is
        the carrier, and it is the part the ACC (a depth-INTEGRATED transport)
        actually reads;
      * shear-dominated       -> the implicit vertical solve or the surface
        stress placement;
      * surface-spike         -> the wind entry;
      * bottom-spike          -> the drag.

    The split is exact by construction (u = ubar + u').  Both sides are
    weighted by the SAME thickness array -- legoESM's own ``H_bathy``/
    ``z_coord`` evaluated at NEMO's restart ssh, NOT NEMO's own e3t -- so the
    weighting cannot differ BETWEEN the two sides, which is all the split
    needs; it is not NEMO's thickness field, and the earlier wording saying so
    was wrong.  The reduction is stated next to every number.
    """
    ic, run_dir = _ARMS[arm]
    env = dict(os.environ)
    env["DINO_1226_IC_STEP"] = str(ic)
    env["DINO_NEMO_RUN_TWIN_STEP1"] = run_dir
    if stress_implicit is not None:
        env["DINO_D180_STRESS_IMPLICIT"] = "1" if stress_implicit else "0"
    else:
        env.pop("DINO_D180_STRESS_IMPLICIT", None)
    code = r"""
import json, os, sys, numpy as np
sys.path.insert(0, %r)
import multistep_replay as m
from legoesm.ocean.vertical import compute_layer_thickness
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.experiments.dino import (
    apply_dino_lat_lon_surface_forcing, dino_lat_lon_model_config,
    dino_lat_lon_surface_forcing_arrays, dino_step_surface_forcing)
m.provenance('phase1b')
g, br, cfg, st = m.build_replay_ic()
_si = os.environ.get('DINO_D180_STRESS_IMPLICIT')
if _si is not None:
    # PRE-REGISTERED ONE-VARIABLE A/B.  surface_stress_implicit=False (the
    # card's default) gives the wind an explicit per-step surface kick;
    # =True deposits it in the top cell of the implicit vertical solve's RHS,
    # which is where NEMO's dynzdf puts it.  Nothing else changes.
    import dataclasses as _dc
    cfg = _dc.replace(cfg, surface_stress_implicit=(_si == '1'))
    print('ABLATION: surface_stress_implicit=' + str(cfg.surface_stress_implicit))
mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
_oi = os.environ.get('DINO_OUTER_INTEGRATOR', '')
if _oi:
    # STEP-COMPOSITION ablation.  NOT one variable, and NOT a test of
    # legoESM-vs-NEMO step composition -- both corrections are from the
    # 2026-08-20 adversarial review and are recorded here so the next reader
    # does not repeat the error:
    #   (a) HISTORICAL (kept for the record): outer_integrator='nemo_mlf'
    #       used to ALSO force a divisor flag the DINO card did not set, so
    #       the arm moved two fields at once.  That flag is gone -- NEMO's
    #       e3w(Kmm) divisor now belongs to the NEMO implicit-ZDF identity,
    #       which this card already selects, so the arm is one variable now.
    #   (b) _nemo_mlf_step's own docstring says it is IDENTICAL to
    #       _leapfrog_step except for how the dissipation pass is composed,
    #       and its verification rung is a bit-comparison against it.  Two
    #       legoESM implementations of the same composition agreeing tells us
    #       nothing about whether that composition matches NEMO's.
    if _oi not in ('leapfrog', 'nemo_mlf'):
        raise SystemExit('Unknown DINO_OUTER_INTEGRATOR=' + repr(_oi))
    mc = mc._replace(outer_integrator=_oi,
                     zdf_implicit_solver_evaluation=(
                         'nemo_literal' if _oi == 'nemo_mlf'
                         else mc.zdf_implicit_solver_evaluation))
    print('ABLATION: outer_integrator=' + str(mc.outer_integrator)
          + ' zdf_implicit_solver_evaluation='
          + str(mc.zdf_implicit_solver_evaluation))
model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
sf = dino_step_surface_forcing(forcing) if bool(getattr(cfg,'wind_through_step',False)) else None
dt = 2700.0
kt = m.IC_STEP + 1
placement = getattr(cfg, 'surface_tendency_placement', 'applied_now')
ext = None
if placement == 'leapfrog_rhs':
    st, ext = apply_dino_lat_lon_surface_forcing(st, forcing, br.z_coord, cfg, dt,
                                                 t_seconds=kt*dt, return_rate=True)
else:
    st = apply_dino_lat_lon_surface_forcing(st, forcing, br.z_coord, cfg, dt,
                                            t_seconds=kt*dt)
st = model.step(st, dt, surface_forcing=sf, external_tracer_rate=ext)
ns = m.nemo_now_state_at(kt)
umask = np.asarray(g.umask) > 0.5
vmask = np.asarray(g.vmask) > 0.5
# SAME thickness on both sides: NEMO's own ssh at this step.  Only the
# velocity differs, so the depth-mean split cannot be contaminated by a
# thickness difference.
e3 = np.asarray(compute_layer_thickness(ns.ssh, st.H_bathy.data, br.z_coord,
                                        min_water_column_m=mc.min_water_column_m))
out = {}
for tag, lego, nemo, msk in (
        ('u', np.asarray(st.u.data)[:, 1:, :], ns.u, umask),
        ('v', np.asarray(st.v.data)[1:, :, :], ns.v, vmask)):
    d = np.where(msk, lego - nemo, 0.0)
    h = np.where(msk, e3, 0.0)
    H = h.sum(axis=-1)
    dbar = np.divide(( d * h ).sum(axis=-1), H, out=np.zeros_like(H), where=H > 0)
    dshear = np.where(msk, d - dbar[..., None], 0.0)
    wet2 = H > 0
    out[tag] = dict(
        max_total=float(np.abs(d[msk]).max()),
        max_depthmean=float(np.abs(dbar[wet2]).max()),
        max_shear=float(np.abs(dshear[msk]).max()),
        rms_total=float(np.sqrt((d[msk]**2).mean())),
        rms_depthmean=float(np.sqrt((dbar[wet2]**2).mean())),
        rms_shear=float(np.sqrt((dshear[msk]**2).mean())),
        per_level_rms=[float(np.sqrt((d[..., k][msk[..., k]]**2).mean()))
                       if msk[..., k].any() else 0.0
                       for k in range(d.shape[-1])],
        argmax=[int(x) for x in np.unravel_index(
            int(np.argmax(np.where(msk, np.abs(d), -np.inf))), d.shape)],
        max_abs_state=float(np.abs(nemo[msk]).max()),
    )
print('@@JSON@@' + json.dumps(out))
""" % (_THIS_DIR,)
    proc = subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True)
    _require_clean(proc, arm)
    hit = [l for l in proc.stdout.splitlines() if l.startswith("@@JSON@@")]
    for l in proc.stdout.splitlines():
        if not l.startswith("@@JSON@@"):
            print(f"  [{arm}] {l}")
    if not hit:
        print(proc.stderr[-4000:])
        raise SystemExit("phase1b produced no result")
    import json
    res = json.loads(hit[-1][len("@@JSON@@"):])
    print(f"\nPHASE 1b  arm={arm}: ONE step, max/rms over WET FACES of "
          f"(legoESM - NEMO), split u = depth-mean + shear on NEMO's own e3")
    print("  comp  reduction        total    depth-mean         shear   "
          "depth-mean share of rms")
    for tag in ("u", "v"):
        r = res[tag]
        for red in ("max", "rms"):
            share = (r[f"{red}_depthmean"] / r[f"{red}_total"]
                     if r[f"{red}_total"] else float("nan"))
            print(f"  {tag}     {red:<12s}{r[f'{red}_total']:13.4e}"
                  f"{r[f'{red}_depthmean']:14.4e}{r[f'{red}_shear']:14.4e}"
                  f"{share:14.3f}")
        print(f"        argmax(j,i,k)={tuple(r['argmax'])}  "
              f"max|NEMO {tag}|={r['max_abs_state']:.4f} m/s")
        pl = r["per_level_rms"]
        top = sorted(range(len(pl)), key=lambda k: -pl[k])[:5]
        print("        per-level rms, 5 largest levels: "
              + ", ".join(f"k={k}:{pl[k]:.3e}" for k in top))
    return 0


# ---------------------------------------------------------------------------
# PHASE 1c -- does the per-step divergence build the ACC excess?
# ---------------------------------------------------------------------------
def phase1c(arm: str = "d180", n_steps: int = 4) -> int:
    """Per-step ACC of legoESM and of NEMO, from the same starting state.

    The 90-day gap being chased is +1.87 Sv of ACC, already fully present at
    day 30 (960 steps) and flat thereafter.  If the step-by-step divergence
    Phase 1 measures is what builds it, the ACC difference has to open at a
    rate that reaches ~1.87 Sv inside those 960 steps -- roughly 2e-3 Sv per
    step on average.  A per-step difference orders below that, or one that
    oscillates in sign, does NOT build the gap and points at an equilibration
    response instead.  This measures the rate directly rather than inferring
    it.

    ONE metric, imported not copied: ``acc_thermal_wind.acc_full`` (the
    recorded campaign metric -- full-section zonal transport, e3t_1d
    weighting, median over longitudes 2..-2), applied to BOTH sides with the
    SAME wet-u mask, so the only difference between the two numbers is the
    velocity field.
    """
    ic, run_dir = _ARMS[arm]
    env = dict(os.environ)
    env["DINO_1226_IC_STEP"] = str(ic)
    env["DINO_NEMO_RUN_TWIN_STEP1"] = run_dir
    code = r"""
import json, sys, numpy as np
sys.path.insert(0, PROBE_DIR)
import multistep_replay as m
import acc_thermal_wind as A
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.experiments.dino import (
    apply_dino_lat_lon_surface_forcing, dino_lat_lon_model_config,
    dino_lat_lon_surface_forcing_arrays, dino_step_surface_forcing)
m.provenance('phase1c')
g, br, cfg, st = m.build_replay_ic()
mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
sf = dino_step_surface_forcing(forcing) if bool(getattr(cfg,'wind_through_step',False)) else None
dt = 2700.0
placement = getattr(cfg, 'surface_tendency_placement', 'applied_now')
umask = np.asarray(g.umask) > 0.5


def lego_u(state):
    # (nlat, nlon+1, nz) u-faces -> NEMO's 52 u-columns, the SAME slice
    # multistep_replay uses for its own umask comparison.
    return np.asarray(state.u.data)[:, 1:, :]


rows = []
u0 = lego_u(st)
# Day-0 row: BOTH sides are the same array, so its gap is 0.0 BY
# CONSTRUCTION and proves nothing.  It is printed only as the common
# starting value.  The real day-0 gate is verify_day0_matches_restart
# inside build_replay_ic, which is what asserts the bridge equals NEMO.
rows.append(dict(kt=m.IC_STEP, acc_lego=A.acc_full(u0, umask),
                 acc_nemo=float("nan")))
for k in range(1, N_STEPS + 1):
    kt = m.IC_STEP + k
    ext = None
    if placement == 'leapfrog_rhs':
        st, ext = apply_dino_lat_lon_surface_forcing(st, forcing, br.z_coord, cfg,
                                                     dt, t_seconds=kt*dt,
                                                     return_rate=True)
    else:
        st = apply_dino_lat_lon_surface_forcing(st, forcing, br.z_coord, cfg, dt,
                                                t_seconds=kt*dt)
    st = model.step(st, dt, surface_forcing=sf, external_tracer_rate=ext)
    ns = m.nemo_now_state_at(kt)
    ul = lego_u(st)
    if not np.isfinite(ul).all() or not np.isfinite(ns.u).all():
        raise SystemExit('non-finite velocity at kt=%d' % kt)
    rows.append(dict(kt=kt, acc_lego=A.acc_full(ul, umask),
                     acc_nemo=A.acc_full(ns.u, umask)))
# per-level relative size of the final-step difference
d = np.where(umask, ul - ns.u, 0.0)
lev = []
for kk in range(d.shape[-1]):
    msk = umask[..., kk]
    if not msk.any():
        lev.append((0.0, 0.0)); continue
    lev.append((float(np.sqrt((d[..., kk][msk]**2).mean())),
                float(np.sqrt((ns.u[..., kk][msk]**2).mean()))))
print('@@JSON@@' + json.dumps(dict(rows=rows, per_level=lev)))
"""
    code = code.replace("PROBE_DIR", repr(_THIS_DIR)).replace(
        "N_STEPS", str(n_steps))
    proc = subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True)
    _require_clean(proc, arm)
    hit = [l for l in proc.stdout.splitlines() if l.startswith("@@JSON@@")]
    for l in proc.stdout.splitlines():
        if not l.startswith("@@JSON@@") and "provenance" not in l:
            print(f"  [{arm}] {l}")
    if not hit:
        print(proc.stderr[-4000:])
        raise SystemExit("phase1c produced no result")
    import json
    res = json.loads(hit[-1][len("@@JSON@@"):])
    print(f"\nPHASE 1c  arm={arm}: ACC [Sv], acc_thermal_wind.acc_full, "
          f"same wet-u mask on both sides")
    print("     kt      ACC legoESM      ACC NEMO        gap [Sv]   "
          "gap step-to-step")
    prev = None
    for r in res["rows"]:
        if not np.isfinite(r["acc_nemo"]):
            print(f"  {r['kt']:7d}{r['acc_lego']:16.6f}"
                  f"{'(same array)':>15}{'':>16}   <- start, not a gate")
            continue
        gap = r["acc_lego"] - r["acc_nemo"]
        d = "" if prev is None else f"{gap - prev:+16.6e}"
        print(f"  {r['kt']:7d}{r['acc_lego']:16.6f}{r['acc_nemo']:15.6f}"
              f"{gap:+16.6e}{d}")
        prev = gap
    print("\n  per-level rms after the last step: "
          "diff vs NEMO's own u, and the ratio")
    print("     k        rms|d_u|      rms|u NEMO|    relative")
    for k, (dr, nr) in enumerate(res["per_level"]):
        if nr <= 0:
            continue
        if k < 6 or k % 6 == 0:
            print(f"  {k:4d}{dr:16.4e}{nr:16.4e}{dr/nr:12.4f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", default="0", choices=["0", "1", "1b", "1c"],
                    help="0 = NEMO-side control, 1 = free-running walk")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--arms", default="d180,y20")
    ap.add_argument("--stress-implicit", default=None,
                    choices=["0", "1"],
                    help="phase 1b only: force DINOConfig.surface_stress_implicit "
                         "(one-variable A/B against the card default)")
    args = ap.parse_args(argv)
    provenance()
    if args.phase == "0":
        return phase0()
    if args.phase == "1c":
        rc = 0
        for arm in args.arms.split(","):
            rc |= phase1c(arm, n_steps=args.steps)
        return rc
    if args.phase == "1b":
        rc = 0
        si = None if args.stress_implicit is None else (args.stress_implicit == "1")
        for arm in args.arms.split(","):
            rc |= phase1b(arm, stress_implicit=si)
        return rc
    return phase1(n_steps=args.steps, arms=tuple(args.arms.split(",")))


if __name__ == "__main__":
    raise SystemExit(main())
