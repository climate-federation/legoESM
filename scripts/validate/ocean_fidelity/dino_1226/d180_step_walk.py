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

    The split is exact by construction (u = ubar + u'), both sides use the
    SAME thickness field (NEMO's, from its own restart ssh) so the weighting
    cannot differ, and the reduction is stated next to every number.
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
    # PRE-REGISTERED ONE-VARIABLE A/B on the STEP COMPOSITION: the DINO card
    # runs the two-pass leapfrog (a full pipeline pass at Nnn, then a second
    # dissipation-only pass at Nbb); 'nemo_mlf' is the single-pass stpmlf
    # transcription NEMO itself runs.  nemo_mlf hard-requires the NEMO
    # e3w(Kmm) divisor at construction, so force it with the switch.
    if _oi not in ('leapfrog', 'nemo_mlf'):
        raise SystemExit('Unknown DINO_OUTER_INTEGRATOR=' + repr(_oi))
    mc = mc._replace(outer_integrator=_oi,
                     implicit_vmix_e3t_now_divisor=(
                         True if _oi == 'nemo_mlf'
                         else mc.implicit_vmix_e3t_now_divisor))
    print('ABLATION: outer_integrator=' + str(mc.outer_integrator)
          + ' implicit_vmix_e3t_now_divisor='
          + str(mc.implicit_vmix_e3t_now_divisor))
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", default="0", choices=["0", "1", "1b"],
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
    if args.phase == "1b":
        rc = 0
        si = None if args.stress_implicit is None else (args.stress_implicit == "1")
        for arm in args.arms.split(","):
            rc |= phase1b(arm, stress_implicit=si)
        return rc
    return phase1(n_steps=args.steps, arms=tuple(args.arms.split(",")))


if __name__ == "__main__":
    raise SystemExit(main())
