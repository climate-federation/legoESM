"""#1492 item 2.2: the 90-DAY TWIN ACCEPTANCE GATE.

Compares a candidate legoESM configuration's 90-day twin (state-initialized
from NEMO's developed day-180 restart via ``kamm_twin_90d.py --save-3d``)
against NEMO's OWN day-90 state from the SAME restart, on 4 metrics, and
classifies each PASS/FAIL against staged thresholds derived from the #1492
2.1 ensemble noise floor.

Metrics (validated 2026-08-06, S/N 14-22000 at day 90 -- see #1492 2.2
comment / commit 21a7457cf; all imported from the recorded harnesses, none
re-derived here):
  * ACC [Sv]              -- acc_thermal_wind.acc_full (the compare_fullframe
                             formula: full-section zonal transport, e3t_1d
                             weighting, median over longitudes 2..-2)
  * upper contrast        -- band meridional density contrast, thickness-
  * deep  contrast [kg/m3]   weighted above/below the 1400 m split
                             (acc_thermal_wind.contrast_profile+depth_split)
  * southern-band surface sigma MAX and MEAN [kg/m3] -- the mean had the
                             best S/N (22000 at day 90)

The density-class CENSUS is EXCLUDED from this gate: its 90-day (and even
1-year) signal is exactly zero -- dense classes are gap-protected/empty at
this branch state -- so it remains a multi-year campaign metric only
(#1492 2.2 comment).

Thresholds: the 2.1 micro-ensemble noise floor (3 members, 1e-14 T
perturbations -- n=3 caveat inherited: the floor is a small-sample estimate),
staged 5x / 2x / 1x.  Default gate level 5x; select with --level.
  floor: ACC 0.091 Sv | upper 1.1e-4 | deep 4.5e-5 | surface sigma 9.5e-5

Baseline: NEMO's own 90-day continuation from the day-180 restart
(``DINO_00005760_restart``, kt 5760): ``RUN_90D_TWIN/DINO_00008640_restart_*``
(kt 5760 + 90 d x 32 steps/d = 8640; 16 per-rank tiles, stitched with
``rebuild_nemo_restart.rebuild``, the same stitcher deep_box_heat_budget.py /
multistep_replay.py use).  The restart's NOW time level (tn/sn/un) is used --
the same registry level ``kamm_twin_90d.py`` snapshots from the legoESM state
(``st.T.data`` etc.).  The metric protocol uses mesh_mask REFERENCE geometry
(e3t_1d for ACC, e3t_0 elsewhere) on BOTH sides, per the recorded
acc_thermal_wind protocol, so no restart e3t(t) handling is needed (both
restart e3t time levels exist but neither enters these metrics).
``RUN_TWIN_Y20_BUDGET/`` is a DIFFERENT run (y20-branch budget dumps) -- not
a valid baseline for this gate.

Instrument self-checks (fatal, run before any comparison, verbatim from the
recorded 2.2 protocol): NEMO y10 ACC through this harness must reproduce
121.07 Sv; the band volume must reproduce 2.694775e16 m3.

All metric arithmetic is float64 (twin snapshots are stored float32; cast up
on load -- the fp32 quantum is orders below every floor).

Usage
-----
  # gate an existing twin npz (kamm_twin_90d.py --save-3d output):
  acceptance_gate_90d.py CANDIDATE.npz [--level 5]

  # produce the twin first (subprocess kamm_twin_90d.py), then gate it:
  acceptance_gate_90d.py CANDIDATE.npz --run-recipe nemo_dino_kamm_mlf \
      --surface-tendency-placement leapfrog_rhs [--level 5]

  # non-vacuity: prove the gate CAN fail (synthetic violation, no twin needed)
  acceptance_gate_90d.py --self-test

Self-test scope: it exercises the metric/classification/threshold path on the
NEMO baseline (identical pair PASSes, perturbed fields FAIL); it does NOT
exercise ``load_candidate`` (the twin-npz reader) -- that path is covered by
the real gate runs on candidate npz files.

Exit status: 0 = all metrics PASS (or self-test passed), 1 = any FAIL.
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)                       # acc_thermal_wind (sibling)
sys.path.insert(0, os.path.dirname(_DIR))      # rebuild_nemo_restart (parent)
import acc_thermal_wind as A  # noqa: E402  (recorded harness, imported not copied)
import kamm_twin_90d as _twin  # noqa: E402  (one shared seasonal-clock guard)

DINO = A.DINO
RUN_90D_TWIN = os.environ.get("DINO_NEMO_RUN_90D_TWIN", f"{DINO}/RUN_90D_TWIN")
N20_T = f"{DINO}/RUN_20Y_REBUILD/DINO_1y_00060101_00201230_grid_T.nc"
N20_U = f"{DINO}/RUN_20Y_REBUILD/DINO_1y_00060101_00201230_grid_U.nc"

KT_RESTART = 5760                    # DINO_00005760_restart = day-180 twin IC
STEPS_PER_DAY = 32                   # 32 x 2700 s (kamm_twin_90d.py protocol)
KT_DAY90 = KT_RESTART + 90 * STEPS_PER_DAY          # = 8640

BAND_VOLUME_RECORDED = 2.694775e16   # m3 (recorded 2.2-protocol control)
ACC_Y10_RECORDED = 121.07            # Sv  (recorded harness control)

# 2.1 micro-ensemble noise floor (#1492; n=3 -- small-sample caveat).
FLOORS = {"acc": 0.091, "up": 1.1e-4, "deep": 4.5e-5,
          "smax": 9.5e-5, "smean": 9.5e-5}
LABELS = {"acc": "ACC [Sv]", "up": "upper contrast <1400m [kg/m3]",
          "deep": "deep contrast >1400m [kg/m3]",
          "smax": "S-band surface sigma MAX [kg/m3]",
          "smean": "S-band surface sigma MEAN [kg/m3]"}
KEYS = ("acc", "up", "deep", "smax", "smean")


# ------------------------------------------------------------------ metrics ---
# Thin composition of the recorded acc_thermal_wind harness functions -- the
# same bundle the validated 2.2 S/N probe used (numerics all live in A.*).
def surface_sigma_south(st, wet):
    jmid = (A.J0 + A.J1) // 2
    s0 = (A.rho_of(st, wet) - A.RHO0)[A.J0:jmid + 1, :, 0]
    s0 = s0[np.isfinite(s0)]
    return float(s0.max()), float(s0.mean())


def metrics(st, wet):
    up, deep = A.depth_split(A.contrast_profile(A.rho_of(st, wet), wet), wet)
    smax, smean = surface_sigma_south(st, wet)
    return {"acc": A.acc_full(st["u"], A.umask), "up": up, "deep": deep,
            "smax": smax, "smean": smean}


# ------------------------------------------------------------------ loaders ---
def load_candidate(path, day=90):
    """{"T","S","u","land_mask"} (fp64) from a kamm_twin_90d --save-3d npz."""
    d = np.load(path)
    # #1455: the twin harness stamps WHICH vertical ladders the bridge handed
    # legoESM (nemo_ladder_mode). PRINT it, never refuse on it -- both ladders
    # are legitimately scoreable and the gate's job is to say which grid a score
    # was earned on, not to pick one. An artifact written before the stamp
    # existed does NOT record its grid at all -- it may have been run with
    # LEGOESM_NEMO_E3T set to anything -- so it is reported as unknown, never
    # guessed at from whatever the default was on the day.
    # It prints BEFORE the seasonal-clock refusal below, so a candidate that
    # is refused still records which grid it ran on.
    ladder = (str(d["nemo_ladder_mode"]) if "nemo_ladder_mode" in d.files
              else "UNSTAMPED -- this artifact predates the nemo_ladder_mode "
                   "stamp and does not record which vertical ladders it ran "
                   "on; read its run log")
    print(f"vertical ladder of this candidate: {ladder}", flush=True)
    # #1640 (GLM): the LABEL above is a taxonomy; this is the IDENTITY. Two
    # runs with the same label and different numbers get different hashes, so
    # "same grid" is decidable instead of asserted.
    lhash = (str(d["vertical_ladder_sha256"])
             if "vertical_ladder_sha256" in d.files
             else "UNSTAMPED -- predates the content hash; its ladder ARRAYS "
                  "cannot be compared to any other run's")
    print(f"vertical ladder content hash: {lhash}", flush=True)
    # #1455 season-bug guard (extend-only): NEMO's analytic surface forcing is
    # a function of the day of year through the absolute step index
    # (usrdef_sbc.F90:536), and the day-180 restart carries adatrj=180.0, so a
    # candidate forced on any other day of the year is a cross-season confound
    # rather than a fidelity measurement. The guard lives in kamm_twin_90d so
    # every scorer applies the same one; it refuses an artifact that carries no
    # clock stamps and one whose clock is not the restart's own.
    import os as _os
    if _os.environ.get("DINO_GATE_ALLOW_LEGACY_CLOCK") != "1":
        _twin.assert_nemo_seasonal_clock(d, path)

    key = f"u3d_day{day}"
    if key not in d:
        raise SystemExit(f"{path} has no {key} -- run kamm_twin_90d.py with "
                         f"--save-3d --days >= {day}")
    lU = d[key].astype(np.float64)          # (199,53,36) u-faces, full depth
    u = lU[:, 1:53, :].copy()               # faces 1..52 <-> NEMO u-cols
    u[:, 47, :] = lU[:, 48, :]              # acc_thermal_wind.load_lego, verbatim
    return {"T": d[f"T3d_day{day}"].astype(np.float64),
            "S": d[f"S3d_day{day}"].astype(np.float64),
            "u": u, "land_mask": d["land_mask"].astype(np.float64)}


def load_nemo_day90(run_dir=RUN_90D_TWIN, kt=KT_DAY90):
    """NEMO's own day-90 state (NOW level tn/sn/un) from the stitched restart."""
    from rebuild_nemo_restart import rebuild
    pattern = f"{run_dir}/DINO_{kt:08d}_restart_*.nc"
    if not glob.glob(pattern):
        raise SystemExit(f"NEMO day-90 baseline not found: {pattern}")
    raw = rebuild(pattern, ["tn", "sn", "un"])   # (lev,y,x) fp64, NaN on land
    yxz = lambda a: np.moveaxis(a, 0, -1)                        # noqa: E731
    return {"T": yxz(raw["tn"]), "S": yxz(raw["sn"]), "u": yxz(raw["un"])}


# -------------------------------------------------------------- self-checks ---
def instrument_self_checks(wet):
    """Fatal harness controls, verbatim from the recorded 2.2 protocol."""
    acc10 = A.acc_full(A.load_nemo(N20_T, 4, N20_U)["u"], A.umask)
    print(f"[self-check 1] NEMO y10 ACC through THIS harness: {acc10:.2f} Sv "
          f"vs recorded {ACC_Y10_RECORDED} Sv")
    if abs(acc10 - ACC_Y10_RECORDED) > 0.5:
        raise SystemExit("HARNESS FAILS ITS OWN ACC CONTROL")
    import netCDF4 as nc
    mm = nc.Dataset(f"{DINO}/RUN_TRAJ/mesh_mask.nc")
    e1t = np.asarray(mm["e1t"][0]).squeeze()
    e2t = np.asarray(mm["e2t"][0]).squeeze()
    band = slice(A.J0, A.J1 + 1)
    cell_vol = (e1t * e2t)[:, :, None] * A.e3t0
    vol = float(np.sum(np.where(wet[band], cell_vol[band], 0.0)))
    print(f"[self-check 2] band volume: {vol:.6e} m3 vs recorded "
          f"{BAND_VOLUME_RECORDED:.6e} "
          f"(rel {abs(vol - BAND_VOLUME_RECORDED) / BAND_VOLUME_RECORDED:.2e})")
    if abs(vol - BAND_VOLUME_RECORDED) / BAND_VOLUME_RECORDED > 1e-3:
        raise SystemExit("HARNESS FAILS BAND-VOLUME CONTROL")
    print("[self-checks PASS]\n")


# --------------------------------------------------------------------- gate ---
def classify(cand_m, nemo_m, level):
    """[(key, cand, nemo, |diff|, threshold, passed)] at level x the floor."""
    return [(k, cand_m[k], nemo_m[k], abs(cand_m[k] - nemo_m[k]),
             level * FLOORS[k], abs(cand_m[k] - nemo_m[k]) <= level * FLOORS[k])
            for k in KEYS]


def print_gate(rows, level, tag="", certified=True):
    """Print the metric table, and the PASS/FAIL tally ONLY when certified.

    ``certified=False`` (#1640) prints the same numbers but withholds every
    verdict token -- the per-row ``PASS``/``FAIL`` flags AND the tally line.
    A row status is a verdict, so suppressing only the tally would still have
    issued one five times over.  The scores stay visible because an off-claim
    candidate is still worth scoring; what it cannot buy is a certification.
    """
    print(f"{'metric':<34}{'candidate':>14}{'NEMO d90':>14}{'|diff|':>12}"
          f"{'thresh':>12}  status")
    for k, c, n, d, t, ok in rows:
        if certified:
            flag = f"PASS({level}x)" if ok else "FAIL  <--"
        else:
            # NOT a verdict: how the number sits against a threshold that does
            # not apply to this candidate's grid/precision.
            flag = f"(within {level}x)" if ok else f"(over {level}x)"
        print(f"{LABELS[k]:<34}{c:>14.6f}{n:>14.6f}{d:>12.3e}{t:>12.3e}  {flag}")
    n_pass = sum(ok for *_, ok in rows)
    n_fail = len(rows) - n_pass
    if not certified:
        return n_fail
    # fidelity_bar_gate.py tally-line convention (matched in style, not imported)
    print(f"\n{tag}GATE 90D-TWIN: PASS {n_pass} | FAIL {n_fail} | "
          f"level {level}x | total {len(rows)}")
    return n_fail


def self_test(level):
    """Non-vacuity: the gate MUST pass an identical pair and MUST fail a
    synthetic violation (candidate fields perturbed by >>10x every threshold)."""
    wet = A.tmask                      # band rows are fully wet: == tmask&land
    instrument_self_checks(wet)
    nemo = load_nemo_day90()
    nemo_m = metrics(nemo, wet)

    ident = classify(metrics(nemo, wet), nemo_m, 1)
    assert all(ok for *_, ok in ident), "identical fields must PASS at 1x"
    print("[self-test] identical candidate PASSes at 1x: OK")

    # Synthetic violation: +1 mm/s barotropic u (~10 Sv >> 10 x 5 x 0.091 Sv);
    # -0.5 K on the southern half-band, full depth (~+0.08 kg/m3 sigma >>
    # 10 x 5 x every density floor). Perturb FIELDS, not metric values.
    bad = {k: v.copy() for k, v in nemo.items()}
    bad["u"] = bad["u"] + 1e-3
    jmid = (A.J0 + A.J1) // 2
    bad["T"][A.J0:jmid + 1] -= 0.5
    rows = classify(metrics(bad, wet), nemo_m, level)
    n_fail = print_gate(rows, level, tag="[self-test synthetic violation] ")
    for k, _, _, d, t, _ in rows:
        assert d > 10.0 * t, f"{k}: perturbation {d:.3e} not >10x threshold {t:.3e}"
    assert n_fail == len(rows), "synthetic violation must FAIL every metric"
    print("SELF-TEST PASS (synthetic violation FAILs the gate; identical "
          "candidate PASSes)")
    return 0


def run_candidate_twin(candidate_path, recipe, placement, bridge_before):
    cmd = [sys.executable, f"{_DIR}/kamm_twin_90d.py", recipe, candidate_path,
           "--days", "90", "--save-3d"]
    if bridge_before:
        cmd.append("--bridge-before")
    if placement:
        cmd += ["--surface-tendency-placement", placement]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("candidate", nargs="?", default=None,
                   help="kamm_twin_90d.py --save-3d output npz to gate")
    p.add_argument("--level", type=int, default=5, choices=(5, 2, 1),
                   help="staged gate level: threshold = level x the 2.1 noise "
                        "floor (default 5)")
    p.add_argument("--run-recipe", default=None,
                   help="produce the candidate twin first by running "
                        "kamm_twin_90d.py with this recipe (90 d, --save-3d)")
    p.add_argument("--surface-tendency-placement", default=None,
                   choices=("applied_now", "leapfrog_rhs"),
                   help="forwarded to kamm_twin_90d.py with --run-recipe")
    p.add_argument("--bridge-before", default=True,
                   action=argparse.BooleanOptionalAction,
                   help="forwarded to kamm_twin_90d.py with --run-recipe "
                        "(default on: nemo_dino_kamm_mlf requires it)")
    p.add_argument("--self-test", action="store_true",
                   help="synthetic-violation non-vacuity check (no twin needed)")
    args = p.parse_args(argv)

    if args.self_test:
        return self_test(args.level)
    if args.candidate is None:
        p.error("candidate npz required (or --self-test)")
    if args.run_recipe:
        run_candidate_twin(args.candidate, args.run_recipe,
                           args.surface_tendency_placement, args.bridge_before)

    cand = load_candidate(args.candidate)
    wet = A.tmask & (cand["land_mask"] > 0.5)[:, :, None]   # SAME mask, both sides
    instrument_self_checks(wet)
    nemo = load_nemo_day90()
    print(f"candidate: {args.candidate}")
    print(f"baseline : {RUN_90D_TWIN}/DINO_{KT_DAY90:08d}_restart_* "
          f"(NEMO day-90 continuation of the day-180 restart, NOW level)\n")
    rows = classify(metrics(cand, wet), metrics(nemo, wet), args.level)

    # #1640: SCORE anything, CERTIFY only the claim's grid and precision.
    # The gate's job is to say which grid a score was earned on, so an
    # off-claim candidate is still scored and printed in full -- but the
    # PASS/FAIL verdict is WITHHELD, because a gate that will score anything
    # cannot certify a claim that depends on being on the reference's ladders
    # at double precision.
    #
    # THIS IS A BEHAVIOUR CHANGE OFF THE CLAIM'S GRID, not a no-op, and an
    # earlier comment here wrongly called it one. Any recorded protocol that
    # sweeps ladder modes and reads a verdict per arm -- notably the
    # `off/e3t_only/gdept_only/both` sweep in `d180_step_walk.py` and the
    # exit-status expectations in `PREREG_gate90_ladder_promotion.md` -- now
    # gets UNCERTIFIED (and exit 3) on its non-`both` arms. That is the
    # intended consequence of the review finding, not a regression: those arms
    # were being read as verdicts on a claim they cannot support. The metric
    # numbers in those tables are unchanged and still printed.
    #
    # Exit 3, not 2: argparse's own `p.error` exits 2, so 2 would make
    # "uncertified" indistinguishable from "bad CLI arguments".
    with np.load(args.candidate) as _stamps:
        ok, reasons, _ladder, _dtype = (
            _twin.certifiable_grid_and_precision(_stamps))
    if not ok:
        print_gate(rows, args.level, certified=False)
        print("\nGATE 90D-TWIN: UNCERTIFIED -- no PASS/FAIL verdict issued "
              "(exit 3).")
        for r in reasons:
            print(f"  - {r}")
        print("  The metric table above is a real score on the grid the "
              "candidate actually ran on; it is NOT a verdict on the claim, "
              "which is about "
              f"nemo_ladder_mode={_twin.CLAIM_LADDER_MODE!r} at "
              f"control_dtype={_twin.CLAIM_CONTROL_DTYPE!r}.")
        return 3
    n_fail = print_gate(rows, args.level)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
