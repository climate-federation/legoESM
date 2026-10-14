#!/usr/bin/env bash
# Grid ladder G2, round-2 ACQUISITION -- OPERATOR-EXECUTED ONLY.
#
#   preflight (default):  run_r2.sh
#   acquire:              run_r2.sh --run
#   re-admit:             run_r2.sh --admit-existing
#
# ONE more run of the round-1 record binary, no build: twin c, 10 steps, with
# NEMO's own restart written after EVERY step (nn_stocklist = 1..10).  Round 2
# found that the card matches NEMO after one step on level 1 and on every 2-D
# field, and leaves the bar on the second step; what it then reads and the
# record does not hold is levels 2..30 after one step.  Consecutive restarts
# give all 30 levels at entry AND exit of every step 2..10.
#
# WHAT THE RUN GAVE (round 3): NEMO cannot write a list of consecutive
# restarts.  rst_opn opens the file of nitrst at kt == nitrst - 1
# (restart.f90:119) and the list pointer moves only after the write at
# kt == nitrst (restart.f90:188, :199-202).  With the list starting at 1 no
# step equals nitrst - 1 = 0, so nothing opens until the end-of-run fallback
# of the same line fires at kt = 10 and creates the file NAMED for step 1;
# kt = 10 is not nitrst = 1, so rst_write only defines its variables and the
# file holds NO record.  The record therefore has no all-level state at any
# odd step.  (A list of odd steps from 3 on would open: 3,5,7,9.)
#
# Admission: twin c's step streams kt = 1..10 are byte-identical to twin a's
# and its solver substep streams equal twin a's on the consumed support (a
# third run of the same trajectory); the file named for step 1 must hold no
# record and be the only restart twin c has; the all-level states are twin
# b's restarts of steps 2, 4, 6, 8, 10, each the recorded after-step state.
set -Eeuo pipefail
trap 'printf "REFUSE: unhandled command failure at run_r2.sh line %s\n" "$LINENO" >&2; exit 1' ERR

export PATH=/home/dbalwada/legoESM/.venv/bin:/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
readonly NEMO_ROOT=${NEMO_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}
readonly RUN_BIN=$NEMO_ROOT/cfgs/ORCA2G1_OMIP_L1_P5/BLD/bin/nemo.exe
readonly RUN_SHA=9b96f1427503da2773ce6c17621dd431c3261792d11a39c218f64d88e3f00778
readonly SHIPPED_DOMCFG=/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0/ORCA_R2_zps_domcfg.nc
readonly SHIPPED_DOMCFG_SHA=7125f7a54e8693ff8f1327b878d2258d258878bc7302ac123ac671f2339d2839
readonly ROUND1=${ROUND1:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round10/oracle_g2_r10}
readonly EVIDENCE=${EVIDENCE:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round11/oracle_g2_r11_consecutive}
readonly TWIN_C=$EVIDENCE/p3_c_10step

mode=${1:---preflight-only}
case "$mode" in --run|--preflight-only|--admit-existing) ;; *)
  printf 'Usage: %s [--run|--preflight-only|--admit-existing]\n' "$0" >&2; exit 64 ;; esac
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo_root=$(git -C "$here" rev-parse --show-toplevel)
[[ -z "$(git -C "$repo_root" status --porcelain --untracked-files=all)" ]] \
  || { printf 'REFUSE: legoESM worktree is dirty; commit the acquisition tool first\n' >&2; exit 66; }
readonly DECK=$here/namelist_cfg CHECKER=$here/../nemo_testcase_g1_orca2grid/check_records.py
sha_of() { sha256sum "$1" | awk '{print $1}'; }
for f in "$DECK" "$CHECKER" "$RUN_BIN" "$SHIPPED_DOMCFG" "$NEMO_ROOT/cfgs/SHARED/namelist_ref" \
         "$ROUND1/p3_a_100step/oracle_g1step_kt00000010.bin" \
         "$ROUND1/p3_b_10step/ORCA2G1_00000010_restart.nc"; do
  [[ -f "$f" ]] || { printf 'REFUSE: missing input %s\n' "$f" >&2; exit 67; }
done
[[ "$(sha_of "$RUN_BIN")" == "$RUN_SHA" && "$(sha_of "$SHIPPED_DOMCFG")" == "$SHIPPED_DOMCFG_SHA" ]] \
  || { printf 'REFUSE: the record binary or the shipped domain file changed\n' >&2; exit 67; }
assignments() { sed -e 's/!.*//' -e 's/[[:space:]]*$//' "$1" | grep -v '^$'; }
rendered=$(mktemp)
trap 'rm -f "$rendered"' EXIT
sed -E -e 's/^(   nn_itend    = ).*/\1    10/' -e 's/^(   nn_stock    = ).*/\1    10/' \
       -e 's/^(   nn_stocklist = ).*/\11,2,3,4,5,6,7,8,9,10/' "$DECK" >"$rendered"
[[ "$(diff "$DECK" "$rendered" | grep -c '^>')" -eq 3 ]] \
  || { printf 'REFUSE: rendering did not change exactly the 3 run-length lines\n' >&2; exit 68; }
printf 'PREFLIGHT: record binary and domain pinned; deck = the committed G2 deck, 10 steps, a restart after every step; commit %s\n' \
  "$(git -C "$repo_root" rev-parse HEAD)"

admit() {
  [[ "$(sha_of "$TWIN_C/domain_cfg.nc")" == "$SHIPPED_DOMCFG_SHA" && "$(sha_of "$TWIN_C/nemo")" == "$RUN_SHA" ]] \
    || { printf 'REFUSE: twin c did not run the pinned binary on the shipped domain\n' >&2; exit 73; }
  grep -qx 'RUN_DONE' "$TWIN_C/run.user.log" && [[ "$(tr -d ' \n' <"$TWIN_C/time.step")" == 10 ]] \
    && ! grep -q 'E R R O R' "$TWIN_C/ocean.output" \
    || { printf 'REFUSE: twin c is not a finished 10-step run\n' >&2; exit 73; }
  cmp -s <(assignments "$TWIN_C/namelist_cfg" | grep -Ev 'nn_itend|nn_stock') \
         <(assignments "$DECK" | grep -Ev 'nn_itend|nn_stock') \
    || { printf 'REFUSE: twin c ran a deck that is not the committed one\n' >&2; exit 73; }
  python3 -I - "$(dirname "$CHECKER")" "$TWIN_C" "$ROUND1/p3_a_100step" "$ROUND1/p3_b_10step" \
      "$EVIDENCE/g2_round11_consecutive_admission.json" "${PLANT:-none}" <<'PY'
import json
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import check_records as cr
c, a, b, out, plant = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5], sys.argv[6]
try:
    # twin c against twin a: every step stream byte for byte (the plant flips
    # one bit of twin a's copy), the solver streams on the consumed support
    result = cr.check(c, a, 10, 10, "twin-ulp" if plant == "twin-ulp" else "none")
    result["substeps"] = cr.check_substeps(c, a, [1, 2])
    import netCDF4
    import numpy as np
    # twin c's only restart file is the one NAMED for step 1, opened at
    # kt = 10 by the end-of-run fallback and never written (header comment)
    names = sorted(q.name for q in c.glob("*_restart.nc"))
    cr.require(names == ["ORCA2G1_00000001_restart.nc"],
               f"twin c holds restart files {names}, not the one empty file")
    with netCDF4.Dataset(c / names[0]) as h:
        records = len(h.dimensions["time_counter"])
    if plant == "empty":
        records = 1
    cr.require(records == 0, "twin c's file named for step 1 holds a record: "
               "read which step it is before using it")
    log = (c / "ocean.output").read_text().splitlines()
    opened = [log[n + 1].split()[-1] for n, line in enumerate(log)
              if "open ocean restart NetCDF file" in line]
    cr.require(opened == ["10"], f"NEMO opened restart files at kt {opened}, not once at 10")
    result["twin_c_restart"] = {"file": names[0], "time_records": 0, "opened_at_kt": 10}
    # the all-level states of this trajectory are twin b's: each restart is
    # the after-step state of the step stream just verified against twin a
    # (sea level, both barotropic velocities and level 1 of u, v bit for bit)
    for step in (2, 4, 6, 8, 10):
        path = b / f"ORCA2G1_{step:08d}_restart.nc"
        cr.require(path.is_file(), f"twin b has no restart at step {step}")
        head, frame = cr.read_stream(cr.step_path(c, step).read_bytes(), path.name)
        js, je, i_s, ie = head["Njs0"] - 1, head["Nje0"], head["Nis0"] - 1, head["Nie0"]
        with netCDF4.Dataset(path) as h:
            for var, name, lev in (("sshn", "f_ssh_bb", None), ("uu_n", "f_uu_b_bb", None),
                                   ("vv_n", "f_vv_b_bb", None), ("un", "f_uu_k1_bb", 0),
                                   ("vn", "f_vv_k1_bb", 0)):
                got = np.asarray(h.variables[var][0], dtype=np.float64)
                got = got if lev is None else got[lev]
                if plant == "restart" and step == 2 and var == "un":
                    got = got.copy()
                    got.flat[got.size // 2] += 1.0
                cr.require(np.array_equal(got, frame[name][js:je, i_s:ie]),
                           f"{path.name}: {var} is not the recorded after-step state")
    result["restarts_equal_step_streams"] = [2, 4, 6, 8, 10]
except cr.Refuse as exc:
    sys.exit(f"REFUSE: {exc}")
if plant != "none":
    sys.exit("REFUSE: the plant stayed green")
Path(out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
print("G2_TWIN_C_ADMITTED step streams 1..10 equal twin a; NO odd-step restart exists "
      "(the file named for step 1 holds no record); all-level states = twin b's 2,4,6,8,10")
PY
}

if [[ "$mode" == --admit-existing ]]; then
  [[ -d "$TWIN_C" ]] || { printf 'REFUSE: nothing to admit, %s is absent\n' "$TWIN_C" >&2; exit 69; }
  for plant in twin-ulp restart empty; do
  if PLANT=$plant admit >/dev/null 2>&1; then printf 'REFUSE: the %s plant stayed green\n' "$plant" >&2; exit 72; fi
done
  admit; printf 'ACQUISITION_DONE %s\n' "$EVIDENCE"; exit 0
fi
[[ ! -e "$EVIDENCE" ]] || { printf 'REFUSE: %s exists; use --admit-existing\n' "$EVIDENCE" >&2; exit 69; }
if [[ "$mode" != --run ]]; then
  printf 'WOULD: mpirun -np 1 ./nemo (ORCA2G1_OMIP_L1_P5) in %s\nPREFLIGHT_OK\n' "$TWIN_C"; exit 0
fi
mkdir -p "$TWIN_C"
cp "$rendered" "$TWIN_C/namelist_cfg"
cp -L "$NEMO_ROOT/cfgs/SHARED/namelist_ref" "$TWIN_C/namelist_ref"
cp "$SHIPPED_DOMCFG" "$TWIN_C/domain_cfg.nc"
cp "$RUN_BIN" "$TWIN_C/nemo"
git -C "$repo_root" rev-parse HEAD >"$EVIDENCE/legoesm_commit"
(
  cd "$TWIN_C"
  printf 'RUN_STARTED_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >run.user.log
  mpirun -np 1 --oversubscribe ./nemo >>run.user.log 2>&1
  printf 'RUN_FINISHED_UTC=%s\nRUN_DONE\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>run.user.log
)
for plant in twin-ulp restart empty; do
  if PLANT=$plant admit >/dev/null 2>&1; then printf 'REFUSE: the %s plant stayed green\n' "$plant" >&2; exit 72; fi
done
admit
printf 'ACQUISITION_DONE %s\n' "$EVIDENCE"
