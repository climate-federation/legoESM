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
# Admission: twin c's step streams kt = 1..10 are byte-identical to twin a's
# (so the extra restart writes change nothing), its solver substep streams
# equal twin a's on the consumed support, and its restarts at steps
# 2, 4, 6, 8, 10 equal twin b's variable by variable.
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
    result["restarts_equal_twin_b"] = cr.restarts_identical(b, c, [2, 4, 6, 8, 10])
    for step in range(1, 11):
        cr.require((c / f"ORCA2G1_{step:08d}_restart.nc").is_file(),
                   f"twin c has no restart at step {step}")
except cr.Refuse as exc:
    sys.exit(f"REFUSE: {exc}")
if plant != "none":
    sys.exit("REFUSE: the plant stayed green")
Path(out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
print("G2_CONSECUTIVE_RESTARTS_ADMITTED steps 1..10; step streams equal twin a; restarts 2,4,6,8,10 equal twin b")
PY
}

if [[ "$mode" == --admit-existing ]]; then
  [[ -d "$TWIN_C" ]] || { printf 'REFUSE: nothing to admit, %s is absent\n' "$TWIN_C" >&2; exit 69; }
  if PLANT=twin-ulp admit >/dev/null 2>&1; then printf 'REFUSE: the twin plant stayed green\n' >&2; exit 72; fi
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
if PLANT=twin-ulp admit >/dev/null 2>&1; then printf 'REFUSE: the twin plant stayed green\n' >&2; exit 72; fi
admit
printf 'ACQUISITION_DONE %s\n' "$EVIDENCE"
