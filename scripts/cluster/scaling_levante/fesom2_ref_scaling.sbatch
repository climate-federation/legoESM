#!/bin/bash -l
#SBATCH --job-name=fesom2_ref
#SBATCH --account=bb1596
#SBATCH --partition=compute
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --time=02:30:00
#SBATCH --output=fesom2_ref.%j.log
# ===========================================================================
# FESOM2 (native Fortran, AWI prebuilt) CORE2-mesh strong-scaling REFERENCE
# for the ocean scaling plots (user directive 2026-08-10: put the SOTA
# unstructured-ocean dycore on the same axes as our lanes).
#
# Binary: /work/ab0995/a270301/fesom2/bin/fesom.x (OpenMPI 4.1.2 spack,
# absolute rpaths — no modules needed). Mesh: CORE2 (126 858 vertices,
# 47 levels) with prebuilt METIS partitions dist_{32,64,128,256,512}.
# Forcing: JRA55-do 1958, IC: PHC3.0 — the AWI papers' hindcast config,
# namelists copied verbatim from /work/ab0995/a270301/fesom2/config with
# ONLY ResultPath / run length / restart cadence patched.
#
# TIMING METHOD (init-free): each rung runs TWICE, 2 model days and 4
# model days (36 steps/day). step_cost = (T4 - T2) / 72 — mesh read,
# partition setup and forcing open cancel in the difference, and the
# method needs no assumptions about FESOM's log format.
# ===========================================================================
set -uo pipefail
FESOM_SRC=/work/ab0995/a270301/fesom2
BIN=$FESOM_SRC/bin/fesom.x
BASE=/scratch/b/b381103/legoesm_scaling/fesom2_ref_j${SLURM_JOB_ID}
mkdir -p "$BASE"; echo "base=$BASE"
RUNGS="${RUNGS:-32 64 128 256 512}"
rc=0

make_rundir () {  # np days -> prints rundir
  local NP=$1 DAYS=$2
  local D="$BASE/np${NP}_d${DAYS}"
  mkdir -p "$D/out"
  cp "$FESOM_SRC"/config/namelist.* "$D/" 2>/dev/null
  # Patch ONLY: ResultPath, run length (days), restarts off.
  python3 - "$D" "$DAYS" <<'PYEOF'
import re, sys
d, days = sys.argv[1], sys.argv[2]
p = f"{d}/namelist.config"
s = open(p).read()
s = re.sub(r"ResultPath\s*=\s*'[^']*'", f"ResultPath       = '{d}/out/'", s)
s = re.sub(r"run_length\s*=\s*\S+", f"run_length        = {days}", s)
s = re.sub(r"run_length_unit\s*=\s*'\w'", "run_length_unit   = 'd'", s)
s = re.sub(r"restart_length_unit\s*=\s*'\w+'",
           "restart_length_unit     = 'off'", s)
s = re.sub(r"raw_restart_length_unit\s*=\s*'\w+'",
           "raw_restart_length_unit = 'off'", s, count=1)
open(p, "w").write(s)
PYEOF
  printf '0 1 1958\n0 1 1958\n' > "$D/out/fesom.clock"
  echo "$D"
}

declare -A WALL
for NP in $RUNGS; do
  if [ ! -d /pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2/dist_${NP} ]; then
    echo "np=$NP: no dist_${NP} partition — skipped"; continue
  fi
  for DAYS in 2 4; do
    D=$(make_rundir "$NP" "$DAYS")
    echo "=== fesom2 np=$NP days=$DAYS ==="
    t0=$(date +%s.%N)
    ( cd "$D" && srun --ntasks="$NP" --kill-on-bad-exit=1 "$BIN" ) \
      > "$D/fesom.log" 2>&1 || { echo "np=$NP d=$DAYS FAILED"; rc=1; \
        tail -20 "$D/fesom.log"; continue 2; }
    t1=$(date +%s.%N)
    WALL[${NP}_${DAYS}]=$(awk -v a="$t1" -v b="$t0" 'BEGIN{print a-b}')
    echo "np=$NP days=$DAYS wall=${WALL[${NP}_${DAYS}]}s"
  done
done

echo "=== RECEIPTS (step cost = (T4-T2)/72, 36 steps/day) ==="
OUT="$BASE/fesom2_ref_scaling.jsonl"; : > "$OUT"
for NP in $RUNGS; do
  T2=${WALL[${NP}_2]:-}; T4=${WALL[${NP}_4]:-}
  [ -z "$T2" ] || [ -z "$T4" ] && { echo "np=$NP incomplete"; continue; }
  python3 - "$NP" "$T2" "$T4" >> "$OUT" <<'PYEOF'
import json, subprocess, sys
np_, t2, t4 = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
ms = (t4 - t2) / 72.0 * 1000.0
sypd = 86400.0 / ((t4 - t2) / 2.0) / 365.0 if t4 > t2 else float("nan")
print(json.dumps({
    "component": "fesom2_native_ref", "grid": "fesom_core2_triangular",
    "n_ranks": np_, "n_cells": 126858, "n_levels": 47,
    "steps_per_day": 36, "time_per_step_ms": round(ms, 2),
    "sypd": round(sypd, 3), "method": "T4_minus_T2_over_72",
    "binary": "/work/ab0995/a270301/fesom2/bin/fesom.x",
    "mesh": "CORE2", "forcing": "JRA55-do-1958", "machine": "levante_compute",
}))
PYEOF
done
cat "$OUT"
grep -q time_per_step_ms "$OUT" && echo "FESOM2_REF_DONE" || \
  { echo "FESOM2_REF_EMPTY"; rc=1; }
exit $rc
