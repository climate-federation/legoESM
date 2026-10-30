#!/bin/bash
# Full-model gates of the FV3-duo land device split, production C24 CAM6 deck
# (land on), 1 simulated day from a fresh start, the command line of the run
# in $RUN (its run_manifest.json) with --days/--checkpoint-days 1.
#   MODE=gpu   (1 node, 2 A100, --exclusive): main ($MAIN_SHA) vs this tree on
#              1 GPU, every checkpoint leaf BITWISE; this tree on 1 vs 2 GPUs
#              with LEGOESM_PHASE_TIMERS=1 (land s/step), 2-GPU run finite.
#   MODE=cpu1  (1 node): main vs this tree on 1 CPU device, BITWISE.
#   MODE=cpu6  (1 node): this tree, 6 host devices (face layout), finite.
#   MODE=win24 (3 nodes x 8 ranks x 4 cpus, CPU gloo, M7 harness settings):
#              24 processes vs 1 process with 24 host devices, windows kt=2
#              pad=4, CLUBB cloud fraction OFF in both arms (the windows
#              layout refuses it), every checkpoint leaf BITWISE.
# sbatch examples:
#   MODE=gpu   sbatch -N1 --gres=gpu:2 --constraint=a100 --exclusive -t 06:00:00 THIS
#   MODE=cpu6  sbatch -N1 -n1 -c 24 --mem=120G -t 12:00:00 THIS
#   MODE=win24 sbatch -N3 -n24 --ntasks-per-node=8 -c 4 --exclusive -t 12:00:00 THIS
#SBATCH --account=glab
#SBATCH --job-name=land_split_gates
#SBATCH --partition=short
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/land_shard/gates_%j.log
set -uo pipefail
WT=${WT:-$(cd "$(dirname "$0")/../../.." && pwd)}
RUN=${RUN:-/burg-archive/glab/users/pg2328/fv3_duo_gaps/valid_main60/duo_60d_soilic}
LAND_IC=${LAND_IC:-/burg-archive/glab/users/pg2328/legoESM_run60b/data/lmip_soil_ic/soil_ic_fv3duo24_era5T_19790101.npz}
MAIN_SHA=${MAIN_SHA:-890090a9b3}
OUT=${OUT_ROOT:-/burg-archive/glab/users/pg2328/fv3_duo_gaps/land_shard}/gates_$SLURM_JOB_ID
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
mkdir -p "$OUT"
export JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1 LEGOESM_JIT_CACHE_DIR=""
pp() { local t=$1; echo "$t/packages/core:$t/packages/atmosphere:$t/packages/coupler:$t/packages/ice:$t/packages/land:$t/packages/ocean:$t/packages/ml:$t/packages/tools:$t/src:$t"; }
argv() {   # the run's command line, 1 day, output redirected
  "$PY" - "$RUN/run_manifest.json" "$1" "$LAND_IC" <<'PYEOF'
import json, shlex, sys
a = shlex.split(json.load(open(sys.argv[1]))["run"]["command_line"])[1:]
for flag, val in (("--output", sys.argv[2]), ("--days", "1"), ("--checkpoint-days", "1")):
    a[a.index(flag) + 1] = val
a += ["--land-ic", sys.argv[3]]
print(" ".join(shlex.quote(x) for x in a))
PYEOF
}
arm() {    # arm NAME TREE [env...] -- [srun/launcher prefix...] -- [extra run_amip flags...]
  local name=$1 tree=$2; shift 2
  local envs=() pre=() extra=()
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do pre+=("$1"); shift; done; shift
  extra=("$@")
  echo "===== arm $name: $(git -C "$tree" rev-parse --short HEAD) dirty=$(git -C "$tree" status --porcelain -uno | wc -l) ${envs[*]} ${pre[*]} ${extra[*]}"
  ( cd "$tree" && eval "env PYTHONPATH=$(pp "$tree") ${envs[*]} ${pre[*]} $PY -u scripts/run/run_amip.py $(argv "$OUT/$name") ${extra[*]}" ) > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "arm $name rc=$rc  $(grep -m1 'fv3_duo layout' "$OUT/$name.log" | sed 's/^ *//' | cut -c1-110)"
  grep -m1 -E "land: every device solves" "$OUT/$name.log"
  grep -E "phase timers \(last 24 steps" "$OUT/$name.log" | sed -E 's/.*  land ([0-9.]+) .*total ([0-9.]+).*/land \1 total \2/' | tail -n +2 | tr '\n' ';'; echo
  return $rc
}
compare() {  # compare A B MODE(bitwise|finite)
  "$PY" - "$OUT/$1" "${2:+$OUT/$2}" "$3" <<'PYEOF'
import glob, sys, numpy as np
def last(d):
    fs = sorted(glob.glob(f"{d}/checkpoint_day_*.npz"))
    if not fs:
        print(f"REFUSED: no checkpoint under {d}"); sys.exit(2)
    return fs[-1], np.load(fs[-1], allow_pickle=False)
fa, a = last(sys.argv[1])
if sys.argv[3] == "finite":
    bad = [k for k in a.files if a[k].dtype.kind == "f" and not np.isfinite(a[k]).all()
           and not k.startswith("land_ml_canopy_x")]
    print(f"FINITE {fa.split('/')[-2]}: {len(a.files)} leaves, non-finite: {bad or 'none'}")
    sys.exit(1 if bad else 0)
fb, b = last(sys.argv[2])
if fa.split("/")[-1] != fb.split("/")[-1]:
    print("REFUSED: different final checkpoints", fa, fb); sys.exit(2)
skip = {"_git_sha"}
keys = sorted((set(a.files) | set(b.files)) - skip)
diff = [k for k in keys if k not in a.files or k not in b.files
        or a[k].shape != b[k].shape or a[k].dtype != b[k].dtype
        or a[k].tobytes() != b[k].tobytes()]
print(f"BITWISE {fa.split('/')[-2]} vs {fb.split('/')[-2]} ({fa.split('/')[-1]}): "
      f"{len(keys)} leaves, differing: {diff or 'none'}")
sys.exit(1 if diff else 0)
PYEOF
}
RC=0
case "${MODE:?MODE=gpu|cpu1|cpu6|win24}" in
gpu)
  export JAX_PLATFORMS=cuda
  nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c
  MAINWT=$OUT/wt_main
  git -C "$WT" worktree add --detach "$MAINWT" "$MAIN_SHA" > /dev/null 2>&1 || exit 2
  arm main1 "$MAINWT" CUDA_VISIBLE_DEVICES=0 -- -- || RC=1
  arm new1 "$WT" CUDA_VISIBLE_DEVICES=0 LEGOESM_PHASE_TIMERS=1 -- -- || RC=1
  arm new2 "$WT" CUDA_VISIBLE_DEVICES=0,1 LEGOESM_PHASE_TIMERS=1 -- -- || RC=1
  compare new1 main1 bitwise || RC=1
  compare new2 "" finite || RC=1
  git -C "$WT" worktree remove --force "$MAINWT" ;;
cpu1)       # CPU is run-to-run deterministic: main vs this tree, 1 device
  export JAX_PLATFORMS=cpu
  MAINWT=$OUT/wt_main
  git -C "$WT" worktree add --detach "$MAINWT" "$MAIN_SHA" > /dev/null 2>&1 || exit 2
  arm main1 "$MAINWT" -- -- || RC=1
  arm new1 "$WT" -- -- || RC=1
  compare new1 main1 bitwise || RC=1
  git -C "$WT" worktree remove --force "$MAINWT" ;;
cpu6)
  export JAX_PLATFORMS=cpu
  arm cpu6 "$WT" "XLA_FLAGS=--xla_force_host_platform_device_count=6" -- -- || RC=1
  compare cpu6 "" finite || RC=1 ;;
win24)
  export JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_max_isa=AVX"
  export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
  source /etc/profile.d/modules.sh; module load openmpi/gcc/64/4.1.7a1
  for v in $(env | sed -n 's/^\(MPITRAMPOLINE_[A-Z_]*\)=.*/\1/p'); do unset "$v"; done
  export JAX_CPU_COLLECTIVES_IMPLEMENTATION=gloo
  # gate deck only: the windows layout refuses CLUBB cloud fraction
  WIN="--fv3-duo-windows 2 --fv3-duo-window-pad 4 --no-use-clubb-cloud-fraction"
  arm win1 "$WT" "XLA_FLAGS='--xla_cpu_max_isa=AVX --xla_force_host_platform_device_count=24'" -- srun -N 1 -n 1 --mpi=pmix -- $WIN || RC=1
  arm win24 "$WT" -- srun -N "$SLURM_JOB_NUM_NODES" -n 24 --mpi=pmix --kill-on-bad-exit=0 "$WT/scripts/cluster/fv3_native/bind_rank.sh" -- --distributed --distributed-mode spmd $WIN || RC=1
  compare win24 win1 bitwise || RC=1
  compare win24 "" finite || RC=1 ;;
*) echo "unknown MODE=$MODE"; exit 2 ;;
esac
echo "GATES MODE=$MODE rc=$RC"
exit $RC
