#!/bin/bash -l
#SBATCH --job-name=mpas_s9_ens
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=mpas_s9_ens.%j.log
# ENSEMBLE-PARALLELISM RECEIPT (codex 2026-07-31 consult lever #1; design
# fixed per codex round-20 items 9-12).  The tile floor caps ONE
# trajectory's strong scaling; past it the honest use of 128 GPUs is K
# independent replicas at the per-trajectory sweet spot.  This job
# measures the only thing that can refute that: FABRIC CONTENTION between
# replicas sharing the IB tree.
#
#   solo_pre  : ONE 32-GPU s9 run, other 24 nodes idle
#   phase B   : FOUR concurrent 32-GPU s9 runs on disjoint 8-node sets
#               (SLURM_STEP_NODELIST is per-step -> per-replica
#               coordinator autodetect; jobid-derived port shared but
#               hosts differ)
#   solo_post : solo again AFTER phase B (brackets ordering/thermal
#               drift; contrast uses mean of the two solos)
#
# Falsifiability, written BEFORE submit:
#   numbers : 2 solo + 4 replica steady_median_ms
#   CONFIRM : max(replica) <= 1.10 x mean(solo) -> guaranteed aggregate
#             >= 4/1.10 = 3.64x the 32-GPU solo rate (~19.9 GC/s if solo
#             reproduces 5.47) = ~3.3x the observed 128-GPU
#             single-trajectory rate.  NOT "4x": 1.10 is the bar, the
#             margin below it is the measured contention.
#   REFUTE  : any replica > 1.10x solo -> contention term, quantified
#             per replica.
# Protocol: config identical to job 26600095 np32 rung (sfc, lloyd 0,
# f32, padded-128 reorder) EXCEPT steps 5000 / warmup 100 so the stepping
# window (~60 s at 12.5 ms/step) dwarfs launch skew between replicas --
# overlap is EVIDENCED, not assumed, by the per-step Start/End + NodeList
# table sacct prints at the end.  Absolute ms/step is therefore only
# compared WITHIN this job (solo vs replicas), never against the
# steps-12 ladder rows.
# INTERPRETATION LIMIT (codex r21 item 4): this design SCREENS for a
# co-execution penalty vs the solo brackets; if a penalty appears, its
# attribution (shared IB fabric vs node/topology placement vs drift)
# needs the per-step nodelist table + follow-up, and the solo/replica
# ordering is bracketed (pre+post), not fully counterbalanced.
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/mpas_s9_ens_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
rc=0

run_arm () { # tag  (one 32-GPU replica on 8 disjoint nodes)
  echo "[$1] launch epoch=$(date +%s.%N)"
  # rank 0 prints its step's nodelist BEFORE exec'ing the bench — the
  # per-replica disjointness + coordinator-host evidence codex asked for
  # ("$0" inside bash -c is $PY, passed as the first post-script arg).
  JAX_ENABLE_X64=0 srun --nodes=8 --ntasks=32 --ntasks-per-node=4 \
      --gpus-per-node=4 --gpu-bind=none --exact --kill-on-bad-exit=1 \
      --job-name="arm_$1" \
    bash -c '[ "${SLURM_PROCID:-1}" = 0 ] && echo "[step $ARM_TAG] nodelist=$SLURM_STEP_NODELIST"; exec "$0" "$@"' \
    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
      --multicontroller --n-devices 32 \
      --subdivision 9 --nlev 26 --steps 5000 --warmup 100 --lloyd 0 \
      --partition-method sfc --reorder-for 128 \
      --out "$OUTDIR/$1.jsonl"
  s=$?
  echo "[$1] exit=$s epoch=$(date +%s.%N)"
  return $s
}

echo "=== solo_pre: 1x32 GPUs, 24 nodes idle ==="
ARM_TAG=solo_pre; export ARM_TAG
run_arm solo_pre || { echo "solo_pre FAILED"; rc=1; }

echo "=== phase B: 4 concurrent 32-GPU replicas ==="
pids=(); tags=(rep0 rep1 rep2 rep3)
for T in "${tags[@]}"; do ARM_TAG=$T run_arm "$T" & pids+=($!); done
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "${tags[$i]} FAILED"; rc=1; }
done

echo "=== solo_post: 1x32 GPUs, 24 nodes idle ==="
ARM_TAG=solo_post; export ARM_TAG
run_arm solo_post || { echo "solo_post FAILED"; rc=1; }

echo "=== RESULTS ==="
for T in solo_pre rep0 rep1 rep2 rep3 solo_post; do
  F="$OUTDIR/$T.jsonl"
  "$PY" -c "
import json,math,sys
try:
    d=json.loads(open('$F').readline())
    ms=d['steady_median_ms']
    assert math.isfinite(ms) and ms > 0, f'non-finite {ms}'
except Exception as e:
    print('$T: MISSING/INVALID ->', e); sys.exit(1)
print(f'$T: {ms:8.2f} ms  {d.get(\"mcells_per_s\",0)/1000:.2f} GC/s')" \
    || { echo "$T receipt invalid"; rc=1; }
done

echo "=== overlap evidence: per-step nodelist + wall window ==="
sacct -j "$SLURM_JOB_ID" \
  --format=JobID%18,JobName%12,NodeList%45,Start,End,State -P \
  || { echo "sacct overlap table UNAVAILABLE"; rc=1; }
echo "DONE rc=$rc"; exit $rc
