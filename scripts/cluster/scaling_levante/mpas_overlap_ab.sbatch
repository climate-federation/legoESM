#!/bin/bash -l
#SBATCH --job-name=ovl_ab
#SBATCH --account=bd1179_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:15:00
#SBATCH --exclude=l50003,l50139,l50027,l50009,l50100
#SBATCH --output=ovl_ab.%j.log
# ===========================================================================
# DOES THE INTERIOR/RIM OVERLAP CUT THE ICOSAHEDRAL STEP? Isolated A/B.
#
# The overlap hides the ppermute halo behind the interior tendency. It needs
# TWO things: compact partitions (metis -- a stringy sfc partition has almost
# no interior, proven by the sizing probe) and the latency-hiding scheduler
# (LEGOESM_XLA_OVERLAP=1). To isolate the overlap's own effect, BOTH arms run
# with the scheduler ON and metis; the only variable is the overlap switch.
# A third arm (scheduler off, overlap off) is the current production baseline,
# so the total improvement is also readable.
#
# The compile-time HLO proof already fired (async collective-permute with
# interior fusions between start and done, job 27166926); this prices it.
#
# A PRIORI GATE, on the overlap arm vs the scheduler-on/overlap-off arm at
# 64 GPUs (the isolated effect):
#   CONFIRMS: overlap is at least 5% faster. Then the wire is being hidden and
#     the lever is real; take it to 128 where communication is a bigger share.
#   REFUTES: within 1.5% (arm spread is ~0.5%). Then XLA fired the async form
#     but the interior pass does not actually run concurrently on the GPU, or
#     the rim recompute eats the saving; report and stop.
#
# SUBMIT: sbatch scripts/cluster/scaling_levante/mpas_overlap_ab.sbatch
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
# The s9 lloyd-0 mesh is cached HERE (not ~/.cache, where the bench defaults); the working MPAS scaling jobs all set this.
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="$SCRATCH/legoesm_scaling/ovl_ab_j${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"; mkdir -p "$LEGOESM_JIT_CACHE_DIR"
NODES="${SLURM_JOB_NUM_NODES:-16}"; GPN=4; NDEV=$((NODES*GPN))
export NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16
rc=0

run_arm () {  # xla_overlap mpas_overlap tag
  local XO=$1 MO=$2 TAG=$3
  echo "--- $TAG xla_overlap=$XO mpas_overlap=$MO $(date +%T)"
  printf 'xla_overlap=%s mpas_overlap=%s metis 16ch\n' "$XO" "$MO" \
    > "$OUTDIR/${TAG}.env"
  LEGOESM_XLA_OVERLAP=$XO LEGOESM_MPAS_HALO_OVERLAP=$MO JAX_ENABLE_X64=0 \
    timeout --kill-after=30 900 srun --export=ALL --nodes="$NODES" --ntasks="$NDEV" \
      --ntasks-per-node="$GPN" --gpus-per-node="$GPN" --gpu-bind=none \
      --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
      --multicontroller --n-devices "$NDEV" --subdivision 9 --nlev 26 --lloyd 0 \
      --steps 60 --warmup 15 --halo-strategy ppermute \
      --partition-method metis \
      --out "$OUTDIR/${TAG}.jsonl" || { echo "ARM $TAG FAILED"; rc=1; }
  sleep 10
}

# Discard warm-up, then palindrome over the three conditions x two reps.
run_arm 1 0 warmup_discard
for A in "0 0 prod_a" "1 0 lhs_a" "1 1 ovl_a" "1 1 ovl_b" "1 0 lhs_b" "0 0 prod_b"; do
  run_arm $A
done

echo "=== RECEIPTS ==="
"$PY" - "$OUTDIR" <<'PYEOF'
import json, os, sys
outdir = sys.argv[1]
def med(tag):
    p = os.path.join(outdir, tag + ".jsonl")
    if not os.path.isfile(p): return None
    r = [json.loads(l) for l in open(p) if l.startswith("{")]
    r = [x for x in r if x.get("steady_median_ms")]
    return float(r[-1]["steady_median_ms"]) if r else None
def pair(n):
    a, b = med(n+"_a"), med(n+"_b")
    if a is None or b is None: return None, None
    m = 0.5*(a+b); return m, abs(a-b)/m
rows = {n: pair(n) for n in ("prod","lhs","ovl")}
print(f"{'arm':>6} {'ms/step':>9} {'spread':>7}")
for n,(m,s) in rows.items():
    if m is not None: print(f"{n:>6} {m:9.4f} {s:7.2%}")
if rows["ovl"][0] and rows["lhs"][0]:
    iso = rows["ovl"][0]/rows["lhs"][0] - 1
    print(f"\nisolated overlap effect (ovl vs lhs): {iso:+.2%}")
    if rows["prod"][0]:
        tot = rows["ovl"][0]/rows["prod"][0] - 1
        print(f"total vs production (ovl vs prod): {tot:+.2%}")
    err = rows["ovl"][1] + rows["lhs"][1]
    if abs(iso) < err:
        print("VERDICT: within arm spread -- INCONCLUSIVE, rerun"); sys.exit(3)
    if iso <= -0.05:
        print("VERDICT: CONFIRMS -- overlap hides the wire; take it to 128")
    elif iso >= -0.015:
        print("VERDICT: REFUTES -- overlap does not pay at 64"); sys.exit(6)
    else:
        print("VERDICT: PARTIAL -- small gain, reprice at 128")
else:
    print("NO VERDICT: missing arms"); sys.exit(4)
PYEOF
exit $rc
