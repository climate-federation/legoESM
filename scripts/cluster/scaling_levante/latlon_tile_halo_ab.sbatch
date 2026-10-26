#!/bin/bash -l
# Did rewriting the tiled halo assembly make it faster, or only move fewer bytes?
#
# The byte census says the rewritten body moves 68.7 MB of local data where the
# old one moved 194 MB, for the same 0.81 MB on the wire. A byte count is not a
# time -- that is exactly the mistake this line of work already made once -- so
# both bodies run here, on the same GPUs, in the same allocation, under the SAME
# instrument, which is copied from this branch into the old checkout so the two
# arms differ in the halo body and in nothing else.
#
# The instrument matters as much as the arms. Timing one call to the body is a
# measurement of host dispatch on four devices from one process; it reported
# 680 microseconds "per collective" for collectives carrying 0.8 MB in total.
# Both arms run the fixed version, which applies the body to several independent
# field groups inside one jit call and takes the one-group versus many-groups
# difference.
#
# Falsifiable, written before submit:
#   rewritten body faster by roughly the byte ratio -> the assembly WAS the cost
#     and the tiled lane is worth re-measuring against latitude bands at 64 GPUs.
#   rewritten body within noise of the old one -> the local bytes are not what
#     this body spends its time on either, and the next step is a kernel profile
#     rather than another rewrite.
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --job-name=ll_tile_ab
#SBATCH --output=ll_tile_ab.%j.log
#SBATCH --exclude=l50003,l50139,l50027,l50009,l50100
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
OLD_REPO="${OLD_REPO:-/work/bd1083/b309178/diffESM/legoesm_wt_prepack}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
export LEGOESM_JIT_CACHE_DIR="$SCRATCH/legoesm_jit_cache_j${SLURM_JOB_ID}"
mkdir -p "$LEGOESM_JIT_CACHE_DIR"
export NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_P2P_NET_CHUNKSIZE=131072
OUTDIR="$SCRATCH/legoesm_scaling/ll_tile_ab_j${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

# The two arms differ ONLY in which checkout the halo body is imported from.
# The instrument is byte-identical in both, which is checked here rather than
# assumed -- a stale copy in the old checkout would silently compare two
# different measurements.
NEW_SCRIPT="$SUBMIT_DIR/scripts/bench/bench_latlon_tile_halo_micro.py"
OLD_SCRIPT="$OLD_REPO/scripts/bench/bench_latlon_tile_halo_micro.py"
cmp -s "$NEW_SCRIPT" "$OLD_SCRIPT" || {
  echo "instrument differs between the two checkouts; copy it across first"
  exit 2
}
echo "arms: new=$SUBMIT_DIR  old=$OLD_REPO"
echo "  new body sha: $(cd "$SUBMIT_DIR" && git rev-parse --short HEAD)"
echo "  old body sha: $(cd "$OLD_REPO" && git rev-parse --short HEAD)"

pp_for() {  # build the federation PYTHONPATH for a checkout
  local R=$1 PP="$R/src"
  for p in atmosphere core coupler ice land ml ocean tools; do PP="$PP:$R/packages/$p"; done
  echo "$PP"
}
rc=0
# Interleaved, not blocked: the two arms alternate so a drift on the node does
# not land entirely on whichever ran second.
for REP in 1 2; do
  for ARM in old new; do
    if [ "$ARM" = new ]; then R="$SUBMIT_DIR"; else R="$OLD_REPO"; fi
    for FIELDS in 6 12; do
      echo "--- arm=$ARM rep=$REP fields=$FIELDS $(date +%T)"
      PYTHONPATH="$(pp_for "$R")" LEGOESM_GIT_SHA="$(cd "$R" && git rev-parse --short HEAD)" \
        timeout 900 srun --export=ALL --nodes=1 --ntasks=1 --gpus-per-node=4 \
          --gpu-bind=none \
        "$PY" "$NEW_SCRIPT" \
          --p-lat 2 --p-lon 2 --n-lat 512 --n-lon 1024 --nlev 26 \
          --fields "$FIELDS" --groups 8 --reps 50 --warmup 10 \
          --out "$OUTDIR/${ARM}_f${FIELDS}_r${REP}.jsonl" \
        || { echo "ARM $ARM f$FIELDS r$REP FAILED"; rc=1; }
    done
  done
done

"$PY" - "$OUTDIR" <<'PYSUM'
import json, sys, glob, os, collections
outdir = sys.argv[1]
rows = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(outdir, "*.jsonl"))):
    r = json.loads(open(f).read().splitlines()[-1])
    arm = os.path.basename(f).split("_")[0]
    rows[(arm, r["fields"])].append(r)
if not rows:
    print("NO RECEIPTS"); sys.exit(1)
print(f"{'fields':>6} {'arm':>5} {'packed ms/group':>16} {'per-field ms/group':>19} {'sha':>10}")
for (arm, nf), rs in sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    pk = [x["arms"]["packed"]["marginal_ms"] for x in rs]
    pf = [x["arms"]["per-field"]["marginal_ms"] for x in rs]
    print(f"{nf:>6} {arm:>5} {sum(pk)/len(pk):>16.4f} {sum(pf)/len(pf):>19.4f} "
          f"{rs[0].get('git_sha'):>10}   (n={len(rs)}, packed {['%.4f' % v for v in pk]})")
for nf in sorted({k[1] for k in rows}):
    if ("old", nf) in rows and ("new", nf) in rows:
        o = sum(x["arms"]["packed"]["marginal_ms"] for x in rows[("old", nf)]) / len(rows[("old", nf)])
        n = sum(x["arms"]["packed"]["marginal_ms"] for x in rows[("new", nf)]) / len(rows[("new", nf)])
        om = sum(x["arms"]["packed"]["marginal_ms_midpoint"] for x in rows[("old", nf)]) / len(rows[("old", nf)])
        nm = sum(x["arms"]["packed"]["marginal_ms_midpoint"] for x in rows[("new", nf)]) / len(rows[("new", nf)])
        print(f"{nf} fields, midpoint arm: old {om:.4f} -> new {nm:.4f} ms  ({om/nm:.2f}x)")
        print(f"{nf} fields: old {o:.4f} -> new {n:.4f} ms/group  ({o/n:.2f}x)")
PYSUM
echo "DONE rc=$rc"; exit $rc
