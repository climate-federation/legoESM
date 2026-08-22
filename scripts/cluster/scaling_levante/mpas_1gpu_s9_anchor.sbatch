#!/bin/bash -l
# THE MISSING DENOMINATOR: subdivision 9 on ONE GPU, under today's flags.
#
# The distance-to-limit table quotes the lat-lon lane against perfect scaling
# of its single-GPU step, and has to leave the icosahedral row blank because
# no matched single-GPU measurement exists for this configuration. An older
# 174 ms figure is on record but from a different date and a different flag
# set, and quoting it next to today's numbers would be exactly the
# cross-protocol comparison this campaign keeps having to retract.
#
# Same mesh, same level count, same halo switches and the same
# shard-autotuning setting as the 64- and 128-GPU sweeps, so the ratio is a
# ratio and not a confound. Two timed arms plus a discarded warm-up.
#
# Whole-mesh state at subdivision 9 with 26 levels is a few gigabytes, so this
# wants an 80 GB card; the small partition's 40 GB ones may refuse it.
#SBATCH --job-name=mpas_s9_1gpu
#SBATCH --account=bd1179_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --exclude=l50003,l50139,l50027,l50009,l50100
#SBATCH --output=mpas_s9_1gpu.%j.log
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
export LEGOESM_XLA_OVERLAP=0
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="$SCRATCH/legoesm_scaling/mpas_s9_1gpu_j${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"; mkdir -p "$LEGOESM_JIT_CACHE_DIR"

rc=0
run_arm () {  # tag
  echo "--- $1 $(date +%T)"
  LEGOESM_MPAS_WIDE_HALO=1 LEGOESM_MPAS_RAGGED_HALO=0 \
  LEGOESM_MPAS_HALO_BALLAST=1 LEGOESM_MPAS_HALO_NOCOMM=0 \
  LEGOESM_MPAS_HALO_NOSTAGE=0 JAX_ENABLE_X64=0 \
    timeout --kill-after=30 1500 srun --export=ALL --nodes=1 --ntasks=1 \
      --gpus-per-node=1 --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
      --n-devices 1 --subdivision 9 --nlev 26 \
      --steps 40 --warmup 5 --lloyd 0 --partition-method sfc \
      --out "$OUTDIR/${1}.jsonl" || { echo "ARM $1 FAILED"; rc=1; }
  sleep 10
}
run_arm warmup_discard
run_arm anchor_a
run_arm anchor_b

"$PY" - "$OUTDIR" <<'PYEOF'
import json, os, statistics, sys
outdir = sys.argv[1]
vals = []
for tag in ("anchor_a", "anchor_b"):
    path = os.path.join(outdir, tag + ".jsonl")
    if not os.path.isfile(path):
        print("MISSING: %s.jsonl" % tag)
        continue
    rec = json.loads([l for l in open(path).read().splitlines() if l.strip()][-1])
    # A single-GPU row that quietly ran a different mesh is not a denominator.
    for name, got, want in (("devices", rec.get("n_devices"), 1),
                            ("subdivision", rec.get("subdivision"), 9),
                            ("levels", rec.get("nlev"), 26),
                            ("platform", rec.get("platform"), "gpu")):
        if got != want:
            print("WRONG ARM: %s ran with %s = %r, expected %r"
                  % (tag, name, got, want))
            break
    else:
        vals.append(float(rec["steady_median_ms"]))
        print("%-10s %9.3f ms" % (tag, vals[-1]))
if len(vals) == 2:
    med = statistics.median(vals)
    spread = 100.0 * (max(vals) - min(vals)) / min(vals)
    print("\nsubdivision 9, 26 levels, ONE GPU: %.3f ms  (spread %.2f%%)"
          % (med, spread))
    for nd in (64, 128):
        print("  perfect scaling to %3d GPUs would be %6.3f ms" % (nd, med / nd))
else:
    print("\nnot enough arms for a denominator")
    sys.exit(3)
PYEOF
SUMRC=$?
if [ "$rc" -ne 0 ] || [ "$SUMRC" -ne 0 ]; then echo "ANCHOR FAILED"; exit 1; fi
echo DONE
