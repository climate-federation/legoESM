#!/bin/bash -l
#SBATCH --job-name=halo_c64
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --exclude=l50100
#SBATCH --output=halo_c64.%j.log
# Ragged-vs-coloured microbench at nd=32/64 — the ZERO-SLICE-COST
# discriminator for the s9@64 production loss (job 26824688: ratio
# 1.220 vs the 0.686 WIN at s8@16). Hypothesis (GLM-5.2's make-or-break
# warning): XLA/NCCL does not prune zero-size ragged slices, so a
# device pays (n_dev - degree) no-op sends — 5 at nd16 (cheap, ragged
# won 0.475) but 54 at nd64.
# Falsifiability BEFORE submit, at FIXED degree 10 and FIXED payload:
#   CONFIRMS zero-slice cost: ragged per-fill GROWS with nd at fixed
#     degree (nd64 >> nd16's 478 us) while ppermute stays ~flat-ish;
#     ragged/ppermute ratio at nd64 >= ~1 matches the production loss.
#   REFUTES: ragged per-fill ~flat in nd (then the production loss is
#     elsewhere: the s_max padding gather (int32[64,2,149795] stacked
#     metadata!) or payload-size effects — next discriminator would be
#     a padded-gather A/B).
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_XLA_OVERLAP=0
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/halo_c64_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"
rc=0

# degree-10 symmetric alias-free offset sets
run_leg () {  # nd nodes offsets
  local ND=$1 NODES=$2 OFFS=$3
  srun --nodes="$NODES" --ntasks="$ND" --ntasks-per-node=4 \
      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_halo_collectives.py \
      --multicontroller --n-devices "$ND" --offsets "$OFFS" \
      --strategies ppermute,ragged \
      --out "$OUTDIR/nd${ND}.jsonl" || { echo "nd$ND FAILED"; rc=1; }
}
run_leg 32  8 "1,-1,2,-2,5,-5,7,-7,6,-6"
run_leg 64 16 "1,-1,2,-2,5,-5,7,-7,6,-6"

"$PY" - "$OUTDIR" <<'EOF' || rc=1
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
for nd in (32, 64):
    rows = {json.loads(l)["strategy"]: json.loads(l)
            for l in open(out / f"nd{nd}.jsonl")}
    pp, rg = rows["ppermute"], rows["ragged"]
    assert pp["n_devices"] == nd and pp["verified_exact"] and rg["verified_exact"]
    print(f"nd{nd}: ppermute({pp['n_collectives_per_fill']}r) "
          f"{pp['per_fill_us']:9.1f} us | ragged {rg['per_fill_us']:9.1f} us "
          f"| ratio {rg['per_fill_us']/pp['per_fill_us']:.3f}")
print("(nd16 reference from 26818265: ppermute 1008.0, ragged 478.4, ratio 0.475)")
EOF
echo "DONE rc=$rc"; exit $rc
