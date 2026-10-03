#!/bin/bash -l
#SBATCH --job-name=gpoly_l10_acc
#SBATCH --account=bb1596
#SBATCH --partition=compute
#SBATCH --constraint=1024G
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=gpoly_l10_acc_%j.log
# ===========================================================================
# Does gpoly@15 still beat poly@20 on the FINER mesh? L10 (10.5M cells),
# 512 emulated devices (= the ladder's 20k cells/device), real captured
# Helmholtz systems, f32 and f64, plus L9 at dt=600 s (a longer step than
# the 300 s the first measurement used).
#
# PRE-REGISTERED: PASS if, on every captured system, gpoly:4 at M=15 has
# rel_res <= poly:4 at M=20 (f64) and dEta_max <= poly:4 at M=20 (f32).
# FAIL otherwise -> the default move (PR #1859) is not merged as is.
# CPU on a 1 TB node: the eager capture of one L9 system already exceeds an
# A100's 80 GB (job 27841012 OOM), and L10 is 4x that.
# Submit:  sbatch --export=ALL,LEGOESM_REPO=$PWD <this>
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export JAX_PLATFORMS=cpu
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || exit 2
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
OUT="$SCRATCH/legoesm_scaling/ca_pcg_preflight/l10_j${SLURM_JOB_ID}"; mkdir -p "$OUT"
echo "sha $(git -c safe.directory='*' rev-parse --short HEAD 2>/dev/null) out $OUT"
CONV=scripts/validate/ocean_fidelity/barotropic_pcg_convergence.py
ITERS=10,15,20,30
run() {  # run <tag> <args...>
  local tag="$1"; shift
  echo "=== $tag $(date +%T)"
  "$PY" "$CONV" --iters "$ITERS" --out "$OUT/$tag.json" "$@" > "$OUT/$tag.log" 2>&1 \
    || echo "FAILED $tag"
  grep -E "^ +[0-9]+ +(15|20) " "$OUT/$tag.log"
}
for prec in f32 f64; do
  x=(); [[ $prec == f64 ]] && x=(--x64)
  sys="$OUT/systems_L10_nd512_$prec.npz"
  run "L10_${prec}_poly4" --subdivision 10 --emulate-devices 512 --steps 3 \
      --precond poly:4 --save-systems "$sys" "${x[@]}"
  run "L10_${prec}_gpoly4" --subdivision 10 --emulate-devices 512 \
      --precond gpoly:4 --load-systems "$sys" "${x[@]}"
  sys9="$OUT/systems_L9_nd128_dt600_$prec.npz"
  run "L9dt600_${prec}_poly4" --subdivision 9 --emulate-devices 128 --steps 3 \
      --dt 600 --precond poly:4 --save-systems "$sys9" "${x[@]}"
  run "L9dt600_${prec}_gpoly4" --subdivision 9 --emulate-devices 128 --dt 600 \
      --precond gpoly:4 --load-systems "$sys9" "${x[@]}"
done
echo "=== DONE $OUT"
