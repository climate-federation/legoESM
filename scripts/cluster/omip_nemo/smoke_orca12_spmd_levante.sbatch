#!/bin/bash -l
#SBATCH --job-name=orca12_smk
#SBATCH --account=bd1179_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=2
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=/work/bd1083/b309178/diffESM/legoesm_pg/ocean_hires_runs/logs/orca12_smoke.%j.log
# ORCA12 (NOC ORCA0083-N06, 1/12 deg tripolar, 75 levels) feasibility SMOKE on
# Levante GPUs: run_omip.py tripole lane, lat-band SPMD route-B multicontroller
# (one process per GPU), JRA55-do 1958 forcing, slab sea ice, fp64.
# SIZING ONLY: reports peak GPU memory, s/step and finiteness. NOT a production
# year -- the T-pivot fold seam of this mesh is only approximated (see PR).
# Submit with --nodes=8 (16 GPUs at TPN=2) or --nodes=16 (32 GPUs); DT/STEPS/TPN via --export.
set -uo pipefail
WT=${LEGOESM_REPO:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/agent-ac78430029eb34e21}
PY=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python
R=/work/bd1083/b309178/diffESM/legoesm_pg/ocean_hires_runs
M=$R/meshes/ORCA0083-N06
DT=${DT:-150}
STEPS=${STEPS:-200}
# Tasks per node: each process builds the GLOBAL setup in HOST RAM (#1370) --
# ~130 GB RSS per process at ORCA12 (job 27326645: 4/node = OUT_OF_MEMORY on a
# 512 GB node). 2 per node fits; the other two GPUs of the node stay idle.
TPN=${TPN:-2}
# Probe knobs: DIAG_EVERY (blowup-onset resolution, 1 = every step), EXTRA
# (extra run_omip flags, e.g. "--jra55-no-sponge" or
# "--jra55-zero-surface-fluxes"), MEMPROBE=1 (per-device live-array report).
DIAG_EVERY=${DIAG_EVERY:-20}
EXTRA=${EXTRA:-}
[ -n "${MEMPROBE:-}" ] && export LEGOESM_OMIP_MEMPROBE=1
NGPU=$(( SLURM_JOB_NUM_NODES * TPN ))
DAYS=$(awk -v s="$STEPS" -v d="$DT" 'BEGIN {printf "%.10f", s*d/86400.0}')
# H_max must equal the dz-file total to 1e-6 relative (run_omip's dz-ref gate).
# awk, not python: the batch node's system python has no numpy.
HMAX=$(awk '{s+=$1} END {printf "%.9f", s}' "$M/orca12_L75_dz.txt")
[ -n "$HMAX" ] || { echo "HMAX empty"; exit 2; }
OUT=$R/smoke_orca12/n${NGPU}_dt${DT}_j${SLURM_JOB_ID}
mkdir -p "$OUT" $R/logs
cd "$WT"
export PYTHONPATH="$WT/packages/atmosphere:$WT/packages/core:$WT/packages/coupler:$WT/packages/ice:$WT/packages/land:$WT/packages/ml:$WT/packages/ocean:$WT/packages/tools:$WT/src:$WT"
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export LEGOESM_LATLON_SPMD_FUSED_HALO=1
echo "=== ORCA12 smoke: nodes=$SLURM_JOB_NUM_NODES gpus=$NGPU dt=$DT steps=$STEPS days=$DAYS diag_every=$DIAG_EVERY extra=[$EXTRA] memprobe=${MEMPROBE:-0} $(date) ==="
echo "mesh: $M (sha256 in $M/SHA256SUMS)"; cat $M/PROVENANCE.txt
( while true; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr '\n' ' '; echo; sleep 15; done ) > $OUT/gpumem_node0.txt &
MON=$!
t0=$(date +%s)
# per-rank stderr files: the failing rank's real XLA error (OOM etc.) survives
# the coordination-service teardown noise in the aggregated log.
srun --ntasks=$NGPU --ntasks-per-node=$TPN --gpus-per-node=$TPN --gpu-bind=none --kill-on-bad-exit=1 \
  --error="$OUT/rank_%t.err" \
  "$PY" scripts/run/run_omip.py --grid tripole \
    --tripole-mesh "$M/mesh_hgr.nc:$M/mesh_zgr.nc:$M/mask.nc" --tripole-strip-north-rows 1 \
    --tripole-fold-convention "(n_lon-i)%n_lon" \
    --nlev 75 --dz-ref-file $M/orca12_L75_dz.txt --H-max "$HMAX" \
    --dt $DT --days $DAYS --precision fp64 \
    --forcing-mode jra55_do_tropical --jra55-cache $R/forcing/jra55_do_fesom_1958_1958_1deg_noleap.zarr \
    --jra55-sea-ice --ice-dynamics none --ice-categories 1 \
    --vertical-mixing-scheme kpp --tripole-closed-seas marmara,black_sea \
    --woa-init --woa-t $R/forcing/phc3_winter_woa_format_nofabricated.nc --woa-s $R/forcing/phc3_winter_woa_format_nofabricated.nc --woa-void-fill \
    --enable-latlon-spmd --multicontroller \
    --diag-every $DIAG_EVERY $EXTRA --output "$OUT" 2>&1 | tee $OUT/run.log | grep -v "^WARNING\|^I0000\|^E0000" | tail -80
rc=${PIPESTATUS[0]}
kill $MON 2>/dev/null
echo "run_rc=$rc wall_s=$(( $(date +%s) - t0 ))"
echo "peak GPU mem MiB (node 0, per GPU):"; python3 - "$OUT/gpumem_node0.txt" <<'EOF'
import sys, collections, re
peak = collections.defaultdict(int)
for line in open(sys.argv[1]):
    for i, m in re.findall(r"(\d+),\s*(\d+)", line):   # nvidia-smi CSV: "0, 123"
        peak[int(i)] = max(peak[int(i)], int(m))
print(dict(sorted(peak.items())))
EOF
echo "=== DONE $(date) ==="; exit $rc
