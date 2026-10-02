#!/bin/bash
#SBATCH --account=glab
#SBATCH --job-name=duo_dt_ladder
#SBATCH --partition=short
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=11:00:00
#SBATCH --array=0-2
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/a_ladder/slurm_%A_%a.out
# Decision A (deck ledger, user-approved 2026-10-01): the dt ladder for the
# FV3 duo C24 L32 CAM6 deck = MPAS's rule -- 5-day arms at the 1800-s
# divisors around the driver's printed CFL clamp (349 s at C24): 225 / 300 /
# 360 s, physics every 1800 s and radiation hourly on every arm.  The driver
# REDUCES a dt above its clamp (logged "CFL: reducing dt"), so the 360 arm's
# RESOLVED dt is read from the log, not assumed.  Adopt the largest stable
# arm below the clamp.  One process on one A100 (fp64 column lane).
set -uo pipefail
DTS=(225 300 360)
DT=${DTS[$SLURM_ARRAY_TASK_ID]}
PHYS=$((1800 / DT)); RAD=$((3600 / DT))
WT=${WT:-/burg-archive/glab/users/pg2328/legoESM_codex_m8}
export LEGOESM_REPO="$WT"
export PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH="$WT/packages/core:$WT/packages/atmosphere:$WT/packages/coupler:$WT/packages/ice:$WT/packages/land:$WT/packages/ocean:$WT/packages/ml:$WT/packages/tools:$WT/src"
export AMIP_DATA=/burg-archive/glab/users/ac5006/legoESM/data/amip
export AMIP_SURFDATA=/burg-archive/glab/users/pg2328/legoESM/data/legoesm_surfdata_c260716.nc
export CLM_SURFDATA=/burg-archive/glab/users/pg2328/legoESM/data/clm/surfdata_1.9x2.5_16pfts_CMIP6_simyr2000.nc
export JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1 LEGOESM_JIT_CACHE_DIR=""
OUT=/burg-archive/glab/users/pg2328/fv3_duo_gaps/a_ladder/dt${DT}
mkdir -p "$OUT"
cd "$WT"; echo "sha $(git rev-parse --short HEAD) dirty $(git status --porcelain | grep -v '^??' | wc -l) host $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) dt=$DT phys=$PHYS rad=$RAD"
source "$WT/config/amip/amip_production.ginsburg.sh"
"$PY" -u scripts/run/run_amip.py --config config/amip/amip_production_fv3duo_c24.yaml \
  "${AMIP_PATH_FLAGS[@]}" --dt "$DT" --physics-update-steps "$PHYS" --rad-update-steps "$RAD" \
  --days "${DAYS:-5}" --checkpoint-days 5 --diag-days 1 --output "$OUT"
echo "rc=$?"
grep -m1 "CFL: reducing dt" "$OUT"/*.log 2>/dev/null || true
