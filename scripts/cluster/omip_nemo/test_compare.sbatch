#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=omip_cmp
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cmp_%j.out

# Validate compare_omip_nemo.py end-to-end on existing smoke artifacts
# (legoESM 10-day final vs NEMO 1-month smoke grid_T). Not scientific (different
# durations/ICs) -- just checks load/regrid/metrics/plots/report all work.
set +e
cd /burg-archive/glab/users/pg2328/legoESM
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMO_SMOKE=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_SMOKE/ORCA1_1m_20000101_20000201_grid_T.nc
echo "=== HOST ==="; hostname; date
$PY scripts/validate/compare_omip_nemo.py \
  --legoesm-snapshot results/omip_nemo/legoesm_tripole/snapshot_final.npz \
  --nemo-gridt "$NEMO_SMOKE" --nemo-time-idx 0 \
  --output-dir results/omip_nemo/compare_smoketest 2>&1 | tail -30
echo "cmp_rc=${PIPESTATUS[0]}"
echo "=== outputs ==="; ls -la results/omip_nemo/compare_smoketest/ 2>/dev/null
echo "=== DONE ==="
