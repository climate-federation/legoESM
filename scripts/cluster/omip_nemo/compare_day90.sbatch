#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=omip_cmp90
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cmp90_%j.out

# FIRST real end-to-end comparison: legoESM tripole day-90 (rest-state IC,
# in-step CORE-II forcing) vs NEMO ORCA1 month-3. Heavily caveated (rest-state
# vs Gouretski IC; 90-day spinup) -- proves the pipeline + yields real numbers.
set +e
cd /burg-archive/glab/users/pg2328/legoESM
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMO=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc
echo "=== HOST ==="; hostname; date
$PY scripts/validate/compare_omip_nemo.py \
  --legoesm-snapshot results/omip_nemo/test_newforcing_tripole/snapshot_final.npz \
  --nemo-gridt "$NEMO" --nemo-time-idx 2 \
  --output-dir results/omip_nemo/compare_day90 2>&1 | tail -25
echo "cmp_rc=${PIPESTATUS[0]}"
echo "=== outputs ==="; ls -la results/omip_nemo/compare_day90/ 2>/dev/null
echo "=== DONE ==="
