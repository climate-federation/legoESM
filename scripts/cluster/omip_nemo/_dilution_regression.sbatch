#!/bin/bash
#SBATCH --job-name=dilreg
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/dilreg_%j.out
# GLM falsifier (same-data form): virtual-minus-real vs NEMO-minus-real dS.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=$ROOT/results/omip_nemo
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
for ML in 30 10 1.5; do
  echo "##### ML depth $ML m, days 15->30 (recs 2->5)"
  $PY scripts/validate/ocean_fidelity/surface_dilution_regression.py \
    --virtual-early $R/${VIRT:-trp_unified180}/snapshot_day0015.npz --virtual-late $R/${VIRT:-trp_unified180}/snapshot_day0030.npz \
    --real-early $R/${REAL:-trp_fwnemo_180d}/snapshot_day0015.npz --real-late $R/${REAL:-trp_fwnemo_180d}/snapshot_day0030.npz \
    --nemo-gridt $GT --nemo-rec-early 2 --nemo-rec-late 5 --ml-depth $ML 2>&1 | grep -v Warning
done
echo "##### ML 30 m, days 30->90 (recs 5->17)"
$PY scripts/validate/ocean_fidelity/surface_dilution_regression.py \
  --virtual-early $R/${VIRT:-trp_unified180}/snapshot_day0030.npz --virtual-late $R/${VIRT:-trp_unified180}/snapshot_day0090.npz \
  --real-early $R/${REAL:-trp_fwnemo_180d}/snapshot_day0030.npz --real-late $R/${REAL:-trp_fwnemo_180d}/snapshot_day0090.npz \
  --nemo-gridt $GT --nemo-rec-early 5 --nemo-rec-late 17 --ml-depth 30 2>&1 | grep -v Warning
echo DONE
