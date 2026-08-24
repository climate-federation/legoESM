#!/bin/bash
#SBATCH --job-name=arcfct
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/arcfct_%j.out
#
# THE MIXING-LENGTH A/B, scored on the halocline metric -- using arms that
# ALREADY EXIST, so this costs no GPU.
#
# THE MISMATCH: NEMO ORCA1 runs nn_mxl = 2 (namelist_cfg:444).  Our control
# runs --tke-mxl-choice 3.  The numbering is Veros-derived and does NOT map
# 1:1 (config.py:246-250): our choice 4 IS NEMO nn_mxl=2 (a SINGLE length,
# l_eps = l_k = min(lup,ldn)), while our choice 3 is NEMO nn_mxl=3, whose l_k
# is the LARGER geometric mean.  K = l_k*sqrt(e), so we have been running the
# longer mixing length -- the right sign for the observed over-mixing.
#
# METRIC (pre-registered): top-19 m Arctic volume-weighted salinity SPREAD.
#   IC day 0      0.740   <- what we start with
#   NEMO month 1  0.827   <- the target
#   control d30   0.018   <- what we currently end with
# CONFIRMS: mxl4 spread is substantially closer to 0.74/0.83 than 0.018.
# REFUTES : mxl4 spread is still ~0.02, i.e. the length choice is not the
#           lever and the remaining suspect is the DISSIPATION.
# INDETERMINATE: a partial move (say 0.05-0.15) -- report the number, do not
#           call it a fix.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
export PYTHONPATH="$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
MESH=$REPO/data/grids/eORCA1.2_mesh_mask.nc
R=$REPO/results/omip_nemo
for lab in fct2_d30; do
  SNAP=$R/nemolev_trp_$lab/snapshot_final.npz
  [ -r "$SNAP" ] || { echo "SKIP $lab (no snapshot)"; continue; }
  echo; echo "########## $lab"
  $PY -u scripts/validate/ocean_fidelity/arctic_salt_vs_volume.py \
    --snapshot "$SNAP" --mesh-mask "$MESH" --label "$lab" --profile \
    --out-json $R/arctic_salt_volume/$lab.json
done
echo "=== rc=$? ==="
