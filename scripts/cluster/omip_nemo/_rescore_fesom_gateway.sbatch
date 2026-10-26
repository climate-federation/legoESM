#!/bin/bash
#SBATCH --job-name=fesgw
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/fesgw_%j.out
#
# THE MATCHED CONTROL FOR THE FESOM FORCING ARM.
#
# The new FESOM arm reads the corrected CORE-II fields and is scored against
# NEMO's matched cold start.  Its predecessors were scored against the
# five-year January climatology, so the improvement cannot be read off the two
# numbers -- the reference changed as well as the forcing.  This re-scores the
# predecessors, from their own day-30 snapshots, against the SAME reference
# record the new arm used, so forcing is the only thing that differs.
#
#   fesom_anchor_d30  raw winds, Dirichlet surface TKE + mixing-length anchor
#   fesom_dirich_d30  raw winds, Dirichlet without the anchor
#   fesom_nyfice_d30  raw winds, NEMO January ice state
#   fesom_nyf_d30     raw winds, the FESOM reference card
#
# `fesom_anchor_d30` is the one that matches the new arm's configuration, so
# that pair is the one-variable comparison; the others are context.
set -uo pipefail
R=/burg-archive/glab/users/pg2328/legoESM; cd $R || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$R/src:$(ls -d $R/packages/*/ | sed 's:/$::' | tr '\n' ':')$R:$R/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
GW_T=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc

for ARM in fesom_anchor_d30 fesom_dirich_d30 fesom_nyfice_d30 fesom_nyf_d30; do
  SNAP=$R/results/omip_nemo/$ARM/snapshot_day0030.npz
  echo
  echo "############ $ARM vs GATEWAY rec 5 ############"
  [ -f "$SNAP" ] || { echo "SKIP: no $SNAP"; continue; }
  $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
    --nemo-gridt "$GW_T" --nemo-time-idx 5 --freeze-clamp-C -1.8 --sss-faithful \
    --grid-label ${ARM}_gw \
    --output-dir $R/results/omip_nemo/compare_gw_$ARM 2>&1 | tail -34
  echo "### $ARM rc=$?"
done
echo DONE
