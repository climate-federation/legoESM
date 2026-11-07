#!/bin/bash
#SBATCH --job-name=schoist5
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/schoist5_%j.out
#
# THE MISSING CONTROL.  _score_mxl2_d5.sbatch scored the mxl_choice=2 arm at
# day 5 and reported Southern-Ocean MLD bias -19.8 m (lego 19.9 vs NEMO 39.7,
# corr 0.673).  That number was then read against the day-30 arms
# (nemolev_trp_mxl3ctl_d30 +2.3 m, nemolev_trp_mxl4_d30 +1.4 m) -- but those
# differ from mxl2_d5 in TWO fields, `--tke-mxl-choice` AND `--years`
# (0.0137 = 5 d vs 0.0821 = 30 d), so the comparison is a LEAD-TIME confound,
# not an mxl_choice result.
#
# nemolev_trp_hoist_d5 is the TRUE one-variable control: the manifest diff vs
# nemolev_trp_mxl2_d5 is exactly `--tke-mxl-choice 3` vs `2` and NOTHING else.
# It was never run through compare_omip_nemo, so the d5 A/B has no baseline.
# This job supplies it, with the protocol copied BYTE-FOR-BYTE from
# _score_mxl2_d5.sbatch (same --nemo-gridt, --nemo-month 1, --res-deg 1.0).
#
# PRE-REGISTERED metric: MLD bias in band antarctic_S_of_45S.
#   mxl_choice=2, d5 (measured): -19.8 m, corr 0.673
#   CONFIRMS "lead-time artifact, mxl_choice is NOT a Southern-Ocean lever":
#       control d5 bias also strongly negative, |bias(ctl) - bias(mxl2)| < 5 m.
#   REFUTES: control d5 bias >= -8 m, i.e. choice 2 by itself removes ~12 m of
#       Southern-Ocean mixed layer.  Then mxl_choice IS the SO lever and the
#       d5 arm is a real defect, not a spin-up stage.
#   INDETERMINATE: 5-12 m separation -- report the number, claim no direction.
#
# Secondary (no pre-registration, reported only): SST/SSS bands and zonal means,
# for the same one-variable pair.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=$REPO/results/omip_nemo
MESH=$REPO/data/grids/eORCA1.2_mesh_mask.nc
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc
echo "HEAD: $(git -C $REPO log --oneline -1)"
NFAIL=0
run () { echo; echo "########## $1"; shift; "$@"; local rc=$?; echo "=== rc=$rc"; [ $rc -eq 0 ] || NFAIL=$((NFAIL+1)); }

run "CONTROL (mxl_choice=3) d5 vs NEMO month 1 -- identical protocol to mxl2_d5" \
  $PY -u scripts/validate/compare_omip_nemo.py \
    --legoesm-snapshot $R/nemolev_trp_hoist_d5/snapshot_final.npz \
    --nemo-gridt "$GT" --nemo-month 1 --res-deg 1.0 \
    --grid-label "tripole (mxl_choice=3 control, d5)" \
    --output-dir $R/compare_hoist_d5_m01

echo; echo "=== ALL DONE (failed stages: $NFAIL) ==="
exit $((NFAIL > 0))
