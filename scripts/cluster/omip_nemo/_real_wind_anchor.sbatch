#!/bin/bash
#SBATCH --job-name=anchortest
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=0:45:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/anchortest_%j.out
#
# Does the 0.86 zero-step length ratio survive NEMO's REAL wind-dependent
# surface anchor?  The probe had been using the windless rn_mxl0 floor
# (0.04 m) where NEMO uses zraug*taum = 7.95*taum -- about 1.6 m under a
# 0.2 N/m2 wind.  A too-short anchor shortens lup near the surface and so
# UNDERSTATES our l_k, which is the direction of the deficit.
#   ARTIFACT : the ratio rises toward 1.0 -- the model's length is fine and
#              the day-30 Southern Ocean shallowing after the sqrt(2) fix is a
#              real physical response, not an exposed length bug.
#   REAL     : the ratio stays near 0.86 -- our length really is ~14% short
#              and that is what the amplitude fix uncovered.
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/legoESM; cd $WT || exit 1
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
T=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD2
/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python -u \
  scripts/validate/ocean_fidelity/nemo_zero_step_closure.py \
  --restart-npz $T/restart_8760_full.npz \
  --mesh-mask $WT/data/grids/eORCA1.2_mesh_mask.nc \
  --trd-tfile $T/ORCA1_1h_20010101_20010101_trd1h_T.nc --trd-rec 0 \
  --out-json $WT/results/omip_nemo/zero_step_closure/real_anchor.json 2>&1 \
  | grep -aE "^\[anchor\]|^\[C[123]\]|^band|^antarctic|^SH_mid|^tropics|^NH_mid|^arctic|^ONE STEP|FATAL"
echo "=== rc=${PIPESTATUS[0]} ==="
