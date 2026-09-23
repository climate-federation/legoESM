#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=n2conv
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:25:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/n2conv_%j.out
#
# WHY don't we convect at the equatorial surface when NEMO does 44% of the
# time?  The 5-day-mean K verdict found our surface avm/avt 4x too small and
# conv-frac 0.00 vs NEMO 0.44.  Convection on the production card is handled
# ONLY by the TKE closure's mixing-length blowup where N2<0 (convection="none",
# no separate adjustment).  So either (a) our near-surface N2 is >=0 where
# NEMO's is <0 -- an N2/EOS defect the closure can't overcome, or (b) our N2
# matches NEMO's instability but the closure fails to turn it into convective
# K -- a closure defect.  This reads existing snapshots, runs no model.
#
# Production already uses n2_mode="nemo_bn2" (the corrected N2), so the OLD
# insitu-compressibility bug may be fixed; this checks whether it still bites.
#
# DISCRIMINATOR: our N2 (via the model's OWN compute_buoyancy_frequency_nemo_
# bn2) vs NEMO's published bn2, matched box + record, near surface.
#   ours >= 0 where NEMO < 0  -> N2/EOS defect (the trigger never sees the
#                                instability) = case (a)
#   ours ~ NEMO both < 0      -> closure defect (sees it, doesn't convect) =
#                                case (b)
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
GW_W=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_W.nc
# day-30 snapshot from the kmean arm (production card, merged #1645).
SNAP=$ROOT/results/omip_nemo/nemolev_trp_kmean_d30/snapshot_day0030.npz
[ -f "$SNAP" ] || { echo "no day-30 snapshot"; exit 1; }
$PY -u scripts/validate/ocean_fidelity/equatorial_n2_vs_nemo.py \
  --snapshot "$SNAP" --nemo-gridw "$GW_W" --rec 5 --depth-max 150 2>&1 | tail -80
echo DONE
