#!/bin/bash
#SBATCH --job-name=mxlkdisc
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/mxlkdisc_%j.out
#
# GLM'S DISCRIMINATOR, RUN ON OUTPUT THAT ALREADY EXISTS.
#
# The TKE mixing-length arm deepened the nino3 mixed layer by 0.42 m and moved
# day-5 SST by +0.002 C.  Two readings survive that: the mixed-layer lever is
# weak, or dsigma=0.01 MLD is the wrong diagnostic -- a threshold crossing that
# can slide by metres in weak stratification while the turbulent mixing depth
# and the heat flux do not move at all.
#
# The check that separates them is the closure's OWN heat diffusivity profile
# in the cold-tongue box, mixing length 3 versus 4:
#   K_T moved   -> the lever is real but weak, and MLD is reporting it faithfully
#   K_T unchanged while MLD moved -> the diagnostic moved, not the physics, and
#                                    the arm tested nothing
#
# This costs no GPU time.  Both snapshots already carry the model's own
# K_M_diag / K_H_diag, which is what the probe reads; an earlier check of mine
# looked for "A_v"/"K_v" (the names in the probe's docstring) and wrongly
# reported the fields absent.
#
# Record 0 is the five-day window ENDING at day 5, matching both snapshots.
# The warm-pool box is the within-oracle control: if the profiles agree there
# too, a cold-tongue-only comparison would be reading a constant.
set -uo pipefail
R=/burg-archive/glab/users/pg2328/legoESM; cd $R || exit 1
M=/burg-archive/glab/users/pg2328/_omip_merge
echo "MAIN HEAD: $(git log --oneline -1)"
echo "PROBE TREE: $(git -C $M log --oneline -1)"
export PYTHONPATH="$R/src:$(ls -d $R/packages/*/ | sed 's:/$::' | tr '\n' ':')$R:$R/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_W.nc
CTL=$R/results/omip_nemo/trp_faithz0_d30/snapshot_day0005.npz
ARM=$R/results/omip_nemo/trp_mxl4_d5/snapshot_final.npz

for f in "$GW" "$CTL" "$ARM"; do
  [ -r "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done
echo "CTL (mixing length 3): $CTL"
echo "ARM (mixing length 4): $ARM"

echo "############ cold tongue, 200-260E ############"
$PY -u $M/scripts/validate/ocean_fidelity/equatorial_diffusivity_oracle.py \
  --nemo-gridw "$GW" --rec 0 --lon-west 200 --lon-east 260 \
  --legoesm-snapshot "$CTL" "$ARM"
echo "### rc=$?"

echo
echo "############ warm pool, 140-180E, within-oracle control ############"
$PY -u $M/scripts/validate/ocean_fidelity/equatorial_diffusivity_oracle.py \
  --nemo-gridw "$GW" --rec 0 --lon-west 140 --lon-east 180 \
  --legoesm-snapshot "$CTL" "$ARM"
echo "### rc=$?"
echo DONE
