#!/bin/bash
#SBATCH --job-name=mxlflux
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/mxlflux_%j.out
#
# CODEX'S CHECK, the half my first discriminator did not cover.
#
# The K-profile run settled one branch: the closure's heat diffusivity DID
# move between mixing length 3 and 4 (down ~11% above 13 m, up 6-18% at
# 15-21 m), so the dsigma=0.01 mixed-layer depth was reporting real physics
# and not a threshold sliding in weak stratification.
#
# What it did NOT settle is whether that diffusivity change reaches the HEAT
# BUDGET.  Entrainment cooling goes as the diffusivity TIMES the temperature
# gradient at the mixing base, and a large fractional change in K buys nothing
# where the water is already well mixed.  This probe computes the downgradient
# flux F = -K dT/dz per column before averaging, which is why it is the right
# instrument: a mean diffusivity times a mean gradient is not a mean flux.
#
# READ IT: flux through 15-25 m moved between the arms -> the lever is real,
# merely weak, and pushing it harder is worth costing.  Flux unchanged while
# K moved -> the closure rearranged its diffusivity where the water carries no
# temperature contrast, and depth is the wrong variable entirely.
#
# Record 0:1 is the five-day window ENDING at day 5, matching both snapshots.
set -uo pipefail
R=/burg-archive/glab/users/pg2328/legoESM; cd $R || exit 1
M=/burg-archive/glab/users/pg2328/_omip_merge
echo "MAIN HEAD: $(git log --oneline -1)"
echo "PROBE TREE: $(git -C $M log --oneline -1)"
export PYTHONPATH="$R/src:$(ls -d $R/packages/*/ | sed 's:/$::' | tr '\n' ':')$R:$R/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
CTL=$R/results/omip_nemo/trp_faithz0_d30/snapshot_day0005.npz
ARM=$R/results/omip_nemo/trp_mxl4_d5/snapshot_final.npz

for f in "$GT" "$CTL" "$ARM"; do
  [ -r "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

for pair in "mxl3_ctl:$CTL" "mxl4_arm:$ARM"; do
  LAB=${pair%%:*}; SNAP=${pair#*:}
  echo "############ $LAB ############"
  echo "$SNAP"
  $PY -u $M/scripts/validate/ocean_fidelity/equatorial_thermocline.py \
    --legoesm-snapshot "$SNAP" --nemo-gridt "$GT" --nemo-t-recs 0:1 \
    --label "$LAB" 2>&1 | tail -220
  echo "### $LAB rc=$?"
done
echo DONE
