#!/bin/bash
#SBATCH --job-name=eucpost
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/eucpost_%j.out
#
# IS THE EQUATORIAL UNDERCURRENT STILL SHORT AFTER THE WIND FIX?
#
# Today's committed measurement (b92451ca8) showed the equatorial diffusivity
# deficit is a CONSEQUENCE: the mixing length the closure used is the
# stability-limited one (median l_implied/l_b = 1.697 over 20-73 m), so K_M is
# a slave to the turbulent energy and the cause lives in e's SOURCES.
#
# The obvious source candidate -- too little wind -- is ALREADY FIXED and must
# not be re-opened: ORCA1's namelist reads the Large & Yeager bias-corrected
# _MOD fields, we had been forcing from the raw ones (U_10 4.736 m/s vs
# U_10_MOD 6.279 over nino3, +33%), and core2_nyf_cache_dir() now prefers the
# corrected cache. The production baseline's manifest resolves to
# .../forcing/core2_nyf_mod, so this run already HAS the correction.
#
# What has NOT been measured is whether the undercurrent, and therefore the
# vertical shear that feeds turbulence production, is still short AFTERWARDS.
# Pre-fix arms had the EUC core at 0.354 m/s against NEMO's 0.496, i.e. 71%.
#
# NO NEW PROBE IS WRITTEN. This runs the committed instrument
# scripts/validate/ocean_fidelity/equatorial_thermocline.py, whose EUC block
# already refuses an unmatched comparison: passing --nemo-ufile without
# --nemo-w-recs raises, because an earlier unmatched read produced a bogus
# "FESOM EUC 78% of NEMO". Record 5 of the 5-day means is day 30, the same day
# as snapshot_final.npz and the record every scorecard on this thread uses.
#
# PRE-REGISTERED, so the answer cannot be argued afterwards. On the EUC core
# speed at 200E/220E/240E, ours against NEMO's, matched day 30:
#   * ours < 80% of NEMO  -> the shear deficit SURVIVES the wind fix. The
#     turbulence-production shortfall then has a resolved-flow cause, and the
#     next question is why the undercurrent is weak when the wind driving it
#     is now correct.
#   * ours >= 95% of NEMO -> shear is FINE and the hypothesis is dead: the
#     missing turbulent energy must come from somewhere other than the mean
#     shear, and the eliminated-lever list needs a genuinely new entry.
#   * 80-95% -> partial; report the number, claim no direction, and do not
#     launch anything on it.
# The pre-fix 71% is the reference point: anything near it means the wind
# correction did not propagate into the current, which would itself be a
# finding about the momentum path rather than about turbulence.
#
# This is a MEASUREMENT with an existing, already-reviewed instrument, so it
# does not need a reviewer pass first; the reviewers come in before any FIX
# proposed on the strength of it.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY
SNAP=$ROOT/results/omip_nemo/trp_orca1faith_d30/snapshot_final.npz
MESH=$ROOT/data/grids/eORCA1.2_mesh_mask.nc

ls -l "$SNAP" "$MESH" 2>&1 | cat

$PY -u scripts/validate/ocean_fidelity/equatorial_thermocline.py \
  --legoesm-snapshot "$SNAP" \
  --nemo-gridt "$GW/ORCA1_5d_20000101_20000331_grid_T.nc" \
  --nemo-t-recs 5 \
  --nemo-wfile "$GW/ORCA1_5d_20000101_20000331_grid_W.nc" \
  --nemo-w-var wo --nemo-w-recs 5 \
  --nemo-ufile "$GW/ORCA1_5d_20000101_20000331_grid_U.nc" \
  --nemo-vfile "$GW/ORCA1_5d_20000101_20000331_grid_V.nc" \
  --meridional-lon 200,220,240 \
  --mesh-mask "$MESH" \
  --label trp_orca1faith_d30 2>&1 | cat
echo DONE
