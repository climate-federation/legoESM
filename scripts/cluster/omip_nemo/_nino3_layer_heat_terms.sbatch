#!/bin/bash
#SBATCH --job-name=nino3trm
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/nino3trm_%j.out
#
# WHICH TERM COOLS THE LAYER BENEATH THE TRAPPED HEAT?
#
# Measured tonight (518ca4646): in the nino3 box our heat piles into the TOP
# TWO METRES (+0.415 C over days 25-30) while every layer below ten metres
# cools -- 10-50 m by 0.283 C -- and the column as a whole loses heat. So the
# extra heat is not a source; it is failing to leave the surface downward.
#
# Both reviewers say that state measurement CANNOT name the operator. Codex:
# the bins are cell groups with moving boundaries, and adiabatic heave changes
# a bin's temperature with no mixing deficit at all. GLM: shallow horizontal
# warm-water convergence produces the same signature with mixing untouched.
# Both name the same decisive instrument, a matched timestep-accumulated closed
# budget, which needs applied surface fluxes the run does not export.
#
# THIS IS NOT THAT INSTRUMENT, and it does not pretend to be. It is the
# existing per-term split inside equatorial_thermocline.py, run inside the ONE
# LAYER ITS OWN DOCSTRING AUTHORISES.
#
# THE PROBE'S OWN MEASURED LIMIT, quoted rather than paraphrased (--heat-budget
# help, measured 2026-09-08): "neither side can be closed from the archived
# fields. Ours stores INSTANTANEOUS fluxes, so the terms cannot match a 15-day
# mean tendency; the oracle publishes 5-day MEANS, and a mean diffusivity times
# a mean gradient is not a mean flux -- where its mixing is intermittent the two
# are anticorrelated (avt at 50 m in the cold tongue: median 4.1e-04, max
# 3.8e+01 m2/s), which is what produced a +852 K/month diffusive term at
# 50-150 m. The terms are usable only for ORDER and SIGN in a layer with no
# intermittent mixing (15-25 m), never as a closure."
#
# So the layer is 15-25 m, chosen by the instrument and not by me, and it
# happens to sit inside the 10-50 m layer that cools. Only ORDER and SIGN are
# read. No term is quoted as a closure.
#
# PRE-REGISTERED READING, written before submission:
#
#   VOID FIRST. The probe prints its own control: the three terms summed
#   against the observed dT/dt from the previous snapshot. Its own rule is
#   "a residual LARGER than the vertical term voids the comparison". If that
#   fires, the answer is "this instrument cannot read this layer" and nothing
#   else in the output is interpreted. This band CAN fire and is checked first.
#
#   TRAPPING SUPPORTED   our vertical-diffusion term is smaller in magnitude
#                        than NEMO's by more than a factor 2, while the two
#                        advective terms agree in sign and to within a factor
#                        2. Heat is not being carried down where NEMO carries
#                        it, and advection is not the difference.
#
#   ADVECTION LEADS      the horizontal-advection terms differ between the two
#                        sides by MORE than the vertical-diffusion terms do.
#                        Then GLM's shallow warm convergence is the bigger
#                        discrepancy and vertical mixing is the wrong lever,
#                        whatever the diffusion terms do.
#
#   NEITHER              every term agrees within a factor 2, or the two
#                        differences are comparable. The split does not
#                        isolate a term and will be reported as not isolating
#                        one, not rounded toward trapping.
#
# A sign disagreement on any term is reported as a CONVENTION question first,
# not as physics -- that trap has already cost this campaign once.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge
cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
echo "dirty files: $(git status --porcelain | wc -l)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_orca1faithtint_d30
G=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY
GT=$G/ORCA1_5d_20000101_20000331_grid_T.nc
GW=$G/ORCA1_5d_20000101_20000331_grid_W.nc
GU=$G/ORCA1_5d_20000101_20000331_grid_U.nc
GV=$G/ORCA1_5d_20000101_20000331_grid_V.nc
MESH=/burg-archive/glab/users/pg2328/legoESM/data/grids/eORCA1.2_mesh_mask.nc
SNAP=$R/snapshot_day0030.npz
PREV=$R/snapshot_day0025.npz
for f in "$GT" "$GW" "$GU" "$GV" "$MESH" "$SNAP" "$PREV"; do
  [ -e "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

# Day 30 -> the NEMO record ENDING on day 30 is 30/5-1 = 5.
rec=5
echo "day 30, NEMO record $rec (days $((rec*5))-$(((rec+1)*5)))"

# nino3 longitudes are 150W-90W = 210-270E on the probe's 0..360 convention.
$PY -u scripts/validate/ocean_fidelity/equatorial_thermocline.py \
  --legoesm-snapshot "$SNAP" --label "nino3_d30_15to25m" \
  --nemo-gridt "$GT" --nemo-t-recs "${rec}:$((rec+1))" \
  --nemo-wfile "$GW" --nemo-w-var wo --nemo-w-recs "${rec}:$((rec+1))" \
  --nemo-ufile "$GU" --nemo-vfile "$GV" \
  --mesh-mask "$MESH" \
  --box-budget 210,270 \
  --budget-layer 15,25 \
  --budget-lat-halfwidth 5 \
  --heat-budget \
  --heat-budget-prev "$PREV" \
  --lat-halfwidth 2.0
rc=$?
echo "=== rc=$rc ==="
exit $rc
