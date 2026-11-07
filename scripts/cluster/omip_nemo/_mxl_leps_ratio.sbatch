#!/bin/bash
#SBATCH --job-name=lepsratio
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/lepsratio_%j.out
#
# THE CHEAP FALSIFIER for the nn_mxl=2 day-90 arm (job 9398236).
#
# That arm costs ~30 h to answer "does NEMO's nn_mxl=2 cure the Southern Ocean
# autumn over-deepening".  This job costs minutes and answers the PRIOR
# question the arm assumes: is the difference between nn_mxl=2 and nn_mxl=3
# seasonally selective at all?  The two settings share the eddy-coefficient
# length and differ only in the dissipation length, sqrt(lup*ldn) vs
# min(lup,ldn), so their ratio R >= 1 measures the whole separation.
#
# CONTROLLED BY CONSTRUCTION: day 30 and day 90 are two snapshots of the SAME
# run (nemolev_trp_mlefix_d90) -- one config, one IC, one code revision, two
# times.  Nothing but the season differs.
#
# PRE-REGISTERED READ, Southern Ocean band, 0-100 m, at the WINDLESS l_sfc
# (the optimistic end -- a bigger wind anchor only shrinks R):
#   MECHANISM LIVE : R(d90) >= 1.3 AND R(d90) - R(d30) >= 0.15.  The settings
#       genuinely diverge in autumn and agree in summer, which is the pattern
#       the measured MLD errors show (+19 m at d90, +1.4 m at d30).
#   MECHANISM DEAD : R(d90) <= 1.1.  nn_mxl cannot move a 19 m mixed layer by
#       changing dissipation a few percent -> CANCEL job 9398236 and go to the
#       next suspect (NEMO nn_mle=1 uses hmlef, not the turbulent hmld, for the
#       Fox-Kemper streamfunction; and the ~3x tracer-diffusivity excess from
#       Stage A, whose target is contaminated by EVD+IWM inside NEMO's avt).
#   INDETERMINATE  : 1.1 to 1.3 -- let the arm run, report the number, claim
#       no direction.
# NOTE the day-30 value is a PREDICTION CHECK with a known answer: the measured
# day-30 MLD separation between the two settings is 0.9 m, so R(d30) must be
# near 1.  If the probe reports a large R at day 30 it is contradicting a
# measurement and the PROBE is wrong, not the model.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=$REPO/results/omip_nemo
MESH=$REPO/data/grids/eORCA1.2_mesh_mask.nc
ARM=$R/nemolev_trp_mlefix_d90
echo "HEAD: $(git -C $REPO log --oneline -1)"
NFAIL=0
run () { echo; echo "########## $1"; shift; "$@"; local rc=$?; echo "=== rc=$rc"; [ $rc -eq 0 ] || NFAIL=$((NFAIL+1)); }

for D in 0030 0060 0090; do
  run "l_eps ratio, day $D (same run, season is the only difference)" \
    $PY -u scripts/validate/ocean_fidelity/mxl_dissipation_length_ratio.py \
      --snapshot $ARM/snapshot_day$D.npz --mesh-mask "$MESH" \
      --label mlefix_d$D \
      --out-json $R/mxl_leps_ratio/mlefix_d$D.json
done

echo; echo "=== ALL DONE (failed stages: $NFAIL) ==="
exit $((NFAIL > 0))
