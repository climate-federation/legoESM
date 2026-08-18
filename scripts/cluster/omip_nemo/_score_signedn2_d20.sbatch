#!/bin/bash
#SBATCH --job-name=scored20
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/scored20_%j.out
#
# DAY-20 INSURANCE. The matched control (9412975) is tight on walltime: day 15
# at 6h15 elapsed, needing ~5.1h more against 5.5h remaining. If it dies short
# of day 30 the pre-registered day-30 A/B is lost, but BOTH arms snapshot every
# 10 days, so day 20 is a valid matched pair at a shorter lead time. Scoring
# the signed side now so the comparison is one job away either way.
#
# NOT a substitute for the day-30 verdict -- shorter lead time, and the
# pre-registered thresholds were set at day 30.
#
# This is NOT the A/B -- the matched insitu control (9412975) is still
# running and the one-variable verdict waits for it. This answers the
# separate and independently useful question the task actually asks: how
# close is SST/SSS/MLD to NEMO right now.
#
# Same protocol as every other arm scored in this campaign (NEMO month 1,
# freeze clamp -1.8 C, --sss-faithful), so it is comparable to them.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
export PYTHONPATH="$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc
SNAP=$ROOT/results/omip_nemo/nemolev_trp_signedn2_d30/snapshot_day0020.npz
echo "HEAD: $(git -C $ROOT log --oneline -1)"
[ -f "$SNAP" ] || { echo "FATAL: no snapshot $SNAP"; exit 1; }
[ -f "$GT" ] || { echo "FATAL: no NEMO grid_T $GT"; exit 1; }
$PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
  --nemo-gridt "$GT" --nemo-month 1 --freeze-clamp-C -1.8 --sss-faithful \
  --grid-label signedn2_d20 \
  --output-dir $ROOT/results/omip_nemo/compare_signedn2_d20_m01 2>&1 | tail -45
echo "=== rc=$?"
