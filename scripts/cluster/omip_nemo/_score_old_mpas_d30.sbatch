#!/bin/bash
#SBATCH --job-name=oldmpascore
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/oldmpascore_%j.out
#
# Score the PRE-HARMONISATION MPAS run against NEMO at day 30.
#
# Why this run exists as a scoring target at all: _mpas_mixing_harm_d30.sbatch
# pre-registers CLOSER / NEUTRAL / WORSE as movement "toward the tripole's
# value RELATIVE TO THE OLD MPAS RUN". Two of those three numbers were already
# in hand (harmonised MPAS, and the tripole once it lands); the third was not,
# so the card's own bands could not be evaluated. This supplies it.
#
# mpas_orca1faithtint_d30 is the right reference, verified rather than assumed:
# its manifest passes NONE of --A-h, --C-smag-lap, --gm-treguier,
# --gm-kappa-min, which is exactly the condition the 2026-09-20 manifest audit
# described when it found the MPAS lane running library defaults.
#
# PROTOCOL IS HELD BYTE-IDENTICAL to the harmonised MPAS scorecard
# (results/omip_nemo/compare/mpas_mixharm_d30/report.json records
# nemo_time_idx = 5): same tool, same NEMO file, same record, same grid label,
# and every other option left at its default on both sides. A scorecard
# computed at a different record or resolution would be a confound, not a
# comparison.
#
# The command is echoed below before it runs. Recovering the harmonised run's
# exact invocation required reading its output JSON because the job log did not
# contain the command line; this card does not repeat that gap.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_fluxacc_wt
MAIN=/burg-archive/glab/users/pg2328/legoESM
cd $ROOT || exit 1
echo "=== HEAD $(git -C $ROOT rev-parse HEAD)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
export JAX_PLATFORMS=cpu
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
SNAP=$MAIN/results/omip_nemo/mpas_orca1faithtint_d30/snapshot_day0030.npz
[ -f "$SNAP" ] || { echo "REFUSING: $SNAP absent"; exit 3; }
CMD="$PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot $SNAP --nemo-gridt $GT --nemo-time-idx 5 --grid-label mpas --output-dir $MAIN/results/omip_nemo/compare/mpas_old_d30"
echo "CMD: $CMD"
$CMD
rc=$?; echo "=== rc=$rc ==="
exit $rc
