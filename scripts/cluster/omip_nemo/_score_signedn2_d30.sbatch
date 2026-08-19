#!/bin/bash
#SBATCH --job-name=scoenso
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/scoenso_%j.out
#
# ABSOLUTE skill of the signed-TEOS-10 arm 9409057 against NEMO at day 30.
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
SNAP=$ROOT/results/omip_nemo/nemolev_trp_signedn2_d30/snapshot_day0030.npz
echo "HEAD: $(git -C $ROOT log --oneline -1)"
[ -f "$SNAP" ] || { echo "FATAL: no snapshot $SNAP"; exit 1; }
[ -f "$GT" ] || { echo "FATAL: no NEMO grid_T $GT"; exit 1; }
$PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
  --nemo-gridt "$GT" --nemo-month 1 --freeze-clamp-C -1.8 --sss-faithful \
  --grid-label signedn2_d30 \
  --output-dir $ROOT/results/omip_nemo/compare_signedn2_d30_m01_enso 2>&1 | tail -60
echo "=== rc=$?"
