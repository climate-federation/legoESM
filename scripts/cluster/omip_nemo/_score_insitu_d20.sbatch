#!/bin/bash
#SBATCH --job-name=scoctl20
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/scoctl20_%j.out
#
# The insitu CONTROL at day 20 -- completing the first fully matched
# one-variable A/B scorecard (n2_mode only; 3 of 556 config leaves differ).
# The signed side was scored by _score_signedn2_d20.sbatch under the identical
# protocol, so these two are directly comparable.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
export PYTHONPATH="$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc
SNAP=$ROOT/results/omip_nemo/nemolev_trp_insitun2_d30/snapshot_day0020.npz
[ -f "$SNAP" ] || { echo "FATAL: no control day-20 snapshot"; exit 1; }
$PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
  --nemo-gridt "$GT" --nemo-month 1 --freeze-clamp-C -1.8 --sss-faithful \
  --grid-label insitun2_d20 \
  --output-dir $ROOT/results/omip_nemo/compare_insitun2_d20_m01 2>&1 | tail -55
echo "=== rc=$?"
