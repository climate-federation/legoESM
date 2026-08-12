#!/bin/bash
#SBATCH --job-name=cmpadv
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cmpadv_%j.out
#
# Stage C: our tracer-advection operator vs NEMO's ttrd_totad on the RUN_TRD2
# hourly file.  Advection is the FIRST tracer operator of the RK3 stage, so
# no reconstruction is needed -- the state it acts on is record r itself.
# Two schemes scored: fct2 (= NEMO's nn_fct_h/v=2, tests our implementation)
# and superbee (our production choice, measures what that choice costs).
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
export PYTHONPATH="$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
TRD2=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD2
MESH=/burg-archive/glab/users/pg2328/legoESM/data/grids/eORCA1.2_mesh_mask.nc

$PY -u scripts/validate/ocean_fidelity/compare_advection_nemo.py \
  --tfile $TRD2/ORCA1_1h_20010101_20010101_trd1h_T.nc \
  --ufile $TRD2/ORCA1_1h_20010101_20010101_trd1h_U.nc \
  --vfile $TRD2/ORCA1_1h_20010101_20010101_trd1h_V.nc \
  --mesh-mask "$MESH" --rec 1 \
  --output-dir $REPO/results/omip_nemo/advection_match
rc=$?
echo "=== rc=$rc ==="
exit $rc
