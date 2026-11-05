#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/burg/glab/users/pg2328/oracle_audit/f4/logs/clubb_impl_%j.out
# Stage A = committed run.sh steps (dump_cam_settings.py, then the generator from
# clubb_release/run_scripts with the clubb venv python), 3 times: plainA, plainB
# (SPREAD_MODE unset: the committed generator path) and pert (SPREAD_MODE=pert).
# Stage B = analyze_clubb.py in the legoESM env.
set -euo pipefail
export CLUBB_FIXTURE_IMPLEMENTED=1
WT=/burg/glab/users/pg2328/wt_f4
D=/burg/glab/users/pg2328/oracle_audit/f4/clubb_impl
CLUBB_ROOT=/burg/glab/users/pg2328/oracle_audit/spread/clubb_core/clubb_release
LEGO_PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=$D/runs; mkdir -p $R
lscpu | grep -E "Model name|Flags" | cut -c1-200
SET=$R/settings; mkdir -p $SET
PYTHONPATH=$(ls -d $WT/packages/* | tr '\n' ':')$WT JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  $LEGO_PY $WT/scripts/validate/oracle/clubb_core/dump_cam_settings.py $SET
cd $CLUBB_ROOT/run_scripts
export PYTHONDONTWRITEBYTECODE=1
for m in plainA plainB; do
  env -u SPREAD_MODE CLUBB_ROOT=$CLUBB_ROOT $CLUBB_ROOT/.venv-python/bin/python $D/gen_spread_impl.py $SET $R/$m.npz
done
SPREAD_MODE=pert CLUBB_ROOT=$CLUBB_ROOT $CLUBB_ROOT/.venv-python/bin/python $D/gen_spread_impl.py $SET $R/pert.npz
echo STAGE_A_DONE
cd $D
PYTHONPATH=$(ls -d $WT/packages/* | tr '\n' ':')$WT JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  $LEGO_PY $D/analyze_clubb_impl.py $R $R/result.json
echo STAGE_B_DONE
