#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=probe_edmf
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/probe_edmf_%j.out
set +e
WT=/burg/glab/users/pg2328/scmrce_wt5
cd $WT
echo "PINNED sha: $(git rev-parse HEAD)"
export JAX_PLATFORMS=cpu
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
JAX_ENABLE_X64=1 $PY /burg-archive/glab/users/pg2328/legoESM/scripts/tmp/_probe_edmf_fixture.py 2>&1 | tail -60
echo ALLDONE
