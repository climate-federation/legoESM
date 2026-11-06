#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/burg/glab/users/pg2328/oracle_audit/f4/logs/reanalyze_%j.out
H=/burg/glab/users/pg2328/oracle_audit/f4
WT=/burg/glab/users/pg2328/wt_f4
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH=$(ls -d $WT/packages/* | tr "\n" ":")$WT:$H
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
for f in "$@"; do echo "=== $f"; $PY $H/analyze.py $f $H/runs/$f $H/runs/$f/buildgen.log 2>&1 | grep -E "CONTROLS|CLASS_COUNTS|discrete changes|port .* spread|C5 " | cut -c1-300; done
echo REANALYZE_DONE
