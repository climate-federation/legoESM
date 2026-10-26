#!/bin/bash
#SBATCH --job-name=teos10
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/teos10_%j.out
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/legoESM; cd $WT || exit 1
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "HEAD: $(git -C $WT log --oneline -1)"
$PY -m pytest -q tests/ocean/unit/test_teos10_rab_bn2.py -v 2>&1 | tail -25
echo "=== new tests rc=$? ==="
$PY -m pytest -q tests/ocean/unit -k "eos or bn2 or nemo_identity" 2>&1 | tail -8
echo "=== eos suites rc=$? ==="
