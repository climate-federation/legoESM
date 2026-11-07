#!/bin/bash
#SBATCH --job-name=clipfix
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/clipfix_%j.out
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/legoESM; cd $WT || exit 1
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "HEAD: $(git -C $WT log --oneline -1)"
$PY -m pytest -q tests/ocean/unit/test_teos10_rab_bn2.py -v 2>&1 | tail -22
echo "=== teos rc=$? ==="
$PY -m pytest -q tests/ocean/unit/test_mpas_tke.py tests/ocean/unit/test_tripole_vmix_cli.py \
   tests/unit/test_run_omip_core2_kappa_convention_cli.py tests/ocean/unit/test_tke_nemo_identity.py 2>&1 | tail -8
echo "=== suites rc=$? ==="
