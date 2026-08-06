#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=zdf_probe
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/zdf_probe_%j.out
set +e
ROOT=/burg/glab/users/pg2328/legoESM; cd $ROOT
export PYTHONPATH="$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== tendency_probe: zdf (NEMO ttrd_zdf analogue) tracer tendency ==="
$PY -m pytest tests/ocean/unit/test_tendency_probe.py -x -q 2>&1 \
  | grep -vE "FutureWarning|warnings.warn|XLA|cuda|CUDA|Plugin|tensorflow" | tail -30
echo "=== DONE ==="
