#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=iso_nh
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/scmrce_iso_nh_%j.out
set +e
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
T=tests/atmosphere/hydrostatic/unit/test_convection.py::TestIntegration::test_nonhydrostatic_nonzero_heating
for WT in /burg/glab/users/pg2328/scmrce_baseline /burg/glab/users/pg2328/scmrce_wt6; do
  cd $WT
  echo "######## $WT  sha=$(git rev-parse --short HEAD) ########"
  export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')
  JAX_ENABLE_X64=1 $PY -m pytest "$T" -q -p no:cacheprovider 2>&1 | tail -18
  echo "   exit=${PIPESTATUS[0]}"
done
echo ALLDONE
