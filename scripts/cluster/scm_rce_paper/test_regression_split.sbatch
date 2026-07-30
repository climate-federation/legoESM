#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=scmrce_reg
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/scmrce_reg_%j.out
set +e
WT=/burg/glab/users/pg2328/scmrce_wt6
cd $WT
echo "PINNED sha: $(git rev-parse HEAD)"; echo "host: $(hostname)"
export JAX_PLATFORMS=cpu
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
# Split per file: a hard fault in one must not hide the others (the r4 run
# crashed with a fault dump that tail -25 truncated, so the block was
# UNVERIFIED, not green).
for f in tests/unit/test_tiedtke.py tests/unit/test_emanuel.py \
         tests/unit/test_zhang_mcfarlane.py tests/unit/test_kain_fritsch.py \
         tests/unit/test_bechtold_implicit_flux.py \
         tests/atmosphere/hydrostatic/unit/test_edmf_convection_824.py \
         tests/atmosphere/hydrostatic/unit/test_convection.py \
         tests/unit/test_bechtold.py ; do
  echo "############ $f ############"
  JAX_ENABLE_X64=1 $PY -m pytest $f -q -p no:cacheprovider 2>&1 | tail -6
  echo "   exit=${PIPESTATUS[0]}"
done
echo ALLDONE
