#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --gres=gpu:1
#SBATCH --job-name=initwind
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/initwind_%j.out
set +e
# The per-case initial-wind tests live in test_wangara_day33_forcing.py, which
# neither of the other les_scm test blocks collects -- so they were written and
# not run. Run them, plus the case-registry tests that build the same columns.
: "${E_WT:?set E_WT}"
cd "$E_WT" || exit 1
echo "sha: $(git rev-parse HEAD)  host: $(hostname)"
export PYTHONPATH=$E_WT/src:$(ls -d "$E_WT"/packages/*/ | tr '\n' ':')
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python -m pytest -q \
    tests/atmosphere/test_wangara_day33_forcing.py \
    tests/validation/test_scm_wangara.py 2>&1 | tail -30
echo "DONE $(date +%T)"
