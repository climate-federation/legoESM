#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=grad_cond
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/grad_cond_%j.out
set +e
# NO grep on the driver's output. The previous attempt at this measurement
# filtered its own stdout to the lines it expected, so when every arm died on a
# refused --hours the log showed four section headers and nothing else -- a
# guard that hides the failure it exists to surface. Print everything; the
# interesting lines are few.
: "${E_WT:?set E_WT}"
cd "$E_WT" || exit 1
echo "PINNED sha: $(git rev-parse HEAD)  host: $(hostname)"
export PYTHONPATH=$E_WT/src:$(ls -d "$E_WT"/packages/*/ | tr '\n' ':')
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
for S in clubb smagorinsky; do
  for C in bomex cbl; do
    echo "########## $S on $C ##########"
    $PY scripts/validate/scm_gradient_conditioning.py \
        --scheme "$S" --case "$C" --hours 0.25 0.5 1 2 4 2>&1 \
      | grep -vE "slow_operation|^E[0-9]{4}|XLA has|If you|constant-folding"
    echo "  rc=$?"
  done
done
echo "DONE $(date +%T)"
