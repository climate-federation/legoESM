#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=zthread1b
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=2:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/zthread1b_%j.out
# Stage-1b: is the broad-slice SIGABRT mine or pre-existing?  Same pytest -k
# slice twice: clean HEAD worktree (no threading edits) vs the working tree.
# 120G (the 48G job died with a glibc abort = likely OOM in one big jax
# process), and full tails so the dying test is visible this time.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
WT=/local/zthread_wt_$SLURM_JOB_ID
git worktree add --detach "$WT" HEAD >/dev/null 2>&1 || { echo "FATAL worktree"; exit 1; }

run_slice () {  # label  tree
  echo "########## $1 ($2) ##########"
  ( cd "$2" && \
    PYTHONPATH="$(ls -d $2/packages/*/ | sed 's:/$::' | tr '\n' ':')$2" \
    $PY -m pytest -q tests/ocean/unit -k "tke or implicit or ab2 or leapfrog or mlf or step" \
    -x 2>&1 | tail -25 )
  echo "### $1 rc=$?"
}
run_slice CLEAN_HEAD "$WT"
run_slice WORKING_TREE "$ROOT"
git worktree remove --force "$WT" >/dev/null 2>&1 || true
echo DONE
