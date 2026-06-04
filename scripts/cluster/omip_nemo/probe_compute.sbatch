#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=omip_probe
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/probe_%j.out

# OMIP-faithful iter-1 probe: determine the data-staging path before
# committing to a NEMO acquisition strategy.
#   1. Does a compute node have outbound internet? (decides where multi-GB
#      NEMO/container/forcing downloads must run vs the login node.)
#   2. Does jn2808's conda env import pg2328's legoesm source correctly?
#   3. Is singularity usable on the compute node (container path)?
set +e
echo "=== HOST ==="; hostname; date

echo "=== COMPUTE-NODE INTERNET ==="
timeout 20 curl -sI https://forge.nemo-ocean.eu 2>&1 | head -1
echo "curl_forge_rc=$?"
timeout 20 git ls-remote https://github.com/alexis-barge/ORCA1.git 2>&1 | head -1
echo "git_orca1_rc=$?"
timeout 20 curl -sI https://github.com 2>&1 | head -1
echo "curl_github_rc=$?"

echo "=== LEGOESM IMPORT (pg2328 src via jn2808 env) ==="
export PYTHONPATH=/burg-archive/glab/users/pg2328/legoESM/src:$PYTHONPATH
JAX_PLATFORMS=cpu /burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python -c \
  "import legoesm, jax; print('legoesm:', legoesm.__file__); print('jax:', jax.__version__)" 2>&1 | tail -5

echo "=== SINGULARITY ==="
module load singularity/3.7.1 2>&1 | tail -2
singularity --version 2>&1
echo "singularity_rc=$?"
# Test container pull+run on compute node (exercises registry internet + runtime).
timeout 120 singularity exec docker://alpine:3.19 echo "container_ok" 2>&1 | tail -3
echo "container_exec_rc=$?"
echo "=== DONE ==="
