#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=e025_4gpu_mn
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/e025_4gpu_mn_%j.out

# Multi-NODE 4-GPU eORCA025 (1/4 deg) tripole SMOKE for the lat-band SPMD ocean
# step under jax.distributed.  4 GPUs across 2 nodes (2 GPUs/node) -- the f64 1/4
# deg step needs ~37 GB/GPU at N=4, which fits the 48 GB RTX8000 but OOMs at N<=2
# on a single node (~75 GB).  One MPI process per GPU; the bootstrap (--distributed)
# initializes jax.distributed with rank-0's hostname as coordinator, then
# jax.devices() spans all 4 GPUs and create_latlon_mesh shards the "lat" band axis
# across both nodes.  SMOKE only (10 sim-days): reports steps/s + per-GPU memory +
# that it fits; NOT the full multi-year integration.
set +e
# Run from the MAIN checkout so the gitignored runtime data (the 7.3 GB eORCA025
# mesh, CORE-II forcing, WOA) at data/ resolve via the existing relative paths --
# a git worktree does NOT carry gitignored data.  But run the WORKTREE's EDITED
# packages by PREPENDING them to PYTHONPATH (the env's editable install points at a
# third checkout), so the multi-node --distributed code under test is what runs.
MAIN=/burg-archive/glab/users/pg2328/legoESM
WT=/burg-archive/glab/users/pg2328/legoESM/.claude/worktrees/multinode
cd "$MAIN"
export PYTHONPATH="$WT/packages/core:$WT/packages/ocean:$WT/packages/atmosphere:$WT/packages/coupler:$WT/packages/ice:$WT/packages/land:$WT/packages/ml:$WT/packages/tools:$PYTHONPATH"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo "=== ALLOC ==="; date
echo "nodes: $SLURM_JOB_NODELIST  ntasks: $SLURM_NTASKS  gpus/node: 2"
scontrol show hostnames "$SLURM_JOB_NODELIST"

# Per-process GPU selection: each task sees BOTH node GPUs; the bootstrap passes
# local_device_ids=SLURM_LOCALID to jax.distributed.initialize so each process
# claims its OWN local GPU.  We deliberately do NOT use --gpu-bind=single:1 here:
# re-indexing every task's GPU to 0 makes jax's cross-node global-topology
# key-value exchange DEADLOCK (DEADLINE_EXCEEDED on key cuda:global_topology,
# observed at 4 GPUs / 2 nodes) because the co-located processes cannot
# distinguish their devices.  Declaring local_device_ids explicitly is the
# jax-on-SLURM idiom that makes the topology exchange deterministic.  (We also do
# NOT write a node-local /tmp wrapper -- /tmp is per-node, so the 2nd node's ranks
# could not exec it -> earlier PMIx hang.)
LOG=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/e025_4gpu_mn_${SLURM_JOB_ID}_full.log

# srun launches one process per task (4 total, 2 per node).  jax.distributed
# discovers rank/coordinator from MPI (mpi4py reads the SLURM PMI env) and the
# per-process GPU from SLURM_LOCALID; the SPMD halo is pure-JAX ppermute/psum
# across processes (no mpi4jax).  Env is passed inline via `env` (no wrapper file).
# tee the full log so we capture the REAL srun rc (the `| tail` would mask it) AND
# scan for the success/abort markers.  The DRIVER script (with the --distributed
# edits) lives in the WORKTREE; cwd is MAIN only for the relative data/ paths.
srun --mpi=pmix \
  env JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
  $PY "$WT/scripts/run/run_omip_core2.py" \
    --grid tripole \
    --mesh data/grids/eORCA025_mesh_mask.nc \
    --n-gpus 4 --distributed \
    --barotropic-solver implicit_cn \
    --smoke --dt 600 --diag-every-days 2 \
    --output results/omip_nemo/e025_4gpu_multinode_smoke 2>&1 | tee "$LOG" | tail -120
rc=${PIPESTATUS[0]}

echo "e025_4gpu_mn_rc=${rc}"
echo "=== smoke verdict ==="
if [ "$rc" -ne 0 ]; then
  echo "[SMOKE-FAIL] srun exited rc=${rc}"
elif grep -q "\[ABORT\]" "$LOG"; then
  echo "[SMOKE-FAIL] run hit a non-finite [ABORT]"; rc=1
elif grep -q "\[done\]" "$LOG"; then
  echo "[SMOKE-OK] reached [done]; steps/s + projected wall-time above"
else
  echo "[SMOKE-INCOMPLETE] no [done]/[ABORT] marker (check $LOG)"; rc=1
fi
echo "steps/s + per-GPU memory are in the run log: $LOG"
echo "=== GPU MEM (last sample, this node) ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv 2>&1 | head -4
echo "=== DONE ==="
exit ${rc}
