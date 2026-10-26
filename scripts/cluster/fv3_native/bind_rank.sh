#!/bin/bash
# Pin this SLURM task to its own --cpus-per-task cores (task/cgroup here
# binds nothing, so every rank otherwise sees all 32 cores and XLA:CPU's
# Eigen pool spawns 32 threads per rank).  Usage: srun ... bind_rank.sh CMD...
C=${SLURM_CPUS_PER_TASK:-4}
exec taskset -c $((SLURM_LOCALID * C))-$((SLURM_LOCALID * C + C - 1)) "$@"
