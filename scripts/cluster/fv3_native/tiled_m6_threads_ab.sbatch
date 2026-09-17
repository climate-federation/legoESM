#!/bin/bash
#SBATCH --account=glab --job-name=m6_thrAB --partition=short --time=04:00:00
#SBATCH --nodes=7 --ntasks=54 --cpus-per-task=4 --exclusive
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/m6_threads_ab_%j.log
# A/B inside ONE allocation (same nodes, same hour): the XLA:CPU intra-op
# thread pool at its default (every core) vs pinned to --cpus-per-task.
# Both arms are the C96 kt=3 54-rank row, --timing 10, bitwise-gated.
G=/burg-archive/glab/users/pg2328/fv3_duo_gaps
S=/burg-archive/glab/users/pg2328/legoESM_fv3jax/scripts/cluster/fv3_native/tiled_m6_model_gate_mpi.sbatch
for arm in B_bind A_nobind; do
  echo "########## ARM $arm $(date +%H:%M:%S)"
  if [ $arm = B_bind ]; then export BIND=1; else export BIND=0; fi
  export N=96 KT=3 PAD=11 REF=$G/ladder_ref_c96_v2.npz GATE_EXTRA="--timing 10"
  bash "$S" 2>&1 | grep -E "rss window step|TIMING|VERDICT|SRUN_RC|bind=|Unknown|error|taskset" | grep -v "rank [1-9]" | cut -c1-200
done
echo DONE threads_ab
