#!/bin/bash
#SBATCH --account=glab --job-name=m6_nopadAB --partition=short --time=04:00:00
#SBATCH --nodes=7 --ntasks=54 --cpus-per-task=4 --exclusive
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/m6_nopad_ab_%j.log
# M8 go/no-go (codex 2026-09-06), ONE allocation: the C96 kt=3 54-rank row
# vs its DIAGNOSTIC arm that skips the per-firing pad refresh (NOT a ladder
# row; the diag arm is expected to fail the bitwise gate and its timing is
# printed as a decomposition).  Pre-registered: t_nopad ~ 0.12 + 0.12 *
# 509/1305 = 0.167 s/step if the step is collective-COUNT bound; ~0.24 s
# refutes it (then packing/batching the pad refresh cannot pay).  The
# 'pack' arm is M8-A itself (HLO census 1305 -> 926 collectives/step).
G=/burg-archive/glab/users/pg2328/fv3_duo_gaps
S=/burg-archive/glab/users/pg2328/legoESM_fv3jax/scripts/cluster/fv3_native/tiled_m6_model_gate_mpi.sbatch
for arm in baseline pack nopad; do
  echo "########## ARM $arm $(date +%H:%M:%S)"
  EXTRA="--timing 20"; [ $arm = nopad ] && EXTRA="--timing 20 --diag-no-pad-refresh"; [ $arm = pack ] && EXTRA="--timing 20 --pack-pad-refresh"
  export N=96 KT=3 PAD=11 REF=$G/ladder_ref_c96_v2.npz GATE_EXTRA="$EXTRA"
  bash "$S" 2>&1 | grep -E "preflight|SLOWEST|PREFLIGHT|rss window step|TIMING|VERDICT|SRUN_RC|bind=|Unknown|error" | grep -v "^\[m6\] rank [1-9]" | cut -c1-200
done
echo DONE nopad_ab
