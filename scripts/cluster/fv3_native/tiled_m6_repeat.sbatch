#!/bin/bash
#SBATCH --account=glab --job-name=m6_rep --partition=short --time=03:00:00
#SBATCH --nodes=7 --ntasks=54 --cpus-per-task=4 --exclusive
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/m6_repeat_%j.log
# NOISE FLOOR: the SAME configuration timed N times inside ONE allocation.
# The M8 ladder's rungs (0.237 / 0.226 / 0.211 / 0.222 s/step) were each
# measured in a DIFFERENT allocation, so none of them is attributable
# until the spread of a repeated identical row is known (2026-09-07).
G=/burg-archive/glab/users/pg2328/fv3_duo_gaps
S=/burg-archive/glab/users/pg2328/legoESM_fv3jax/scripts/cluster/fv3_native/tiled_m6_model_gate_mpi.sbatch
for r in 1 2 3; do
  echo "########## REPEAT $r $(date +%H:%M:%S)"
  export N=96 KT=3 PAD=11 REF=$G/ladder_ref_c96_v2.npz GATE_EXTRA="--timing 20 --refresh-band 4"
  bash "$S" 2>&1 | grep -E "preflight:|TIMING|VERDICT|SRUN_RC" | grep -v "^\[m6\] rank [1-9]" | sort -u | cut -c1-190
done
echo DONE m6_repeat
