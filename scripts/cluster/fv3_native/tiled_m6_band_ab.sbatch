#!/bin/bash
#SBATCH --account=glab --job-name=m6_bandAB --partition=short --time=03:00:00
#SBATCH --nodes=7 --ntasks=54 --cpus-per-task=4 --exclusive
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/m6_bandAB_%j.log
# BAND A/B inside ONE allocation, interleaved full/band/band/full so a
# drift shows up.  The band refresh is the only flag-selectable M8 lever
# and so the only one whose gain can be attributed: within-allocation
# repeatability is +-1 ms (job 9675236: 0.218/0.220/0.219) but the SAME
# code measured 0.211 and 0.222 on different node sets (2026-09-07).
G=/burg-archive/glab/users/pg2328/fv3_duo_gaps
S=/burg-archive/glab/users/pg2328/legoESM_fv3jax/scripts/cluster/fv3_native/tiled_m6_model_gate_mpi.sbatch
export N=96 KT=3 PAD=11 REF=$G/ladder_ref_c96_v2.npz
for arm in full band band2 full2; do
  case $arm in
    band|band2) export GATE_EXTRA="--timing 20 --refresh-band 4" ;;
    *)          export GATE_EXTRA="--timing 20" ;;
  esac
  echo "########## ARM $arm  GATE_EXTRA=$GATE_EXTRA  $(date +%H:%M:%S)"
  bash "$S" 2>&1 | grep -E "preflight:|M8-C|TIMING|VERDICT|SRUN_RC" \
    | grep -v "^\[m6\] rank [1-9]" | sort -u | cut -c1-190
done
echo DONE m6_bandAB
