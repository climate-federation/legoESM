#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=reff1521eq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/results/dycore_1521/eqsweep_%j.out
#
# #1521: does the droplet-radius attribution survive on an EQUILIBRATED state,
# and on THIS machine?
#
# WHY THIS ARM EXISTS.  The issue's cheapest surviving discriminator is
# "prescribe the droplet number (or the observed effective radius) directly,
# on this machine, and see where the bias lands" -- a substitution against a
# fixed number, so the synthetic aerosol here only has to be HELD FIXED, not
# correct.  Its offline half is already instrumented
# (scripts/validate/amip_bias/reff_rsut_sensitivity.py) and was run once on a
# DAY-12 checkpoint.  Day 12 is a spin-up state: its cloud field is not the one
# the +36.6 W/m2 was scored on, and the radius-to-flux mapping is nonlinear and
# column-dependent, so the sensitivity measured there does not automatically
# transfer.
#
# This machine now carries a 100-year AMIP chain, so the same sweep can be run
# on a genuinely equilibrated state (day 36500, i.e. year 100) and on a
# mid-chain one (day 18250, year 50) as a second sample.  Two checkpoints
# rather than one so a difference between them is visible as a difference
# rather than mistaken for the answer.
#
# PRE-REGISTERED READING, written before submission:
#   * d(rsut)/d(r_eff) of order -3 to -5 W/m2/um over 8-14 um, i.e. -15 to -20
#     W/m2 available from 7.84 -> 11-14 um  => the radius attribution is the
#     right SIZE on an equilibrated state too, and it transfers to this
#     machine.  The remaining ~15-20 W/m2 anvil residual stands as attributed.
#   * sensitivity well under -1.5 W/m2/um  => the radius cannot buy the
#     residual here, and the attribution needs rework on this machine
#     regardless of what the other machine measured.
#   * the two checkpoints disagree by more than a factor of 1.5  => the
#     sensitivity is state-dependent at that level and no single number should
#     be quoted from either.
#
# WHAT THIS CANNOT DO, stated so the number is not over-read.  It is a
# RADIATION-KERNEL substitution on a frozen cloud field: everything except the
# radius handed to the kernel is byte-identical between arms.  It therefore
# measures the instantaneous radiative sensitivity and NOT the coupled
# response -- the cloud field does not get to adjust.  The full prescribed-
# droplet-number RUN is still the thing that answers "where does the bias
# land", and it is not this job.
#
# The probe's own frame (from its docstring) also applies: single homogeneous
# column per cell, so this is the plane-parallel-side value and an UPPER bound
# on production's, which additionally applies two_region inhomogeneity and
# McICA subcolumns.  Only the DIFFERENCES between arms are the measurement.
set -uo pipefail
export LEGOESM_REPO=/burg-archive/glab/users/pg2328/_issues_wt
source /burg-archive/glab/users/pg2328/legoESM/scripts/cluster/scaling_ginsburg/_env.sh
cd "$REPO"

CKPT_DIR=/burg-archive/glab/users/pg2328/legoESM/results/amip_mpas_100yr
OUT=/burg-archive/glab/users/pg2328/legoESM/results/dycore_1521
mkdir -p "$OUT"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export PYTHONUNBUFFERED=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo "sha=$(git rev-parse --short HEAD) host=$(hostname) $(date -u)"
nvidia-smi --query-gpu=name --format=csv,noheader || true

for day in 36500 18250; do
  CK="$CKPT_DIR/checkpoint_day_$(printf %05d "$day").npz"
  [ -f "$CK" ] || CK="$CKPT_DIR/checkpoint_day_${day}.npz"
  if [ ! -f "$CK" ]; then
    echo "=== day $day: NO CHECKPOINT at $CK -- skipped, and said so ==="
    continue
  fi
  echo "=== day $day ($CK) $(date -u) ==="
  $PY -u scripts/validate/amip_bias/reff_rsut_sensitivity.py "$CK" \
      --mesh-level 5 --radii-um 8 10 11 12 14 --n-times 8 \
      || echo "[day $day] sweep exited nonzero"
done

echo "EQUILIBRATED REFF SWEEP DONE $(date -Is)"
