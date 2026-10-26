#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=hs1029lid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/results/dycore_1029/lid_%j.out
#
# #1029 follow-up: the sigma arm is NOT a pressure-gradient knob, it is a LID
# knob -- so separate the two.
#
# WHY THIS EXISTS.  The previous job (seed discriminator) found the survival
# times order with the offline terrain-PGF rest residual:
#
#   arm                          rest residual   died at
#   B  L40 sigma  dt 200 s        1.37e-04         8.6 d
#   A  L40 hybrid dt 200 s        1.84e-05        54.6 d
#   C  L40 hybrid dt 100 s        1.84e-05        74.2 d   (timestep control)
#   D  L80 hybrid dt 100 s        6.36e-06        94.2 d
#
# Adversarial review then measured what ELSE those arms changed, and it is a
# lot.  The top FULL level sits at:
#
#   sigma  L40  sigma_full[0] = 2.24e-02  -> lid ~28.5 km
#   hybrid L40  sigma_full[0] = 1.16e-04  -> lid ~68.0 km
#   hybrid L80  sigma_full[0] = 1.53e-05  -> lid ~83.1 km
#
# The top-sponge coefficient and profile are both referenced to
# ``sigma_full[0]``, so swapping the coordinate changes the lid height, the
# sponge strength AND where it acts -- and halving dt also moves the polar
# filter cutoff and doubles the viscosity cap.  None of the four arms is a
# one-variable arm for the residual.  The ordering is real; the attribution
# is not earned.
#
# THE DISCRIMINATOR, from the review: run the sigma arm with NO MOUNTAIN.
#
# PRE-REGISTERED READING:
#   * E (sigma, h_0 = 0) SURVIVES 200 d, or dies far later than B's 8.6 d
#       => B's early death needs the terrain; the sigma lid alone is not
#          unstable, and terrain dependence holds on BOTH coordinates.  It
#          still does not single out the PGF truncation term over any other
#          terrain-mediated mechanism, and the write-up must say so.
#   * E dies at about B's 8.6 d with no mountain
#       => B measured a sigma/lid instability, not terrain at all, and the
#          residual ordering across A/B/D is a coincidence of three confounded
#          arms.  Retract the ordering claim.
#   * F (hybrid L80 at the BASE dt) reproduces the old ladder's early death
#       => confirms that the old "more levels dies sooner" result was the
#          vertical Courant number, on this tree, in this job -- so the
#          reversal seen in D is a dt effect and not a tree difference.
#
# G is the flat-mountain hybrid control, already known to survive 200 d; it is
# rerun here only so all four readings come from ONE job on ONE tree.
set -uo pipefail
export LEGOESM_REPO=/burg-archive/glab/users/pg2328/_issues_wt
source /burg-archive/glab/users/pg2328/legoESM/scripts/cluster/scaling_ginsburg/_env.sh
cd "$REPO"

OUT=/burg-archive/glab/users/pg2328/legoESM/results/dycore_1029/sigma_lid
mkdir -p "$OUT"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export PYTHONUNBUFFERED=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo "sha=$(git rev-parse --short HEAD) host=$(hostname) $(date -u)"
nvidia-smi --query-gpu=name --format=csv,noheader || true

run_arm () {   # $1 label, $2 nlev, $3 vcoord, $4 dt_scale, $5 h_0
  echo "=== arm $1 (nlev=$2 coord=$3 dt_scale=$4 h_0=$5) $(date -u) ==="
  LEGOESM_MATRIX_NLEV="$2" LEGOESM_MATRIX_VCOORD="$3" \
  LEGOESM_MATRIX_DT_SCALE="$4" LEGOESM_TOPO_H0="$5" \
    $PY -u scripts/matrix/run_atmosphere_test_matrix.py \
      --only hydro --test "=held_suarez_topo" --grid latlon \
      --no-cross-grid-plots -o "$OUT/$1" \
      || echo "[arm $1] matrix exited nonzero -- read $OUT/$1/summary.json"
  grep -hE "BLOWUP at step|XFAIL|XPASS|PASS \||FAIL \||max\|v\|" \
      "$OUT/$1"/*.log "$OUT/$1/summary.txt" 2>/dev/null | head -4
}

run_arm E_sigma40_flat      40 sigma  1.0 0.0
run_arm F_hybrid80_dt1      80 hybrid 1.0 2000.0
run_arm G_hybrid40_flat     40 hybrid 1.0 0.0

echo "=== failure step per arm ==="
for a in E_sigma40_flat F_hybrid80_dt1 G_hybrid40_flat; do
  echo "--- $a"
  grep -rhoE "BLOWUP at step [0-9]+" "$OUT/$a" 2>/dev/null | head -1
  find "$OUT/$a" -name summary.json -exec head -c 700 {} \; 2>/dev/null
  echo
done
echo "SIGMA LID DISCRIMINATOR DONE $(date -Is)"
