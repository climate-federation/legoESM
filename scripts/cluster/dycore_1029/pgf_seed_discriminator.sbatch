#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=hs1029seed
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/results/dycore_1029/seed_%j.out
#
# #1029 -- is the terrain pressure-gradient TRUNCATION ERROR what seeds the
# held_suarez_topo runaway?  The level-count ladder could not answer it, and
# this job says why and replaces it.
#
# WHAT IS ALREADY MEASURED, offline, at this case's own geometry (72x144,
# 2000 m mountain, fp64, t=0 tendencies from exact rest, no integration):
#
#     coord   nlev   stratified rest residual   isothermal control
#     sigma     40         1.369e-04 m/s^2            6.1e-16
#     hybrid    40         1.839e-05 m/s^2            1.4e-15
#     hybrid    80         6.357e-06 m/s^2            1.6e-15
#
# The isothermal control is machine zero, so the instrument is sound; the
# stratified residual is a CONVERGENT vertical-quadrature truncation, not a
# discrete cancellation failure (that claim was retracted in 0c170ecc8).
#
# WHY THE EXISTING LEVEL LADDER (job 9632033) CANNOT SETTLE IT.  It ran 40 and
# 80 levels at the SAME timestep and got BLOWUP at step 23600 and step 2200 --
# i.e. cutting the spurious acceleration threefold made the run die ten times
# SOONER.  That is the opposite of the prediction, but it is also confounded:
# doubling the level count at fixed dt halves the layer thickness and so
# doubles the vertical Courant number.  Two variables moved, so neither
# reading is earned.
#
# THE TWO CLEAN LEVERS, pre-registered before submission:
#
#   B (sigma, 40 levels, SAME dt, SAME mountain, SAME grid) moves the terrain
#     PGF residual by 7.4x UPWARDS and moves the vertical CFL barely at all.
#     This is the lever the level ladder wanted and could not get.
#   D (hybrid, 80 levels, dt x0.5) restores the vertical Courant number of the
#     40-level control while cutting the residual 3x, with C (hybrid, 40
#     levels, dt x0.5) as the timestep-only control that says how much of D is
#     just the smaller dt.
#
# PRE-REGISTERED READING RULE, written before any arm returned:
#   * B survives 200 d, or dies MATERIALLY LATER than A (>2x the step count),
#     while carrying 7.4x the spurious acceleration
#       => the terrain-PGF truncation error is REFUTED as what sets the
#          failure.  Say so and stop working it.
#   * B dies materially EARLIER than A
#       => consistent with the residual being the seed; CONFIRMS the direction
#          (does not prove the term -- a momentum-budget diagnostic would).
#   * D (matched vertical CFL, 3x smaller residual) dies at or before A's step
#       => also refutes the residual, and independently of B.
#   * D survives or dies much later AND C (dt only) does not, by itself,
#       produce the same shift => the level count is doing real work and the
#       residual stays live.
#   * B and D disagree => report BOTH, name the disagreement, pick neither.
#
# A RUNS FIRST so today's code is shown to still carry the defect before any
# treatment means anything.  ONE VARIABLE per arm against A, injected through
# the environment so no committed default moves.
#
# WHY NOT CHEAPER: the quantity wanted is a failure STEP, which only exists in
# an integration; the rest-state residual that motivates it is already
# measured offline and is quoted above.  ~2-7 min per arm on one GPU.
set -uo pipefail
export LEGOESM_REPO=/burg-archive/glab/users/pg2328/_issues_wt
source /burg-archive/glab/users/pg2328/legoESM/scripts/cluster/scaling_ginsburg/_env.sh
cd "$REPO"

OUT=/burg-archive/glab/users/pg2328/legoESM/results/dycore_1029/pgf_seed
mkdir -p "$OUT"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export PYTHONUNBUFFERED=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo "sha=$(git rev-parse --short HEAD) host=$(hostname) $(date -u)"
nvidia-smi --query-gpu=name --format=csv,noheader || true
$PY -c "import jax; print('jax', jax.__version__, jax.devices())"

run_arm () {   # $1 label, $2 nlev, $3 vcoord, $4 dt_scale
  echo "=== arm $1 (nlev=$2 coord=$3 dt_scale=$4) $(date -u) ==="
  LEGOESM_MATRIX_NLEV="$2" LEGOESM_MATRIX_VCOORD="$3" \
  LEGOESM_MATRIX_DT_SCALE="$4" \
    $PY -u scripts/matrix/run_atmosphere_test_matrix.py \
      --only hydro --test "=held_suarez_topo" --grid latlon \
      --no-cross-grid-plots -o "$OUT/$1" \
      || echo "[arm $1] matrix exited nonzero -- read $OUT/$1/summary.json"
  grep -hE "BLOWUP at step|XFAIL|XPASS|PASS \||FAIL \||max\|v\|" \
      "$OUT/$1"/*.log "$OUT/$1/summary.txt" 2>/dev/null | head -4
}

run_arm A_control_hybrid40_dt1   40 hybrid 1.0
run_arm B_sigma40_dt1            40 sigma  1.0
run_arm C_hybrid40_dt0p5         40 hybrid 0.5
run_arm D_hybrid80_dt0p5         80 hybrid 0.5

echo "=== the deliverable: failure step per arm ==="
for a in A_control_hybrid40_dt1 B_sigma40_dt1 C_hybrid40_dt0p5 D_hybrid80_dt0p5; do
  echo "--- $a"
  grep -hoE "BLOWUP at step [0-9]+" "$OUT/$a"/*/*/*.log "$OUT/$a"/*.log 2>/dev/null | head -1
  find "$OUT/$a" -name summary.json -exec head -c 700 {} \; 2>/dev/null
  echo
done
echo "PGF SEED DISCRIMINATOR DONE $(date -Is)"
