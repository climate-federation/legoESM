#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=hs1029cliff
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/results/dycore_1029/cliff_%j.out
#
# #1029: TWO failure modes, and one pre-registered falsifier for the split.
#
# THE READING THAT PRODUCED THIS JOB.  Seven arms are on the issue.  Multiply
# each arm's offline terrain-PGF rest residual by its survival time and the
# spurious momentum it accumulated before dying comes out nearly constant:
#
#   arm  residual [m/s^2]  died [d]   residual x time [m/s]
#   B      1.37e-04           8.6        102
#   A      1.84e-05          54.6         87
#   C      1.84e-05          74.2        118
#   D      6.36e-06          94.2         52
#   F      6.36e-06           5.1        2.8   <-- 30x outlier
#
# and the two arms whose blow-up wind was recorded died at |v| = 165.8 and
# 169.0 m/s.  So A/B/C/D look like ONE mechanism -- a steady spurious torque
# integrating, undamped, to a critical wind -- and F looks like a DIFFERENT
# one.  Held-Suarez damps momentum only below sigma = 0.7
# (`k_v = k_f max(0, (sigma-0.7)/0.3)`, held_suarez.py:204), so above that
# there is nothing to stop a steady torque growing linearly, which is what
# the "residual x time is constant" collapse says is happening.
#
# F is hybrid L80 at the BASE timestep.  Its top layer is 7.4x thinner than
# the 40-level control's, and halving dt there bought 18.5x more life (5.1 ->
# 94.2 days) where the same halving at 40 levels bought only 1.36x.  A
# response that superlinear in dt is a stability THRESHOLD, not a truncation
# error -- so F is read as the vertical Courant limit in the thin top layer,
# crossed at dt = 200 s and not at dt = 100 s.
#
# THE FALSIFIER, written before the run.  If F is a threshold, the arm
# between the two -- 80 levels at dt = 150 s -- must land on ONE SIDE of it,
# not in the middle:
#   * H dies in ~4-8 days   => the threshold is below 150 s; THRESHOLD, and
#                              the two-mechanism reading stands.
#   * H survives ~70-100 d  => the threshold is above 150 s; THRESHOLD, same
#                              conclusion, other side.
#   * H dies somewhere around 20-50 days, i.e. INTERPOLATING smoothly between
#     5 and 94 => NOT a threshold.  The two-mechanism reading is wrong and
#     the F outlier needs another explanation.  Say so.
#
# I is the same question asked of the 40-level control, where the reading
# predicts NO cliff: 40 levels at dt = 150 s should sit smoothly between the
# 54.6-day and 74.2-day arms (~60-70 days).  If I ALSO jumps, dt is doing
# something at both level counts and the thin-layer story is wrong.
#
# WHY NOT CHEAPER: the quantity is a failure day, which only exists in an
# integration.  The residuals are already measured offline.  ~1-8 min per arm.
set -uo pipefail
export LEGOESM_REPO=/burg-archive/glab/users/pg2328/_issues_wt
source /burg-archive/glab/users/pg2328/legoESM/scripts/cluster/scaling_ginsburg/_env.sh
cd "$REPO"

OUT=/burg-archive/glab/users/pg2328/legoESM/results/dycore_1029/cliff
mkdir -p "$OUT"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export PYTHONUNBUFFERED=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo "sha=$(git rev-parse --short HEAD) host=$(hostname) $(date -u)"
nvidia-smi --query-gpu=name --format=csv,noheader || true

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

run_arm H_hybrid80_dt0p75  80 hybrid 0.75
run_arm I_hybrid40_dt0p75  40 hybrid 0.75

echo "=== failure step per arm ==="
for a in H_hybrid80_dt0p75 I_hybrid40_dt0p75; do
  echo "--- $a"
  grep -rhoE "BLOWUP at step [0-9]+" "$OUT/$a" 2>/dev/null | head -1
  find "$OUT/$a" -name summary.json -exec head -c 700 {} \; 2>/dev/null
  echo
done
echo "CLIFF AND CLOCK DONE $(date -Is)"
