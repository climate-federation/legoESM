#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=hs1029h0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/results/dycore_1029/flat_mountain/%x_%j.out
# #1029 -- THE arm the existing flat/topo pair cannot supply.
#
# The pair already on disk (results/dycore_1029/localisation) compares
# `held_suarez_topo` (BLOWUP, NaN in 155 s) against `held_suarez` (PASS, 200
# days, max|v| 66.0 / 59.6 m/s, mass drift exactly 0).  It differs in TWO
# variables: the case's init/topography AND the top-sponge mitigation, which is
# gated on the case (`sponge_coeff = ... if _topo else 0.0`) -- so the flat arm
# ran with NO sponge while the topo arm ran WITH one.
#
# This runs the TOPO case with the mountain set to zero.  Same init routine,
# same sponge, same everything; one variable: h_0.
#
#   PRE-REGISTERED READING
#   h_0 = 0 SURVIVES 200 days  -> the mountain is necessary; the terrain
#       pressure-gradient term is confirmed as the seed and the resolution
#       arms should be redone at fixed vertical CFL.
#   h_0 = 0 STILL BLOWS UP     -> the terrain term is exonerated; the seed is
#       in the topo case's init/jet or its sponge, and the wind maximum
#       sitting at 255-258 E against a mountain at 270 E is the next thread.
#
# Control arm at the registry height is run FIRST so this job reproduces the
# known blowup on today's code before the treatment is believed.
set -uo pipefail
export LEGOESM_REPO=/burg-archive/glab/users/pg2328/_iss_wt
source /burg-archive/glab/users/pg2328/legoESM/scripts/cluster/scaling_ginsburg/_env.sh
cd "$REPO"
OUT=/burg-archive/glab/users/pg2328/legoESM/results/dycore_1029/flat_mountain
mkdir -p "$OUT"
echo "sha=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD) host=$(hostname) $(date -u)"
export JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
$PY -c "import jax; print('jax', jax.__version__, jax.devices())"

run_arm () {   # $1 = label, $2 = h_0 [m]
  echo "=== arm $1 (held_suarez_topo, h_0=$2 m) $(date -u) ==="
  LEGOESM_TOPO_H0="$2" $PY -u scripts/matrix/run_atmosphere_test_matrix.py \
      --only hydro --test "=held_suarez_topo" --grid latlon \
      --no-cross-grid-plots -o "$OUT/$1" \
      || echo "[arm $1] matrix exited nonzero -- see $OUT/$1/summary.json"
  grep -E "XFAIL|PASS|FAIL|BLOWUP|max\|v\|" "$OUT/$1/summary.txt" 2>/dev/null | tail -3
}

run_arm control_h0_2000 2000.0
run_arm treatment_h0_0     0.0
echo "FLAT MOUNTAIN ARM DONE $(date -Is)"
