#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=hs1029lev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/results/dycore_1029/lev_%j.out
#
# #1029 LEVEL-COUNT ARM -- is the terrain pressure-gradient truncation error the
# seed of the held_suarez_topo runaway?  (User-approved 2026-09-04.)
#
# WHAT IS ALREADY MEASURED.  At this case's own geometry (72x144, hybrid,
# 2000 m mountain) a stratified atmosphere at exact rest feels a spurious
# acceleration of 14.8 / 3.5 / 1.2 m/s per day at 20 / 40 / 80 levels, against
# 2.5e-10 m/s/day for the isothermal state the existing regression tests use.
# The case runs 40 levels.
#
# THE PREDICTION, written before the run.  If that ~3.5 m/s/day is the seed,
# then trebling it (20 levels) must kill the run EARLIER and cutting it to a
# third (80 levels) must delay it or remove it.  Specifically:
#   CONFIRMS: 80-level arm survives 200 d, or dies materially later than the
#             40-level arm's ~day 96; and the 20-level arm dies EARLIER.
#   REFUTES:  all three arms die at about the same day.  The spurious
#             acceleration then varies 12-fold across the arms while the death
#             day does not, so it is not what sets the failure.
#   AMBIGUOUS: 80 survives but 20 does not die earlier -- report as such and do
#             not pick the half that agrees.
#
# ONE VARIABLE = the level count.  Same grid, coordinate, mountain, length,
# timestep and diffusion; only DEFAULT_NLEV differs, injected per arm through
# the environment so no committed default moves.
#
# WHY NOT CHEAPER: the quantity wanted is a failure DAY, which only exists in an
# integration.  The offline tendency measurement is already done and is what
# motivated this.  ~10 min per arm on one A100 (the 40-level arm NaN'd in 2.9
# min of wall in job 9631822).
set -uo pipefail

ROOT=/burg-archive/glab/users/pg2328/legoESM
cd "$ROOT" || exit 2
OUT="${OUT:-$ROOT/results/dycore_1029/levels}"
mkdir -p "$OUT"

export PYTHONPATH="$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/ac5006/legoESM/.venv/bin/python

echo "sha=$(git rev-parse --short HEAD)  host=$(hostname)  $(date -u)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

run_arm () {   # $1 = nlev
  echo "=== arm nlev=$1 $(date -u) ==="
  LEGOESM_MATRIX_NLEV="$1" "$PY" -u scripts/matrix/run_atmosphere_test_matrix.py \
      --only hydro --test "=held_suarez_topo" --grid latlon \
      --no-cross-grid-plots -o "$OUT/nlev$1" \
      || echo "[arm nlev=$1] matrix exited nonzero -- see summary.json"
}

for n in 20 40 80; do run_arm "$n"; done

echo "=== death day per arm (the deliverable) ==="
for n in 20 40 80; do
  echo "--- nlev=$n"
  grep -hE "BLOWUP at step|XFAIL|XPASS|PASS \|" "$OUT/nlev$n"/*.log 2>/dev/null | head -4
  find "$OUT/nlev$n" -name summary.json -exec sh -c \
    'echo "  $1"; head -c 900 "$1"; echo' _ {} \; 2>/dev/null | head -20
done
echo "DONE $(date -u)"
