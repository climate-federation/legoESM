#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=hs1029ens
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/results/dycore_1029/ens_%j.out
#
# #1029 -- the width that every arm on this issue is missing.
#
# The thread now holds twelve single-member arms (nine on the dt x level ladder,
# three on the mountain-height axis) and every difference between them has been
# read as signal.  None of them has an error bar.  A blow-up TIME in a nonlinear
# core is a chaotic functional of the initial condition, so before any of those
# differences means anything the member-to-member scatter has to be measured --
# on THIS case, at THIS resolution, not assumed from experience elsewhere.
#
# Stage 1, the deliverable: five members at h_0 = 2000 m and five at 1000 m,
# differing ONLY in the seed of the initial temperature perturbation (a new
# LEGOESM_TOPO_SEED probe override, default unchanged).  The spread within each
# group is the resolution of the whole issue.  Reading rule, pre-registered:
#   * within-group spread much smaller than the 71.5 vs 54.6 d gap between the
#     groups  -> the mountain-height dependence is real and can be fitted.
#   * spread comparable to that gap -> the h_0 dependence is NOT established,
#     and neither is the "cliff" at 80 levels or any other pairwise difference
#     on this issue; they all need members before they can be read.
# The second outcome would be the more valuable one, and it is the one I expect
# to be resisted, so it is written down here before the numbers exist.
#
# Stage 2: bracket the 4000 m arm, which died at day 0.46 with max|v| = 4.0 m/s
# -- no jet runaway at all, i.e. a different and much faster failure.  Members
# at 2800 / 3200 / 3600 m separate two readings of that: a CLIFF between the
# slow mode and the fast one (survival falls off abruptly somewhere in the
# bracket) versus SATURATION (survival slides down smoothly and 4000 m is just
# the end of the same curve).  Only the first justifies excluding 4000 m from
# the scaling fit, which is what I want to do and have not yet earned.
#
# COST: thirteen idealized 200-day arms on one GPU, no forcing files, ~4-12 min
# each.  No cheaper offline test exists -- the quantity is a survival time, and
# a spread in survival times needs the times.
set -uo pipefail

ROOT=/burg-archive/glab/users/pg2328/_sweep0923_wt
cd "$ROOT" || exit 2
OUT="${OUT:-/burg-archive/glab/users/pg2328/legoESM/results/dycore_1029/ensemble_$SLURM_JOB_ID}"
mkdir -p "$OUT"

export PYTHONPATH="$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/ac5006/legoESM/.venv/bin/python
[ -x "$PY" ] || { echo "FATAL: interpreter $PY missing"; exit 2; }

echo "sha=$(git -C "$ROOT" rev-parse --short HEAD)$([ -n "$(git -C "$ROOT" status --porcelain)" ] && echo -dirty)  host=$(hostname)  $(date -u)"
"$PY" -c "import jax; print('jax', jax.__version__, jax.devices())"

run_arm () {   # $1 = label, $2 = h_0, $3 = seed ("" for the init default)
    echo "=== arm $1 (h_0=$2 seed=${3:-default}) $(date -u) ==="
    if [ -n "$3" ]; then export LEGOESM_TOPO_SEED="$3"; else unset LEGOESM_TOPO_SEED; fi
    LEGOESM_TOPO_H0="$2" "$PY" -u scripts/matrix/run_atmosphere_test_matrix.py \
        --only hydro --test "=held_suarez_topo" --grid latlon \
        --no-cross-grid-plots -o "$OUT/$1" 2>&1 \
      | grep -E "BLOWUP|XFAIL|XPASS|LEGOESM_TOPO|Total:|max\|v\|"
    echo "[arm $1] done"
}

echo "############ STAGE 1: ensemble spread at two mountain heights ############"
for s in 42 1234 2718 31415 65537; do
  run_arm "h2000_s${s}" 2000 "$s"
done
for s in 42 1234 2718 31415 65537; do
  run_arm "h1000_s${s}" 1000 "$s"
done

echo "############ STAGE 2: bracket the fast mode between 2000 and 4000 m ############"
for h in 2800 3200 3600; do
  run_arm "h${h}_sdef" "$h" ""
done

echo "############ SUMMARY ############"
grep -E "^=== arm|BLOWUP at step|XPASS \|" "$SLURM_SUBMIT_DIR/../../results/dycore_1029/ens_$SLURM_JOB_ID.out" 2>/dev/null || true
echo "DONE $(date -u)"
