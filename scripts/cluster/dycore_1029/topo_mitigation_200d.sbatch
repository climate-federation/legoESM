#!/usr/bin/env bash
#SBATCH --account=bb1596
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=hs1029
#
# #1029 — does the wired top-sponge mitigation ARREST the latlon
# held_suarez_topo blow-up over a full 200-day run WITHOUT over-damping the
# resolved jet?  This is the experiment the matrix known-failure entry itself
# names as owed (run_atmosphere_test_matrix.py, the held_suarez_topo/latlon
# note): pass -> XPASS (loud) -> remove the XFAIL; only-delays -> stays XFAIL
# and the generator fix (hybrid PGF hybrid_factor <-> Simmons-Burridge
# geopotential, #1078/A_half) is owed.
#
# CONTROLLED COMPARISON, ONE VARIABLE = the topography:
#   arm TOPO = held_suarez_topo, latlon, hybrid, 200 d (mitigation wired).
#   arm FLAT = held_suarez      , latlon, hybrid, 200 d (jet-strength control).
# Everything else (grid, coord, days, dt, precision) matrix-default and
# byte-identical between the arms.  DECIDING NUMBERS, read from each arm's
# summary.json:
#   * TOPO verdict: XPASS (survives 200 d, blow-up arrested) vs XFAIL (still
#     NaNs ~day 96 -> only delayed, generator fix owed).
#   * jet: TOPO max|u| within ~20% of FLAT max|u| = not over-damped; TOPO
#     max|u| collapsed toward zero = the sponge bought stability by killing the
#     jet (a false PASS the matrix tripwire is meant to catch).
set -euo pipefail

WT=/work/bd1083/b309178/diffESM/prmerge/fix
PY=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python
OUT="${OUT:-${WT}/results/dycore_1029}"

cd "${WT}"
PYTHONPATH="$(ls -d "${WT}"/packages/*/ | paste -sd:)":"${WT}/src":"${WT}"
export PYTHONPATH
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1

echo "sha=$(git rev-parse --short HEAD)$([ -n "$(git status --porcelain)" ] && echo -dirty)  host=$(hostname)  $(date -u)"
nvidia-smi --query-gpu=name --format=csv,noheader || true
"${PY}" -c "import jax; print('jax', jax.__version__, jax.devices())"

run_arm () {   # $1 = label, $2 = case
    local label="$1" case="$2"
    echo "=== arm ${label} (case=${case}) $(date -u) ==="
    "${PY}" -u scripts/matrix/run_atmosphere_test_matrix.py \
        --only hydro --test "=${case}" --grid latlon \
        --no-cross-grid-plots -o "${OUT}/${label}" \
        || echo "[arm ${label}] matrix exited nonzero — see ${OUT}/${label}/summary.json"
}

run_arm topo held_suarez_topo
run_arm flat held_suarez

echo "=== summaries ==="
for a in topo flat; do
    echo "--- ${a} ---"
    find "${OUT}/${a}" -name summary.json -exec cat {} \; 2>/dev/null || echo "  (no summary.json)"
done
echo "DONE $(date -u)"
