#!/bin/bash
#SBATCH --account=bb1596
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --job-name=hs1029exp
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs/dycore_1029_exponent/exp_%j.out
#
# #1029 -- does B = eta**2 instead of eta**3 cure the lat-lon held_suarez_topo
# runaway?  (User decision 2026-10-06: measure first, no default change.)
#
# ONE variable: LEGOESM_MATRIX_TRANSITION_EXPONENT (2 vs 3, set explicitly in
# BOTH arms so each log proves the override reached the model).  A + B is
# identical between arms (pinned by tests/unit/test_issue_1029_exponent_probe.py),
# so level pressures and sigma_full (hence the top sponge) agree where
# p_s = p_ref; where p_s != p_ref the level pressures differ by
# (B3 - B2)(p_s - p_ref), up to ~29 hPa near 700 hPa over the mountain -- that
# terrain-following change IS the treatment.  Same tree, same case (h_0 =
# 2000 m registry value), 40 levels and 200 days pinned on the command line,
# base dt, default blow-up check (every 100 steps = 0.23 d resolution), same
# ten initial-perturbation seeds in both arms (paired), GPU assignment rotated.
#
# PRE-REGISTERED, written before submission:
# (a) the number: survival time in days per (seed, exponent), right-censored
#     at 200 d (a completed run).  20 numbers, plus #survivors per arm.
# (b) Survivor = status PASS with no BLOWUP; a FAIL without BLOWUP, a crash,
#     or a log without the override line is INVALID, never a survivor.
#     CONFIRMED (cures it): exp-2 >= 8/10 survive AND exp-3 <= 2/10.
#     NOT CURED, delays it: exp-2 < 8/10 AND one-sided paired sign test
#     (exp2 outlives exp3) p <= 0.05.
#     REFUTED: exp-2 <= 2/10 AND sign test p > 0.2.  Else INCONCLUSIVE.
#     Controls: exp-3 seed 42 run twice must give the same blow-up day
#     (determinism); exp-3 seed 42 vs the issue's 54.6 d is REPORTED (drift),
#     not a verdict input -- the verdict compares arms inside this job only.
#     Mechanistic prior: at h_0 = 2000 m p_s >= 803.6 hPa > the 663.9 hPa
#     negative-mass threshold, so exponent 2 cannot act through layer-mass
#     sign; a CONFIRM says the coordinate change cures it, not which term.
# (c) why nothing cheaper: survival time on this case scatters 5.5x across
#     seeds (CV 0.81, n=5), so a single pair cannot be read; the offline
#     rest-state PGF residual was refuted as a predictor of survival (arm F,
#     2026-09-21), so it cannot stand in for the survival time.
#
# COST: 20 idealized 200-day arms, 4 at a time (one per GPU), no forcing files.
set -uo pipefail
export PATH=/sw/spack-levante/git-2.43.7-2ofazl/bin:$PATH
ROOT=/work/bd1083/b309178/diffESM/legoesm_pg/wt_i1029
OUT=${OUT:-/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs/dycore_1029_exponent/job_$SLURM_JOB_ID}
mkdir -p "$OUT"
cd "$ROOT" || exit 2
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | tr '\n' ':')"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python
echo "sha=$(git -c safe.directory=$ROOT rev-parse --short HEAD)$([ -n "$(git -c safe.directory=$ROOT status --porcelain --untracked-files=no)" ] && echo -dirty) host=$(hostname) $(date -u)" | tee "$OUT/provenance.txt"

export LEGOESM_MATRIX_NLEV=40
run_arm () {  # $1 gpu, $2 exponent, $3 seed, $4 tag
  CUDA_VISIBLE_DEVICES=$1 LEGOESM_MATRIX_TRANSITION_EXPONENT=$2 LEGOESM_TOPO_SEED=$3 \
    "$PY" -u scripts/matrix/run_atmosphere_test_matrix.py --only hydro \
      --test "=held_suarez_topo" --grid latlon --days 200 --no-cross-grid-plots \
      -o "$OUT/$4" > "$OUT/$4.log" 2>&1
}

ARMS=()
for s in 42 1234 2718 31415 65537 7 99 2024 8191 104729; do
  ARMS+=("3 $s e3_s$s" "2 $s e2_s$s")
done
ARMS+=("3 42 e3_s42dup")
i=0; b=0; PIDS=(); TAGS=()
for a in "${ARMS[@]}"; do
  set -- $a
  run_arm $(( (i + b) % 4 )) "$1" "$2" "$3" &
  PIDS+=($!); TAGS+=("$3"); i=$((i + 1))
  if (( i % 4 == 0 )) || (( i == ${#ARMS[@]} )); then
    for k in "${!PIDS[@]}"; do
      wait "${PIDS[$k]}"; rc=$?
      echo "[${TAGS[$k]}] exit=$rc $(date -u)"
      echo "${TAGS[$k]}=$rc" >> "$OUT/exit_codes.txt"
    done
    PIDS=(); TAGS=(); b=$((b + 1))
  fi
done
"$PY" scripts/validate/issue_1029_exponent_survival.py "$OUT" | tee "$OUT/verdict.txt"
RC=$?
echo "DONE $(date -u) reader-exit=$RC"
exit $RC
