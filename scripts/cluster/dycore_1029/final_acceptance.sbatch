#!/bin/bash
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --job-name=hs1029fin
#SBATCH --nodes=1
#SBATCH --gpus=3
#SBATCH --cpus-per-task=24
#SBATCH --mem=150G
#SBATCH --time=01:30:00
#SBATCH --output=/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs/dycore_1029_anatomy/fin_%j.out
#
# #1029 acceptance on the FINAL setup (maintainer: one run per exponent + flat):
# DCMIP mountain (3pi/4, pi/16, 0N) + del-4 on u, v and layer-mass-weighted T
# (nu_del4 2.88e15 m^4/s, per-row pole cap), matrix runner, no overrides
# except the exponent.  Default seed only: del-4 erases the grid-scale seed
# perturbation, so seeds collapse to one trajectory (measured, job 27943516).
# PRE-REGISTERED:
# (a) per run: completes 200 d or blow-up day; final max|v|; runner status.
# (b) topo exponent 3 and exponent 2: CURED if each completes 200 d with
#     final max|v| < 90 m/s (exp-3 hybrid reports XPASS against the
#     known-failure entry).  NOT CURED if either blows up.
#     Flat held_suarez (sigma + hybrid): PASSES if both complete 200 d and the
#     jet max is >= 0.8 x the no-del-4 climate (sigma 66.0 -> 52.8 m/s,
#     hybrid 59.6 -> 47.7 m/s).
# (c) no cheaper test: survival is the quantity in question; one member each
#     at ~5-12 min of A100 per 200-d run.
# COST: 3 GPUs x <= 30 min = <= 1.5 GPU-h (expected ~0.3).
set -uo pipefail
ROOT=/work/bd1083/b309178/diffESM/legoesm_pg/wt_1029
OUT=/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs/dycore_1029_anatomy/fin_$SLURM_JOB_ID
mkdir -p $OUT; cd $ROOT
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | tr '\n' ':')"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python
echo "sha=$(git -c safe.directory=$ROOT rev-parse --short HEAD) dirty=$(git -c safe.directory=$ROOT status --porcelain --untracked-files=no | wc -l) $(date -u)"
run () { CUDA_VISIBLE_DEVICES=$1 env $4 $PY -u scripts/matrix/run_atmosphere_test_matrix.py --only hydro --test "=$2" --grid latlon --days 200 --no-cross-grid-plots -o $OUT/$3 > $OUT/$3.log 2>&1; }
run 0 held_suarez_topo topo_exp3 LEGOESM_MATRIX_TRANSITION_EXPONENT=3 &
run 1 held_suarez_topo topo_exp2 LEGOESM_MATRIX_TRANSITION_EXPONENT=2 &
run 2 held_suarez flat LEGOESM_NOOP=1 &
wait
for a in topo_exp3 topo_exp2 flat; do echo "== $a: $(grep -m1 'dissipation' $OUT/$a.log | cut -c1-90) $(grep -m1 'TRANSITION_EXPONENT' $OUT/$a.log | cut -c1-60)"; grep -E "BLOWUP at|^\s+\*?\*?(PASS|FAIL|XFAIL|XPASS)\s+\|" $OUT/$a.log | cut -c1-200; done
echo DONE $(date -u)
