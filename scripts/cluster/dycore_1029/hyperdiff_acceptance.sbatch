#!/bin/bash
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --job-name=hs1029acc
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs/dycore_1029_anatomy/acc_%j.out
#
# #1029 acceptance, FINAL setup: DCMIP mountain (R_m 3pi/4, zeta pi/16, 0N)
# + del-4 hyperdiffusion on u, v, T (nu_del4 = matrix value 2.88e15 m^4/s,
# per-row pole cap), set explicitly for both lat-lon HS cases.
#
# STAGE 1 -- paired 30-d run (probe), seeds 42 / 65537 / 7, del-4 on vs off,
# same tree, same mountain; only nu_del4 differs.
#   (a) number: mean lowest-10-level 2dx zonal-wind content (u2dx_low) over
#       days 10-30, per seed and arm; heat leak of the del-4 T term [W/m2].
#   (b) CONFIRM del-4 acts on the growing mode: on/off ratio <= 0.3 in all 3
#       seeds.  REFUTE: ratio > 0.7 in >= 2 seeds.  Else INCONCLUSIVE.
#       GATE: stage 2 runs ONLY on CONFIRM.
# STAGE 2 -- 200-d matrix runs (the runner, no override):
#   topo hybrid exponent 3, 6 seeds; topo exponent 2, same 6 seeds;
#   flat held_suarez (sigma + hybrid).
#   (a) survival day per seed (censored 200 d), final max|v|, runner status.
#   (b) CURED (per exponent): >= 5/6 survive 200 d with final max|v| < 90.
#       NOT CURED: <= 1/6 survive.  Else PARTIAL.
#       Flat re-check PASSES if both complete 200 d and the jet max is
#       >= 0.8 x the no-del-4 climate (sigma 66.0 -> 52.8, hybrid 59.6 -> 47.7).
# (c) no cheaper test: survival scatters 5.5x across seeds; the 30-d stage is
#     the cheap discriminator and gates the rest.
# COST: stage 1 ~0.2 GPU-h; stage 2 14 arms <= ~2.5 GPU-h.
set -uo pipefail
ROOT=/work/bd1083/b309178/diffESM/legoesm_pg/wt_1029
OUT=/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs/dycore_1029_anatomy/acc_$SLURM_JOB_ID
mkdir -p $OUT; cd $ROOT
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | tr '\n' ':')"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python
echo "sha=$(git -c safe.directory=$ROOT rev-parse --short HEAD) dirty=$(git -c safe.directory=$ROOT status --porcelain --untracked-files=no | wc -l) $(date -u)"
P=scripts/cluster/dycore_1029/burst_anatomy_probe.py
pair () { CUDA_VISIBLE_DEVICES=$1 $PY -u $P --seed $2 $3 --every 100 --fine-from 999999 --max-steps 12960 --out $OUT/pair_$4 > $OUT/pair_$4.log 2>&1; }
pair 0 42 --nu4-matrix on42 & pair 1 42 "" off42 & pair 2 65537 --nu4-matrix on65537 & pair 3 65537 "" off65537 & wait
pair 0 7 --nu4-matrix on7 & pair 1 7 "" off7 & wait
VERDICT=$($PY - "$OUT" <<'PYEOF'
import json, sys, math
out = sys.argv[1]; ratios = []
def load(t):
    return [json.loads(l) for l in open(f"{out}/pair_{t}/diag.jsonl")]
for s in (42, 65537, 7):
    on, off = load(f"on{s}"), load(f"off{s}")
    m = lambda L: sum(d["u2dx_low"] for d in L if 10 <= d["day"] <= 30) / max(1, sum(1 for d in L if 10 <= d["day"] <= 30))
    a, b = m(on), m(off)
    r = a / b if b > 0 else float("nan")
    leak = [d["heat_leak_Wm2"] for d in on]
    print(f"seed {s}: u2dx on {a:.3f} off {b:.3f} ratio {r:.3f}; leak on mean {sum(leak)/len(leak):.3e} W/m2, max|.| {max(abs(x) for x in leak):.3e}; off last day {off[-1]['day']:.1f} umax {off[-1]['umax']:.1f}; on last day {on[-1]['day']:.1f} umax {on[-1]['umax']:.1f}", file=sys.stderr)
    ratios.append(r)
ok = all(r <= 0.3 for r in ratios)
bad = sum(1 for r in ratios if not (r <= 0.7)) >= 2
print("CONFIRM" if ok else ("REFUTE" if bad else "INCONCLUSIVE"))
PYEOF
)
echo "STAGE1 VERDICT: $VERDICT"
[ "$VERDICT" = "CONFIRM" ] || { echo "stage 2 skipped by gate"; echo DONE $(date -u); exit 0; }
cat > $OUT/arms.txt <<ARMS
t3_s42 held_suarez_topo 3 42
t3_s65537 held_suarez_topo 3 65537
t3_s7 held_suarez_topo 3 7
t3_s99 held_suarez_topo 3 99
t3_s1234 held_suarez_topo 3 1234
t3_s2024 held_suarez_topo 3 2024
t2_s42 held_suarez_topo 2 42
t2_s65537 held_suarez_topo 2 65537
t2_s7 held_suarez_topo 2 7
t2_s99 held_suarez_topo 2 99
t2_s1234 held_suarez_topo 2 1234
t2_s2024 held_suarez_topo 2 2024
flat held_suarez - -
ARMS
run_arm () { local g=$1 tag=$2 case=$3 ex=$4 seed=$5; local env=(CUDA_VISIBLE_DEVICES=$g)
  [ "$ex" != "-" ] && env+=(LEGOESM_MATRIX_TRANSITION_EXPONENT=$ex)
  [ "$seed" != "-" ] && env+=(LEGOESM_TOPO_SEED=$seed)
  env "${env[@]}" $PY -u scripts/matrix/run_atmosphere_test_matrix.py --only hydro --test "=$case" --grid latlon --days 200 --no-cross-grid-plots -o $OUT/$tag > $OUT/$tag.log 2>&1; }
for g in 0 1 2 3; do ( awk -v g=$g 'NR%4==g' $OUT/arms.txt | while read -r l; do run_arm $g $l; done ) & done
wait
for f in $OUT/t*.log $OUT/flat.log; do echo "== $(basename $f .log): $(grep -m1 'dissipation' $f | cut -c1-90) $(grep -m1 'TRANSITION_EXPONENT' $f | cut -c1-60)"; grep -E "BLOWUP at|^\s+\*?\*?(PASS|FAIL|XFAIL|XPASS)\s+\|" $f | tail -3 | cut -c1-200; done
echo DONE $(date -u)
