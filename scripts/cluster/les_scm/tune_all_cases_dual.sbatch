#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100"
#SBATCH --job-name=les_dual
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
# MEASURED: the completed 8-step 9-scheme 5-case run (9333698) took 21h43,
# while the per-scheme tuning wall summed to ~17 min. The cost is therefore
# COMPILATION of the per-(scheme,case) rollout graph, not the optimizer steps,
# so raising 8 -> 40 steps adds little. 72 h is the Ginsburg maximum and
# _write_outputs runs ONCE after the last scheme, so a timeout loses the whole
# job: take the partition max whenever the only artifact lands at the end.
#SBATCH --time=72:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/les_dual_%j.out
set +e
# ONE parameter set per scheme, fitted jointly to every usable LES case.
#
# WHY THIS RE-RUN, and why the previous ranking is not citable:
#  1. The completed run used a pin that PREDATES two tuner fixes -- a blown-up
#     candidate could be accepted as an "improvement" when the current loss
#     exceeded the non-finite penalty, and the tuned evaluation's non-finite
#     flag was dropped so a blown-up arm stayed rankable.
#  2. It ran 8 steps and 6 of 9 schemes were still descending at step 8.
#
# AGGREGATION is the variable under test and the ONLY difference between the
# two arms below. 'mean' weights every regime's absolute normalized error
# equally, which measured 43% of the joint loss on ekman against 3.1% on
# bomex; 'default_relative' gives every regime equal improvement headroom.
# Both numbers are reported by both arms, so they are directly comparable.
: "${DUAL_WT:?set DUAL_WT to the pinned worktree to run from}"
WT=$DUAL_WT
cd $WT || exit 1
echo "PINNED sha: $(git rev-parse HEAD)"; echo "host: $(hostname)"; nvidia-smi -L
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=/burg-archive/glab/users/pg2328/legoESM/results
CASES="bomex:$R/les_ref/bomex,rico:$R/les_ref/rico,gabls1:$R/les_ref/gabls1,cbl:$R/les_ref/cbl,ekman:$R/les_ref/ekman"
STEPS=${STEPS:-40}

run_arm () {          # $1 = aggregation
  local agg=$1
  local out=$R/scm_les_turbulence/dual_$agg
  echo "================= ARM $agg ================="
  rm -rf $out
  local t0=$SECONDS
  # PIPESTATUS, not $?: piping through tail records TAIL's status, so an OOM
  # or a traceback would report rc=0 and a green Slurm job.
  $PY scripts/run/run_scm_les_turbulence_tuning.py \
      --cases "$CASES" --outdir "$out" --nlev 48 --steps $STEPS \
      --joint-aggregation "$agg" 2>&1 | tail -400
  local rc=${PIPESTATUS[0]}
  echo "ARM $agg rc=$rc wall=$((SECONDS - t0))s"
  [ "$rc" -ne 0 ] && { echo "ARM $agg FAILED -- do not cite this arm"; return; }

  $PY scripts/plot/plot_scm_les_tuning_summary.py "$out"
  echo "  summary plot rc=$?"
  for c in $($PY -c "
import json
print(' '.join(json.load(open('$out/tuned_parameters.json'))['protocol']['cases']))
"); do
    $PY scripts/plot/plot_scm_les_turbulence.py "$out" --case "$c" \
        --out "$out/profiles_vs_les_$c.png" >/dev/null && echo "  plotted $c"
  done
  $PY scripts/validate/scm_les_score_decomposition.py "$out" \
      --json-out "$out/score_decomposition.json" | tail -60
  echo "=== ranking ($agg) ==="; cat "$out/ranking.csv"
}

run_arm mean
run_arm default_relative
echo "DONE $(date +%T)"
