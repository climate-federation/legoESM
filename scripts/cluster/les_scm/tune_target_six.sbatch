#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100"
#SBATCH --job-name=les_six
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
# The completed 8-step 9-scheme 5-case run took 21h43 while the per-scheme
# tuning wall summed to ~17 min, so the cost is COMPILATION of the
# per-(scheme,case) rollout graph. Six cases and 40 steps therefore cost about
# 6/5 of that, not 5x. _write_outputs runs ONCE after the last scheme, so a
# timeout loses the job: take the partition maximum.
#SBATCH --time=72:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/les_six_%j.out
set +e
# ONE parameter set per turbulence scheme, fitted jointly to the SIX target
# regimes -- three dry, three wet:
#
#   dry   ekman    neutral shear layer        scored u, v
#         gabls1   stable boundary layer      scored theta, u, v
#         wangara  diurnal convective BL      scored theta, u, v
#   wet   bomex    shallow cumulus            scored theta, qv
#         astex    stratocumulus              scored theta, qv
#         dycoms   nocturnal stratocumulus    scored theta, qv
#
# Both stratocumulus decks set doradsimple, and the SCM arm runs that SAME
# Stevens (2005) longwave kernel, so cloud-top radiative cooling is present on
# both sides rather than only in the LES.
#
# --surface-flux-to-closure is ON: without it every closure computes a surface
# heat flux of exactly zero on a prescribed-flux case, which measured a 39%
# score change on YSU and moved it two ranks.
#
# AGGREGATION is the only difference between the two arms. Both report both
# numbers, so they are directly comparable.
: "${SIX_WT:?set SIX_WT to the pinned worktree}"
WT=$SIX_WT
cd $WT || exit 1
echo "PINNED sha: $(git rev-parse HEAD)"; echo "host: $(hostname)"; nvidia-smi -L
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=/burg-archive/glab/users/pg2328/legoESM/results
STEPS=${STEPS:-40}

CASES="ekman:$R/les_ref/ekman,gabls1:$R/les_ref/gabls1,wangara:$R/les_ref/wangara"
CASES="$CASES,bomex:$R/les_ref/bomex,astex:$R/les_ref/astex,dycoms:$R/les_ref/dycoms"

# PREFLIGHT: refuse to start if any reference is missing. Six cases is a long
# job and discovering a missing deck after the queue wait wastes the slot.
for c in ekman gabls1 wangara bomex astex dycoms; do
  if [ ! -d "$R/les_ref/$c" ]; then
    echo "MISSING LES reference: $R/les_ref/$c -- refusing to start"; exit 2
  fi
done
echo "all six references present"

run_arm () {
  local agg=$1
  local out=$R/scm_les_turbulence/six_$agg
  echo "================= ARM $agg ================="
  rm -rf $out
  local t0=$SECONDS
  # NO `| tail`: a pipe buffers the driver's progress until it exits, so a
  # long run shows nothing and a death shows nothing either. PIPESTATUS is
  # unnecessary without the pipe.
  $PY scripts/run/run_scm_les_turbulence_tuning.py \
      --cases "$CASES" --outdir "$out" --nlev 48 --steps $STEPS \
      --joint-aggregation "$agg" --surface-flux-to-closure
  local rc=$?
  echo "ARM $agg rc=$rc wall=$((SECONDS - t0))s"
  [ "$rc" -ne 0 ] && { echo "ARM $agg FAILED -- do not cite this arm"; return; }

  $PY scripts/plot/plot_scm_les_tuning_summary.py "$out"
  echo "  summary plot rc=$?"
  for c in ekman gabls1 wangara bomex astex dycoms; do
    $PY scripts/plot/plot_scm_les_turbulence.py "$out" --case "$c" \
        --out "$out/profiles_vs_les_$c.png" >/dev/null 2>&1 \
        && echo "  plotted $c"
  done
  $PY scripts/validate/scm_les_score_decomposition.py "$out" \
      --json-out "$out/score_decomposition.json" | tail -80
  echo "=== ranking ($agg) ==="; cat "$out/ranking.csv"
}

run_arm mean
run_arm default_relative
echo "DONE $(date +%T)"
