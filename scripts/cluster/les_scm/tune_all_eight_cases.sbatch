#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100"
#SBATCH --job-name=les_eight
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/les_eight_%j.out
set +e
# ONE parameter set per turbulence closure, fitted jointly to EVERY LES
# reference we have -- four dry and four wet:
#
#   dry   ekman     neutral shear layer          scored u, v
#         gabls1    stable boundary layer        scored theta, u, v
#         cbl       Nieuwstadt free convection   scored theta
#         wangara   diurnal convective BL        scored theta, u, v
#   wet   bomex     shallow trade cumulus        scored theta, qv
#         rico      precipitating trade cumulus  scored theta, qv
#         astex     transitioning stratocumulus  scored theta, qv
#         dycoms    nocturnal stratocumulus      scored theta, qv
#
# Both stratocumulus decks set `dolongwave = .true., doradsimple = .true.` and
# the SCM arm runs that SAME Stevens (2005) kernel, so cloud-top radiative
# cooling is on both sides rather than only in the LES.
#
# WALLTIME, sized from the ONE measurement available (job 9333698: 9 schemes,
# 5 cases, 8 steps, 21 h 43). That run's cost is dominated by XLA compilation
# of the per-(scheme, case) rollout graph, which scales with schemes x cases
# and not with optimizer steps: 5 -> 8 cases is 1.6x the compiles, giving
# ~35 h, plus ~13 h of execution at 40 steps over 1.47x the model steps per
# pass. ~48 h expected against a 72 h limit. The driver now writes its full
# report after EVERY scheme, so an overrun costs the unfinished schemes rather
# than the whole campaign.
#
# --steps 40, not 8: in job 9333698 six of nine schemes were still accepting
# an update at step 8, so that run reported a truncated fit as a converged one.
#
# --surface-flux-to-closure is ON. Without it every closure computes a surface
# heat flux of exactly zero on a prescribed-flux case, which disables the
# defining pathway of every flux-driven nonlocal scheme; measured as a 39%
# score change on YSU, moving it two ranks.
#
# --joint-aggregation default_relative, ONE arm only. 'mean' weights each
# regime's ABSOLUTE normalized error equally, which on the five-case run put
# 43% of the joint loss on ekman (70% for tke) against 3.1% on bomex -- the
# "joint" fit was largely an ekman fit. 'default_relative' minimizes the mean
# of s_i/s_i(default), so every regime enters with equal improvement headroom,
# which is what "one set of parameters that works across all the cases" means.
# BOTH numbers are reported either way, and score_tuned is always the plain
# mean so the headline default-vs-tuned comparison keeps its units.
: "${E_WT:?set E_WT to the pinned worktree}"
WT=$E_WT
cd "$WT" || exit 1
echo "PINNED sha: $(git rev-parse HEAD)"; echo "host: $(hostname)"; nvidia-smi -L
export PYTHONPATH=$WT/src:$(ls -d "$WT"/packages/*/ | tr '\n' ':')$PYTHONPATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=/burg-archive/glab/users/pg2328/legoESM/results
CASE_LIST="ekman gabls1 cbl wangara bomex rico astex dycoms"
STEPS=${STEPS:-40}
AGG=${AGG:-default_relative}
OUT=${OUT:-$R/scm_les_turbulence/all_eight}

# PREFLIGHT: refuse to start if any reference is missing. Eight cases is a long
# job and discovering a missing deck after the queue wait wastes the slot.
CASES=""
for c in $CASE_LIST; do
  if [ ! -d "$R/les_ref/$c" ]; then
    echo "MISSING LES reference: $R/les_ref/$c -- refusing to start"; exit 2
  fi
  CASES="${CASES:+$CASES,}$c:$R/les_ref/$c"
done
echo "all eight references present"
echo "cases: $CASES"

rm -rf "$OUT"
t0=$SECONDS
# NO pipe to tail: a pipe buffers the driver's progress until it exits, so a
# long run shows nothing and a death shows nothing either. Without the pipe,
# $? is the driver's own status.
$PY scripts/run/run_scm_les_turbulence_tuning.py \
    --cases "$CASES" --outdir "$OUT" --nlev 48 --steps "$STEPS" \
    --joint-aggregation "$AGG" --surface-flux-to-closure
rc=$?
echo "driver rc=$rc wall=$((SECONDS - t0))s"

# The plots run even on rc != 0: the per-scheme flush means a killed or partly
# failed run still holds every scheme that finished, and those are exactly the
# ones worth looking at.
$PY scripts/plot/plot_scm_les_tuning_summary.py "$OUT"
echo "  summary plot rc=$?"
for c in $CASE_LIST; do
  $PY scripts/plot/plot_scm_les_turbulence.py "$OUT" --case "$c" \
      --out "$OUT/profiles_vs_les_$c.png" && echo "  plotted $c"
done
$PY scripts/validate/scm_les_score_decomposition.py "$OUT" \
    --json-out "$OUT/score_decomposition.json" | tail -90
echo "=== ranking ==="; cat "$OUT/ranking.csv"
echo "DONE $(date +%T)"
exit $rc
