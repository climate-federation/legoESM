#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --job-name=tqtune_arm
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --array=0-19%8
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/tqtune_arm_%A_%a.out
set +e
# ---------------------------------------------------------------------------
# A-PRIORI + TUNED SCM-RCE equilibria for all 10 convection schemes, tuned to
# the CRM's TEMPERATURE AND HUMIDITY profiles, scored against the EXTERNAL
# RCEMIP SAM_CRM RCE_small300 reference.
#
# FIFTY tasks = 10 schemes x 5 SEEDS.  The extra seeds are not redundancy: a
# uniform search over 20-26 parameters leaves draws ~0.76 of the range apart on
# every axis, so a single seed's ranking is partly a lottery.  Two seeds make
# rank REVERSALS visible; they do not make the search converged, and the report
# must say so.
#
# WHAT DIFFERS FROM THE 2026-08-13 anchor-off arm (labelled, not silent):
#   --objective thermo      full-column T + TROPOSPHERIC RH, each on a fixed
#                           physical tolerance (1 K, 5 % RH).  The historical
#                           `combined` score is 87-100 % condensate, so it is
#                           not a temperature or humidity target.
#   --param-set physical    every aggressive-tier parameter EXCEPT those
#                           categorised `numerics`, PLUS the tier-0 parameters
#                           excluded only because their AD gradient vanishes --
#                           the CAPE trigger of eight of the ten schemes.
#   --tune-refine-frac 0.35 the last third of the budget is a local coordinate
#                           sweep around the incumbent instead of more random
#                           draws.
# Everything else is byte-identical to that arm, so the a-priori columns are
# directly comparable and the change in the tuned columns is attributable.
#
# TUNE-BUDGET: 12 evaluations per tunable parameter, clamped to [48, 240], and
# COMPUTED FROM THE LIVE REGISTRY at the SAME parameter set the run uses -- a
# budget derived at a different set would silently under- or over-spend.
# At the measured ~171 s per 100-day evaluation, 240 evaluations is ~11.4 h.
# ---------------------------------------------------------------------------
WT=${WT:?export WT=<pinned worktree>}
PIN=${PIN:?export PIN=<sha> the worktree must be at}
ARM=${ARM:-implicit_flux}
REPO=/burg-archive/glab/users/pg2328/legoESM
REF=$REPO/results/rcemip_ref_sam300
BL_ANCHOR=${BL_ANCHOR:--1}
OBJECTIVE=${OBJECTIVE:-thermo}
PARAM_SET=${PARAM_SET:-physical}
REFINE_FRAC=${REFINE_FRAC:-0.35}
EVALS_PER_PARAM=${EVALS_PER_PARAM:-12}
EVALS_MIN=${EVALS_MIN:-48}
EVALS_MAX=${EVALS_MAX:-240}

SCHEMES=(sbm dca kuo mass_flux edmf zhang_mcfarlane kain_fritsch emanuel tiedtke bechtold)
# FIVE independent searches per scheme: two give a disagreement, five give a
# distribution, which is what makes "this scheme improved" separable from
# "this search got lucky". Array index -> (scheme, seed) as idx%10, idx/10.
SEEDS=(20260815 20260816 20260817 20260818 20260819)
SCHEME=${SCHEMES[$(( SLURM_ARRAY_TASK_ID % 10 ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID / 10 ))]}
# Coriolis parameter [1/s].  RCEMIP is NONROTATING with a freely evolving
# wind; the SCM default 2.5e-5 plus a uniform geostrophic target pins the
# column wind near its initial value and force-feeds moisture into a surface
# layer nothing can ventilate.  Measured on mass_flux: the pin accounts for
# ~78% of the +6.2 g/kg near-surface humidity bias (dqv +6.20 -> +1.37, RH
# 0.933 -> 0.799, SST-T_air 0.63 -> 2.53 K).  CORIOLIS=0 is the corrected
# protocol; the results directory carries it so a corrected run can never be
# compared term-by-term against a pinned one by accident.
CORIOLIS=${CORIOLIS:-0.0}
PROTO_TAG=${PROTO_TAG:-f0}
# Joint multi-SST tuning for generalization. sbatch --export SPLITS a value on
# commas, so the SSTs are passed DASH-joined (SSTS=295-300-305) and converted to
# the comma list the driver's --ssts expects HERE, inside the job. Default 300
# keeps single-SST behaviour. The driver resolves sibling rcemip_ref_sam<SST>
# dirs; the primary reference-dir must match one of the SSTs.
SSTS=${SSTS//-/,}
SSTS=${SSTS:-300}
OUT=$REPO/results/scm_rce_tqtune/arm_${OBJECTIVE}_${PARAM_SET}_${PROTO_TAG}_seed${SEED}

cd $WT || exit 90
HEAD_SHA=$(git rev-parse HEAD)
echo "PINNED sha: $HEAD_SHA"
echo "host: $(hostname)  date: $(date -Is)"
[ "$HEAD_SHA" = "$PIN" ] || { echo "PREFLIGHT FAIL: $HEAD_SHA != $PIN"; exit 91; }
git diff --quiet && git diff --cached --quiet || {
  echo "PREFLIGHT FAIL: DIRTY TREE at $HEAD_SHA — uncommitted edits would run"
  git status --porcelain | head -20
  exit 94
}
[ -d "$REF/snapshots3d" ] || { echo "PREFLIGHT FAIL: no reference at $REF"; exit 94; }
case "$REF" in
  *rcemip1_n128_ocean*|*rcemip1_postfix*)
    echo "PREFLIGHT FAIL: REF points at OUR OWN CRM run, not an oracle"; exit 95;;
esac

export TMPDIR=${SLURM_TMPDIR:-/burg-archive/glab/users/pg2328/_jobtmp/${SLURM_JOB_ID:-$$}}
mkdir -p "$TMPDIR" || { echo "PREFLIGHT FAIL: cannot create TMPDIR $TMPDIR"; exit 89; }
trap 'rm -rf /burg-archive/glab/users/pg2328/_jobtmp/${SLURM_JOB_ID:-$$}' EXIT

export JAX_PLATFORMS=cpu
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$WT
# A persistent XLA:CPU cache shared across heterogeneous burst nodes loads AOT
# results compiled for another CPU (see convtune_verify.sbatch).
export LEGOESM_JIT_CACHE_DIR=""
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

# --- tuning budget from the LIVE registry, at THIS parameter set ------------
# The snippet's EXIT STATUS matters and `| tail -1` would hide it: a snippet
# that prints a number and then dies still yields a plausible count.
NPARAM_RAW=$(JAX_ENABLE_X64=1 $PY - "$SCHEME" "$PARAM_SET" <<'PYEOF' 2>/dev/null
import sys
from scripts.run import run_scm_rce_campaign as camp
cfg = camp.make_physics_config(convection=sys.argv[1])
_c, _s, sub = camp._active_subconfig(cfg, "convection")
key = camp._scheme_key_for_subconfig(sub)
from legoesm.training.param_collector import build_trainable_params
import jax.numpy as jnp
tier, include_tier0, exclude = camp.resolve_param_selection(key, sys.argv[2])
p = build_trainable_params(active_scheme_keys={key}, tier=tier,
                           include_tier0=include_tier0, exclude=exclude,
                           dtype=jnp.float64)
print(len(p.constraints))
PYEOF
)
NPARAM_RC=$?
[ $NPARAM_RC -eq 0 ] || { echo "PREFLIGHT FAIL: parameter count for $SCHEME exited $NPARAM_RC"; exit 96; }
[ "$(printf '%s' "$NPARAM_RAW" | wc -l)" -le 1 ] || { echo "PREFLIGHT FAIL: parameter count printed more than one line: '$NPARAM_RAW'"; exit 96; }
NPARAM=$(printf '%s' "$NPARAM_RAW" | tr -d '[:space:]')
case "$NPARAM" in ''|*[!0-9]*) echo "PREFLIGHT FAIL: could not count params for $SCHEME (got '$NPARAM_RAW')"; exit 96;; esac
# Zero tunables is a REAL state, not an error: `dca` exposes none even in the
# physical set (its only tier-0 entry is the CAPE trigger, which IS admitted --
# so if this prints 0 for dca after the physical set landed, that is itself
# worth reading).  The scheme still runs, with a budget of 1: the tuner returns
# the defaults immediately when there are no constraints.
if [ "$NPARAM" -eq 0 ]; then
  echo "NOTE: $SCHEME has NO tunable parameters in the '$PARAM_SET' set; its"
  echo "      tuned result is its a-priori result BY CONSTRUCTION, not by failure."
fi
EVALS=$(( NPARAM * EVALS_PER_PARAM ))
if [ "$NPARAM" -eq 0 ]; then
  EVALS=1
else
  [ "$EVALS" -lt "$EVALS_MIN" ] && EVALS=$EVALS_MIN
fi
[ "$EVALS" -gt "$EVALS_MAX" ] && EVALS=$EVALS_MAX

echo "SCHEME=$SCHEME  SEED=$SEED  KERNEL_ARM=$ARM  objective=$OBJECTIVE"
echo "PROTOCOL: coriolis=$CORIOLIS ($PROTO_TAG)  SSTs=$SSTS  outdir=$OUT"
echo "param_set=$PARAM_SET  n_params=$NPARAM  tune_evals=$EVALS"
echo "refine_frac=$REFINE_FRAC  bl_anchor_top_m=$BL_ANCHOR"
echo "REFERENCE=$REF"
echo "OUT=$OUT"

JAX_ENABLE_X64=1 $PY $WT/scripts/run/run_scm_rce_convection_intercomparison.py \
  --reference-dir $REF \
  --outdir $OUT \
  --schemes $SCHEME \
  --objective $OBJECTIVE \
  --param-set $PARAM_SET \
  --tune-refine-frac $REFINE_FRAC \
  --subsidence-solve $ARM \
  --microphysics morrison \
  --hard-saturation-adjustment \
  --days 100 \
  --dt 600 \
  --analysis-days 5 \
  --last-reference-files 5 \
  --radiation rrtmgp \
  --scm-convection-substeps 10 \
  --scm-microphysics-substeps 30 \
  --bl-anchor-top-m $BL_ANCHOR \
  --ssts $SSTS \
  --tune-evals $EVALS \
  --tune-seed $SEED \
  --coriolis-s-inv $CORIOLIS \
  --no-merge 2>&1 | grep -v "^E[0-9]\|pjrt_executable\|cpu_aot_loader"
RC=${PIPESTATUS[0]}
echo "ARM_STATUS=$([ $RC -eq 0 ] && echo OK || echo FAILED) (scheme=$SCHEME seed=$SEED rc=$RC)"
echo ALLDONE
# PROPAGATE the failure: printing it while exiting 0 makes sacct report every
# arm COMPLETED, and the merge gate would then be the only thing between a dead
# scheme and a published table.
exit $RC
