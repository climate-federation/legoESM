#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --job-name=convtune_arm
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --array=0-9%5
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/convtune_arm_%A_%a.out
set +e
# ---------------------------------------------------------------------------
# A-PRIORI + TUNED SCM-RCE equilibria for all 10 convection schemes, scored
# against the EXTERNAL RCEMIP SAM_CRM RCE_small300 reference.
#
# CONTROLLED COMPARISON.  Every flag below is pinned here, never defaulted, and
# is byte-identical across the ten schemes.  What varies between array tasks is
# the convection scheme and NOTHING else except the tuning budget, which is
# derived from the scheme's own parameter count (see TUNE-BUDGET below).
#
# Saturation treatment (the point of this arm):
#   --microphysics morrison        SAM M2005 flavour => the column carries ICE,
#                                  so the IFS/SAM homogeneous-freezing
#                                  allowance (rh_homo = 2.583 - T/207.8) is
#                                  LIVE.  Under the historical `kessler`
#                                  default it is inert and the upper
#                                  troposphere is biased for every scheme.
#   --hard-saturation-adjustment   the liquid half: no standing liquid
#                                  super-saturation.
#
# RESOURCES, from the cost probe (jobs 9356449_0/_1) rather than from caution:
# peak RSS 2.41 GB and 169 % CPU for a bechtold + morrison + RRTMGP run.  The
# first submission asked for 120 G and 8 cores; on a busy burst partition that
# is why 9 of 10 array tasks sat in "Nodes required for job are DOWN, DRAINED
# or reserved", so an over-request costs QUEUE TIME, not just accounting.
# 24 G is 10x the measured peak.
#
# TUNE-BUDGET.  The sampler spends its first 1 + 2*n_params evaluations on a
# deterministic one-at-a-time sweep and the rest on joint random draws
# (run_scm_rce_campaign._candidate_values).  A flat 48 therefore gives DCA (1
# parameter) 45 joint draws and Bechtold (20 parameters) 7 — the high-
# dimensional schemes would be ranked by optimiser under-convergence.  The
# budget is 12 evaluations per tunable parameter, clamped to [48, 240], and is
# COMPUTED from the live parameter registry rather than a hardcoded table so it
# cannot go stale.  Acceptance is best-so-far with a fixed seed, so a 48-eval
# result is nested inside a longer one.
# ---------------------------------------------------------------------------
WT=${WT:?export WT=<pinned worktree>}
PIN=${PIN:?export PIN=<sha> the worktree must be at}
ARM=${ARM:-implicit_flux}
REPO=/burg-archive/glab/users/pg2328/legoESM
REF=$REPO/results/rcemip_ref_sam300
BL_ANCHOR=${BL_ANCHOR:-0.0}
# Two labelled variables vs the 2026-08-10 arm, in ONE new baseline rather
# than a comparison against it: the precipitation term is now live (it reports
# the applied microphysics evaluation, so it discriminates between schemes
# instead of adding a constant 0.798 to every score), and BL_ANCHOR=-1
# disables the SST anchor so the sensible heat flux can be nonzero.
OUTTAG=$(awk -v b="$BL_ANCHOR" 'BEGIN{print (b<0)?"_noanchor":""}')
OUT=$REPO/results/scm_rce_convtune/arm_${ARM}_morrison${OUTTAG}
EVALS_PER_PARAM=${EVALS_PER_PARAM:-12}
EVALS_MIN=${EVALS_MIN:-48}
EVALS_MAX=${EVALS_MAX:-240}

cd $WT || exit 90
HEAD_SHA=$(git rev-parse HEAD)
echo "PINNED sha: $HEAD_SHA"
echo "host: $(hostname)  date: $(date -Is)"
[ "$HEAD_SHA" = "$PIN" ] || { echo "PREFLIGHT FAIL: $HEAD_SHA != $PIN"; exit 91; }
[ -d "$REF/snapshots3d" ] || { echo "PREFLIGHT FAIL: no reference at $REF"; exit 94; }
case "$REF" in
  *rcemip1_n128_ocean*|*rcemip1_postfix*)
    echo "PREFLIGHT FAIL: REF points at OUR OWN CRM run, not an oracle"; exit 95;;
esac

SCHEMES=(sbm dca kuo mass_flux edmf zhang_mcfarlane kain_fritsch emanuel tiedtke bechtold)
SCHEME=${SCHEMES[$SLURM_ARRAY_TASK_ID]}

# Node-local scratch has broken this campaign four times now: /local cleaned
# mid-run, /tmp cleaned mid-run, CODEX_HOME vanishing between mkdir and use,
# and now a FULL /tmp on g008 -- "cannot create temp file for here-document:
# No space left on device", which kills the parameter-count snippet before it
# can print anything. Point TMPDIR at durable space so the shell's own
# here-documents have somewhere to live.
export TMPDIR=${SLURM_TMPDIR:-/burg-archive/glab/users/pg2328/_jobtmp/${SLURM_JOB_ID:-$$}}
mkdir -p "$TMPDIR" || { echo "PREFLIGHT FAIL: cannot create TMPDIR $TMPDIR"; exit 89; }
trap 'rm -rf /burg-archive/glab/users/pg2328/_jobtmp/${SLURM_JOB_ID:-$$}' EXIT

export JAX_PLATFORMS=cpu
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$WT
# See convtune_verify.sbatch: a persistent XLA:CPU cache shared across
# heterogeneous burst nodes loads AOT results compiled for another CPU.
export LEGOESM_JIT_CACHE_DIR=""
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

# --- tuning budget from the LIVE registry ---------------------------------
# The snippet's EXIT STATUS matters, and `| tail -1` would hide it: a snippet
# that prints a number and then dies still yields a plausible count.  Capture
# the output, check the status separately, and require exactly one integer.
NPARAM_RAW=$(JAX_ENABLE_X64=1 $PY - "$SCHEME" <<'PYEOF' 2>/dev/null
import sys
from scripts.run import run_scm_rce_campaign as camp
cfg = camp.make_physics_config(convection=sys.argv[1])
_c, _s, sub = camp._active_subconfig(cfg, "convection")
key = camp._scheme_key_for_subconfig(sub)
from legoesm.training.param_collector import build_trainable_params
import jax.numpy as jnp
p = build_trainable_params(active_scheme_keys={key}, tier="extended",
                           dtype=jnp.float64)
print(len(p.constraints))
PYEOF
)
NPARAM_RC=$?
[ $NPARAM_RC -eq 0 ] || { echo "PREFLIGHT FAIL: parameter count for $SCHEME exited $NPARAM_RC"; exit 96; }
[ "$(printf '%s' "$NPARAM_RAW" | wc -l)" -le 1 ] || { echo "PREFLIGHT FAIL: parameter count printed more than one line: '$NPARAM_RAW'"; exit 96; }
NPARAM=$(printf '%s' "$NPARAM_RAW" | tr -d '[:space:]')
case "$NPARAM" in ''|*[!0-9]*) echo "PREFLIGHT FAIL: could not count params for $SCHEME (got '$NPARAM_RAW')"; exit 96;; esac
# Zero tunables is a REAL state, not an error: MEASURED 2026-08-10, `dca`
# reports 0 extended-tier parameters from the live registry (the campaign doc
# claims 1, `cape_threshold` — that claim is stale). Refusing to run such a
# scheme would drop it from the ranking and the merge gate would then refuse
# the whole table, so the arm runs with a budget of 1: tune_category_winner
# returns the defaults immediately when there are no constraints, so a larger
# budget buys nothing. The row then shows tuned == a priori with #params 0,
# which is the honest report of a scheme with no exposed tuning surface.
if [ "$NPARAM" -eq 0 ]; then
  echo "NOTE: $SCHEME has NO extended-tier tunable parameters; its tuned"
  echo "      result is its a-priori result BY CONSTRUCTION, not by failure."
fi
EVALS=$(( NPARAM * EVALS_PER_PARAM ))
if [ "$NPARAM" -eq 0 ]; then
  EVALS=1
else
  [ "$EVALS" -lt "$EVALS_MIN" ] && EVALS=$EVALS_MIN
fi
[ "$EVALS" -gt "$EVALS_MAX" ] && EVALS=$EVALS_MAX

echo "SCHEME=$SCHEME  KERNEL_ARM=$ARM  n_params=$NPARAM  tune_evals=$EVALS  bl_anchor_top_m=$BL_ANCHOR"
echo "REFERENCE=$REF"
echo "OUT=$OUT"

JAX_ENABLE_X64=1 $PY $WT/scripts/run/run_scm_rce_convection_intercomparison.py \
  --reference-dir $REF \
  --outdir $OUT \
  --schemes $SCHEME \
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
  --tune-evals $EVALS \
  --tune-seed 20260810 \
  --no-merge 2>&1 | grep -v "^E[0-9]\|pjrt_executable\|cpu_aot_loader"
RC=${PIPESTATUS[0]}
echo "ARM_STATUS=$([ $RC -eq 0 ] && echo OK || echo FAILED) (scheme=$SCHEME arm=$ARM rc=$RC)"
echo ALLDONE
# PROPAGATE the failure: printing it while exiting 0 makes sacct report every
# arm COMPLETED, which is the state the merge gate would then be the only
# thing standing between a dead scheme and a published table.
exit $RC
