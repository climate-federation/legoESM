#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --job-name=subcloud_arm
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/subcloud_arm_%A_%a.out
set +e
# ---------------------------------------------------------------------------
# THE SUB-CLOUD ARM.  Target, measured (job 9403110): our columns sit at
# RH 0.82-0.99 with the lowest level within ~1 K of the SST and evaporate
# 1.2-1.9 mm/day, against the CRM reference's RH 0.752, 3.06 K and 2.73.
#
# ONE VARIABLE PER TASK, and the tasks are ordered so each isolates one:
#
#   0  emanuel  + clubb, downdraft OFF   <- isolates CLUBB against the
#                                          published louis arm's emanuel row
#   1  emanuel  + clubb, downdraft ON    <- isolates the ported CONVECT
#                                          downdraft against task 0
#   2  bechtold + clubb                  <- a scheme whose downdraft
#                                          evaporation knob is already live
#   3  tiedtke  + clubb                  <- ditto, second opinion
#
# Everything else is byte-identical to the anchor-off arm's protocol, so the
# published louis rows remain the control.
#
# OBJECTIVE = subcloud.  The full-column score is 87-100 % condensate and
# cannot see this layer; the sub-cloud objective scores the layer-mean bias
# and shape of T and q_v on DECLARED scales (1 K, 1 g/kg) plus the surface
# evaporation against the reference's own measured hfls.  Both scores are
# recorded for every run whichever is selected.
#
# TUNE SET = the focused cross-category set, every member of which was checked
# to be READ by executed code.  Six proposed knobs were dropped after that
# check (four unread CLUBB fields, morrison's evap_coeff which the m2005
# branch does not use, and emanuel's downdraft_efficiency while its branch is
# off); see SUBCLOUD_*_INCLUDE.
# ---------------------------------------------------------------------------
WT=${WT:?export WT=<pinned worktree>}
PIN=${PIN:?export PIN=<sha>}
REPO=/burg-archive/glab/users/pg2328/legoESM
REF=$REPO/results/rcemip_ref_sam300
OUT=${OUT:-$REPO/results/scm_rce_convtune/arm_subcloud_clubb}

cd $WT || exit 90
HEAD_SHA=$(git rev-parse HEAD)
echo "PINNED sha: $HEAD_SHA"; echo "host: $(hostname)"; echo "task $SLURM_ARRAY_TASK_ID"
[ "$HEAD_SHA" = "$PIN" ] || { echo "PREFLIGHT FAIL: $HEAD_SHA != $PIN"; exit 91; }
git diff --quiet && git diff --cached --quiet || {
  echo "PREFLIGHT FAIL: DIRTY TREE at $HEAD_SHA — uncommitted edits would run"
  git status --porcelain | head -20
  exit 94
}

case $SLURM_ARRAY_TASK_ID in
  0) SCHEME=emanuel;  DD="--no-emanuel-unsaturated-downdraft"; TAG=emanuel_dd_off ;;
  1) SCHEME=emanuel;  DD="--emanuel-unsaturated-downdraft";    TAG=emanuel_dd_on  ;;
  2) SCHEME=bechtold; DD="--no-emanuel-unsaturated-downdraft"; TAG=bechtold       ;;
  3) SCHEME=tiedtke;  DD="--no-emanuel-unsaturated-downdraft"; TAG=tiedtke        ;;
  *) echo "no such task"; exit 93 ;;
esac
# Separate output dirs: tasks 0 and 1 run the SAME convection scheme and would
# otherwise overwrite one another's scheme_emanuel.json.
TASKOUT=$OUT/$TAG
mkdir -p "$TASKOUT"
echo "scheme=$SCHEME  $DD  out=$TASKOUT"

export JAX_PLATFORMS=cpu
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$WT
export LEGOESM_JIT_CACHE_DIR=""
export MPLBACKEND=Agg
export TMPDIR=${SLURM_TMPDIR:-/burg-archive/glab/users/pg2328/_tmp}
mkdir -p "$TMPDIR"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

# TUNE BUDGET from the LIVE parameter count, not a guess: the focused set is
# built per (turbulence, microphysics, convection) triple and differs between
# tasks, so a flat number would give one task a denser search than another.
NPAR=$(JAX_PLATFORMS=cpu $PY - "$SCHEME" "$DD" <<'EOF'
import sys
sys.path.insert(0, ".")
from scripts.run.run_scm_rce_convection_intercomparison import (
    default_focused_include)
scheme, dd = sys.argv[1], sys.argv[2]
inc = default_focused_include(
    turbulence="clubb", microphysics="morrison", convection=scheme,
    emanuel_unsaturated_downdraft=(dd == "--emanuel-unsaturated-downdraft"))
print(len(inc))
EOF
)
[ -n "$NPAR" ] && [ "$NPAR" -gt 0 ] 2>/dev/null || { echo "ABORT: could not resolve the parameter count ($NPAR)"; exit 94; }
EVALS=$((12 * NPAR))
[ $EVALS -lt 48 ] && EVALS=48
[ $EVALS -gt 240 ] && EVALS=240
echo "focused parameters: $NPAR -> $EVALS evaluations"

JAX_ENABLE_X64=1 $PY $WT/scripts/run/run_scm_rce_convection_intercomparison.py \
  --reference-dir $REF --outdir $TASKOUT \
  --schemes $SCHEME \
  --turbulence clubb --microphysics morrison --hard-saturation-adjustment \
  --radiation rrtmgp --subsidence-solve implicit_flux \
  --bl-anchor-top-m -1.0 --scm-convection-substeps 10 \
  --tune-mode focused --objective subcloud $DD \
  --days 100 --analysis-days 5 --dt 600 \
  --tune-evals $EVALS --tune-seed 20260815 --no-merge 2>&1 \
  | grep -v "^E[0-9]\|pjrt_executable\|cpu_aot_loader\|CUDA_ERROR_NO_DEVICE\|xla_bridge\|plugin_module\|_check_cuda\|cuda_device_count\|Jax plugin configuration"
echo "ARM_RC=${PIPESTATUS[0]}"
echo "COMPLETED"
