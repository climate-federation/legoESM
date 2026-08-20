#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100"
#SBATCH --job-name=les_sfab
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/les_sfab_%j.out
set +e
# ONE variable: whether the deck's PRESCRIBED surface flux reaches the closure.
#
# PREDICTION being tested. With the flux withheld (the historical path) every
# closure computes a surface heat flux of exactly zero. Local closures still
# see the gradient the separately-injected heating produces; flux-driven
# NONLOCAL closures lose their defining pathway -- YSU's convective velocity
# scale, PBL depth, entrainment and countergradient all come from shflx.
#   CONFIRMS the ranking was a harness artifact: the four flux-driven schemes
#     (ysu, holtslag_boville, edmf, mynn25) improve markedly on the dry
#     convective and stable cases, and the rank ORDER changes.
#   REFUTES it: the per-case scores move by less than the spread between
#     adjacent ranks, leaving the order intact.
# Tuning is SKIPPED on purpose: this is a question about the default physics,
# and the optimizer would confound it. That also makes it the cheapest run
# that can answer it -- no offline test can, because the closures have to be
# integrated to respond to a boundary condition.
: "${AB_WT:?set AB_WT to the pinned worktree}"
WT=$AB_WT
cd $WT || exit 1
echo "PINNED sha: $(git rev-parse HEAD)"; echo "host: $(hostname)"; nvidia-smi -L
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=/burg-archive/glab/users/pg2328/legoESM/results
CASES="bomex:$R/les_ref/bomex,rico:$R/les_ref/rico,gabls1:$R/les_ref/gabls1,cbl:$R/les_ref/cbl,ekman:$R/les_ref/ekman"

arm () {   # $1 = label, $2... = the ONE differing flag
  local label=$1; shift
  local out=$R/scm_les_turbulence/sfab_$label
  echo "================= ARM $label ================="
  rm -rf $out
  local t0=$SECONDS
  $PY scripts/run/run_scm_les_turbulence_tuning.py \
      --cases "$CASES" --outdir "$out" --nlev 48 --skip-tuning "$@" 2>&1 | tail -120
  local rc=${PIPESTATUS[0]}
  echo "ARM $label rc=$rc wall=$((SECONDS - t0))s"
  [ "$rc" -ne 0 ] && { echo "ARM $label FAILED -- do not cite"; return; }
  echo "=== ranking ($label) ==="; cat "$out/ranking.csv"
}

arm fluxoff --no-surface-flux-to-closure
arm fluxon  --surface-flux-to-closure
echo "DONE $(date +%T)"
