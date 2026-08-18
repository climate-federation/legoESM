#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=tqtune_gates
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/tqtune_gates_%j.out
set +e
# ---------------------------------------------------------------------------
# Every test that touches the new objective, the parameter selection, the
# refinement, the signature and the ranking figure -- plus the SUITES THAT
# CONSUME what was changed (the campaign's own CLI tests and the collector's),
# because editing build_trainable_params and _SIGNATURE_FIELDS is a change to
# every test in those modules, not only to the ones written here.
# ---------------------------------------------------------------------------
WT=${WT:?export WT=<pinned worktree>}
PIN=${PIN:?export PIN=<sha> the worktree must be at}

cd $WT || exit 90
HEAD_SHA=$(git rev-parse HEAD)
echo "PINNED sha: $HEAD_SHA"
echo "host: $(hostname)  date: $(date -Is)"
[ "$HEAD_SHA" = "$PIN" ] || { echo "PREFLIGHT FAIL: $HEAD_SHA != $PIN"; exit 91; }

export TMPDIR=${SLURM_TMPDIR:-/burg-archive/glab/users/pg2328/_jobtmp/${SLURM_JOB_ID:-$$}}
mkdir -p "$TMPDIR" || { echo "PREFLIGHT FAIL: cannot create TMPDIR"; exit 89; }
trap 'rm -rf /burg-archive/glab/users/pg2328/_jobtmp/${SLURM_JOB_ID:-$$}' EXIT

export JAX_PLATFORMS=cpu
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$WT
export LEGOESM_JIT_CACHE_DIR=""
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

JAX_ENABLE_X64=1 $PY -m pytest -q \
  tests/unit/test_scm_rce_thermo_objective.py \
  tests/unit/test_plot_scm_rce_tq_ranking.py \
  tests/unit/test_scm_rce_tq_objective_probe.py \
  tests/unit/test_scm_rce_tq_seed_agreement.py \
  tests/unit/test_scm_rce_convection_intercomparison_cli.py \
  tests/unit/test_param_collector.py \
  tests/test_param_specs.py \
  tests/unit/test_scm_rce_applied_precip.py \
  2>&1 | tail -30
RC=${PIPESTATUS[0]}
echo "GATES_STATUS=$([ $RC -eq 0 ] && echo OK || echo FAILED) (rc=$RC)"
echo ALLDONE
exit $RC
