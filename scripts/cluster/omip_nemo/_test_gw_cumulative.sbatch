#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=gw_cum
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/gw_cum_%j.out
#
# Test the per-cadence CUMULATIVE gateway dump (PR #1440 follow-up).
# Runs in the PINNED PR worktree; read-only (pytest writes only to --basetemp).
set -uo pipefail
WT=${WT:-/burg-archive/glab/users/pg2328/_gw_pr_wt}
cd "$WT"
export PYTHONPATH="$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
BT=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/_pt_gwcum_${SLURM_JOB_ID}
mkdir -p "$BT"
echo "=== worktree $WT at $(git -C $WT rev-parse HEAD) ==="
echo "=== dirty: [$(git -C $WT status --porcelain --untracked-files=no | tr '\n' ' ')] ==="

# TWO invocations: the three suites in ONE interpreter accumulate enough
# jitted models to abort the process (SIGABRT, job 9311668); process
# isolation is the honest fix, not a bigger box.  BOTH exit codes gate.
$PY -m pytest \
  tests/unit/test_run_omip_core2_gateway_transports.py \
  tests/ocean/unit/test_diagnostics_sections.py \
  -q --no-header -p no:cacheprovider --basetemp="$BT/a" 2>&1 \
  | grep -vE "jax_plugins|xla_bridge|cuda|versions_helpers|_check_cuda|plugin_module" \
  | tail -25
RC_A=${PIPESTATUS[0]}
echo "=== EXIT(gateway suites) $RC_A ==="
$PY -m pytest \
  tests/ocean/unit/test_mass_flux_store.py \
  -q --no-header -p no:cacheprovider --basetemp="$BT/b" 2>&1 \
  | grep -vE "jax_plugins|xla_bridge|cuda|versions_helpers|_check_cuda|plugin_module" \
  | tail -25
RC_B=${PIPESTATUS[0]}
echo "=== EXIT(mass/salt-flux suite) $RC_B ==="
echo "=== EXIT $(( RC_A > RC_B ? RC_A : RC_B )) ==="
echo "=== TEST JOB DONE ==="
