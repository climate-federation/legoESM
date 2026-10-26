#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=merge_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/merge_test_%j.out
set +e
# Does the merge resolution actually run? Both conflicts were in a PHYSICS
# path -- the MPAS turbulence factory, where main's land-flux blending and this
# branch's per-step prescribed surface flux now coexist -- so "it parses" is
# not evidence. Merging a physics conflict untested is how a silently wrong
# surface flux reaches main.
: "${E_WT:?set E_WT}"
cd "$E_WT" || exit 1
echo "MERGED sha: $(git rev-parse HEAD)  host: $(hostname)"
export PYTHONPATH=$E_WT/src:$(ls -d "$E_WT"/packages/*/ | tr '\n' ':')
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export LES_REF_DIR=/burg-archive/glab/users/pg2328/legoESM/results/les_ref

echo "=== the two conflicted paths, and what exercises them ==="
$PY -m pytest -q \
    tests/atmosphere/test_prescribed_surface_flux_to_closure.py \
    tests/atmosphere/test_wangara_day33_forcing.py \
    tests/atmosphere/hydrostatic/unit/test_scm_forcing.py \
    tests/atmosphere/hydrostatic/test_run_scm_les_turbulence_tuning.py \
    2>&1 | tail -12
echo "  rc=$?"

echo "=== MPAS turbulence factory: main's land coupling ==="
$PY -m pytest -q -k "land_flux or shflx_land or turbulence" \
    tests/atmosphere 2>&1 | tail -12
echo "  rc=$?"

echo "=== end to end: one scored case still runs ==="
$PY scripts/run/run_scm_les_turbulence_tuning.py \
    --cases "cbl:$LES_REF_DIR/cbl" --outdir /tmp/merge_smoke_$SLURM_JOB_ID \
    --nlev 48 --skip-tuning --schemes clubb_lite,louis \
    --surface-flux-to-closure 2>&1 \
  | grep -vE "slow_operation|^E[0-9]{4}|XLA has|If you|^$" | tail -12
echo "  rc=$?"
echo "DONE $(date +%T)"
