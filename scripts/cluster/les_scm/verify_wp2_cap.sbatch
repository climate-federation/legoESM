#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --job-name=wp2cap
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/wp2cap_%j.out
set +e
# Does capping wp2 on the prognostic path fix Wangara?
#
# PRE-REGISTERED, before the job ran. Baseline, all at default parameters:
#   wangara  objective 1.0026 (must be exactly 1)  |grad|max 1.33e16  score 15.21
#   bomex    objective 1.0                         |grad|max 2.8e-1   score 0.0753
#   cbl      objective 1.0                         |grad|max 5.3e-1   score 0.9455
#
#   CONFIRMS: wangara's |grad|max falls by many orders (to O(1e3) or below),
#     its objective returns to EXACTLY 1.0, and its score drops well under
#     15.21 -- ideally toward the 0.5-1.4 band the other cases occupy.
#   REFUTES: |grad|max stays ~1e16. The cap is then not the binding constraint
#     and the wp3/skewness branch has a second unbounded quantity.
#   PARTIAL (report as such): gradient tamed but the score stays high, i.e. the
#     runaway is bounded but the solution is still wrong.
#
#   CONTROL, and it matters as much: bomex and cbl must be UNCHANGED to the
#   digits above. wp2 never approaches 1000 m^2/s^2 there, so a cap that moves
#   them is capping something it should not, and the "fix" would be a silent
#   physics change on seven healthy cases.
: "${E_WT:?set E_WT}"
cd "$E_WT" || exit 1
echo "PINNED sha: $(git rev-parse HEAD)  host: $(hostname)"
export PYTHONPATH=$E_WT/src:$(ls -d "$E_WT"/packages/*/ | tr '\n' ':')
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
export LES_REF_DIR=/burg-archive/glab/users/pg2328/legoESM/results/les_ref
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=/burg-archive/glab/users/pg2328/legoESM/results

echo "########## unit: the cap and its call site ##########"
$PY -m pytest -q tests/atmosphere/test_prescribed_surface_flux_to_closure.py \
    2>&1 | tail -6

echo "########## trajectory: is it a forward runaway? ##########"
$PY scripts/validate/clubb_moment_health.py --case wangara --every 240 2>&1 \
  | grep -vE "slow_operation|^E[0-9]{4}|XLA has|If you" | tail -25

for C in wangara bomex cbl; do
  echo "########## $C: score + gradient at defaults ##########"
  $PY scripts/run/run_scm_les_turbulence_tuning.py \
      --cases "$C:$R/les_ref/$C" --outdir "$R/scm_les_turbulence/_wp2cap_$C" \
      --nlev 48 --steps 1 --schemes clubb \
      --joint-aggregation default_relative --surface-flux-to-closure 2>&1 \
    | grep -vE "slow_operation|^E[0-9]{4}|XLA has|If you|^$"
done
echo "DONE $(date +%T)"
