#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=arctic_tune
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/lc_rebase/scripts/tmp/%x_%j.out
set -uo pipefail
WORKDIR=/burg-archive/glab/users/pg2328/lc_rebase; cd "$WORKDIR"
export PYTHONPATH=$(ls -d "$WORKDIR"/packages/*/ | sed 's:/$::' | tr '\n' ':')
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu XLA_FLAGS="--xla_force_host_platform_device_count=1"
PYJ=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
SPIN="--spinup-years 120 --verify-years 30 --dt 3600 --n-layers 10 --soil-depth 3.0"
echo "=== node $(hostname) $(date)  OFF baseline (fixed code) ==="
$PYJ scripts/validate/land_carbon_equilibrium.py $SPIN --output results/rescue_off 2>&1 | grep -E "^\[|biomass="
for NSC in 0.005 0.002 0.001; do
  echo "=== $(date)  ON nsc_ref_labile_frac=$NSC ==="
  $PYJ scripts/validate/land_carbon_equilibrium.py $SPIN \
    --nsc-gated-respiration --cold-deciduous-dormancy --nsc-ref-labile-frac $NSC \
    --output "results/rescue_on_nsc${NSC}" 2>&1 | grep -E "^\[|biomass="
done
echo "EXIT=$? done $(date)"
