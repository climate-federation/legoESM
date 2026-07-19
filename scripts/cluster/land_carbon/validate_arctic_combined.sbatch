#!/bin/bash
# Combined arctic-productivity validation: NSC gate (M1) + cold-deciduous
# dormancy with a LOWERED threshold (265 K = -8 degC, so cold plants
# photosynthesise in their cool growing season) + the LEAF BOOTSTRAP (regrow a
# minimum leaf area from labile so larch/tundra escape the 0-leaf GPP lock).
# OFF baseline then ON at two nsc_ref values -- identical spin config, controlled.
# Criteria: temperate/tropical ~unchanged; boreal_forest / shrubland / arctic_tundra
# / larch reach NONZERO biomass (they die with OFF).
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=arctic_combined
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
THR=265.0    # cold-deciduous dormancy onset [K] (-8 degC, deep-cold only)
echo "=== node $(hostname)  $(date)  OFF baseline ==="
$PYJ scripts/validate/land_carbon_equilibrium.py $SPIN --output results/combined_off 2>&1 | grep -E "^\[|biomass="
for NSC in 0.01 0.02; do
  echo "=== $(date)  ON nsc_ref=$NSC  M2_threshold=$THR  +bootstrap ==="
  $PYJ scripts/validate/land_carbon_equilibrium.py $SPIN \
    --nsc-gated-respiration --cold-deciduous-dormancy \
    --freeze-dormancy-threshold-k $THR --nsc-ref-labile-frac $NSC \
    --output "results/combined_on_nsc${NSC}" 2>&1 | grep -E "^\[|biomass="
done
echo "EXIT=$?  done $(date)"
