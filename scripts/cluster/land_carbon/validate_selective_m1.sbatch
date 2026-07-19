#!/bin/bash
# Follow-up validation: SELECTIVE M1 (reserve-days gate) + WHOLE-PLANT cold-
# deciduous dormancy + leaf bootstrap + M2 threshold 265 K.  Tests whether the
# reserve-days reformulation removes the temperate/tropical regression (a healthy
# tree's inert wood no longer triggers the gate) and whether whole-plant dormancy
# finally revives the woody-deciduous larch.  OFF baseline + ON sweep over
# nsc_reserve_days; identical spin config (controlled).
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=selective_val
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
echo "=== node $(hostname)  $(date)  OFF baseline ==="
$PYJ scripts/validate/land_carbon_equilibrium.py $SPIN --output results/sel_off 2>&1 | grep -E "^\[|biomass="
for DAYS in 5 10 20; do
  echo "=== $(date)  ON nsc_reserve_days=$DAYS  M2=265  +whole-plant-dormancy +bootstrap ==="
  $PYJ scripts/validate/land_carbon_equilibrium.py $SPIN \
    --nsc-gated-respiration --cold-deciduous-dormancy \
    --freeze-dormancy-threshold-k 265.0 --nsc-reserve-days $DAYS \
    --output "results/sel_on_days${DAYS}" 2>&1 | grep -E "^\[|biomass="
done
echo "EXIT=$?  done $(date)"
