#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=smagmask
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/smagmask_%j.out

# Does the Smagorinsky term mask the prescribed equatorial viscosity ramp?
# The A_h ramp is ACTIVE in the arm (manifest: --A-h 20000 + eddy_viscosity_3D
# vs the control's default) yet day-30 nino3 moved only 0.07 C (+3.12 vs the
# control's +3.19).  A_smag = (C_smag*Delta)^2*|D| with Delta = sqrt(area) and
# no 1/pi, so C_smag=3 on a 1-degree cell has a ~1e11 prefactor.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
export PYTHONPATH="$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
VISC=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/eddy_viscosity_3D.nc

echo "=== ARM (ramp + Smag 3.0), d30, vs the prescribed NEMO field it was given ==="
JAX_ENABLE_X64=1 $PY -u scripts/validate/ocean_fidelity/smag_vs_prescribed_viscosity.py \
  --snapshot $ROOT/results/omip_nemo/nemolev_trp_ahpsmag_d90/snapshot_day0030.npz \
  --visc-file "$VISC" --A-h-base 20000 --C-smag 3.0 2>&1 | tail -30
echo "=== arm rc=$?"

echo "=== CONTROL (uniform background + Smag 3.0), d30 ==="
JAX_ENABLE_X64=1 $PY -u scripts/validate/ocean_fidelity/smag_vs_prescribed_viscosity.py \
  --snapshot $ROOT/results/omip_nemo/nemolev_trp_signedn2_d30/snapshot_day0030.npz \
  --A-h-uniform 100000 --C-smag 3.0 2>&1 | tail -30
echo "=== ctl rc=$?"

# DEPTH CHECK (codex attack #3, and my own): the ratio above is at level 0,
# but the Equatorial Undercurrent core sits at 50-150 m. A surface-only ratio
# cannot justify a claim about the undercurrent, so re-measure at the levels
# that actually carry it (17 = 47 m, 23 = 97 m, 27 = 147 m).
for LEV in 17 23 27; do
  echo "=== ARM at level $LEV ==="
  JAX_ENABLE_X64=1 $PY -u scripts/validate/ocean_fidelity/smag_vs_prescribed_viscosity.py \
    --snapshot $ROOT/results/omip_nemo/nemolev_trp_ahpsmag_d90/snapshot_day0030.npz \
    --visc-file "$VISC" --A-h-base 20000 --C-smag 3.0 --level $LEV 2>&1 | grep -E "cap\]|band |eq \||midlat"
done
