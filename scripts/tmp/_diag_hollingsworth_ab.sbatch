#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=holl_ab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/holl_%j.out

# Decisive A/B test of the Hollingsworth-Kallberg KE-gradient fix on the
# realistic-WOA cold-start blowup (the OMIP-faithful blocker).
#
#   ARM 1 (CONTROL):  tripole WOA-init, ke_gradient_scheme=centered
#                     -> reproduces the documented day-1 super-exponential
#                        blowup (centered KE + AL81 PV = Hollingsworth-unstable).
#   ARM 2 (TREATMENT): tripole WOA-init, ke_gradient_scheme=hollingsworth
#                     -> does the consistent KE gradient stabilise it?
#   ARM 3 (TREATMENT): latlon-bathy WOA-init, ke_gradient_scheme=hollingsworth
#                     -> same fix on a regular grid (removes curvilinear confound).
#
# Identical otherwise: WOA18 IC (flood-filled S), dt=600, no nudge/drag, fine
# diagnostics. The runner self-aborts + dumps a blowup snapshot on non-finite.
set +e
cd /burg-archive/glab/users/pg2328/legoESM
export PYTHONPATH=/burg-archive/glab/users/pg2328/legoESM/src:$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== HOST ==="; hostname; date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run_arm () {
  local name="$1"; shift
  echo; echo "############################################################"
  echo "### ARM: $name"
  echo "### args: $*"
  echo "############################################################"
  JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY scripts/run/run_omip_core2.py \
    --woa-init --dt 600 --diag-every-days 0.5 "$@" 2>&1
  echo "[arm_rc:$name]=$?"
}

# ARM 1: control — expect day-1 blowup (matches prior centered-KE runs).
run_arm "tripole_centered_CONTROL" \
  --grid tripole --years 0.0833 --ke-gradient-scheme centered \
  --output results/omip_nemo/holl_ab/tripole_centered

# ARM 2: treatment — tripole hollingsworth (90 days if it survives).
run_arm "tripole_hollingsworth" \
  --grid tripole --years 0.25 --ke-gradient-scheme hollingsworth \
  --output results/omip_nemo/holl_ab/tripole_hollingsworth

# ARM 3: treatment — latlon-bathy hollingsworth.
run_arm "latlon_hollingsworth" \
  --grid latlon_bathy --years 0.25 --ke-gradient-scheme hollingsworth \
  --latlon-res 180x360 \
  --output results/omip_nemo/holl_ab/latlon_hollingsworth

echo; echo "=== ALL ARMS DONE ==="; date
