#!/bin/bash
#SBATCH --job-name=bn2val
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=0:45:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/bn2val_%j.out
#
# Does the TEOS-10 rab actually get us closer to NEMO's OWN rn2?
# Recovered from the restart as rn2 = 2*dissl^2 (zdftke.F90:651,680,717), at
# exactly the instant NEMO's tn/sn were written -- no time alignment.
# PRE-REGISTERED: TEOS-10 wins only if its log10 rms is LOWER than S-EOS in
# EVERY band. A closer mean ratio with a worse log rms is cancellation.
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/legoESM; cd $WT || exit 1
export PYTHONPATH="$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
T=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD2
echo "HEAD: $(git -C $WT log --oneline -1)"
/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python -u \
  scripts/validate/ocean_fidelity/nemo_bn2_validation.py \
  --restart-npz $T/restart_8760_full.npz \
  --mesh-mask $WT/data/grids/eORCA1.2_mesh_mask.nc \
  --out-json $WT/results/omip_nemo/bn2_validation/step8760.json 2>&1 \
  | grep -avE "^Jax plugin|^Traceback|xla_bridge|cuda_versions|RuntimeError: jaxlib|_check_cuda|plugin_module|for d in range|  File \"|    "
echo "=== rc=${PIPESTATUS[0]} ==="
