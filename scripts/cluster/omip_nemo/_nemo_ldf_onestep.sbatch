#!/bin/bash
#SBATCH --job-name=ldf_onestep
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/ldf_onestep_%j.out
# One-step certification of the tripole NEMO lateral-viscosity port against
# NEMO's per-step utrd_ldf/vtrd_ldf (RUN_LDF1TS, first step after restart 288).
# Usage: sbatch --export=ALL,ROOT=<worktree> _nemo_ldf_onestep.sbatch
set -euo pipefail
ROOT=${ROOT:-/burg-archive/glab/users/pg2328/_omip_merge}; cd $ROOT
echo "HEAD: $(git log --oneline -1)  dirty: $(git status --porcelain -uno | wc -l)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
E=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1
I=$E/INPUTS/orca1_inputs/data_repository/input_fields
R=$E/EXP00/RUN_LDF1TS
TU=$(ls $R/ORCA1_*ldf1ts_U.nc); TV=$(ls $R/ORCA1_*ldf1ts_V.nc)
/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python -u scripts/validate/ocean_fidelity/nemo_ldf_onestep_probe.py \
  --mesh /burg-archive/glab/users/pg2328/legoESM/data/grids/eORCA1.2_mesh_mask.nc \
  --domcfg $I/domain_cfg.nc --ldf $I/eddy_viscosity_3D.nc \
  --restart-npz $E/EXP00/RUN_DT150_TRD/restart_288_full.npz \
  --trd-u "$TU" --trd-v "$TV" --rec 0 --rn-shlat 2 ${EXTRA:-}
