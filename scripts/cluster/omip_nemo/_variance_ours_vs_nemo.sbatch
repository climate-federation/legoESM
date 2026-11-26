#!/bin/bash
#SBATCH --job-name=varours
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/varours_%j.out
#
# DOES OUR CLOSURE DESTROY TRACER VARIANCE AT NEMO'S RATE?
#
# The budget on NEMO's archived trends showed VERTICAL MIXING is its dominant
# tracer-variance destroyer (theta: zdf -1.61e10 vs advection -7.05e9 vs
# lateral diffusion -3.37e9). So the question that decides whether our ocean
# is faithful in its TENDENCIES is whether OUR vertical closure removes
# variance at the same rate.
#
# Mode-A evaluates our TKE closure on NEMO'S OWN RESTART STATE, so the state
# is shared by construction and the comparison carries NO state-difference
# confound. chi = 2*integral(theta' * dtheta/dt) dV is computed for our
# closure's tendency and for NEMO's ttrd_zdf on the SAME cells with the SAME
# volumes and the SAME backward-Euler solver.
#
# PRE-REGISTERED, sign and size stated before looking: the TKE thread already
# measured our diffusivity against NEMO's band by band at 0.89-1.49, so if
# the closure is faithful the variance ratio should land in roughly that
# range. A ratio far BELOW 1 means we under-mix and would leave too much
# tracer variance; far ABOVE 1 means we over-mix and would erode water
# masses faster than the oracle. A ratio near 1 CONFIRMS the dominant
# operator is faithful and points any residual bias elsewhere.
set -euo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate:$ROOT/scripts/validate/ocean_fidelity"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

TRD=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD2
OUT=$ROOT/omip_nemo/tke_premise
RST=$OUT/nemo_restart_8760.npz
IRST=$OUT/nemo_ice_restart_8760.npz
IWM=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/zdfiwm_forcing_TRA.nc
# HARDCODED: `ls` lists broken symlinks in the grid directory.
MESH=/burg-archive/glab/users/pg2328/legoESM/data/grids/eORCA1.2_mesh_mask.nc
for f in "$RST" "$IRST" "$IWM" "$MESH" "$TRD/ORCA1_1h_20010101_20010101_trd1h_T.nc"; do
  [ -r "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

$PY -u scripts/validate/ocean_fidelity/compare_tendencies_nemo.py \
  --tfile "$TRD/ORCA1_1h_20010101_20010101_trd1h_T.nc" \
  --ufile "$TRD/ORCA1_1h_20010101_20010101_trd1h_U.nc" \
  --vfile "$TRD/ORCA1_1h_20010101_20010101_trd1h_V.nc" \
  --rec 1 --restart-npz "$RST" --ice-restart-npz "$IRST" --skip-maps \
  --iwm-forcing "$IWM" --iwm-backgrounds \
  --variance-budget "$MESH" \
  --output-dir "$ROOT/omip_nemo/variance_ours" 2>&1 | grep -E "variance|Stage A2|FATAL|Error|Traceback" | tail -30
