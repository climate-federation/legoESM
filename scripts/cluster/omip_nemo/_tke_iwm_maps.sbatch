#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=prmap
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/_omip_merge/omip_nemo/prmap_%j.out
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
TRD=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD2
OUT=$ROOT/omip_nemo/tke_premise
IWM=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/zdfiwm_forcing_TRA.nc
$PY -u scripts/validate/ocean_fidelity/compare_tendencies_nemo.py \
  --tfile "$TRD/ORCA1_1h_20010101_20010101_trd1h_T.nc" \
  --ufile "$TRD/ORCA1_1h_20010101_20010101_trd1h_U.nc" \
  --vfile "$TRD/ORCA1_1h_20010101_20010101_trd1h_V.nc" \
  --rec 1 --restart-npz "$OUT/nemo_restart_8760.npz" \
  --iwm-forcing "$IWM" --iwm-backgrounds \
  --ice-restart-npz "$OUT/nemo_ice_restart_8760.npz" \
  --output-dir "$OUT/maps_iwmbg" 2>&1 | tail -12
echo "rc=${PIPESTATUS[0]}"
ls -la "$OUT/maps_iwmbg"/*.png 2>/dev/null | head -20
