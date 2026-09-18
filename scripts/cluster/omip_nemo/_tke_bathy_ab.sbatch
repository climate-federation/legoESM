#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=bathy
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/_omip_merge/omip_nemo/bathy_%j.out
#
# DOES THE CLOSURE KNOW WHERE THE SEAFLOOR IS?
#
# tke_vertical_mixing takes bottom_level and w_active and the probe passed
# NEITHER, so every column was stepped as if it were full-depth ocean. With
# bottom_level=None the bottom TKE Dirichlet condition is pinned at the LAST
# row of the array unconditionally -- the docstring calls that exact only on a
# flat-bottom column -- and with w_active=None there is no wet-interface mask
# to stop the sweeps at the seafloor either.
#
# ORCA1's Antarctic shelf is a few hundred metres deep on a 75-level ladder
# reaching about 5000 m, so this is wrong by kilometres exactly where the
# residual lives, and nowhere else: the open ocean columns ARE full depth.
#
# Residual after the ice fix and nn_eice: antarctic K_M/avm 19.95, K_H/avt
# 4.65; arctic 3.07 / 0.96; mid-latitudes and tropics within 1.2.
#
# Two arms, ONE variable, both with NEMO's own ice, wave mixing and molecular
# backgrounds. bottom_level and w_active come from the state's OWN land mask
# (NEMO writes NaN on land), so no separate bathymetry file can disagree with
# the state being stepped.
#
# PRE-REGISTERED: the Antarctic and Arctic shelf excesses fall sharply while
# the tropics and sh_mid stay put, because those columns are already
# full-depth and the flag cannot move them. If the polar numbers do not move,
# the bottom boundary is eliminated and the excess is in the closure itself.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo "=== unit gate: the new selector ==="
JAX_PLATFORMS=cpu $PY -m pytest -q tests/ocean/unit/test_nemo_ice_tke_attenuation.py 2>&1 | tail -5
[ ${PIPESTATUS[0]} -eq 0 ] || { echo "FATAL: unit gate failed"; exit 1; }

TRD=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD2
OUT=$ROOT/omip_nemo/tke_premise
RST=$OUT/nemo_restart_8760.npz
IRST=$OUT/nemo_ice_restart_8760.npz
IWM=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/zdfiwm_forcing_TRA.nc
for f in "$RST" "$IRST" "$IWM"; do
  [ -e "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

RC=0
arm() {   # arm <label> <extra flags...>
  local label="$1"; shift
  echo ""
  echo "=== ARM $label : $* ==="
  $PY -u scripts/validate/ocean_fidelity/compare_tendencies_nemo.py \
    --tfile "$TRD/ORCA1_1h_20010101_20010101_trd1h_T.nc" \
    --ufile "$TRD/ORCA1_1h_20010101_20010101_trd1h_U.nc" \
    --vfile "$TRD/ORCA1_1h_20010101_20010101_trd1h_V.nc" \
    --rec 1 --restart-npz "$RST" --ice-restart-npz "$IRST" --skip-maps \
    --iwm-forcing "$IWM" --iwm-backgrounds \
    --output-dir "$OUT/bathy_$label" "$@" 2>&1 | tail -45
  [ ${PIPESTATUS[0]} -eq 0 ] || { echo "FATAL: arm $label failed"; RC=1; }
}

arm nobathy
arm bathy   --use-bathy

echo ""
echo "DONE rc=$RC  (read bathy_*/tendency_match_rec1.json — logs are tail-truncated)"
exit $RC
