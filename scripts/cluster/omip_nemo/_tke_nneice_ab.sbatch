#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=nneice
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/_omip_merge/omip_nemo/nneice_%j.out
#
# ORCA1 selects nn_eice = 3 and we implement only nn_eice = 2.
#
# NEMO attenuates the Langmuir and surface-wave-breaking TKE sources by
# MAX(0, 1 - zice_fra) (zdftke.F90:359, :495-:509). zice_fra is fr_i for
# nn_eice=2 and MIN(4*fr_i, 1) for nn_eice=3, and ORCA1's namelist_cfg:453
# selects 3. The two agree at zero and full ice and differ most in between,
# which is the marginal ice zone and the summer Antarctic shelf -- exactly
# where our residual excess is largest (Antarctic K_M 20x vs Arctic 3.1x
# after the ice fix, job 9875780).
#
# Two arms, ONE variable, both with NEMO's own ice, wave mixing and molecular
# backgrounds.
#
# PRE-REGISTERED: the Antarctic residual should shrink substantially and the
# Arctic barely move, because January Arctic ice is near full cover where the
# two forms already agree. If the Antarctic does NOT move, nn_eice is not the
# residual and this is eliminated.
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
    --output-dir "$OUT/eice_$label" "$@" 2>&1 | tail -20
  [ ${PIPESTATUS[0]} -eq 0 ] || { echo "FATAL: arm $label failed"; RC=1; }
}

arm nneice2
arm nneice3 --cfg-override tke_nn_eice=3

echo ""
echo "DONE rc=$RC  (read eice_*/tendency_match_rec1.json — logs are tail-truncated)"
exit $RC
