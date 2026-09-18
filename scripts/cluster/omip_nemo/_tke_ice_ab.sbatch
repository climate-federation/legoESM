#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=tkeice
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/_omip_merge/omip_nemo/tkeice_%j.out
#
# DOES THE PROBE'S MISSING SEA ICE EXPLAIN THE POLAR DIFFUSIVITY EXCESS?
#
# Measured (job 9875099): on the Antarctic shelf above 100 m our momentum
# diffusivity is 283x NEMO's and our tracer diffusivity 44x; Arctic 42x and
# 6.5x. Mid-latitudes agree to within a factor 1.8. NEMO sits essentially at
# its molecular background in those polar cells and we do not.
#
# run_stage_a2_mode_a passed ice_frac=None, so the closure's (1 - ice
# fraction) damping of the surface TKE flux (tke.py:2090, 2171) was
# identically off. The state is 1 January: the Arctic is fully covered and the
# Antarctic coast partly so. Full wind-driven TKE was being injected into
# water the oracle has under ice, in exactly the two failing bands.
#
# Step 1 stitches NEMO's OWN ice restart at step 8760 -- the SAME instant as
# the ocean restart the probe is already seeded from, so this is the oracle's
# ice, not a climatology or a guess.
#
# Two arms, ONE variable:
#   noice   as before, ice_frac=None                    (the current number)
#   ice     + NEMO's own a_i concentration
# Both carry the wave mixing and molecular backgrounds already established as
# necessary, so ice is the only thing that changes.
#
# PRE-REGISTERED: if the missing ice is the cause, the polar K_M/avm and
# K_H/avt excesses collapse toward the mid-latitude values (0.8-1.8x) while
# the ten already-matching bands stay put. If the polar excess survives, the
# probe is exonerated and the excess is the closure's own.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

TRD=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD2
OUT=$ROOT/omip_nemo/tke_premise
RST=$OUT/nemo_restart_8760.npz
IRST=$OUT/nemo_ice_restart_8760.npz
IWM=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/zdfiwm_forcing_TRA.nc

for f in "$RST" "$IWM"; do
  [ -e "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

echo "=== 1. stitch NEMO's ice restart (a_i = concentration, step 8760) ==="
if [ -f "$IRST" ]; then
  echo "[skip] $IRST already present"
else
  $PY -u scripts/validate/ocean_fidelity/rebuild_nemo_restart.py \
    --pattern "$TRD/ORCA1_00008760_restart_ice_*.nc" \
    --fields a_i \
    --out "$IRST" 2>&1 | tail -15
  [ ${PIPESTATUS[0]} -eq 0 ] || { echo "FATAL: ice stitch failed"; exit 1; }
fi
[ -f "$IRST" ] || { echo "FATAL: no $IRST"; exit 1; }

RC=0
arm() {   # arm <label> <extra flags...>
  local label="$1"; shift
  echo ""
  echo "=== ARM $label : $* ==="
  $PY -u scripts/validate/ocean_fidelity/compare_tendencies_nemo.py \
    --tfile "$TRD/ORCA1_1h_20010101_20010101_trd1h_T.nc" \
    --ufile "$TRD/ORCA1_1h_20010101_20010101_trd1h_U.nc" \
    --vfile "$TRD/ORCA1_1h_20010101_20010101_trd1h_V.nc" \
    --rec 1 --restart-npz "$RST" --skip-maps \
    --iwm-forcing "$IWM" --iwm-backgrounds \
    --output-dir "$OUT/ice_$label" "$@" 2>&1 | tail -20
  [ ${PIPESTATUS[0]} -eq 0 ] || { echo "FATAL: arm $label failed"; RC=1; }
}

arm noice
arm ice    --ice-restart-npz "$IRST"

echo ""
echo "DONE rc=$RC  (read ice_*/tendency_match_rec1.json — logs are tail-truncated)"
exit $RC
