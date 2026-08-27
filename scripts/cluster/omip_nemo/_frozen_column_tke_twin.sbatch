#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=frztke
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/frztke_%j.out
#
# THE DECISIVE FROZEN-COLUMN TWIN: does OUR TKE closure equal NEMO's when both
# see NEMO's OWN equatorial state?  We spin our prognostic TKE energy `en` to
# equilibrium on a FROZEN background (T,S,u,v,wind stress) and read avm/avt.
#
# GATE-FIRST (this campaign has been burned by unvalidated instruments):
#   STEP 1  self-recovery CONTROL -- our closure must reproduce our OWN stored
#           K_M_diag/K_H_diag from our OWN snapshot column, in the 65-105 m
#           entrainment band, within +-25%.  Exit 0 required.
#   STEP 2  NON-VACUITY -- rerun the control with u,v scaled x2; the harness
#           returns 0 ONLY IF the reproduction BREAKS.  Proves the gate can
#           fail (a control that always passes proves nothing).
#   STEP 3  the NEMO twin runs ONLY after 1 AND 2 pass.  It prints avm/avt and
#           the ratio to NEMO grid_W -- NO verdict, interpretation is the
#           reader's.
#
# The bare-column closure omits the additive internal-wave-mixing K that the
# stored/published K includes; STEP 1 MEASURES that residual and gates on it,
# so a large IWM contribution ABORTS here rather than contaminating the twin.
set -uo pipefail
PIN=${1:?usage: sbatch _frozen_column_tke_twin.sbatch <pinned-sha>}
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd "$ROOT" || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)"

export PYTHONPATH="$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate:$ROOT/scripts/validate/ocean_fidelity:$ROOT/scripts/run"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

HARNESS=scripts/validate/ocean_fidelity/frozen_column_tke_twin.py
# Snapshot arm that carries BOTH the stored diffusivities AND T/S/u/v/tke, plus
# its run_manifest.json (the TKE-config source -- no hidden config choice).
ARM=$ROOT/results/omip_nemo/nemolev_trp_kmean_d30
SNAP=$ARM/snapshot_day0030.npz
MANI=$ARM/run_manifest.json
# NEMO 5-day gateway output, record 5 = days 26-30 (all four grids same record).
GW_DIR=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY
STEM=ORCA1_5d_20000101_20000331
GT=$GW_DIR/${STEM}_grid_T.nc
GU=$GW_DIR/${STEM}_grid_U.nc
GV=$GW_DIR/${STEM}_grid_V.nc
GWv=$GW_DIR/${STEM}_grid_W.nc
REC=5

for f in "$SNAP" "$MANI" "$GT" "$GU" "$GV" "$GWv"; do
  [ -e "$f" ] || { echo "FATAL: missing input $f"; exit 1; }
done

echo "===================== STEP 1: self-recovery CONTROL ====================="
$PY -u "$HARNESS" --mode control --manifest "$MANI" --snapshot "$SNAP" \
    --nemo-gridu "$GU" --nemo-gridv "$GV" --rec "$REC" --dt 150
rc1=$?
echo "[gate] control exit = $rc1"
[ "$rc1" -eq 0 ] || { echo "FATAL: control did NOT reproduce our own stored K -- instrument is not trusted; NEMO twin NOT run."; exit 1; }

echo "===================== STEP 2: NON-VACUITY (shear x2) ===================="
$PY -u "$HARNESS" --mode control --manifest "$MANI" --snapshot "$SNAP" \
    --nemo-gridu "$GU" --nemo-gridv "$GV" --rec "$REC" --dt 150 \
    --perturb-shear 2.0
rc2=$?
echo "[gate] non-vacuity exit = $rc2  (0 = reproduction correctly BROKE)"
[ "$rc2" -eq 0 ] || { echo "FATAL: control STILL reproduced stored K under a 2x shear perturbation -- the gate is VACUOUS; NEMO twin NOT run."; exit 1; }

echo "===================== STEP 3: NEMO frozen-column TWIN ===================="
$PY -u "$HARNESS" --mode nemo --manifest "$MANI" \
    --nemo-gridt "$GT" --nemo-gridu "$GU" --nemo-gridv "$GV" --nemo-gridw "$GWv" \
    --rec "$REC" --dt 150
rc3=$?
echo "[done] nemo twin exit = $rc3"
exit $rc3
