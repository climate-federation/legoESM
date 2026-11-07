#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=frztke
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/frztke_%j.out
#
# INSTANT-vs-INSTANT frozen-column TKE twin: does OUR closure equal NEMO's when
# both see NEMO's OWN instantaneous equatorial column WITH NEMO's OWN turbulent
# energy en?  We feed NEMO's day-30 restart (tn,sn,un,vn,en) into our closure's
# DIRECT diagnostic pair (compute_mixing_lengths + compute_K_from_tke -- NO en
# advance, unlike tke_vertical_mixing), and compare to NEMO's OWN closure
# diffusivities avm_k/avt_k (NEMO field_def: "from closure schemes" -- the pure
# TKE part, before the additive IWM and before EVD -> no IWM confound, and no
# 5-day-mean K(mean state) artifact).
#
# GATE-FIRST (this campaign has been burned by unvalidated instruments):
#   STEP 1  self-recovery CONTROL -- our closure, fed our OWN snapshot column and
#           our OWN stored en, reproduces our OWN stored K_M_diag/K_H_diag in the
#           65-105 m entrainment band within the tolerance.  Exit 0 required.
#   STEP 2  NON-VACUITY -- rerun the EXACT STEP-1 assertion (closure/stored band
#           ratio in [1/tol, tol]) under a DETERMINISTIC N2 x4 perturbation
#           (T,S deviation x2), and require that assertion to now FAIL (the
#           ratio falls outside the band).  The harness also MEASURES and prints
#           the actual band N2 ratio (not assumed).  Returns 0 ONLY IF the real
#           gate now rejects -- proving it can reject (codex #5).
#   STEP 3  the NEMO twin runs ONLY after 1 AND 2 pass.  Prints avm/avt and the
#           ratio to NEMO avm_k/avt_k -- NO verdict.
set -uo pipefail
PIN=${1:?usage: sbatch _frozen_column_tke_twin.sbatch <pinned-sha>}
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd "$ROOT" || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)"

export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate:$ROOT/scripts/validate/ocean_fidelity:$ROOT/scripts/run"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

HARNESS=scripts/validate/ocean_fidelity/frozen_column_tke_twin.py
# Our snapshot arm: stored diffusivities + full state (T/S/u/v/tke) + manifest.
ARM=$ROOT/results/omip_nemo/nemolev_trp_kmean_d30
SNAP=$ARM/snapshot_day0030.npz
MANI=$ARM/run_manifest.json
# NEMO day-30 INSTANTANEOUS restart (32 tiles) + the 5-day SBC (taum) + an
# ORCA1 mesh_mask tile (native gdept_1d/gdepw_1d/e3t_1d vertical ladders).
D30=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY_D30
RGLOB="$D30/ORCA1_00000720_restart_oce_*.nc"
SBC=$D30/ORCA1_5d_20000101_20000130_SBC.nc
MESH=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_SMOKE/mesh_mask_0001.nc

for f in "$SNAP" "$MANI" "$SBC" "$MESH"; do
  [ -e "$f" ] || { echo "FATAL: missing input $f"; exit 1; }
done
ls $RGLOB >/dev/null 2>&1 || { echo "FATAL: no restart tiles match $RGLOB"; exit 1; }

echo "===================== STEP 1: self-recovery CONTROL ====================="
$PY -u "$HARNESS" --mode control --manifest "$MANI" --snapshot "$SNAP" \
    --nemo-sbc "$SBC" --rec 4
rc1=$?
echo "[gate] control exit = $rc1"
[ "$rc1" -eq 0 ] || { echo "FATAL: control did NOT reproduce our own stored K -- instrument not trusted; NEMO twin NOT run."; exit 1; }

echo "===================== STEP 2: NON-VACUITY (en x0.1) ======================="
$PY -u "$HARNESS" --mode control --manifest "$MANI" --snapshot "$SNAP" \
    --nemo-sbc "$SBC" --rec 4 --perturb-en 0.1
rc2=$?
echo "[gate] non-vacuity exit = $rc2  (0 = STEP-1 assertion correctly FAILED)"
[ "$rc2" -eq 0 ] || { echo "FATAL: STEP-1 assertion STILL passed under en x0.1 -- the gate is VACUOUS; NEMO twin NOT run."; exit 1; }

echo "===================== STEP 3: NEMO instant frozen-column TWIN ==========="
$PY -u "$HARNESS" --mode nemo --manifest "$MANI" \
    --restart-glob "$RGLOB" --nemo-meshmask "$MESH" --nemo-sbc "$SBC" --rec 4
rc3=$?
echo "[done] nemo twin exit = $rc3"
exit $rc3
