#!/bin/bash
#SBATCH --job-name=fesmod
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=8:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/fesmod_%j.out
#
# ============================================================================
# THE THIRD GRID, ON THE FORCING THE OTHER TWO NOW USE.
#
# The tripole and MPAS arms read the bias-corrected CORE-II cache NEMO's
# namelist names.  Every FESOM arm so far was handed an explicit path to the
# RAW cache, so a FESOM-versus-tripole number was a comparison between two
# different atmospheres -- 33 percent weaker trades on one side -- reported as
# a difference between grids.  This arm drops the explicit path so the driver
# resolves the SAME cache the other two grids do.
#
# CONFIG.  The two branches the earlier FESOM work established as necessary,
# both of them ORCA1-card parity rather than tuning:
#   --tke-surface-bc dirichlet  the NEMO en(1) surface boundary value.  Porting
#       it took FESOM's global SST error from 1.555 to 1.160 and its
#       Southern-Ocean warm bias from +1.76 to +0.45.
#   --tke-mxl0-anchor on        NEMO pairs ln_mxl0 with that boundary value;
#       the driver's own help calls dirichlet-without-it a half-port.
#   --ice-ic nemo               January start from NEMO's own ice state.  The
#       FESOM seed puts ice wherever SST is below zero, which in the Antarctic
#       SUMMER is most of the Southern Ocean, and it melted into a fresh cap
#       that shifted the Southern-Ocean answer.
#
# RESIDUAL DIFFERENCES, stated with every number: FESOM keeps its own PHC
# initial state, its own sea-surface-salinity restoring, runoff and
# chlorophyll files, and runs in single precision, which is its published
# convention.  Those are not matched to the tripole and cannot be by flags.
#
# PRE-REGISTERED, day 30 against NEMO's matched cold start (RUN_GATEWAY 5-day
# grid_T record 5 = days 26-30).  The tripole on the same forcing and the same
# reference scores 0.515 global SST error and +1.42 in the cold tongue.
#   global SST error : CONFIRMED that forcing was a real part of the
#       cross-grid gap if it lands below 0.80.  REFUTED, i.e. the gap is the
#       model rather than its atmosphere, if it stays above 1.05.
#   cold tongue : report.  If FESOM's cold-tongue bias moves the same way the
#       tripole's did, the wind correction is acting through a mechanism both
#       grids share; if it does not, one of the two ports is wrong.
#   Southern-Ocean SST and mixed layer : report.  This is the band the
#       Dirichlet branch was ported to fix and the band the user named.
# ============================================================================
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
FJX=/burg-archive/glab/users/pg2328/fesom_jax_repo
D=$REPO/data/fesom2_core2
cd $REPO || exit 1
PIN=${1:?usage: sbatch _fesom_modfrc_d30.sbatch <legoESM-sha> <fesom-sha>}
FPIN=${2:?usage: sbatch _fesom_modfrc_d30.sbatch <legoESM-sha> <fesom-sha>}
[ "$(git -C $REPO rev-parse HEAD)" = "$PIN" ] || { echo "FATAL: legoESM HEAD != $PIN"; exit 1; }
[ "$(git -C $FJX rev-parse HEAD)" = "$FPIN" ] || { echo "FATAL: fesom HEAD != $FPIN"; exit 1; }
echo "=== HEAD $(git -C $REPO log --oneline -1)"
echo "=== fesom_jax $(git -C $FJX log --oneline -1)"

MI=$D/mesh_ic/core2_mesh_ic
export FESOM_SSS_PATH="$MI/PHC2_salx.nc"
export FESOM_RUNOFF_PATH="$MI/CORE2_runoff.nc"
export FESOM_CHL_PATH="$MI/Sweeney_2005.nc"
export FESOM_PHC_PATH="$MI/phc3.0_winter.nc"
for f in "$FESOM_SSS_PATH" "$FESOM_RUNOFF_PATH" "$FESOM_CHL_PATH"; do
  [ -r "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done
ICE_INIT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/Ice_initialization.nc
[ -r "$ICE_INIT" ] || { echo "FATAL: missing $ICE_INIT"; exit 1; }

export PYTHONPATH="$FJX:$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_ENABLE_X64=0     # fesom-jax published hindcast runs f32; keep ITS convention
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo "=== gate: the driver must resolve the CORRECTED cache with no explicit path ==="
$PY -m pytest -q tests/unit/test_run_fesom_core2_cli.py 2>&1 | tail -3
rc=$?; [ $rc -eq 0 ] || { echo "FATAL: CLI gate failed rc=$rc"; exit 1; }

echo "=== gate: NYF reader ==="
( cd $FJX && JAX_PLATFORMS=cpu PYTHONPATH=$FJX $PY -m pytest -q \
    fesom_jax/tests/test_nyf_forcing.py -p no:cacheprovider 2>&1 | tail -2 )
rc=$?; [ $rc -eq 0 ] || { echo "FATAL: gate failed rc=$rc"; exit 1; }

OUT=$REPO/results/omip_nemo/fesom_modfrc_d30
# NO --nyf-zarr: the driver resolves the shared cache, and prints which one,
# so this arm cannot silently end up on a different atmosphere from the
# tripole and MPAS arms it will be compared against.
$PY -u scripts/run/run_fesom_core2.py \
  --mesh-dir "$MI/mesh_core2" \
  --ic-dir "$MI/ic_core2" \
  --days 30 --dt 1800 --year 1958 \
  --forcing core2_nyf \
  --tke-surface-bc dirichlet --tke-mxl0-anchor on \
  --ice-ic nemo --ice-init-file "$ICE_INIT" \
  --snapshot-every-days 10 \
  --output $OUT
rc=$?
echo "=== run rc=$rc ==="
[ $rc -eq 0 ] || exit $rc

echo "=== provenance: which cache did it actually read? ==="
grep -E "^\[forcing\]" $REPO/omip_nemo/fesmod_${SLURM_JOB_ID}.out | head -3 || true

export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
GW_T=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
SNAP=$OUT/snapshot_day0030.npz
[ -f "$SNAP" ] || { echo "FATAL: no day-30 snapshot -- not scoring a partial arm"; exit 1; }

echo "=== score day 30 vs NEMO's MATCHED cold start (GATEWAY rec 5) ==="
$PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
  --nemo-gridt "$GW_T" --nemo-time-idx 5 --freeze-clamp-C -1.8 --sss-faithful \
  --grid-label FESOM2_modfrc_d30 \
  --output-dir $REPO/results/omip_nemo/compare_gw_fesom_modfrc_d30 2>&1 | tail -40
echo "=== compare rc=$?"
echo DONE
