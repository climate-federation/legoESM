#!/bin/bash
#SBATCH --job-name=mpas_spread2
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/mpasspread2_%j.out
#
# THE HARMONIZED MPAS CARD, EXTENDED TO 180 DAYS (matched pair with the
# tripole trp_visc180 run).  The 30-day A/B (jobs 9561558/9561560) confirmed
# the full harmonized card (mle + monthly-IC + dm2dc + sw-rgb-chl) is
# regression-safe: SST rmse 0.491, bias +0.005, Arctic SSS bias +0.26 (was
# +0.48 pre-harmonization).  The residual nino3 +1.00 at day 30 is the same
# shared warm-tongue transient the tripole shows; the tripole's plateaus
# +1.42 -> +0.70 by day 180, so the real cross-grid + vs-NEMO comparison
# needs matched 180-day trajectories.
#
# PRE-REGISTERED: (a) day-30/60/90 scores vs GATEWAY recs 5/11/17 -- CONFIRM
# harmonization if MPAS tracks the tripole within ~0.2 C in every SST band
# and the nino3 bias shrinks like the tripole's; REFUTE (grid-or-config gap
# remains) if any band diverges >0.5 C from the tripole at matched days.
# (b) our own nino3 trajectory beyond day 90 (plateau test, no oracle).
set -uo pipefail
PIN=${1:?usage: sbatch _mpas_spread2_d30.sbatch <pinned-sha>}
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
OUT=$ROOT/results/omip_nemo/mpas_spread2_d30

echo "=== gate: corrected forcing is the default ==="
JAX_PLATFORMS=cpu $PY -m pytest -q tests/ocean/unit/test_core2_corrected_cache_default.py 2>&1 | tail -2
rc=$?; [ $rc -eq 0 ] || { echo "FATAL: gate rc=$rc"; exit 1; }

JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py --grid mpas \
  --mpas-level 7 --mpas-lloyd 20 --mpas-vmix tke \
  --tke-surface-bc nemo_dirichlet --tke-mxl-choice 3 --tke-prognostic \
  --tke-kappa-convention veros_sqrte \
  --dt 150 --nemo-vertical --partial-cell \
  --mle --mle-ce 0.06 \
  --dm2dc --sw-rgb-chl \
  --geothermal --runoff --sss-restore --sss-restore-tau-days 45.5 \
  --sss-restore-bound-mmday 4 --sss-ice-gate-nemo --prognostic-sea-ice \
  --prognostic-ice-dynamics free_drift --runoff-spread-passes 2 --years 0.0822 --snapshot-every-days 30 \
  --diag-every-days 5 --tracer-advection superbee --runoff-depth-nemo-ini \
  --river-mouth-restoring-gate --bottom-drag-scheme nemo_quadratic \
  --ice-ocean-heat-coeff 70 \
  --woa-init \
  --nemo-monthly-init $NEMOIN/orca1_inputs/data_repository/initial_conditions/woce_temp_monthly_init_4p2.nc \
    $NEMOIN/orca1_inputs/data_repository/initial_conditions/woce_salt_monthly_init_4p2.nc \
  --woa-t /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_t_woa57_annual.nc \
  --woa-s /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_s_woa57_annual.nc \
  --sss-restore-file $NEMOIN/orca1_inputs/data_repository/input_fields/sss_climatology_for_restoring.nc \
  --ice-init $NEMOIN/Ice_initialization.nc \
  --output $OUT 2>&1 | tail -25
run_rc=$?
echo "=== run rc=$run_rc"
[ $run_rc -eq 0 ] || { echo "FATAL: run failed rc=$run_rc"; exit 1; }

export JAX_PLATFORMS=cpu
export JAX_PLATFORMS=cpu
GW_T=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
SNAP=$OUT/snapshot_day0030.npz; [ -f "$SNAP" ] || SNAP=$OUT/snapshot_final.npz
$PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
  --nemo-gridt "$GW_T" --nemo-time-idx 5 --freeze-clamp-C -1.8 --sss-faithful \
  --grid-label mpas_spread2_d30 --output-dir $ROOT/results/omip_nemo/compare_mpas_spread2_d30 2>&1 | tail -38
echo "=== PRE-REGISTERED river-mouth boxes (native cells) vs NEMO ==="
$PY - "$SNAP" <<'PY'
import sys, numpy as np, netCDF4 as nc
GW=("/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/"
    "RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc")
BOX={"Ob/Yenisei":(66,74,66,84),"Kara/Laptev":(70,78,60,140),"Amazon":(-2,5,-53,-45)}
d=nc.Dataset(GW); nlat=np.asarray(d["nav_lat"][:],float)
nlon=np.asarray(d["nav_lon"][:],float); nlon=np.where(nlon>180,nlon-360,nlon)
nS=np.ma.filled(d["sos"][5].astype(float),np.nan); nwet=np.isfinite(nS)&(nS>0)
z=np.load(sys.argv[1]); lat=np.asarray(z["lat_T"],float); lon=np.asarray(z["lon_T"],float)
if np.abs(lat).max()<7: lat,lon=np.degrees(lat),np.degrees(lon)
lon=np.where(lon>180,lon-360,lon); S=np.asarray(z["S"])[...,0]; wet=z["land_mask"]>0.5
print(f"{'box':14s}{'NEMO mean':>11s}{'ours mean':>11s}{'bias':>8s}{'NEMO p05':>10s}{'ours p05':>10s}")
for k,(a,b,c,e) in BOX.items():
    m=(nlat>=a)&(nlat<=b)&(nlon>=c)&(nlon<=e)&nwet
    mm=(lat>=a)&(lat<=b)&(lon>=c)&(lon<=e)&wet
    print(f"{k:14s}{np.nanmean(nS[m]):11.2f}{np.nanmean(S[mm]):11.2f}"
          f"{np.nanmean(S[mm])-np.nanmean(nS[m]):+8.2f}"
          f"{np.nanpercentile(nS[m],5):10.2f}{np.nanpercentile(S[mm],5):10.2f}")
print("\nBASELINE (8 passes, mpas_fullharm d30): Ob/Yenisei mean 30.46 (NEMO 26.68, bias +3.78);")
print("Kara 27.87 (+1.54); Amazon 31.65 (NEMO 33.69, bias -2.04).")
print("PRE-REGISTERED: spreading OWNS the Arctic river bias if Ob/Yenisei bias shrinks by >=1.0 psu;")
print("REFUTED if it moves <0.3. Watch the Amazon for the opposite sign (already too fresh).")
PY
echo DONE
