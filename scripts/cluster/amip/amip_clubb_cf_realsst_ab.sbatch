#!/bin/bash
# REPOINTED 2026-09-23: config/amip/amip_production.yaml became the CAM6
# suite on that date.  This script exists to reproduce a SPECIFIC recorded
# analysis, so it now names config/amip/amip_sundqvist_l36.yaml -- the same
# deck under its new name, byte-unchanged -- rather than silently running a
# different model.
# REAL-SST realism A/B for the CLUBB cloud-fraction -> radiation lever.
# Calibrates the albedo/rsut effect under REAL AMIP forcing (observed SST/SIC +
# CMIP6 solar/ozone/ghg/aerosol + ERA5 IC + ETOPO topography), so the marine-Sc
# decks are the real ones — unlike the analytical mechanism A/B
# (amip_clubb_cf_ab.sbatch), whose over-bright analytical baseline only gives a
# DIRECTIONAL magnitude.  Validated by the 2880-step/2-day real-SST smoke (9074855
# COMPLETED).
#
# CONTROLLED COMPARISON: both arms share the byte-identical production config +
# real forcing + latlon-hydrostatic + bechtold + diagnostic CLUBB + slab land
# tile + --per-step-rollout; the ONLY varied knob is --use-clubb-cloud-fraction.
#   Arm A: --no-use-clubb-cloud-fraction -> RH (sundqvist) grid cloud fraction
#   Arm B: --use-clubb-cloud-fraction     -> diagnostic CLUBB PDF cloud fraction
# Compare timeseries.npz['sw_up_toa'] (=rsut) on the SAME grid/metric/sampling;
# vs observed marine-Sc rsut ~ 100-120 W/m2.
#
# NOTE ON GRID: the C48 cd-grid production baseline is NONHYDROSTATIC (feature not
# wired there); this runs latlon C-grid hydrostatic (the wired path) with the
# SAME real forcing + tuned physics.  Land = slab tile (CLM surfdata not staged;
# the marine-Sc target is over ocean).
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=clubbcf_rab
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM_convcmp/results/clubb_cf_realsst_ab/slurm_%j.out

set -u
WT=/burg-archive/glab/users/pg2328/legoESM_convcmp
export PYTHONPATH="$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT:$WT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
cd "$WT"

DAYS="${DAYS:-20}"
OUT=$WT/results/clubb_cf_realsst_ab
mkdir -p "$OUT"

DATA=/burg-archive/glab/users/ac5006/legoESM/data/amip
FDIR=$DATA/forcing_amip
PATHS=(
  --ic era5 --ic-path "$DATA/era5_ic_1979-01-01_gproxy.zarr"
  --topography "$DATA/etopo_0p25deg.nc"
  --forcing-path "$DATA/tosbcs_input4MIPs_PCMDI-AMIP-1-1-9_gn_187001-202212.nc"
  --sst-var tosbcs --sst-offset 273.15
  --sic-path "$DATA/siconcbcs_PCMDI-AMIP-1-1-9_gn_187001-202212_clip.nc" --sic-var siconcbcs
  --solar-file "$FDIR/solar_amip_1979-2014.nc"
  --ozone-file "$FDIR/ozone_amip_clim.nc"
  --ghg-file "$FDIR/ghg_amip_1979-2014.nc"
  --aerosol-file "$FDIR/aerosol_amip_clim.nc"
)
COMMON=(
  --config config/amip/amip_sundqvist_l36.yaml
  "${PATHS[@]}"
  --grid-type latlon --discretization latlon_cgrid --resolution 24 --nlev 40 --dt 60
  --turbulence clubb
  --no-use-multilayer-land --slab-land-active
  --per-step-rollout
  --days "$DAYS"
)

echo "=== $($PY --version); DAYS=$DAYS; $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null|head -1) ==="

echo "############ Arm A (baseline: RH sundqvist cloud fraction) ############"
$PY -u scripts/run/run_amip.py "${COMMON[@]}" \
    --no-use-clubb-cloud-fraction \
    --output "$OUT/armA_sundqvist_cf"
rcA=$?; echo "### Arm A exit=$rcA"

echo "############ Arm B (treatment: diagnostic CLUBB PDF cloud fraction) ############"
$PY -u scripts/run/run_amip.py "${COMMON[@]}" \
    --use-clubb-cloud-fraction \
    --output "$OUT/armB_clubb_cf"
rcB=$?; echo "### Arm B exit=$rcB"

echo "=== REAL-SST A/B done: armA=$rcA armB=$rcB (0=clean).  Compare"
echo "=== timeseries.npz['sw_up_toa'] between $OUT/armA_sundqvist_cf and $OUT/armB_clubb_cf ==="
[ $rcA -ne 0 ] || [ $rcB -ne 0 ] && exit 1 || exit 0
