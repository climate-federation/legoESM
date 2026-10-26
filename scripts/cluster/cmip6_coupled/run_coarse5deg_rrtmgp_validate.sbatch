#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=cmip6_val
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/results/cmip6_coupled/_probes/cmip6_val_%j.out
# Validation of the water-budget fix (2026-06-15): coarse-5deg fully-coupled
# rrtmgp at L20 (production vertical resolution), unfused radiation for a fast
# compile, WITH microphysics=kessler so the water budget is closed (precip
# flows, q_c bounded, cloud-radiation optics sane).  Checks pr>0, rsdt~340,
# rlut physical.
set +e
cd /burg-archive/glab/users/pg2328/legoESM
export PYTHONPATH=$(ls -d packages/*/ | tr '\n' ':')$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
# real input4MIPs ozone + solar, the same set every AMIP grid runs (2026-10-06)
source config/amip/amip_production.ginsburg.sh
echo "=== HOST ==="; hostname; nvidia-smi -L 2>/dev/null; date
OUT=results/cmip6_coupled/coarse5deg_rrtmgp_fixed
JAX_ENABLE_X64=1 $PY -u scripts/run/run_coupled.py \
  --preset slab_simple --resolution 18 --nlev 20 --dt 450 \
  --days 30 --diag-days 30 --radiation rrtmgp --rad-update-steps 4 \
  --unfused-radiation \
  --convection sbm --turbulence holtslag_boville \
  --gravity-wave-drag hines --clouds sundqvist --microphysics kessler \
  --experiment historical --start-year 1850 \
  "${CMIP6_FORCING_FLAGS[@]}" \
  --cmip-output --cmip-resolution-deg 5.0 --output "$OUT"
echo "run_rc=$?"; date
$PY - <<'EOF'
import numpy as np, xarray as xr, glob
for v in ["rsdt","rsut","rlut","pr","tas","tos"]:
    fs=glob.glob(f"results/cmip6_coupled/coarse5deg_rrtmgp_fixed/cmor/*/{v}_*.nc")
    if not fs: print(f"{v}: NO FILE"); continue
    da=xr.open_dataset(fs[0])[v]; s=86400 if v=="pr" else 1
    w=np.cos(np.deg2rad(da.lat.values))
    m=float((da.isel(time=0).weighted(xr.DataArray(w,dims='lat',coords={'lat':da.lat})).mean(('lat','lon')))*s)
    u='mm/day' if v=='pr' else ('K' if v in('tas','tos') else 'W/m2')
    print(f"{v}: area-mean={m:.2f} {u} raw[min,max]=[{float(da.min())*s:.3g},{float(da.max())*s:.3g}]")
EOF
echo "=== DONE ==="; date
