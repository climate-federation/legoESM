#!/bin/bash
#SBATCH --job-name=jra_0p25
#SBATCH --partition=gpu
#SBATCH --account=bb1596_gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=/work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/scripts/slurm-jra_0p25-%j.out

export JAX_ENABLE_X64=1
PY=/work/bb1596/b384036/b384036/envs/legoesm_cuda/bin/python
cd /work/bb1596/b384036/diffESM/legoesm_ggn_newcopy

$PY /work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/scripts/prepare_omip_forcing.py \
    --source /work/bb1596/b384036/diffESM/data/jra55_ryf9091 \
    --years 1900 1900 --ref-year 1900 \
    --target-resolution-deg 0.25 --lon-offset-cells -0.5 \
    --cache-dir /work/bb1596/b384036/diffESM/data \
    --cache-filename jra55_ryf9091_0p25deg_noleap.zarr

# check: expect 720 x 1440, 2920 records, lon 0, 0.25, 0.5
$PY -c "
import xarray as xr
ds = xr.open_zarr('/work/bb1596/b384036/diffESM/data/jra55_ryf9091_0p25deg_noleap.zarr', decode_times=False)
print(dict(ds.sizes)); print(ds.attrs)
print('lon', ds.lon.values[:3], 'lat', ds.lat.values[:3])
"
du -sh /work/bb1596/b384036/diffESM/data/jra55_ryf9091_0p25deg_noleap.zarr
