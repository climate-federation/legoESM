#!/bin/bash
#SBATCH --job-name=lsvars
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=0:10:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/lsvars_%j.out
set -uo pipefail
cd /burg-archive/glab/users/pg2328/legoESM
/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python -u -c "
import xarray as xr
N='/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF/'
import glob,os
for f in sorted(glob.glob(N+'ORCA1_1m_*.nc')):
    ds=xr.open_dataset(f, decode_times=False)
    v3=[k for k in ds.data_vars if ds[k].ndim>=4 or ('depth' in ' '.join(ds[k].dims))]
    print(os.path.basename(f))
    print('   3-D-ish:', v3[:10])
    if not v3: print('   2-D only:', list(ds.data_vars)[:10])
"
