#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=icevars
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/burg-archive/glab/users/pg2328/_omip_merge/omip_nemo/icevars_%j.out
# List the variables in NEMO's per-rank ice restart at step 8760 -- the SAME
# instant as the ocean restart the Mode-A probe is seeded from. The probe
# currently passes ice_frac=None, i.e. it runs the polar ocean with no sea ice
# at all, in January. Needed to wire the oracle's own ice concentration.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
F=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD2/ORCA1_00008760_restart_ice_0000.nc
$PY -c "
import netCDF4 as nc
ds = nc.Dataset('$F')
for k, v in ds.variables.items():
    if v.ndim >= 2:
        print(f'{k:20s} {str(v.dimensions):55s} {v.shape}')
"
