#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=thincell
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:15:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/thincell_%j.out
# Is the single failing SPMD-equivalence cell (u[11,74,9]) a razor-thin
# partial bottom cell (implicit-solve sensitivity) or a band-cut wiring bug?
# Reads nothing but the test's own synthetic geometry.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
$PY - <<'PY'
import numpy as np
import jax.numpy as jnp
from legoesm.ocean.vertical import create_ocean_z_star, create_partial_cell_coordinate
n_lat, n_lon, nlev = 48, 96, 10
lat_idx = np.arange(n_lat)[:, None]; lon_idx = np.arange(n_lon)[None, :]
H = (4000.0 - 1500.0*np.exp(-((lat_idx-n_lat/2)/8.0)**2)
     - 400.0*np.cos(2*np.pi*lon_idx/n_lon))
zc = create_partial_cell_coordinate(create_ocean_z_star(n_levels=nlev, H_max=4000.0),
                                    jnp.asarray(H))
hp = np.asarray(zc.h_partial); dz = np.asarray(zc.dz_ref); bl = np.asarray(zc.bottom_level)
j, i, k = 11, 74, 9
# the failing u-face sits between cells (11,73) and (11,74)
for col in ((j, 73), (j, 74)):
    jj, ii = col
    print(f"col {col}: H={H[jj,ii]:.1f} bottom_level={bl[jj,ii]} "
          f"h_partial[k=9]={hp[jj,ii,9]:.3f} m  (dz_ref[9]={dz[9]:.1f} m, "
          f"fill={hp[jj,ii,9]/dz[9]:.4f})")
print(f"band geometry: 48 rows / 4 bands = 12 -> row 11 IS band 0's last row")
# distribution: how thin is the thinnest active bottom cell anywhere?
act = np.asarray(zc.is_active)
bot = hp[np.arange(n_lat)[:,None], np.arange(n_lon)[None,:], np.maximum(bl,0)]
wet = bl >= 0
print(f"thinnest active bottom cell: {bot[wet].min():.4f} m; "
      f"cells with fill<0.05: {(bot[wet]/dz[np.maximum(bl,0)][wet] < 0.05).sum()}")
PY
echo DONE
