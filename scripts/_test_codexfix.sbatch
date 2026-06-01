#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cxfix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/scripts/_test_codexfix_%j.out
set +e; cd /burg-archive/glab/users/pg2328/legoESM
export PYTHONPATH=/burg-archive/glab/users/pg2328/legoESM/src:$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== HOST ==="; hostname; date
echo "=== [1] #354 scan tests (refactored block_fn) ==="
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $PY -m pytest -q tests/ocean/unit/test_omip2_forcing_scan.py
echo "scan_rc=$?"
echo "=== [2] convection config: convection-only OceanPhysicsConfig builds + runs on small latlon model ==="
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $PY -c "
import jax.numpy as jnp, numpy as np
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig, EnhancedDiffusionConfig
phys = OceanPhysicsConfig(
    vertical_mixing=VerticalMixingConfig(scheme='none'),
    lateral_mixing=LateralMixingConfig(scheme='none'),
    surface_forcing=SurfaceForcingConfig(scheme='none'),
    bottom_drag=BottomDragConfig(scheme='none'),
    convection=OceanConvectionConfig(scheme='enhanced_diffusion', enhanced_diffusion=EnhancedDiffusionConfig(K_conv=1.0,K_bg=1e-5)),
    shortwave_penetration=None)
g=create_latlon_grid(n_lat=12,n_lon=24); z=create_ocean_z_star(n_levels=4,H_max=4000.0)
s=rest_state_latlon_cgrid_ocean(g,z,T_water_init_C=18.0,T_deep=2.0,S_uniform=35.0,H_max=4000.0)
# statically unstable surface
T=np.asarray(s.T.data).copy(); T[:,:,0]=0.0; T[:,:,1]=10.0
s=s._replace(T=s.T.replace(data=jnp.asarray(T)))
cfg=LatLonCGridOceanConfig(A_h=1e4,B_h=0.0,K_h=0.0,K_bih=0.0,bottom_drag_r=0.0,n_barotropic_substeps=3,barotropic_solver='explicit_substep',physics=phys,implicit_vertical_mixing=True)
m=LatLonCGridOceanModel(g,z,cfg)
s2=m.step(s,300.0)
import jax; s2=jax.block_until_ready(s2)
print('CONVECTION-ONLY config step OK, finite=', bool(jnp.all(jnp.isfinite(s2.T.data))))
" 2>&1 | grep -vE "slow_operation|constant.fold|XLA_FLAGS|isn.t necessarily|If you.d like|^E[0-9]" | tail -5
echo "conv_rc=$?"; echo DONE; date
